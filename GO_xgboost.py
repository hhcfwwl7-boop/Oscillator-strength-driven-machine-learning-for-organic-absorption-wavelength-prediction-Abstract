import joblib
import pandas as pd
import seaborn as sns
import xgboost as xgb
import numpy as np
import optuna
import shap
from matplotlib import pyplot as plt
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
from itertools import combinations
import random
from mrmr import mrmr_regression

random.seed(42)
np.random.seed(42)

data = pd.read_csv('modified_data.csv')

feature_columns = ['dipole moment',
                   'Excitation energie', 'solvent', 'ODI_HOMO',
                   'ODI_LUMO', 'ODI_Mean', 'ODI_Std', 'HOMO',
                   'LUMO', 'HOMO_LUMO_Gap', 'Farthest_Distance',
                   'Mol_Radius', 'Mol_Size_Short', 'Mol_Size_L',
                   'Length_Ratio', 'ESPmin', 'ESPmax', 'Nu', 'Pi',
                   'MPI', 'Nonpolar_Area', 'Polar_Area']

target_column = 'abs'

X = data[feature_columns]
y = data[target_column]

print("\n正在生成原始特征相关性热力图...")
plt.figure(figsize=(16, 12))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 8})
plt.title("Correlation Heatmap of Original Molecular Descriptors", fontsize=14)
plt.tight_layout()

plt.savefig("correlation_heatmap_ABC.png", dpi=300, bbox_inches="tight")
print("✅ 原始特征热力图已保存为: correlation_heatmap_ABC.png")
plt.show()

def generate_enhanced_features(df, target_columns=None, degree=2):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=True,
        include_bias=False
    )
    poly_features = poly.fit_transform(df[target_columns])

    feature_names = []
    for i in range(1, degree + 1):
        feature_names += [
            "_x_".join(comb)
            for comb in combinations(target_columns, i)
        ]

    start_idx = len(target_columns)
    poly_df = pd.DataFrame(
        poly_features[:, start_idx:],
        columns=feature_names[start_idx:]
    )

    return pd.concat([df, poly_df], axis=1)

# 生成增强特征
try:
    X_enhanced = generate_enhanced_features(
        X,
        target_columns=['HOMO', 'LUMO'],
        degree=2
    )
    print(f"原始特征数: {X.shape[1]} → 增强后特征数: {X_enhanced.shape[1]}")
except Exception as e:
    print(f"特征工程失败: {str(e)}")
    exit()

def autocorrelation_filter(df, threshold=0.9, protected_features=None):
    if protected_features is None:
        protected_features = []

    if 'dipole moment' not in protected_features:
        protected_features.append('dipole moment')

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = []
    for column in upper.columns:
        if column in protected_features:
            continue

        if any(upper[column] > threshold):
            to_drop.append(column)

    to_drop = [col for col in to_drop if col not in protected_features]

    print(f"将删除的特征: {to_drop}")
    return df.drop(columns=to_drop)

protected_features = ['dipole moment', 'HOMO', 'LUMO', 'Excitation energie']

X_filtered = autocorrelation_filter(
    X_enhanced,
    threshold=0.9,
    protected_features=protected_features
)

print("自相关过滤后特征数:", X_filtered.shape[1])
print("保留的特征:", X_filtered.columns.tolist())

if 'dipole moment' not in X_filtered.columns:
    print("警告: dipole moment 特征在过滤过程中被意外删除!")
    X_filtered['dipole moment'] = X_enhanced['dipole moment']
    print("已手动添加 dipole moment 特征")

train_autocorr_features = X_filtered.columns.tolist()
print(f"训练数据保留的特征: {train_autocorr_features}")

selected_features = mrmr_regression(
    X=X_filtered,
    y=y,
    K=14
)

print("\n计算所有特征的重要性排序...")

base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    random_state=42,
    tree_method='hist'
)

temp_scaler = QuantileTransformer(n_quantiles=100, output_distribution='normal')
X_temp_scaled = pd.DataFrame(temp_scaler.fit_transform(X_filtered), columns=X_filtered.columns)

X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
    X_temp_scaled, y,
    test_size=0.2,
    random_state=42
)

base_model.fit(X_temp_train, y_temp_train)

importance_scores = base_model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': X_filtered.columns,
    'importance': importance_scores
}).sort_values('importance', ascending=False)

print("\n所有特征的重要性排序:")
print(feature_importance)

print("\nmRMR筛选的14个关键特征:")
print(selected_features)

selected_features = [
    'dipole moment','Excitation energie', 'ODI_LUMO','HOMO','Farthest_Distance',
    'Mol_Size_Short', 'Length_Ratio'
]
print(f"\n当前手动选择的特征：{selected_features}")

X_filtered = X_filtered[selected_features]

scaler = QuantileTransformer(n_quantiles=100, output_distribution='normal')
X_scaled = pd.DataFrame(scaler.fit_transform(X_filtered), columns=X_filtered.columns)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)

def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'alpha': trial.suggest_float('alpha', 1e-3, 100.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-3, 100.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'random_state': 42,
        'n_jobs': 1,
        'tree_method': 'hist',
    }

    model = xgb.XGBRegressor(
        **params,
        n_estimators=2000,
        early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=100, show_progress_bar=True)

final_model = xgb.XGBRegressor(
    **study.best_params,
    objective='reg:squarederror',
    eval_metric='rmse',
    n_estimators=2000,
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=1,
    tree_method='hist'
)
final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

y_pred = final_model.predict(X_test)
print("\n优化后模型性能:")
print(f"R²: {r2_score(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} nm")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} nm")

test_results = pd.DataFrame({
    'Actual': y_test,
    'XGBoost_Pred': y_pred
})
test_results.to_csv('xgb_predictions.csv', index=False)

sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 100, 'savefig.dpi': 300})

plt.figure(figsize=(10, 8))
sns.heatmap(X_filtered.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Molecular Descriptor Correlations")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 9))
sns.pairplot(pd.concat([X_filtered[['HOMO', 'dipole moment']], y], axis=1))
plt.suptitle("Molecular Descriptor Relationships", y=1.02)
plt.tight_layout()
plt.show()

feature_importance = final_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(8, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_filtered.columns)[sorted_idx])
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, bins=20)
plt.title('Prediction Error Distribution')
plt.xlabel('Prediction Error')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('XGBoost Predictions vs Actual')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Prediction Residuals')
plt.grid(True)
plt.tight_layout()
plt.show()

joblib.dump(final_model, 'xgboost_model.pkl')
joblib.dump(scaler, 'quantile_scaler.pkl')
joblib.dump(selected_features, 'selected_features.pkl')
joblib.dump(train_autocorr_features, 'train_autocorr_features.pkl')
joblib.dump(feature_columns, 'original_features.pkl')

print("\n已保存模型和预处理对象:")
print("- xgboost_model.pkl")
print("- quantile_scaler.pkl")
print("- selected_features.pkl")
print("- train_autocorr_features.pkl")
print("- original_features.pkl")

new_data = pd.read_csv('data.csv')

try:
    X_new_base = new_data[feature_columns]
    X_new_enhanced = generate_enhanced_features(
        X_new_base,
        target_columns=['HOMO', 'LUMO'],
        degree=2
    )
    print(f"新数据增强后特征数: {X_new_enhanced.shape[1]}")
except Exception as e:
    print(f"新数据特征工程失败: {str(e)}")
    exit()

X_new_filtered = autocorrelation_filter(X_new_enhanced, threshold=0.9)
print(f"新数据自相关筛选后特征数: {X_new_filtered.shape[1]}")

train_features = X_filtered.columns.tolist()

missing_features = set(train_features) - set(X_new_filtered.columns)
if missing_features:
    print(f"警告: 新数据缺少特征 {missing_features}")
    for feat in missing_features:
        X_new_filtered[feat] = 0
    X_new_filtered = X_new_filtered[train_features]

X_new_selected = X_new_filtered[selected_features]

X_new_scaled = pd.DataFrame(
    scaler.transform(X_new_selected),
    columns=X_new_selected.columns
)

new_predictions = final_model.predict(X_new_scaled)

print("\n新数据预测结果:")
for i, sample in enumerate(new_data['SampleName']):
    print(f"{sample}: {new_predictions[i]:.2f} nm")

print("\n诊断信息:")
print("1. 特征一致性检查:")
print(f"  训练特征数: {len(train_features)} | 新数据特征数: {len(X_new_filtered.columns)}")
print(f"  共同特征: {set(train_features) & set(X_new_filtered.columns)}")

print("\n2. 特征范围对比:")
for feat in selected_features:
    train_min = X_filtered[feat].min()
    train_max = X_filtered[feat].max()
    new_val = X_new_selected[feat].iloc[0]

    status = "✅" if (new_val >= train_min) and (new_val <= train_max) else "❌"
    print(f"  {feat}: 训练范围[{train_min:.2f}, {train_max:.2f}] | 新值={new_val:.2f} {status}")

result_df = new_data[['SampleName']].copy()
result_df['Predicted_abs'] = new_predictions
result_df.to_csv('predictions.csv', index=False)
print("\n预测结果已保存至 predictions.csv")
