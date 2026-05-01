import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import joblib
import shap
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PolynomialFeatures
from sklearn.svm import SVR
import random
from mrmr import mrmr_regression
from itertools import combinations

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

print("\nmRMR筛选的14个关键特征:")
print(selected_features)

selected_features = [
    'HOMO', 'ODI_LUMO', 'Farthest_Distance', 'Mol_Size_Short', 'Length_Ratio','dipole moment',
    'Excitation energie'
]

X_filtered = X_filtered[selected_features]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_filtered), columns=X_filtered.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)

def objective(trial):
    kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])

    C = trial.suggest_float('C', 1e-2, 1e4, log=True)
    epsilon = trial.suggest_float('epsilon', 1e-4, 1.0, log=True)

    gamma = 'scale'
    if kernel in ['rbf', 'poly', 'sigmoid']:
        gamma = trial.suggest_float('gamma', 1e-5, 1e2, log=True)

    degree = 3
    if kernel == 'poly':
        degree = trial.suggest_int('degree', 2, 5)

    coef0 = 0.0
    if kernel == 'sigmoid':
        coef0 = trial.suggest_float('coef0', -1.0, 1.0)

    params = {
        'C': C,
        'epsilon': epsilon,
        'kernel': kernel,
        'gamma': gamma,
        'degree': degree,
        'coef0': coef0,
        'max_iter': 10000,
        'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True),
        'cache_size': 1000
    }

    model = SVR(**params)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X_scaled):
        X_train_fold, X_val_fold = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        try:
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            rmse_scores.append(rmse)
        except:
            rmse_scores.append(float('inf'))

    return np.mean(rmse_scores)

study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)
study.optimize(objective, n_trials=200, show_progress_bar=True)

print("最佳参数:", study.best_params)
print(f"最佳 RMSE: {study.best_value:.4f}")

final_model = SVR(**study.best_params)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

print("\n优化后模型性能:")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f} nm")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} nm")

test_results = pd.DataFrame({
    'Actual': y_test,
    'SVM_Pred': y_pred
})
test_results.to_csv('svm_predictions.csv', index=False)

sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 100, 'savefig.dpi': 300})

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('SVM Predictions vs Actual')
plt.grid(True)
plt.tight_layout()
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(14, 10))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('SVM Prediction Residuals')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
sns.histplot(residuals, kde=True, bins=20)
plt.title('Prediction Error Distribution')
plt.xlabel('Prediction Error')
plt.tight_layout()
plt.show()

if study.best_params['kernel'] == 'linear':
    print("\n特征重要性（线性核系数）:")
    feature_importance = pd.DataFrame({
        'feature': X_filtered.columns,
        'importance': final_model.coef_[0]
    }).sort_values('importance', ascending=False)
    print(feature_importance)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('SVM Feature Importance (Linear Kernel)')
    plt.tight_layout()
    plt.show()
else:
    print("\n注意：非线性SVM核不支持直接的特征重要性分析")

joblib.dump(final_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selected_features, 'selected_features.pkl')
joblib.dump(train_autocorr_features, 'train_autocorr_features.pkl')
joblib.dump(feature_columns, 'original_features.pkl')

print("\n已保存模型和预处理对象:")
print("- svm_model.pkl")
print("- scaler.pkl")
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
result_df.to_csv('CatBoost_predictions.csv', index=False)
print("\n预测结果已保存至 CatBoost_predictions.csv")