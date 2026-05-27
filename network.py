# @Version: 1.0
# @Author : 林子健
# @File : network.py
# @Time : 2026/3/1
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('modified_data.csv')

target_cols = ['dipole moment', 'Excitation energie', 'ODI_LUMO', 'HOMO',
               'Farthest_Distance', 'Mol_Size_Short', 'Length_Ratio']

data = data.dropna(subset=['SMILES'] + target_cols)

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((2048,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp)

print("正在计算分子指纹...")
X_fp = np.array([get_fingerprint(smi) for smi in data['SMILES']])
y_targets = data[target_cols].values

target_scaler = StandardScaler()
y_targets_scaled = target_scaler.fit_transform(y_targets)

joblib.dump(target_scaler, 'surrogate_target_scaler.pkl')

X_train, X_test, y_train, y_test = train_test_split(X_fp, y_targets_scaled, test_size=0.1, random_state=42)

class SurrogateNN(nn.Module):
    def __init__(self):
        super(SurrogateNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # 输出 7 个特征
        )

    def forward(self, x):
        return self.network(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SurrogateNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)

epochs = 150
batch_size = 64
print(f"开始训练代理模型，使用设备: {device}...")

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_t.size()[0])

    for i in range(0, X_train_t.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train_t[indices], y_train_t[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 30 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'surrogate_model_weights.pth')
print("代理模型训练完成！已保存为 'surrogate_model_weights.pth'")