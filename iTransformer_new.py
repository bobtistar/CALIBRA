import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

#epochs = 100으로 수정, ECE 추가, seed 추가

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 1. 환경 설정 (1660 Super - CUDA / M1 Pro - MPS)
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# 2. 데이터 준비 (BTC 수익률)
df = yf.download('BTC-USD', start='2021-01-01', end='2026-01-01', interval='1d')
returns = df['Close'].pct_change().dropna().values.reshape(-1, 1)

scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(1 if data[i + seq_length] > 0 else 0)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 30
X, y = create_sequences(returns_scaled, SEQ_LENGTH)
split = int(len(X) * 0.8)

X_train, X_test = torch.FloatTensor(X[:split]).to(device), torch.FloatTensor(X[split:]).to(device)
y_train, y_test = torch.FloatTensor(y[:split]).to(device), torch.FloatTensor(y[split:]).to(device)

# --- 모델 정의 ---

# 3. DLinear (가장 단순한 선형 모델)
class DLinear(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.linear = nn.Linear(seq_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [Batch, Seq_len, 1] -> [Batch, Seq_len]
        x = x.squeeze(-1)
        return self.sigmoid(self.linear(x))

# 4. iTransformer-Lite (복잡한 최신 모델)
class iTransformerLite(nn.Module):
    def __init__(self, seq_len, d_model=64, nhead=4):
        super().__init__()
        self.enc_embedding = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1) # Inversion
        x = self.enc_embedding(x)
        x = self.transformer_encoder(x)
        return self.sigmoid(self.fc(x[:, 0, :]))

# --- 학습 및 평가 함수 ---

def train_and_eval(model_class, name):
    print(f"\n[{name}] 학습 시작...")
    model = model_class.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        y_prob = model(X_test).squeeze().cpu().numpy()
        y_true = y_test.cpu().numpy()
    
    return y_true, y_prob

def calculate_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_idx = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        if np.any(bin_idx):
            delta = np.abs(np.mean(y_true[bin_idx]) - np.mean(y_prob[bin_idx]))
            ece += delta * (np.sum(bin_idx) / len(y_true))
    return ece

# 실행
true_d, prob_d = train_and_eval(DLinear(SEQ_LENGTH), "DLinear")
true_i, prob_i = train_and_eval(iTransformerLite(SEQ_LENGTH), "iTransformer-Lite")

ece_d = calculate_ece(true_d, prob_d)
ece_i = calculate_ece(true_i, prob_i)

# 시각화
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")

p_true_d, p_pred_d = calibration_curve(true_d, prob_d, n_bins=10)
plt.plot(p_pred_d, p_true_d, "s-", label=f"DLinear (ECE: {ece_d:.4f})")

p_true_i, p_pred_i = calibration_curve(true_i, prob_i, n_bins=10)
plt.plot(p_pred_i, p_true_i, "o-", label=f"iTransformer-Lite (ECE: {ece_i:.4f})")

plt.xlabel("Confidence")
plt.ylabel("Actual Accuracy")
plt.title("Comparison: DLinear vs iTransformer-Lite Calibration")
plt.legend()
plt.grid(True)
plt.show()