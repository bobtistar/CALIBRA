import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

# 1. M1 Pro MPS 장치 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 데이터 준비 (비트코인 수익률)
df = yf.download('BTC-USD', start='2020-01-01', end='2026-01-01', interval='1d')
returns = df['Close'].pct_change().dropna().values.reshape(-1, 1)

# 데이터 정규화
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

# 슬라이딩 윈도우 데이터 생성 (30일 보고 1일 예측)
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = 1 if data[i + seq_length] > 0 else 0  # 분류 문제로 변환 (상승=1, 하락=0)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 30
X, y = create_sequences(returns_scaled, SEQ_LENGTH)

# Train/Test 분리 (시계열이므로 순서 유지)
split = int(len(X) * 0.8)
X_train, X_test = torch.FloatTensor(X[:split]).to(device), torch.FloatTensor(X[split:]).to(device)
y_train, y_test = torch.FloatTensor(y[:split]).to(device), torch.FloatTensor(y[split:]).to(device)

# 3. iTransformer-Lite 모델 정의
class iTransformerLite(nn.Module):
    def __init__(self, seq_len, d_model=64, nhead=4):
        super().__init__()
        # iTransformer의 핵심: 시간 차원을 특징(Feature)으로 취급하여 인버전
        self.enc_embedding = nn.Linear(seq_len, d_model) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, seq_len, 1] -> [batch, 1, seq_len] (Inversion)
        x = x.permute(0, 2, 1) 
        x = self.enc_embedding(x) # [batch, 1, d_model]
        x = self.transformer_encoder(x)
        x = self.fc(x[:, 0, :])
        return self.sigmoid(x)

model = iTransformerLite(SEQ_LENGTH).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 모델 학습
print("학습 시작...")
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

# 5. 예측 및 Calibration 측정
model.eval()
with torch.no_grad():
    y_prob = model(X_test).squeeze().cpu().numpy()
    y_true = y_test.cpu().numpy()

# Reliability Diagram 시각화
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
plt.plot(prob_pred, prob_true, "s-", label="iTransformer-Lite")
plt.xlabel("Confidence (Predicted Probability)")
plt.ylabel("Accuracy (Actual Fraction of Positives)")
plt.title("Reliability Diagram: Real Model on BTC Data")
plt.legend()
plt.grid(True)
plt.show()