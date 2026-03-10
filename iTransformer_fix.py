import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

# ============================================================
# 변경 이력:
#   - epochs = 100으로 수정
#   - ECE 추가
#   - seed 추가
#   - VolatilityEstimator 추가 (RQ2, RQ3용)
#   - AdaptiveTemperatureScaling 모듈 추가
#   - 변동성 구간별 ECE 비교 함수 추가 (evaluate_by_volatility_regime)
#   - 시각화: Calibration + Volatility Regime ECE 비교 subplot 추가
# ============================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 1. 환경 설정 (1660 Super - CUDA / M1 Pro - MPS)
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# 2. 데이터 준비 (BTC 수익률)
df = yf.download('BTC-USD', start='2021-01-01', end='2026-01-01', interval='1d')

if df.empty:
    raise ValueError("yfinance에서 데이터를 받아오지 못했습니다. 네트워크 또는 버전을 확인하세요.")

print(f"데이터 수신 성공: {df.shape}")
returns_raw = df['Close'].pct_change().dropna().values.reshape(-1, 1)

scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns_raw)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(1 if data[i + seq_length] > 0 else 0)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 30
X, y = create_sequences(returns_scaled, SEQ_LENGTH)
split = int(len(X) * 0.8)

X_train = torch.FloatTensor(X[:split]).to(device)
X_test  = torch.FloatTensor(X[split:]).to(device)
y_train = torch.FloatTensor(y[:split]).to(device)
y_test  = torch.FloatTensor(y[split:]).to(device)

# ── 변동성 추정 ──────────────────────────────────────────────

class VolatilityEstimator:
    """Realized Volatility (Rolling Std) 기반 변동성 추정기"""
    def __init__(self, window=20):
        self.window = window

    def compute(self, returns_1d):
        """
        returns_1d: 1D numpy array (raw 수익률)
        returns   : 각 시점의 Rolling Std (변동성)
        """
        vol = np.array([
            np.std(returns_1d[max(0, i - self.window):i]) if i > 0 else 0.0
            for i in range(len(returns_1d))
        ])
        return vol

vol_estimator = VolatilityEstimator(window=20)
returns_1d    = returns_raw.flatten()
volatility    = vol_estimator.compute(returns_1d)          # [T]

# 시퀀스에 대응하는 변동성 (각 시퀀스의 마지막 시점 기준)
vol_seq = volatility[SEQ_LENGTH:]                          # [N]
vol_test = vol_seq[split:]                                 # test 구간

# ── 모델 정의 ────────────────────────────────────────────────

class DLinear(nn.Module):
    """가장 단순한 선형 Baseline 모델"""
    def __init__(self, seq_len):
        super().__init__()
        self.linear  = nn.Linear(seq_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, Seq_len, 1] -> [B, Seq_len]
        return self.sigmoid(self.linear(x.squeeze(-1)))

    def get_logit(self, x):
        return self.linear(x.squeeze(-1))


class iTransformerLite(nn.Module):
    """Inverted Transformer (시간 축을 Feature로 취급)"""
    def __init__(self, seq_len, d_model=64, nhead=4):
        super().__init__()
        self.enc_embedding    = nn.Linear(seq_len, d_model)
        encoder_layer         = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc      = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)          # Inversion: [B, 1, Seq_len]
        x = self.enc_embedding(x)        # [B, 1, d_model]
        x = self.transformer_encoder(x)
        return self.sigmoid(self.fc(x[:, 0, :]))

    def get_logit(self, x):
        x = x.permute(0, 2, 1)
        x = self.enc_embedding(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, 0, :])


# ── Adaptive Temperature Scaling ─────────────────────────────

class AdaptiveTemperatureScaling(nn.Module):
    """
    변동성(Volatility)을 입력받아 최적 Temperature를 학습하는 모듈.
    고변동성 구간 → T 증가 → Confidence 분산 완화 (Calibration 개선)
    """
    def __init__(self, vol_dim=1, hidden=16, min_temp=0.1, max_temp=5.0):
        super().__init__()
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.temp_net = nn.Sequential(
            nn.Linear(vol_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Softplus()   # T > 0 보장
        )

    def forward(self, logits, volatility):
        """
        logits    : [B, 1] raw logit (sigmoid 적용 전)
        volatility: [B, 1] 해당 시점 변동성
        """
        T = self.temp_net(volatility)
        T = torch.clamp(T, self.min_temp, self.max_temp)
        calibrated_prob = torch.sigmoid(logits / T)
        return calibrated_prob, T


# ── 학습 및 평가 함수 ─────────────────────────────────────────

def train_and_eval(model_instance, name):
    print(f"\n[{name}] 학습 시작...")
    model     = model_instance.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss    = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_prob = model(X_test).squeeze().cpu().numpy()
        y_true = y_test.cpu().numpy()

    return model, y_true, y_prob


def train_adaptive_temperature(base_model, name, n_epochs=200):
    """
    base_model을 동결하고 AdaptiveTemperatureScaling만 학습.
    변동성 정보를 활용해 구간별 최적 T를 학습.
    """
    print(f"\n[{name} + ATS] Adaptive Temperature 학습 시작...")
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    ats       = AdaptiveTemperatureScaling().to(device)
    optimizer = optim.Adam(ats.parameters(), lr=0.005)
    criterion = nn.BCELoss()

    # test 변동성 텐서
    vol_tensor = torch.FloatTensor(vol_test).unsqueeze(1).to(device)  # [N_test, 1]

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        with torch.no_grad():
            logits = base_model.get_logit(X_test)          # [N_test, 1]
        prob, _ = ats(logits, vol_tensor)
        loss = criterion(prob.squeeze(), y_test)
        loss.backward()
        optimizer.step()

    ats.eval()
    with torch.no_grad():
        logits     = base_model.get_logit(X_test)
        prob, temp = ats(logits, vol_tensor)
        y_prob_ats = prob.squeeze().cpu().numpy()
        y_true     = y_test.cpu().numpy()
        temp_vals  = temp.squeeze().cpu().numpy()

    return y_true, y_prob_ats, temp_vals


def calculate_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        idx = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if np.any(idx):
            delta = np.abs(np.mean(y_true[idx]) - np.mean(y_prob[idx]))
            ece  += delta * (np.sum(idx) / len(y_true))
    return ece


def evaluate_by_volatility_regime(y_true, y_prob, volatility, threshold, label=""):
    """변동성 구간(저/고)별 ECE를 출력"""
    low_idx  = volatility < threshold
    high_idx = volatility >= threshold
    ece_low  = calculate_ece(y_true[low_idx],  y_prob[low_idx])  if np.any(low_idx)  else float('nan')
    ece_high = calculate_ece(y_true[high_idx], y_prob[high_idx]) if np.any(high_idx) else float('nan')
    print(f"  [{label}] 저변동성 ECE: {ece_low:.4f} | 고변동성 ECE: {ece_high:.4f}")
    return ece_low, ece_high


# ── 실행 ─────────────────────────────────────────────────────

# 기본 모델 학습
model_d, true_d, prob_d = train_and_eval(DLinear(SEQ_LENGTH),       "DLinear")
model_i, true_i, prob_i = train_and_eval(iTransformerLite(SEQ_LENGTH), "iTransformer-Lite")

# Adaptive Temperature Scaling 적용 (iTransformer 대상)
true_ats, prob_ats, temp_vals = train_adaptive_temperature(model_i, "iTransformer-Lite")

# ECE 계산
ece_d   = calculate_ece(true_d,   prob_d)
ece_i   = calculate_ece(true_i,   prob_i)
ece_ats = calculate_ece(true_ats, prob_ats)

print(f"\n{'='*50}")
print(f"  DLinear              ECE: {ece_d:.4f}")
print(f"  iTransformer-Lite    ECE: {ece_i:.4f}")
print(f"  iTransformer + ATS   ECE: {ece_ats:.4f}")
print(f"{'='*50}")

# 변동성 구간별 ECE 비교 (RQ2 / RQ3)
threshold = np.median(vol_test)
print(f"\n[변동성 구간별 ECE 비교] threshold={threshold:.5f}")
ece_d_low,   ece_d_high   = evaluate_by_volatility_regime(true_d,   prob_d,   vol_test, threshold, "DLinear")
ece_i_low,   ece_i_high   = evaluate_by_volatility_regime(true_i,   prob_i,   vol_test, threshold, "iTransformer")
ece_ats_low, ece_ats_high = evaluate_by_volatility_regime(true_ats, prob_ats, vol_test, threshold, "iTransformer+ATS")

# ── 시각화 ───────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Calibration Analysis: DLinear vs iTransformer-Lite vs ATS", fontsize=14)

# --- subplot 1: Reliability Diagram ---
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")

p_true_d, p_pred_d = calibration_curve(true_d, prob_d, n_bins=10)
ax.plot(p_pred_d, p_true_d, "s-", label=f"DLinear (ECE: {ece_d:.4f})")

p_true_i, p_pred_i = calibration_curve(true_i, prob_i, n_bins=10)
ax.plot(p_pred_i, p_true_i, "o-", label=f"iTransformer (ECE: {ece_i:.4f})")

p_true_ats, p_pred_ats = calibration_curve(true_ats, prob_ats, n_bins=10)
ax.plot(p_pred_ats, p_true_ats, "^-", label=f"iTransformer+ATS (ECE: {ece_ats:.4f})")

ax.set_xlabel("Confidence")
ax.set_ylabel("Actual Accuracy")
ax.set_title("Reliability Diagram")
ax.legend(fontsize=8)
ax.grid(True)

# --- subplot 2: Volatility Regime ECE 비교 ---
ax2 = axes[1]
x      = np.arange(3)
labels = ["DLinear", "iTransformer", "iTransformer\n+ATS"]
low_eces  = [ece_d_low,  ece_i_low,  ece_ats_low]
high_eces = [ece_d_high, ece_i_high, ece_ats_high]

w = 0.35
ax2.bar(x - w/2, low_eces,  w, label="Low Volatility",  color="steelblue")
ax2.bar(x + w/2, high_eces, w, label="High Volatility", color="tomato")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("ECE")
ax2.set_title("ECE by Volatility Regime")
ax2.legend()
ax2.grid(True, axis='y')

# --- subplot 3: Adaptive Temperature over Time ---
ax3 = axes[2]
ax3.plot(temp_vals, color="darkorange", linewidth=0.8, label="Adaptive Temperature T")
ax3.axhline(y=1.0, color="k", linestyle="--", linewidth=0.8, label="T=1 (No scaling)")

ax3_twin = ax3.twinx()
ax3_twin.fill_between(range(len(vol_test)), vol_test, alpha=0.2, color="gray", label="Volatility")
ax3_twin.set_ylabel("Volatility", color="gray")

ax3.set_xlabel("Test Time Step")
ax3.set_ylabel("Temperature T")
ax3.set_title("Adaptive Temperature vs Volatility")
ax3.legend(loc="upper left", fontsize=8)
ax3_twin.legend(loc="upper right", fontsize=8)
ax3.grid(True)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/calibration_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n시각화 저장 완료: calibration_analysis.png")