import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# =============================================
# 0. 시드 및 환경 설정
# =============================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# =============================================
# 1. 데이터 준비
# =============================================
df = yf.download('BTC-USD', start='2020-01-01', end='2026-01-01', interval='1d')
close_prices = df['Close'].dropna().values.squeeze()
returns = pd.Series(close_prices).pct_change().dropna().values.reshape(-1, 1)
close_prices = close_prices[1:]  # returns와 길이 맞추기

scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

SEQ_LENGTH = 30

def create_sequences(data, prices, seq_length):
    xs, ys, price_seq = [], [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(1 if data[i + seq_length] > 0 else 0)
        price_seq.append(prices[i + seq_length])
    return np.array(xs), np.array(ys), np.array(price_seq)

X, y, future_prices = create_sequences(returns_scaled, close_prices, SEQ_LENGTH)
split = int(len(X) * 0.8)

X_train = torch.FloatTensor(X[:split]).to(device)
X_test  = torch.FloatTensor(X[split:]).to(device)
y_train = torch.FloatTensor(y[:split]).to(device)
y_test  = torch.FloatTensor(y[split:]).to(device)

test_prices = future_prices[split:]  # 테스트 구간의 실제 가격

# =============================================
# 2. 모델 정의
# =============================================
class DLinear(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.linear = nn.Linear(seq_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.squeeze(-1)
        return self.sigmoid(self.linear(x))


class iTransformerLite(nn.Module):
    def __init__(self, seq_len, d_model=64, nhead=4):
        super().__init__()
        self.enc_embedding = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.enc_embedding(x)
        x = self.transformer_encoder(x)
        return self.sigmoid(self.fc(x[:, 0, :]))

# =============================================
# 3. 학습 및 예측
# =============================================
def train_and_eval(model, name, epochs=100):
    print(f"\n[{name}] learning... (epochs={epochs})")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        y_prob = model(X_test).squeeze().cpu().numpy()
        y_true = y_test.cpu().numpy()

    return y_true, y_prob

# =============================================
# 4. 평가 지표
# =============================================
def calculate_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_idx = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        if np.any(bin_idx):
            delta = np.abs(np.mean(y_true[bin_idx]) - np.mean(y_prob[bin_idx]))
            ece += delta * (np.sum(bin_idx) / len(y_true))
    return ece

def evaluate_classification(y_true, y_prob, threshold=0.5, name="Model"):
    y_pred = (y_prob >= threshold).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    ece  = calculate_ece(y_true, y_prob)
    print(f"\n{'='*40}")
    print(f"[{name}] Classification Performance (threshold={threshold})")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ECE       : {ece:.4f}")
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "ece": ece, "y_pred": y_pred}

# =============================================
# 5. 매매 시뮬레이션
# =============================================
def simulate_trading(y_prob, prices, threshold=0.5, name="Model"):
    """
    threshold 이상이면 매수(롱), 미만이면 홀드(포지션 없음).
    수익률 = 실제 다음날 가격 변동
    """
    n = len(prices)
    signals = (y_prob >= threshold).astype(int)

    # 실제 일별 수익률 (테스트 구간)
    price_returns = np.diff(prices) / prices[:-1]  # 길이 n-1

    # 전략 수익률: signal[t] 기준으로 t+1일 수익률 적용
    strategy_returns = signals[:-1] * price_returns  # 길이 n-1

    # 누적 수익률
    cumulative_strategy = np.cumprod(1 + strategy_returns)
    cumulative_bnh      = np.cumprod(1 + price_returns)   # Buy & Hold

    total_return   = cumulative_strategy[-1] - 1
    bnh_return     = cumulative_bnh[-1] - 1
    n_trades       = np.sum(signals[:-1])
    win_trades     = np.sum((signals[:-1] == 1) & (price_returns > 0))
    win_rate       = win_trades / n_trades if n_trades > 0 else 0

    print(f"\n[{name}] Result of Trading Simulation")
    print(f"  Total Tradings   : {n_trades}회")
    print(f"  win_rate           : {win_rate:.2%}")
    print(f"  total_return : {total_return:.2%}")
    print(f"  Buy&Hold return  : {bnh_return:.2%}")

    return {
        "cumulative_strategy": cumulative_strategy,
        "cumulative_bnh":      cumulative_bnh,
        "price_returns":       price_returns,
        "strategy_returns":    strategy_returns,
        "signals":             signals,
        "total_return":        total_return,
        "bnh_return":          bnh_return,
        "n_trades":            n_trades,
        "win_rate":            win_rate,
    }

# =============================================
# 6. 실행
# =============================================
true_d, prob_d = train_and_eval(DLinear(SEQ_LENGTH),           "DLinear")
true_i, prob_i = train_and_eval(iTransformerLite(SEQ_LENGTH),  "iTransformer-Lite")

THRESHOLD = 0.5
metrics_d = evaluate_classification(true_d, prob_d, THRESHOLD, "DLinear")
metrics_i = evaluate_classification(true_i, prob_i, THRESHOLD, "iTransformer-Lite")

sim_d = simulate_trading(prob_d, test_prices, THRESHOLD, "DLinear")
sim_i = simulate_trading(prob_i, test_prices, THRESHOLD, "iTransformer-Lite")

# =============================================
# 7. 종합 시각화 (5개 패널)
# =============================================
fig = plt.figure(figsize=(20, 22))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

colors = {"DLinear": "#2196F3", "iTransformer": "#FF5722", "BnH": "#4CAF50"}

# ── Panel 1: Calibration Curve ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
p_true_d, p_pred_d = calibration_curve(true_d, prob_d, n_bins=10)
p_true_i, p_pred_i = calibration_curve(true_i, prob_i, n_bins=10)
ax1.plot(p_pred_d, p_true_d, "s-", color=colors["DLinear"],
         label=f"DLinear (ECE={metrics_d['ece']:.4f})", lw=2)
ax1.plot(p_pred_i, p_true_i, "o-", color=colors["iTransformer"],
         label=f"iTransformer (ECE={metrics_i['ece']:.4f})", lw=2)
ax1.set_xlabel("Confidence (Predicted Probability)")
ax1.set_ylabel("Actual Accuracy")
ax1.set_title("Calibration Curve", fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── Panel 2: 예측 확률 분포 ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(prob_d, bins=30, alpha=0.6, color=colors["DLinear"],    label="DLinear",       density=True)
ax2.hist(prob_i, bins=30, alpha=0.6, color=colors["iTransformer"], label="iTransformer", density=True)
ax2.axvline(THRESHOLD, color='red', linestyle='--', lw=1.5, label=f"Threshold={THRESHOLD}")
ax2.set_xlabel("Predicted Probability (↑ = bullish forecast)")
ax2.set_ylabel("Density")
ax2.set_title("Prediction Probability Distribution", fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Panel 3: 누적 수익률 비교 ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :])
x_axis = np.arange(len(sim_d["cumulative_strategy"]))
ax3.plot(x_axis, sim_d["cumulative_bnh"],            color=colors["BnH"],
         lw=2,   linestyle="--", label=f"Buy & Hold ({sim_d['bnh_return']:.1%})")
ax3.plot(x_axis, sim_d["cumulative_strategy"],       color=colors["DLinear"],
         lw=2,   label=f"DLinear Strategy ({sim_d['total_return']:.1%})")
ax3.plot(x_axis, sim_i["cumulative_strategy"],       color=colors["iTransformer"],
         lw=2,   label=f"iTransformer Strategy ({sim_i['total_return']:.1%})")
ax3.axhline(1.0, color='gray', linestyle=':', lw=1)
ax3.set_xlabel("Test Days")
ax3.set_ylabel("Cumulative Return (Base=1.0)")
ax3.set_title("Trading Simulation: Cumulative Returns", fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# ── Panel 4: 분류 성능 지표 Bar Chart ─────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
metric_names = ["Accuracy", "Precision", "Recall", "F1"]
vals_d = [metrics_d["acc"], metrics_d["prec"], metrics_d["rec"], metrics_d["f1"]]
vals_i = [metrics_i["acc"], metrics_i["prec"], metrics_i["rec"], metrics_i["f1"]]
x = np.arange(len(metric_names))
w = 0.35
bars_d = ax4.bar(x - w/2, vals_d, w, label="DLinear",      color=colors["DLinear"],      alpha=0.85)
bars_i = ax4.bar(x + w/2, vals_i, w, label="iTransformer", color=colors["iTransformer"], alpha=0.85)
for bar in bars_d:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8)
for bar in bars_i:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8)
ax4.set_xticks(x)
ax4.set_xticklabels(metric_names)
ax4.set_ylim(0, 1.1)
ax4.set_ylabel("Score")
ax4.set_title("Classification Metrics Comparison", fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# ── Panel 5: 매매 시뮬레이션 요약 Table ───────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')
table_data = [
    ["Model",              "DLinear",                         "iTransformer"],
    ["Total trading times",      f"{sim_d['n_trades']}times",          f"{sim_i['n_trades']}times"],
    ["win rate",              f"{sim_d['win_rate']:.2%}",        f"{sim_i['win_rate']:.2%}"],
    ["total return",       f"{sim_d['total_return']:.2%}",    f"{sim_i['total_return']:.2%}"],
    ["B&H return",        f"{sim_d['bnh_return']:.2%}",      "-"],
    ["Accuracy",          f"{metrics_d['acc']:.4f}",         f"{metrics_i['acc']:.4f}"],
    ["F1 Score",          f"{metrics_d['f1']:.4f}",          f"{metrics_i['f1']:.4f}"],
    ["ECE (↓ is good)",      f"{metrics_d['ece']:.4f}",         f"{metrics_i['ece']:.4f}"],
]
tbl = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#37474F')
        cell.set_text_props(color='white', fontweight='bold')
    elif col == 1:
        cell.set_facecolor('#E3F2FD')
    elif col == 2:
        cell.set_facecolor("#C0E946")
ax5.set_title("Simulation Summary", fontsize=13, fontweight='bold', pad=10)

fig.suptitle("BTC Direction Prediction: DLinear vs iTransformer-Lite\n(Trading Simulation & Calibration Analysis)",
             fontsize=15, fontweight='bold', y=0.98)

plt.savefig("/btc_simulation_result.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n결과 이미지 저장됨.")