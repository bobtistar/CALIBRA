import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else
                      ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# =============================================
# 1. 데이터 준비
# =============================================
df = yf.download('BTC-USD', start='2020-01-01', end='2026-01-01', interval='1d')
close_prices = df['Close'].dropna().values.squeeze()
returns = pd.Series(close_prices).pct_change().dropna().values.reshape(-1, 1)
close_prices = close_prices[1:]

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

# Train / Val / Test 분할 (6:2:2)
# Val은 Calibration 학습에 사용, Test는 최종 평가
n = len(X)
train_end = int(n * 0.6)
val_end   = int(n * 0.8)

X_train_np = X[:train_end];   y_train_np = y[:train_end]
X_val_np   = X[train_end:val_end]; y_val_np = y[train_end:val_end]
X_test_np  = X[val_end:];     y_test_np  = y[val_end:]
test_prices = future_prices[val_end:]

X_train = torch.FloatTensor(X_train_np).to(device)
X_val   = torch.FloatTensor(X_val_np).to(device)
X_test  = torch.FloatTensor(X_test_np).to(device)
y_train = torch.FloatTensor(y_train_np).to(device)

# =============================================
# 2. 모델 정의
# =============================================
class DLinear(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.linear  = nn.Linear(seq_len, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x.squeeze(-1)))
    def get_logit(self, x):
        return self.linear(x.squeeze(-1))

class iTransformerLite(nn.Module):
    def __init__(self, seq_len, d_model=64, nhead=4):
        super().__init__()
        self.enc_embedding     = nn.Linear(seq_len, d_model)
        encoder_layer          = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc      = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.enc_embedding(x)
        x = self.transformer_encoder(x)
        return self.sigmoid(self.fc(x[:, 0, :]))
    def get_logit(self, x):
        x = x.permute(0, 2, 1)
        x = self.enc_embedding(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, 0, :])

# =============================================
# 3. 학습 및 logit / prob 추출
# =============================================
def train_model(model, name, epochs=100):
    print(f"\n[{name}] Training... (epochs={epochs})")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_train).squeeze(), y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")
    return model

def get_probs_and_logits(model, X_tensor):
    model.eval()
    with torch.no_grad():
        prob  = model(X_tensor).squeeze().cpu().numpy()
        logit = model.get_logit(X_tensor).squeeze().cpu().numpy()
    return prob, logit

# =============================================
# 4. Calibration 방법 3가지
# =============================================

# ── 4-1. Temperature Scaling ──────────────────────────────────────────────
class TemperatureScaling:
    """
    Learns optimal temperature T from val set logits.
    T > 1 pulls probabilities toward 0.5 (reduces overconfidence).
    """
    def __init__(self):
        self.T = 1.0

    def fit(self, logits_val, y_val):
        T = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=500)
        logits_t  = torch.FloatTensor(logits_val)
        labels_t  = torch.FloatTensor(y_val)
        criterion = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(logits_t / T, labels_t)
            loss.backward()
            return loss
        optimizer.step(closure)
        self.T = T.item()
        print(f"  [Temperature Scaling] Optimal T = {self.T:.4f}")

    def predict_proba(self, logits):
        return torch.sigmoid(torch.FloatTensor(logits) / self.T).numpy()

# ── 4-2. Platt Scaling ────────────────────────────────────────────────────
class PlattScaling:
    """
    Trains logistic regression on val set logits.
    Recalibrates via sigmoid(a * logit + b).
    """
    def __init__(self):
        self.lr = LogisticRegression()

    def fit(self, logits_val, y_val):
        self.lr.fit(logits_val.reshape(-1, 1), y_val)
        a = self.lr.coef_[0][0]
        b = self.lr.intercept_[0]
        print(f"  [Platt Scaling] a={a:.4f}, b={b:.4f}")

    def predict_proba(self, logits):
        return self.lr.predict_proba(logits.reshape(-1, 1))[:, 1]

# ── 4-3. Isotonic Regression ──────────────────────────────────────────────
class IsotonicCalibration:
    """
    Remaps probabilities via a monotone increasing function fitted on val set.
    Flexible but prone to overfitting with small datasets.
    """
    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds='clip')

    def fit(self, probs_val, y_val):
        self.ir.fit(probs_val, y_val)

    def predict_proba(self, probs):
        return self.ir.predict(probs)

# ── 4-4. Adaptive Scaling ─────────────────────────────────────────────────
class AdaptiveScaling:
    """
    Splits the probability range into bins and learns an independent
    temperature T per bin from the val set.
    More fine-grained than global Temperature Scaling.
    """
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.bin_edges = np.linspace(0, 1, n_bins + 1)
        self.temperatures = np.ones(n_bins)  # default T=1 per bin

    def fit(self, logits_val, y_val):
        probs_val = torch.sigmoid(torch.FloatTensor(logits_val)).numpy()
        for i in range(self.n_bins):
            idx = (probs_val > self.bin_edges[i]) & (probs_val <= self.bin_edges[i + 1])
            if np.sum(idx) < 5:
                continue  # skip bins with too few samples
            logits_bin = torch.FloatTensor(logits_val[idx])
            labels_bin = torch.FloatTensor(y_val[idx])
            T = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
            criterion = nn.BCEWithLogitsLoss()
            def closure():
                optimizer.zero_grad()
                loss = criterion(logits_bin / T, labels_bin)
                loss.backward()
                return loss
            optimizer.step(closure)
            self.temperatures[i] = max(T.item(), 0.1)  # clamp to avoid collapse
        print(f"  [Adaptive Scaling] Bin temperatures: {np.round(self.temperatures, 3)}")

    def predict_proba(self, logits):
        probs_raw = torch.sigmoid(torch.FloatTensor(logits)).numpy()
        probs_cal = probs_raw.copy()
        for i in range(self.n_bins):
            idx = (probs_raw > self.bin_edges[i]) & (probs_raw <= self.bin_edges[i + 1])
            if np.any(idx):
                calibrated = torch.sigmoid(
                    torch.FloatTensor(logits[idx]) / self.temperatures[i]
                ).numpy()
                probs_cal[idx] = calibrated
        return probs_cal

# =============================================
# 5. 평가 함수
# =============================================
def calculate_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        idx = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        if np.any(idx):
            ece += np.abs(np.mean(y_true[idx]) - np.mean(y_prob[idx])) * (np.sum(idx) / len(y_true))
    return ece

def evaluate(y_true, y_prob, label):
    y_pred = (y_prob >= 0.5).astype(int)
    ece  = calculate_ece(y_true, y_prob)
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"  {label:35s} | ECE={ece:.4f} | Acc={acc:.4f} | F1={f1:.4f}")
    return ece, acc, f1

# =============================================
# 6. 학습 + Calibration 실행
# =============================================
model_d = train_model(DLinear(SEQ_LENGTH),          "DLinear")
model_i = train_model(iTransformerLite(SEQ_LENGTH), "iTransformer-Lite")

# Val / Test 확률 & logit 추출
prob_d_val,  logit_d_val  = get_probs_and_logits(model_d, X_val)
prob_i_val,  logit_i_val  = get_probs_and_logits(model_i, X_val)
prob_d_test, logit_d_test = get_probs_and_logits(model_d, X_test)
prob_i_test, logit_i_test = get_probs_and_logits(model_i, X_test)
y_val_np_arr  = y_val_np.astype(float)
y_test_np_arr = y_test_np.astype(float)

# ── iTransformer Calibration 3종 ─────────────────────────────────────────
# Temperature Scaling
ts = TemperatureScaling()
ts.fit(logit_i_val, y_val_np_arr)
prob_i_ts = ts.predict_proba(logit_i_test)

# Platt Scaling
ps = PlattScaling()
ps.fit(logit_i_val, y_val_np_arr)
prob_i_ps = ps.predict_proba(logit_i_test)

# Isotonic Regression
iso = IsotonicCalibration()
iso.fit(prob_i_val, y_val_np_arr)
prob_i_iso = iso.predict_proba(prob_i_test)

# Adaptive Scaling
ada = AdaptiveScaling(n_bins=10)
ada.fit(logit_i_val, y_val_np_arr)
prob_i_ada = ada.predict_proba(logit_i_test)

# ── 결과 출력 ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  Calibration Results Comparison (Test Set)")
print("="*70)
evaluate(y_test_np_arr, prob_d_test,  "DLinear (No Calibration)")
evaluate(y_test_np_arr, prob_i_test,  "iTransformer (No Calibration)")
evaluate(y_test_np_arr, prob_i_ts,    "iTransformer + Temperature Scaling")
evaluate(y_test_np_arr, prob_i_ps,    "iTransformer + Platt Scaling")
evaluate(y_test_np_arr, prob_i_iso,   "iTransformer + Isotonic Regression")
evaluate(y_test_np_arr, prob_i_ada,   "iTransformer + Adaptive Scaling")

# =============================================
# 7. 시각화
# =============================================
fig = plt.figure(figsize=(20, 18))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

COLORS = {
    "DLinear":   "#2196F3",
    "No Cal":    "#FF5722",
    "Temp":      "#9C27B0",
    "Platt":     "#009688",
    "Isotonic":  "#FF9800",
    "Adaptive":  "#E91E63",
    "perfect":   "black",
}

configs = [
    ("DLinear (No Cal)",              prob_d_test,  COLORS["DLinear"],  "s-"),
    ("iTransformer (No Cal)",         prob_i_test,  COLORS["No Cal"],   "o-"),
    ("iTransformer + Temp Scaling",   prob_i_ts,    COLORS["Temp"],     "^-"),
    ("iTransformer + Platt Scaling",  prob_i_ps,    COLORS["Platt"],    "D-"),
    ("iTransformer + Isotonic",       prob_i_iso,   COLORS["Isotonic"], "v-"),
    ("iTransformer + Adaptive",       prob_i_ada,   COLORS["Adaptive"], "P-"),
]

# ── Panel 1: Calibration Curve (전체) ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot([0,1],[0,1], "k--", lw=1.5, label="Perfect Calibration")
for label, prob, color, marker in configs:
    ece = calculate_ece(y_test_np_arr, prob)
    pt, pp = calibration_curve(y_test_np_arr, prob, n_bins=10)
    ax1.plot(pp, pt, marker, color=color, lw=2, label=f"{label}\n(ECE={ece:.4f})", markersize=5)
ax1.set_xlabel("Confidence (Predicted Prob)")
ax1.set_ylabel("Actual Accuracy")
ax1.set_title("Calibration Curve: All Methods", fontsize=13, fontweight='bold')
ax1.legend(fontsize=7.5, loc='upper left')
ax1.grid(True, alpha=0.3)

# ── Panel 2: Probability Distribution 비교 ────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(prob_i_test, bins=30, alpha=0.5, color=COLORS["No Cal"],   label="No Calibration", density=True)
ax2.hist(prob_i_ts,   bins=30, alpha=0.5, color=COLORS["Temp"],     label="Temp Scaling",   density=True)
ax2.hist(prob_i_ps,   bins=30, alpha=0.5, color=COLORS["Platt"],    label="Platt Scaling",  density=True)
ax2.hist(prob_i_iso,  bins=30, alpha=0.5, color=COLORS["Isotonic"], label="Isotonic",       density=True)
ax2.hist(prob_i_ada,  bins=30, alpha=0.5, color=COLORS["Adaptive"], label="Adaptive",       density=True)
ax2.axvline(0.5, color='red', linestyle='--', lw=1.5, label="Threshold=0.5")
ax2.set_xlabel("Predicted Probability")
ax2.set_ylabel("Density")
ax2.set_title("iTransformer: Prob Distribution After Calibration", fontsize=13, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ── Panel 3: ECE 개선 Bar Chart ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
method_labels = ["DLinear\n(No Cal)", "iTransformer\n(No Cal)", "iTransformer\n+Temp", "iTransformer\n+Platt", "iTransformer\n+Isotonic", "iTransformer\n+Adaptive"]
ece_vals   = [calculate_ece(y_test_np_arr, p) for _, p, _, _ in configs]
bar_colors = [COLORS["DLinear"], COLORS["No Cal"], COLORS["Temp"], COLORS["Platt"], COLORS["Isotonic"], COLORS["Adaptive"]]
bars = ax3.bar(method_labels, ece_vals, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, ece_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
             f"{val:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
ax3.set_ylabel("ECE (lower is better)")
ax3.set_title("ECE Comparison Across Calibration Methods", fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, max(ece_vals) * 1.25)

# ── Panel 4: Temperature T 시각적 설명 ────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
logit_range = np.linspace(-5, 5, 300)
T_values = [0.5, 1.0, ts.T, 3.0]
T_labels  = ["T=0.5 (more extreme)", "T=1.0 (original)", f"T={ts.T:.2f} (optimal)", "T=3.0 (more flat)"]
T_colors  = ["#E91E63", "#607D8B", COLORS["Temp"], "#FF9800"]
T_styles  = ["--", "-", "-", "--"]
for T, lbl, col, ls in zip(T_values, T_labels, T_colors, T_styles):
    prob_curve = 1 / (1 + np.exp(-logit_range / T))
    lw = 2.5 if T == ts.T else 1.5
    ax4.plot(logit_range, prob_curve, color=col, lw=lw, linestyle=ls, label=lbl)
ax4.axhline(0.5, color='gray', linestyle=':', lw=1)
ax4.axvline(0.0, color='gray', linestyle=':', lw=1)
ax4.set_xlabel("Logit (raw model output)")
ax4.set_ylabel("Calibrated Probability")
ax4.set_title("Temperature Scaling: Effect of T on Sigmoid", fontsize=13, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# ── Panel 5 & 6: Calibration 전후 누적 수익률 ─────────────────────────────
def cum_returns(y_prob, prices, threshold=0.5):
    signals = (y_prob >= threshold).astype(int)
    price_returns = np.diff(prices) / prices[:-1]
    strategy_returns = signals[:-1] * price_returns
    return np.cumprod(1 + strategy_returns), np.cumprod(1 + price_returns)

ax5 = fig.add_subplot(gs[2, :])
cum_bnh = None
for label, prob, color, marker in configs:
    cum_strat, cum_bnh = cum_returns(prob, test_prices)
    total_ret = cum_strat[-1] - 1
    lw = 2.5 if "Temp" in label else 1.8
    ax5.plot(cum_strat, color=color, lw=lw, label=f"{label} ({total_ret:.1%})")

bnh_ret = cum_bnh[-1] - 1
ax5.plot(cum_bnh, color=COLORS["perfect"], lw=2, linestyle="--", label=f"Buy & Hold ({bnh_ret:.1%})")
ax5.axhline(1.0, color='gray', linestyle=':', lw=1)
ax5.set_xlabel("Test Days")
ax5.set_ylabel("Cumulative Return (Base=1.0)")
ax5.set_title("Trading Simulation: Cumulative Returns After Calibration", fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

fig.suptitle("iTransformer Calibration Analysis\nDLinear vs iTransformer (No Cal / Temp / Platt / Isotonic / Adaptive)",
             fontsize=15, fontweight='bold', y=0.99)

plt.savefig("/btc_calibration_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nCalibration analysis complete!")