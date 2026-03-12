import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score

SEED        = 42
N_BOOTSTRAP = 500   # ECE 95% CI 부트스트랩 횟수 (0 → 비활성화)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else
                      ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# =============================================
# 1. 데이터 준비
# =============================================
df = yf.download('BTC-USD', start='2020-01-01', end='2026-01-01', interval='1d')
df = df.dropna()
close_prices_full = df['Close'].values.squeeze()

# pct_change 대신 np.diff 사용 (동일 결과, 명시적)
raw_returns  = np.diff(close_prices_full) / close_prices_full[:-1]  # (N,)
close_prices = close_prices_full[1:]                                  # aligned (N,)
N            = len(raw_returns)

SEQ_LENGTH = 30
PATCH_SIZE = 5   # PatchTST: 30 // 5 = 6 patches

# ── [수정 P2] Scaler 누수 방지: train 구간만으로 fit ──────────────────────
# 시퀀스 n = N - SEQ_LENGTH, train_end = int(n * 0.6)
# 이에 대응하는 raw 상한 index = train_end + SEQ_LENGTH
_n_seq         = N - SEQ_LENGTH
_train_raw_end = int(_n_seq * 0.6) + SEQ_LENGTH   # exclusive

scaler         = StandardScaler()
scaler.fit(raw_returns[:_train_raw_end].reshape(-1, 1))  # train 구간만 fit
returns_scaled = scaler.transform(raw_returns.reshape(-1, 1))        # (N, 1)


def create_sequences(scaled_data, raw_data, prices, seq_length):
    """
    [수정 P3] 레이블 기준: raw_data[t] > 0 (양의 수익률).

    기존 코드의 scaled_data[t] > 0 문제:
      StandardScaler는 평균(≈ +0.2%/day)을 빼므로
      scaled > 0  ⟺  raw > mean(raw)  ('평균 초과 여부')
      → Trading에서 필요한 '양의 수익' 예측과 불일치.
    """
    xs, ys, price_seq, vol_seq = [], [], [], []
    for i in range(len(scaled_data) - seq_length):
        xs.append(scaled_data[i : i + seq_length])
        ys.append(1 if raw_data[i + seq_length] > 0 else 0)   # FIX: raw > 0
        price_seq.append(prices[i + seq_length])
        vol_seq.append(np.std(raw_data[i : i + seq_length]))   # window vol
    return (np.array(xs), np.array(ys),
            np.array(price_seq), np.array(vol_seq))


X, y, future_prices, vol_all = create_sequences(
    returns_scaled, raw_returns, close_prices, SEQ_LENGTH)

# ── 6:2:2 split ──────────────────────────────────────────────────────────
n         = len(X)
train_end = int(n * 0.6)
val_end   = int(n * 0.8)

X_train_np, y_train_np = X[:train_end],        y[:train_end]
X_val_np,   y_val_np   = X[train_end:val_end], y[train_end:val_end]
X_test_np,  y_test_np  = X[val_end:],          y[val_end:]

test_prices = future_prices[val_end:]
vol_val     = vol_all[train_end:val_end]
vol_test    = vol_all[val_end:]

X_train = torch.FloatTensor(X_train_np).to(device)
X_val   = torch.FloatTensor(X_val_np).to(device)
X_test  = torch.FloatTensor(X_test_np).to(device)
y_train = torch.FloatTensor(y_train_np).to(device)

# [중요] train_model 내부 early stopping에서 사용하므로 훈련 전 선언
y_val_arr  = y_val_np.astype(float)
y_test_arr = y_test_np.astype(float)

# =============================================
# 2. 모델 정의
# =============================================
class DLinear(nn.Module):
    """Linear baseline: seq → single logit."""
    def __init__(self, seq_len):
        super().__init__()
        self.linear  = nn.Linear(seq_len, 1)
        self.sigmoid = nn.Sigmoid()

    def _logit(self, x):
        return self.linear(x.squeeze(-1))

    def forward(self, x):
        return self.sigmoid(self._logit(x))

    def get_logit(self, x):
        return self._logit(x)


class iTransformerLite(nn.Module):
    """
    iTransformer (Liu et al., 2023) — 단변량 이진 분류 버전.
    Variable/time 축을 역전(Inversion): (B, T, C) → (B, C, T).

    ⚠️ [수정 P7] 단변량 한계 (C=1) — 논문에 반드시 명시:
      C=1이면 (B, 1, T) → 토큰이 1개뿐이라 Self-Attention이
      자기 자신에만 적용됨. 이는 수학적으로
        output = TransformerEncoder(embed(x)) ≈ LayerNorm(Linear(x))
      와 동등한 '퇴화(degenerate)' 구조.

      iTransformer의 설계 핵심인 '변수 간 attention'은 C ≥ 2에서 발현됨.
      향후 확장: 거래량·고저가 등 추가 변수로 C를 늘릴 것을 권장.

    현재 실험에서는 PatchTST(항상 T/P개 토큰)와의 구조 차이가
    Calibration에 미치는 영향을 관찰하는 것이 RQ1의 목적.
    """
    def __init__(self, seq_len, d_model=64, nhead=4):
        super().__init__()
        self.enc_embedding       = nn.Linear(seq_len, d_model)
        enc_layer                = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=d_model * 4)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.fc      = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def _encode(self, x):
        x = x.permute(0, 2, 1)         # (B, C=1, T)
        x = self.enc_embedding(x)       # (B, 1, d_model)
        x = self.transformer_encoder(x) # (B, 1, d_model)
        return x[:, 0, :]               # (B, d_model)

    def forward(self, x):   return self.sigmoid(self.fc(self._encode(x)))
    def get_logit(self, x): return self.fc(self._encode(x))


class PatchTST(nn.Module):
    """
    PatchTST (Nie et al., 2023) — 단변량 이진 분류 버전.

    시간축을 PATCH_SIZE 크기의 비겹침(non-overlapping) 패치로 분할.
    각 패치를 하나의 토큰으로 임베딩 → Transformer로 시퀀스 처리.

    iTransformer와의 핵심 구조 차이 (RQ1):
      iTransformer : 변수당 1 토큰 (C=1이면 토큰 1개 → 퇴화 가능)
      PatchTST     : 패치당 1 토큰 (항상 T/P개 → 시간 국소 패턴 포착)
    이 차이가 Calibration에 어떤 영향을 주는지가 연구의 RQ1.
    """
    def __init__(self, seq_len, patch_size=PATCH_SIZE, d_model=64, nhead=4,
                 dropout=0.1):
        super().__init__()
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size"
        self.patch_size  = patch_size
        self.num_patches = seq_len // patch_size      # 30 // 5 = 6

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed   = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=d_model * 4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.fc      = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def _encode(self, x):
        B = x.shape[0]
        x = x.squeeze(-1)                                        # (B, T)
        x = x.reshape(B, self.num_patches, self.patch_size)      # (B, P, patch_size)
        x = self.patch_embed(x) + self.pos_embed                 # (B, P, d_model)
        x = self.transformer_encoder(x)                          # (B, P, d_model)
        return x.mean(dim=1)                                     # mean pooling

    def forward(self, x):   return self.sigmoid(self.fc(self._encode(x)))
    def get_logit(self, x): return self.fc(self._encode(x))


# =============================================
# 3. 학습 (Mini-batch DataLoader + Early Stopping)
# =============================================
def train_model(model, name, epochs=300, batch_size=32, patience=20):
    """
    [수정 P6] Full-batch GD → Mini-batch DataLoader + Early Stopping.

    변경 사항:
      - shuffle=False : 시계열 순서 보존 (미래 데이터 누수 방지)
      - ReduceLROnPlateau : val loss 정체 시 lr 자동 0.5배 감소
      - best_state 복원 : 과적합 방지 (val loss 최소 시점 가중치 보존)
      - patience=20 : 20 epoch 개선 없으면 조기 종료
    """
    print(f"\n[{name}] Training... (max={epochs} epochs, patience={patience})")
    model      = model.to(device)
    optimizer  = optim.Adam(model.parameters(), lr=0.001)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer, mode='min', patience=5, factor=0.5)
    criterion  = nn.BCELoss()

    dataset    = TensorDataset(X_train, y_train)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_val_t    = torch.FloatTensor(y_val_arr).to(device)

    best_val   = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val).squeeze(), y_val_t).item()
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch+1:>3d}]  "
                  f"Train={ep_loss/len(loader):.4f}  "
                  f"Val={val_loss:.4f}  Best={best_val:.4f}")

        if no_improve >= patience:
            print(f"  → Early stop @ epoch {epoch+1}  (best val={best_val:.4f})")
            break

    model.load_state_dict(best_state)
    return model


def get_probs_and_logits(model, X_tensor):
    model.eval()
    with torch.no_grad():
        prob  = model(X_tensor).squeeze().cpu().numpy()
        logit = model.get_logit(X_tensor).squeeze().cpu().numpy()
    return prob, logit


# =============================================
# 4. Calibration — 공통 헬퍼
# =============================================
def _fit_temperature(logits_arr, labels_arr):
    """LBFGS로 최적 T를 찾는 공통 함수."""
    T         = torch.nn.Parameter(torch.ones(1))
    opt       = torch.optim.LBFGS([T], lr=0.01, max_iter=500)
    criterion = nn.BCEWithLogitsLoss()
    lg_t      = torch.FloatTensor(logits_arr)
    lb_t      = torch.FloatTensor(labels_arr)

    def closure():
        opt.zero_grad()
        loss = criterion(lg_t / T, lb_t)
        loss.backward()
        return loss

    opt.step(closure)
    return max(T.item(), 0.1)


# ── 4-1. Temperature Scaling ─────────────────────────────────────────────
class TemperatureScaling:
    """전역 단일 T. Val logits에서 LBFGS로 최적화."""
    def __init__(self): self.T = 1.0

    def fit(self, logits_val, y_val):
        self.T = _fit_temperature(logits_val, y_val)
        print(f"  [Temp Scaling]     T = {self.T:.4f}")

    def predict_proba(self, logits):
        return torch.sigmoid(torch.FloatTensor(logits) / self.T).numpy()


# ── 4-2. Platt Scaling ───────────────────────────────────────────────────
class PlattScaling:
    """Logistic regression on val logits: sigmoid(a·logit + b)."""
    def __init__(self): self.lr = LogisticRegression()

    def fit(self, logits_val, y_val):
        self.lr.fit(logits_val.reshape(-1, 1), y_val)
        a, b = self.lr.coef_[0][0], self.lr.intercept_[0]
        print(f"  [Platt Scaling]    a={a:.4f}, b={b:.4f}")

    def predict_proba(self, logits):
        return self.lr.predict_proba(logits.reshape(-1, 1))[:, 1]


# ── 4-3. Isotonic Regression ─────────────────────────────────────────────
class IsotonicCalibration:
    """단조증가 비모수 재매핑. 소규모 데이터에서 과적합 위험."""
    def __init__(self): self.ir = IsotonicRegression(out_of_bounds='clip')

    def fit(self, probs_val, y_val):   self.ir.fit(probs_val, y_val)
    def predict_proba(self, probs):    return self.ir.predict(probs)


# ── 4-4. Adaptive Scaling (분위수 기반 bin) ──────────────────────────────
class AdaptiveScaling:
    """
    [수정 P4] 균등 bin → 분위수(quantile) 기반 bin.

    기존 np.linspace(0,1,11) 문제:
      BTC 이진 분류 모델의 출력 확률은 0.4~0.6 구간에 집중됨.
      균등 분할 시 극단 bin들이 거의 비어 T=1 유지 → 사실상 TemperatureScaling과 동등.

    수정: np.percentile로 각 bin에 동일한 샘플 수 보장.
      → 모든 bin에서 최소 n/n_bins 개의 샘플로 T 최적화 가능.
    """
    def __init__(self, n_bins=10):
        self.n_bins    = n_bins
        self.bin_edges = None
        self.temps     = np.ones(n_bins)

    def fit(self, logits_val, y_val):
        probs_val = torch.sigmoid(torch.FloatTensor(logits_val)).numpy()
        # 분위수 기반 bin 경계 (각 bin ≈ 동일 샘플 수)
        self.bin_edges = np.percentile(probs_val,
                                       np.linspace(0, 100, self.n_bins + 1))
        self.bin_edges[0]  = 0.0
        self.bin_edges[-1] = 1.0 + 1e-9   # 상한 경계값 포함

        for i in range(self.n_bins):
            idx = ((probs_val >  self.bin_edges[i]) &
                   (probs_val <= self.bin_edges[i + 1]))
            if idx.sum() < 5:
                continue
            self.temps[i] = _fit_temperature(logits_val[idx], y_val[idx])
        print(f"  [Adaptive Scaling] bin temps: {np.round(self.temps, 3)}")

    def predict_proba(self, logits):
        probs_raw = torch.sigmoid(torch.FloatTensor(logits)).numpy()
        probs_cal = probs_raw.copy()
        for i in range(self.n_bins):
            idx = ((probs_raw >  self.bin_edges[i]) &
                   (probs_raw <= self.bin_edges[i + 1]))
            if np.any(idx):
                probs_cal[idx] = torch.sigmoid(
                    torch.FloatTensor(logits[idx]) / self.temps[i]).numpy()
        return probs_cal


# ── 4-5. Volatility-Adaptive Scaling ────────────────────────────────────
class VolatilityAdaptiveScaling:
    """
    이분산적 비정상성(Heteroskedastic Non-Stationarity)을 직접 다루는 캘리브레이터.

    Val set을 rolling-window 표준편차의 중앙값으로
    저변동성 / 고변동성 두 구간으로 분리하고,
    각 구간에서 독립적인 T_low, T_high를 LBFGS로 최적화.

    AdaptiveScaling과의 차이:
      AdaptiveScaling  — 내생적 기준 (모델 출력 확률)
      VolAdaptive      — 외생적 기준 (시장 변동성 regime)

    기대: 고변동성 구간에서 모델이 더 과신 → T_high > T_low
    """
    def __init__(self):
        self.T_low         = 1.0
        self.T_high        = 1.0
        self.vol_threshold = None

    def fit(self, logits_val, y_val, vol_val):
        self.vol_threshold = float(np.median(vol_val))
        low_mask  = vol_val <= self.vol_threshold
        high_mask = vol_val >  self.vol_threshold
        if low_mask.sum()  >= 5:
            self.T_low  = _fit_temperature(logits_val[low_mask],  y_val[low_mask])
        if high_mask.sum() >= 5:
            self.T_high = _fit_temperature(logits_val[high_mask], y_val[high_mask])
        print(f"  [Vol-Adaptive]     threshold={self.vol_threshold:.5f}, "
              f"T_low={self.T_low:.4f}, T_high={self.T_high:.4f}")

    def predict_proba(self, logits, vol):
        probs     = np.zeros(len(logits))
        low_mask  = vol <= self.vol_threshold
        high_mask = vol >  self.vol_threshold
        if np.any(low_mask):
            probs[low_mask]  = torch.sigmoid(
                torch.FloatTensor(logits[low_mask])  / self.T_low).numpy()
        if np.any(high_mask):
            probs[high_mask] = torch.sigmoid(
                torch.FloatTensor(logits[high_mask]) / self.T_high).numpy()
        return probs


# =============================================
# 5. 평가 함수
# =============================================
def calculate_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        idx = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if idx.any():
            ece += (np.abs(np.mean(y_true[idx]) - np.mean(y_prob[idx]))
                    * idx.sum() / len(y_true))
    return ece


def calculate_mce(y_true, y_prob, n_bins=10):
    """MCE: 최악 bin의 갭. 극단적 과신/과소신을 ECE보다 민감하게 포착."""
    bins = np.linspace(0, 1, n_bins + 1)
    mce  = 0.0
    for i in range(n_bins):
        idx = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if idx.any():
            mce = max(mce, np.abs(np.mean(y_true[idx]) - np.mean(y_prob[idx])))
    return mce


def brier_score(y_true, y_prob):
    """Brier Score = MSE(prob, label). 낮을수록 좋음."""
    return float(np.mean((y_prob - y_true) ** 2))


def ece_bootstrap_ci(y_true, y_prob, n_bootstrap=N_BOOTSTRAP, ci=0.95):
    """
    [수정 P5] Bootstrap으로 ECE 95% 신뢰구간 추정.

    필요성:
      Test set ~430개, n_bins=10 → bin당 ~43 샘플.
      ECE 추정의 표준오차가 커서 소수점 3자리 차이는 통계적으로 무의미.
      CI가 겹치면 두 방법 간 차이는 유의하지 않음을 논문에 명시해야 함.
    """
    rng  = np.random.default_rng(SEED)
    n    = len(y_true)
    ecks = np.array([
        calculate_ece(y_true[ix], y_prob[ix])
        for ix in (rng.choice(n, n, replace=True) for _ in range(n_bootstrap))
    ])
    lo = np.percentile(ecks, (1 - ci) / 2 * 100)
    hi = np.percentile(ecks, (1 + ci) / 2 * 100)
    return float(np.mean(ecks)), lo, hi


def evaluate(y_true, y_prob, label):
    y_pred = (y_prob >= 0.5).astype(int)
    ece    = calculate_ece(y_true, y_prob)
    mce    = calculate_mce(y_true, y_prob)
    bs     = brier_score(y_true, y_prob)
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, zero_division=0)

    ece_lo = ece_hi = None
    ci_str = ""
    if N_BOOTSTRAP > 0:
        _, ece_lo, ece_hi = ece_bootstrap_ci(y_true, y_prob)
        ci_str = f" [{ece_lo:.4f},{ece_hi:.4f}]"

    print(f"  {label:42s} | ECE={ece:.4f}{ci_str} | MCE={mce:.4f} | "
          f"Brier={bs:.4f} | Acc={acc:.4f} | F1={f1:.4f}")
    return {"ECE": ece, "ECE_lo": ece_lo, "ECE_hi": ece_hi,
            "MCE": mce, "Brier": bs, "Acc": acc, "F1": f1}


def evaluate_by_volatility(y_true, y_prob, vol):
    """
    변동성 구간별 ECE/MCE/Brier 분리 측정 (RQ2, RQ3 핵심 실험).
    threshold = test set vol 중앙값.
    """
    med     = float(np.median(vol))
    results = {}
    for regime, mask in [("LowVol",  vol <= med),
                          ("HighVol", vol >  med)]:
        if mask.sum() < 3:
            continue
        ece = calculate_ece(y_true[mask], y_prob[mask])
        mce = calculate_mce(y_true[mask], y_prob[mask])
        bs  = brier_score(y_true[mask],   y_prob[mask])
        results[regime] = {"ECE": ece, "MCE": mce, "Brier": bs,
                           "n": int(mask.sum())}
    return results


# =============================================
# 6. 학습 + Calibration 실행
# =============================================
model_d = train_model(DLinear(SEQ_LENGTH),          "DLinear")
model_i = train_model(iTransformerLite(SEQ_LENGTH), "iTransformer-Lite")
model_p = train_model(PatchTST(SEQ_LENGTH),         "PatchTST")

# ── Val / Test 확률 & logit 추출 ─────────────────────────────────────────
prob_d_val,  logit_d_val  = get_probs_and_logits(model_d, X_val)
prob_i_val,  logit_i_val  = get_probs_and_logits(model_i, X_val)
prob_p_val,  logit_p_val  = get_probs_and_logits(model_p, X_val)

prob_d_test, logit_d_test = get_probs_and_logits(model_d, X_test)
prob_i_test, logit_i_test = get_probs_and_logits(model_i, X_test)
prob_p_test, logit_p_test = get_probs_and_logits(model_p, X_test)

# ── iTransformer: 5종 Calibration ────────────────────────────────────────
print("\n" + "="*60 + "\n  iTransformer Calibration\n" + "="*60)
ts_i  = TemperatureScaling();        ts_i.fit(logit_i_val, y_val_arr)
ps_i  = PlattScaling();              ps_i.fit(logit_i_val, y_val_arr)
iso_i = IsotonicCalibration();       iso_i.fit(prob_i_val, y_val_arr)
ada_i = AdaptiveScaling();           ada_i.fit(logit_i_val, y_val_arr)
vol_i = VolatilityAdaptiveScaling(); vol_i.fit(logit_i_val, y_val_arr, vol_val)

prob_i_ts  = ts_i.predict_proba(logit_i_test)
prob_i_ps  = ps_i.predict_proba(logit_i_test)
prob_i_iso = iso_i.predict_proba(prob_i_test)
prob_i_ada = ada_i.predict_proba(logit_i_test)
prob_i_vol = vol_i.predict_proba(logit_i_test, vol_test)

# ── PatchTST: 5종 Calibration ────────────────────────────────────────────
print("\n" + "="*60 + "\n  PatchTST Calibration\n" + "="*60)
ts_p  = TemperatureScaling();        ts_p.fit(logit_p_val, y_val_arr)
ps_p  = PlattScaling();              ps_p.fit(logit_p_val, y_val_arr)
iso_p = IsotonicCalibration();       iso_p.fit(prob_p_val, y_val_arr)
ada_p = AdaptiveScaling();           ada_p.fit(logit_p_val, y_val_arr)
vol_p = VolatilityAdaptiveScaling(); vol_p.fit(logit_p_val, y_val_arr, vol_val)

prob_p_ts  = ts_p.predict_proba(logit_p_test)
prob_p_ps  = ps_p.predict_proba(logit_p_test)
prob_p_iso = iso_p.predict_proba(prob_p_test)
prob_p_ada = ada_p.predict_proba(logit_p_test)
prob_p_vol = vol_p.predict_proba(logit_p_test, vol_test)

# ── 결과 출력 ────────────────────────────────────────────────────────────
RESULT_ROWS = [
    ("DLinear (No Cal)",                   prob_d_test),
    ("iTransformer (No Cal)",              prob_i_test),
    ("iTransformer + Temp Scaling",        prob_i_ts),
    ("iTransformer + Platt Scaling",       prob_i_ps),
    ("iTransformer + Isotonic",            prob_i_iso),
    ("iTransformer + Adaptive (prob-bin)", prob_i_ada),
    ("iTransformer + Vol-Adaptive",        prob_i_vol),
    ("PatchTST (No Cal)",                  prob_p_test),
    ("PatchTST + Temp Scaling",            prob_p_ts),
    ("PatchTST + Platt Scaling",           prob_p_ps),
    ("PatchTST + Isotonic",                prob_p_iso),
    ("PatchTST + Adaptive (prob-bin)",     prob_p_ada),
    ("PatchTST + Vol-Adaptive",            prob_p_vol),
]

ci_hdr = "  [ECE 95% CI in brackets]" if N_BOOTSTRAP > 0 else ""
print("\n" + "="*105)
print(f"  Calibration Results (Test Set){ci_hdr}")
print("="*105)
res = {}
for label, prob in RESULT_ROWS:
    res[label] = evaluate(y_test_arr, prob, label)

# ── 변동성 구간별 ECE (RQ2 & RQ3 핵심 실험) ──────────────────────────────
VOL_REGIME_ROWS = [
    ("iTransformer (No Cal)",       prob_i_test),
    ("iTransformer + Temp Scaling", prob_i_ts),
    ("iTransformer + Vol-Adaptive", prob_i_vol),
    ("PatchTST (No Cal)",           prob_p_test),
    ("PatchTST + Temp Scaling",     prob_p_ts),
    ("PatchTST + Vol-Adaptive",     prob_p_vol),
]

print("\n" + "="*95)
print("  Volatility-Regime ECE  [threshold = median(vol_test)]  (RQ2 & RQ3)")
print("="*95)
vol_regime_res = {}
for label, prob in VOL_REGIME_ROWS:
    vr = evaluate_by_volatility(y_test_arr, prob, vol_test)
    vol_regime_res[label] = vr
    for regime, m in vr.items():
        print(f"  {label:42s} [{regime:7s}] "
              f"ECE={m['ECE']:.4f}  MCE={m['MCE']:.4f}  "
              f"Brier={m['Brier']:.4f}  n={m['n']}")

# =============================================
# 7. 시각화 (4행 × 2열 + 마지막 행 전체)
# =============================================
COLORS = {
    "DLinear":  "#2196F3",
    "iT_NC":    "#FF5722",
    "iT_Temp":  "#9C27B0",
    "iT_Platt": "#009688",
    "iT_Iso":   "#FF9800",
    "iT_Ada":   "#E91E63",
    "iT_Vol":   "#795548",
    "PT_NC":    "#F44336",
    "PT_Temp":  "#673AB7",
    "PT_Platt": "#00BCD4",
    "PT_Iso":   "#8BC34A",
    "PT_Ada":   "#FF5252",
    "PT_Vol":   "#4CAF50",
}

# short label → RESULT_ROWS key 매핑 (Panel 3 CI error bar에서 사용)
_SHORT_TO_FULL = {
    "DLinear\nNoCal":  "DLinear (No Cal)",
    "iT\nNoCal":       "iTransformer (No Cal)",
    "iT+Temp":         "iTransformer + Temp Scaling",
    "iT+Platt":        "iTransformer + Platt Scaling",
    "iT+Iso":          "iTransformer + Isotonic",
    "iT+Ada":          "iTransformer + Adaptive (prob-bin)",
    "iT+VolAda":       "iTransformer + Vol-Adaptive",
    "PT\nNoCal":       "PatchTST (No Cal)",
    "PT+Temp":         "PatchTST + Temp Scaling",
    "PT+Platt":        "PatchTST + Platt Scaling",
    "PT+Iso":          "PatchTST + Isotonic",
    "PT+Ada":          "PatchTST + Adaptive (prob-bin)",
    "PT+VolAda":       "PatchTST + Vol-Adaptive",
}

fig = plt.figure(figsize=(22, 28))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.33)

# ── Panel 1: Calibration Curve — iTransformer ────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect")
i_cfgs = [
    ("DLinear",        prob_d_test, COLORS["DLinear"],  "s-"),
    ("iT NoCal",       prob_i_test, COLORS["iT_NC"],    "o-"),
    ("iT+Temp",        prob_i_ts,   COLORS["iT_Temp"],  "^-"),
    ("iT+Platt",       prob_i_ps,   COLORS["iT_Platt"], "D-"),
    ("iT+Isotonic",    prob_i_iso,  COLORS["iT_Iso"],   "v-"),
    ("iT+Adaptive",    prob_i_ada,  COLORS["iT_Ada"],   "P-"),
    ("iT+VolAdaptive", prob_i_vol,  COLORS["iT_Vol"],   "h-"),
]
for lbl, prob, col, mk in i_cfgs:
    ece    = calculate_ece(y_test_arr, prob)
    pt, pp = calibration_curve(y_test_arr, prob, n_bins=10)
    ax1.plot(pp, pt, mk, color=col, lw=2,
             label=f"{lbl} (ECE={ece:.4f})", markersize=5)
ax1.set_xlabel("Confidence"); ax1.set_ylabel("Actual Accuracy")
ax1.set_title("Calibration Curve: iTransformer", fontsize=13, fontweight='bold')
ax1.legend(fontsize=7.5, loc='upper left'); ax1.grid(True, alpha=0.3)

# ── Panel 2: Calibration Curve — PatchTST ────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect")
p_cfgs = [
    ("DLinear",        prob_d_test, COLORS["DLinear"],  "s-"),
    ("PT NoCal",       prob_p_test, COLORS["PT_NC"],    "o-"),
    ("PT+Temp",        prob_p_ts,   COLORS["PT_Temp"],  "^-"),
    ("PT+Platt",       prob_p_ps,   COLORS["PT_Platt"], "D-"),
    ("PT+Isotonic",    prob_p_iso,  COLORS["PT_Iso"],   "v-"),
    ("PT+Adaptive",    prob_p_ada,  COLORS["PT_Ada"],   "P-"),
    ("PT+VolAdaptive", prob_p_vol,  COLORS["PT_Vol"],   "h-"),
]
for lbl, prob, col, mk in p_cfgs:
    ece    = calculate_ece(y_test_arr, prob)
    pt, pp = calibration_curve(y_test_arr, prob, n_bins=10)
    ax2.plot(pp, pt, mk, color=col, lw=2,
             label=f"{lbl} (ECE={ece:.4f})", markersize=5)
ax2.set_xlabel("Confidence"); ax2.set_ylabel("Actual Accuracy")
ax2.set_title("Calibration Curve: PatchTST", fontsize=13, fontweight='bold')
ax2.legend(fontsize=7.5, loc='upper left'); ax2.grid(True, alpha=0.3)

# ── Panel 3: ECE Bar Chart + Bootstrap CI error bars ─────────────────────
ax3 = fig.add_subplot(gs[1, 0])
all_cfgs = [
    ("DLinear\nNoCal",  prob_d_test, COLORS["DLinear"]),
    ("iT\nNoCal",       prob_i_test, COLORS["iT_NC"]),
    ("iT+Temp",         prob_i_ts,   COLORS["iT_Temp"]),
    ("iT+Platt",        prob_i_ps,   COLORS["iT_Platt"]),
    ("iT+Iso",          prob_i_iso,  COLORS["iT_Iso"]),
    ("iT+Ada",          prob_i_ada,  COLORS["iT_Ada"]),
    ("iT+VolAda",       prob_i_vol,  COLORS["iT_Vol"]),
    ("PT\nNoCal",       prob_p_test, COLORS["PT_NC"]),
    ("PT+Temp",         prob_p_ts,   COLORS["PT_Temp"]),
    ("PT+Platt",        prob_p_ps,   COLORS["PT_Platt"]),
    ("PT+Iso",          prob_p_iso,  COLORS["PT_Iso"]),
    ("PT+Ada",          prob_p_ada,  COLORS["PT_Ada"]),
    ("PT+VolAda",       prob_p_vol,  COLORS["PT_Vol"]),
]
lbls3    = [c[0] for c in all_cfgs]
ece_vals = [calculate_ece(y_test_arr, c[1]) for c in all_cfgs]
cols3    = [c[2] for c in all_cfgs]
x3       = np.arange(len(lbls3))
bars     = ax3.bar(x3, ece_vals, color=cols3, alpha=0.85,
                   edgecolor='white', linewidth=1.0)

for bar, val in zip(bars, ece_vals):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.002,
             f"{val:.3f}", ha='center', va='bottom',
             fontsize=7, fontweight='bold', rotation=40)

# [수정 P5] Bootstrap 95% CI error bars
if N_BOOTSTRAP > 0:
    lo_errs, hi_errs = [], []
    for short_lbl in lbls3:
        full_lbl = _SHORT_TO_FULL[short_lbl]
        r        = res[full_lbl]
        lo_errs.append(r["ECE"] - r["ECE_lo"])
        hi_errs.append(r["ECE_hi"] - r["ECE"])
    ax3.errorbar(x3, ece_vals, yerr=[lo_errs, hi_errs],
                 fmt='none', ecolor='black', capsize=3, elinewidth=1.2,
                 label="95% CI")
    ax3.legend(fontsize=8)

ax3.set_xticks(x3); ax3.set_xticklabels(lbls3, fontsize=7.5)
ax3.set_ylabel("ECE ↓"); ax3.grid(True, alpha=0.3, axis='y')
ax3.set_title("ECE: All Methods" + (" (with 95% CI)" if N_BOOTSTRAP > 0 else ""),
              fontsize=13, fontweight='bold')
ax3.set_ylim(0, max(ece_vals) * 1.45)
ax3.axvline(6.5, color='gray', linestyle='--', lw=1, alpha=0.5)
ax3.text(3.0, max(ece_vals) * 1.33, "iTransformer",
         ha='center', fontsize=8.5, color='#555')
ax3.text(9.5, max(ece_vals) * 1.33, "PatchTST",
         ha='center', fontsize=8.5, color='#555')

# ── Panel 4: MCE & Brier Score ────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
mce_vals   = [calculate_mce(y_test_arr, c[1]) for c in all_cfgs]
brier_vals = [brier_score(y_test_arr,   c[1]) for c in all_cfgs]
w4 = 0.35
ax4.bar(x3 - w4/2, mce_vals,   w4, label='MCE',         color='#5C6BC0', alpha=0.85)
ax4.bar(x3 + w4/2, brier_vals, w4, label='Brier Score', color='#EF5350', alpha=0.85)
ax4.set_xticks(x3); ax4.set_xticklabels(lbls3, fontsize=7.5)
ax4.set_ylabel("Score ↓"); ax4.legend(fontsize=9)
ax4.set_title("MCE & Brier Score: All Methods", fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.axvline(6.5, color='gray', linestyle='--', lw=1, alpha=0.5)
ax4.text(3.0, max(max(mce_vals), max(brier_vals)) * 1.05,
         "iTransformer", ha='center', fontsize=8.5, color='#555')
ax4.text(9.5, max(max(mce_vals), max(brier_vals)) * 1.05,
         "PatchTST",     ha='center', fontsize=8.5, color='#555')

# ── Panel 5: Volatility-Regime ECE (RQ2 & RQ3 핵심) ─────────────────────
ax5 = fig.add_subplot(gs[2, 0])
vr_lbls  = ["iT\nNoCal", "iT+Temp", "iT+VolAda",
            "PT\nNoCal", "PT+Temp", "PT+VolAda"]
vr_probs = [prob_i_test, prob_i_ts, prob_i_vol,
            prob_p_test, prob_p_ts, prob_p_vol]
vr_cols  = [COLORS["iT_NC"],  COLORS["iT_Temp"], COLORS["iT_Vol"],
            COLORS["PT_NC"],  COLORS["PT_Temp"], COLORS["PT_Vol"]]

med_vol = float(np.median(vol_test))
low_m   = vol_test <= med_vol
high_m  = vol_test >  med_vol
ece_low  = [calculate_ece(y_test_arr[low_m],  p[low_m])  for p in vr_probs]
ece_high = [calculate_ece(y_test_arr[high_m], p[high_m]) for p in vr_probs]

x5 = np.arange(len(vr_lbls)); w5 = 0.35
b_lo = ax5.bar(x5 - w5/2, ece_low,  w5, color=vr_cols, alpha=0.55,
               edgecolor='white', label=f"Low Vol  (n={low_m.sum()})")
b_hi = ax5.bar(x5 + w5/2, ece_high, w5, color=vr_cols, alpha=1.00,
               edgecolor='white', hatch='//', label=f"High Vol (n={high_m.sum()})")

for bars_g, vals in [(b_lo, ece_low), (b_hi, ece_high)]:
    for bar, val in zip(bars_g, vals):
        ax5.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.003,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=7.5)

ax5.set_xticks(x5); ax5.set_xticklabels(vr_lbls, fontsize=9)
ax5.set_ylabel("ECE ↓")
ax5.set_title("Volatility-Regime ECE: Low vs High Vol\n"
              "(RQ2 & RQ3 — Heteroskedastic Non-Stationarity)",
              fontsize=13, fontweight='bold')
ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3, axis='y')
ax5.axvline(2.5, color='gray', linestyle='--', lw=1, alpha=0.5)
ymax5 = max(max(ece_low), max(ece_high))
ax5.set_ylim(0, ymax5 * 1.40)
ax5.text(1.0, ymax5 * 1.28, "iTransformer",
         ha='center', fontsize=8.5, color='#555')
ax5.text(4.0, ymax5 * 1.28, "PatchTST",
         ha='center', fontsize=8.5, color='#555')

# ── Panel 6: Temperature Effect 비교 ─────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
logit_range = np.linspace(-5, 5, 300)
T_combos = [
    (0.5,    "T=0.5 (overconfident)",         "#E91E63",         "--", 1.5),
    (1.0,    "T=1.0 (original)",              "#607D8B",         "-",  1.5),
    (ts_i.T, f"T={ts_i.T:.2f} (iT optimal)", COLORS["iT_Temp"], "-",  2.5),
    (ts_p.T, f"T={ts_p.T:.2f} (PT optimal)", COLORS["PT_Temp"], "-",  2.5),
    (3.0,    "T=3.0 (underconfident)",        "#FF9800",         "--", 1.5),
]
for T, lbl, col, ls, lw in T_combos:
    ax6.plot(logit_range, 1 / (1 + np.exp(-logit_range / T)),
             color=col, lw=lw, linestyle=ls, label=lbl)
ax6.axhline(0.5, color='gray', linestyle=':', lw=1)
ax6.axvline(0.0, color='gray', linestyle=':', lw=1)
ax6.set_xlabel("Logit"); ax6.set_ylabel("Calibrated Probability")
ax6.set_title("Temperature Effect:\niTransformer vs PatchTST Optimal T",
              fontsize=13, fontweight='bold')
ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

# ── Panel 7: 누적 수익률 (연속 포지션 사이징) ────────────────────────────
def cum_returns(y_prob, prices):
    """
    [수정 P1] 이진 시그널(≥ 0.5) → 연속 포지션 사이징.

    기존 이진 방식의 문제:
      Temperature Scaling은 sigmoid(z/T) > 0.5 ⟺ z > 0 (T > 0이면 항상 성립)
      → 로짓 부호(sign)를 보존하므로 threshold=0.5 기준 시그널이
        uncalibrated와 동일 → iT NoCal / iT+Temp / iT+VolAda 수익률 동일.

    수정: 확률을 연속 포지션으로 변환.
      position = 2 * prob - 1  ∈ [-1, +1]
        prob=0.80 → +0.60  (강한 롱)
        prob=0.50 → 0.00   (중립)
        prob=0.20 → -0.60  (숏)
      Calibration으로 확률 정밀도가 개선되면 포지션 크기가 달라져
      수익률 차이가 드러남 → 연구 목적에 부합.
    """
    positions  = 2 * y_prob - 1                # [-1, 1]
    price_rets = np.diff(prices) / prices[:-1]
    strat_rets = positions[:-1] * price_rets
    return np.cumprod(1 + strat_rets), np.cumprod(1 + price_rets)

ax7 = fig.add_subplot(gs[3, :])
cum_cfgs = [
    ("DLinear",        prob_d_test, COLORS["DLinear"],  1.6),
    ("iT NoCal",       prob_i_test, COLORS["iT_NC"],    1.8),
    ("iT+Temp",        prob_i_ts,   COLORS["iT_Temp"],  2.0),
    ("iT+VolAdaptive", prob_i_vol,  COLORS["iT_Vol"],   2.0),
    ("PT NoCal",       prob_p_test, COLORS["PT_NC"],    1.8),
    ("PT+Temp",        prob_p_ts,   COLORS["PT_Temp"],  2.0),
    ("PT+VolAdaptive", prob_p_vol,  COLORS["PT_Vol"],   2.0),
]
cum_bnh = None
for lbl, prob, col, lw in cum_cfgs:
    cum_strat, cum_bnh = cum_returns(prob, test_prices)
    ax7.plot(cum_strat, color=col, lw=lw,
             label=f"{lbl} ({cum_strat[-1]-1:.1%})")
ax7.plot(cum_bnh, color='black', lw=2, linestyle='--',
         label=f"Buy & Hold ({cum_bnh[-1]-1:.1%})")
ax7.axhline(1.0, color='gray', linestyle=':', lw=1)
ax7.set_xlabel("Test Days"); ax7.set_ylabel("Cumulative Return (Base=1.0)")
ax7.set_title(
    "Trading Simulation: Continuous Position Sizing  (position = 2·prob − 1)\n"
    "DLinear / iTransformer / PatchTST — Calibration 효과가 포지션 크기에 직접 반영됨",
    fontsize=13, fontweight='bold')
ax7.legend(fontsize=9); ax7.grid(True, alpha=0.3)

fig.suptitle(
    "Calibration Analysis: DLinear vs iTransformer vs PatchTST\n"
    "BTC-USD  |  Heteroskedastic Non-Stationarity  |  "
    "Temp / Platt / Isotonic / Adaptive(quantile) / Vol-Adaptive Scaling",
    fontsize=14, fontweight='bold', y=0.999)

plt.savefig("btc_calibration_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nCalibration analysis complete!")
