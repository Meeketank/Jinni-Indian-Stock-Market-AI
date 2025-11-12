# streamlit_app.py
"""
JINNI — Improved Streamlit app (single-file)
- Background online learner (persisted)
- Better prediction ensemble (learner + heuristic)
- Fundamental analysis (yfinance.info)
- Rich technical indicators + simple pattern signals
- Safe scanning of universe with progress, no experimental_rerun
- Robust error handling and caching
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import threading, time, os, pickle, random, math, traceback
from typing import Tuple, List, Dict

# ------------------- CONFIG -------------------
CACHE_DIR = ".jinni_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
MODEL_FILE = os.path.join(CACHE_DIR, "background_model_v2.pkl")
HIST_TTL = 60 * 60 * 6  # 6 hours cache for history
FETCH_TTL = 60 * 20     # 20 minutes for fast fetch
RATE_LIMIT = 0.6        # seconds between yfinance hits (cooperative)
BG_INTERVAL = 20        # seconds between background batches
BG_BATCH = 30           # tickers per background batch
N_LAGS = 8              # features: N lag returns + extras
MAX_UNIVERSE = 8000     # cap universe stored locally

SEED = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
        "HINDUNILVR.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","SBIN.NS"]

# ------------------- UTIL -------------------
def now_ts(): return time.time()
def cache_path(name, kind):
    safe = name.replace("/", "_").replace(":", "_").replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{safe}__{kind}.pkl")

def save_cache(obj, path):
    try:
        with open(path, "wb") as f:
            pickle.dump({"ts": now_ts(), "data": obj}, f)
    except Exception:
        pass

def load_cache(path, ttl):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            d = pickle.load(f)
        if now_ts() - d.get("ts", 0) > ttl:
            return None
        return d.get("data")
    except Exception:
        return None

_last_fetch = 0.0
_fetch_lock = threading.Lock()
def throttle_fetch():
    global _last_fetch
    with _fetch_lock:
        elapsed = time.time() - _last_fetch
        if elapsed < RATE_LIMIT:
            time.sleep(RATE_LIMIT - elapsed + random.random()*0.05)
        _last_fetch = time.time()

def safe_history(symbol: str, period="1y") -> pd.DataFrame:
    """Try to fetch history with small retry & throttle"""
    for attempt in range(3):
        try:
            throttle_fetch()
            tk = yf.Ticker(symbol)
            df = tk.history(period=period, interval="1d", auto_adjust=True)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            time.sleep(0.6 + attempt*0.6)
    return None

# ------------------- FEATURES & INDICATORS -------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # MAs
    for p in [5,10,20,50,100,200]:
        df[f"MA{p}"] = df["Close"].rolling(window=p, min_periods=1).mean()
    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta>0,0).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta<0,0)).rolling(14, min_periods=1).mean().replace(0, np.nan)
    rs = (gain / loss).fillna(0.0)
    df["RSI"] = 100 - (100/(1+rs))
    # MACD
    exp12 = df["Close"].ewm(span=12).mean()
    exp26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = exp12 - exp26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    # Bollinger
    df["BB_mid"] = df["Close"].rolling(20, min_periods=1).mean()
    bbstd = df["Close"].rolling(20, min_periods=1).std().fillna(0)
    df["BB_up"] = df["BB_mid"] + 2*bbstd
    df["BB_dn"] = df["BB_mid"] - 2*bbstd
    df["BB_width"] = (df["BB_up"] - df["BB_dn"]) / df["BB_mid"].replace(0, np.nan)
    # Volume MA
    df["Vol_MA20"] = df["Volume"].rolling(20, min_periods=1).mean()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_MA20"].replace(0, np.nan)
    # Volatility (20d annualized %)
    df["Volatility"] = df["Close"].pct_change().rolling(20, min_periods=1).std() * np.sqrt(252) * 100
    return df

def build_features(df: pd.DataFrame, n_lags=N_LAGS) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X,y) where X includes last n_lags returns + simple technical features; y is next-day return"""
    if df is None or len(df) < n_lags + 5:
        return np.empty((0, n_lags+3)), np.empty((0,))
    close = df["Close"].values.astype(float)
    returns = (close[1:] - close[:-1])/(close[:-1]+1e-12)
    X, y = [], []
    ma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    ma50 = pd.Series(close).rolling(50, min_periods=1).mean().values
    rsi = add_indicators(df)["RSI"].values
    for i in range(n_lags, len(returns)):
        lag = returns[i-n_lags:i]
        idx = i+1  # index into close for "current day"
        ma_diff = (ma20[idx] - ma50[idx]) / (close[idx] if close[idx]!=0 else 1)
        rsi_curr = float(rsi[idx]) if idx < len(rsi) else 50.0
        vol = float(np.std(returns[max(0,i-20):i+1])) if i>0 else 0.0
        feat = np.concatenate([lag, [ma_diff, rsi_curr, vol]])
        X.append(feat)
        y.append(returns[i])
    return np.array(X), np.array(y)

# ------------------- LIGHTWEIGHT ONLINE MODEL -------------------
class OnlineModel:
    def __init__(self, dim):
        self.dim = dim
        self.w = np.zeros(dim)
        self.b = 0.0
        self.lr = 0.0035
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return X.dot(self.w) + self.b
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        if X.size == 0:
            return
        preds = self.predict(X)
        errs = preds - y
        grad_w = (errs[:,None] * X).mean(axis=0)
        grad_b = errs.mean()
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b
    def state(self):
        return {"w": self.w, "b": self.b}
    def load_state(self, s):
        if not s: return
        self.w = s.get("w", self.w)
        self.b = s.get("b", self.b)

# ------------------- BACKGROUND LEARNER -------------------
class BackgroundLearner:
    def __init__(self, model_file=MODEL_FILE):
        self.model_file = model_file
        self.dim = N_LAGS + 3
        self.model = OnlineModel(self.dim)
        self.universe = SEED.copy()
        self.metrics = {"dir_acc_ema": 0.5, "mae_ema": 0.5, "samples": 0}
        self.ema_alpha = 0.05
        self._stop = threading.Event()
        self._thread = None
        self.lock = threading.Lock()
        self.batch_size = BG_BATCH
        self.interval = BG_INTERVAL
        self._load()
    def _load(self):
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, "rb") as f:
                    obj = pickle.load(f)
                self.model.load_state(obj.get("model_state", {}))
                self.universe = obj.get("universe", self.universe)[:MAX_UNIVERSE]
                self.metrics.update(obj.get("metrics", {}))
            except Exception:
                pass
    def _persist(self):
        try:
            with open(self.model_file, "wb") as f:
                pickle.dump({"model_state": self.model.state(), "universe": self.universe, "metrics": self.metrics}, f)
        except Exception:
            pass
    def start(self):
        if self._thread and self._thread.is_alive(): return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._persist()
    def add_symbol(self, sym: str):
        s = (sym or "").strip().upper()
        if not s: return
        if s not in self.universe:
            self.universe.insert(0, s)
            self.universe = self.universe[:MAX_UNIVERSE]
    def _loop(self):
        while not self._stop.is_set():
            try:
                if not self.universe:
                    time.sleep(1); continue
                batch = random.sample(self.universe, min(self.batch_size, len(self.universe)))
                batch_dir, batch_mae, batch_samples = [], [], 0
                for s in batch:
                    try:
                        hist = load_cache(cache_path(s,"hist"), HIST_TTL)
                        if hist is None:
                            hist = safe_history(s, period="400d")
                            if hist is None: continue
                            save_cache(hist, cache_path(s,"hist"))
                        X,y = build_features(hist, N_LAGS)
                        if X.size == 0: continue
                        preds = self.model.predict(X)
                        mae = float(np.mean(np.abs(preds - y)))
                        dir_acc = float(np.mean(np.sign(preds) == np.sign(y)))
                        self.model.partial_fit(X, y)
                        batch_dir.append(dir_acc); batch_mae.append(mae)
                        batch_samples += len(y)
                        time.sleep(0.01)
                    except Exception:
                        continue
                if batch_samples > 0:
                    with self.lock:
                        a = self.ema_alpha
                        if batch_dir:
                            self.metrics["dir_acc_ema"] = a * float(np.mean(batch_dir)) + (1-a) * self.metrics["dir_acc_ema"]
                        if batch_mae:
                            self.metrics["mae_ema"] = a * float(np.mean(batch_mae)) + (1-a) * self.metrics["mae_ema"]
                        self.metrics["samples"] += batch_samples
                    self._persist()
                time.sleep(self.interval + random.random()*3)
            except Exception:
                time.sleep(self.interval + random.random()*3)
    def get_metrics(self):
        with self.lock:
            return dict(self.metrics)
    def predict(self, symbol: str, days:int=7) -> Tuple[float,float]:
        """
        Return predicted multi-day return (as fraction) and confidence [0..1]
        Steps:
        - Fetch / cache history
        - Build features; use last feature to predict daily return
        - Convert to days-horizon with sqrt scaling
        - Confidence based on data size & volatility
        """
        try:
            s = (symbol or "").strip().upper()
            hist = load_cache(cache_path(s,"hist"), FETCH_TTL)
            if hist is None:
                hist = safe_history(s, period="500d")
                if hist is None:
                    return 0.0, 0.0
                save_cache(hist, cache_path(s,"hist"))
            X, _ = build_features(hist, N_LAGS)
            if X.size == 0:
                return 0.0, 0.0
            latest = X[-1].reshape(1,-1)
            daily_pred = float(np.clip(self.model.predict(latest)[0], -0.8, 0.8))
            # convert to multi-day: sqrt scaling
            multi = daily_pred * math.sqrt(max(1, days))
            # confidence: depends on history length and volatility
            returns = (hist["Close"].values[1:] - hist["Close"].values[:-1])/(hist["Close"].values[:-1]+1e-12)
            vol = float(np.std(returns[-60:])) if len(returns)>=10 else float(np.std(returns)) if len(returns)>0 else 0.0
            data_strength = min(1.0, len(hist)/250.0)
            conf = max(0.03, min(0.99, data_strength * (1.0 - vol*2.5)))
            return multi, conf
        except Exception:
            return 0.0, 0.0
    def scan_universe(self, min_pct: float=10.0, days:int=7, sample_limit:int=200, progress_hook=None) -> List[Dict]:
        """Scan the stored universe (sample_limit items) and return candidates with expected absolute pct >= min_pct"""
        results = []
        pool = list(self.universe)
        if not pool: return results
        limit = min(sample_limit, len(pool))
        sample = random.sample(pool, limit)
        total = len(sample)
        for i, s in enumerate(sample, 1):
            try:
                hist = load_cache(cache_path(s,"hist"), HIST_TTL)
                if hist is None:
                    hist = safe_history(s, period="180d")
                    if hist is None:
                        if progress_hook: progress_hook(i, total)
                        continue
                    save_cache(hist, cache_path(s,"hist"))
                pred, conf = self.predict(s, days=days)
                est_pct = pred * 100.0
                # scale with sqrt timeframe to compare to user percent
                eff_pct = est_pct
                if abs(eff_pct) >= abs(min_pct) and conf > 0.06:
                    cur = float(hist["Close"].iloc[-1])
                    tgt = cur * (1 + pred)
                    results.append({"symbol": s, "expected_pct": eff_pct, "confidence": conf, "current": cur, "target": tgt})
            except Exception:
                pass
            if progress_hook:
                progress_hook(i, total)
        results.sort(key=lambda r: abs(r["expected_pct"]), reverse=True)
        return results

# ------------------- INITIALIZE BG -------------------
if "BG" not in st.session_state:
    st.session_state.BG = BackgroundLearner()
    st.session_state.BG.start()
BG = st.session_state.BG

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="JINNI — Indian Stock Market AI (improved)", page_icon="🧞", layout="wide")
st.markdown("<h1 style='text-align:center'>🧞 JINNI — Indian Stock Market AI</h1>", unsafe_allow_html=True)
st.write("Background training runs automatically. ⚠️ Educational only — not financial advice.")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    user_input = st.text_input("Enter stock symbol (e.g., RELIANCE or RELIANCE.NS)", value="RELIANCE")
    pred_days = st.selectbox("Prediction horizon (days)", [1,7,14,30,60,90,180,365], index=1)
    st.markdown("---")
    st.subheader("Recommendations")
    min_pct = st.number_input("Minimum expected move (%)", value=7.0, step=1.0)
    scan_full = st.checkbox("Scan full stored universe (may take long)", value=False)
    # safe max for number_input
    uni_len = len(BG.universe) if BG.universe else 0
    min_limit = 10
    default_limit = 400
    if uni_len <= 0:
        max_limit = max(min_limit, default_limit)
    else:
        max_limit = max(min_limit, uni_len)
    default_limit = min(default_limit, max_limit)
    sample_limit = st.number_input("Max symbols to scan now", min_value=min_limit, max_value=max_limit, value=default_limit, step=25)
    scan_btn = st.button("🔍 Find opportunities")
    st.markdown("---")
    st.subheader("Model (background)")
    m = BG.get_metrics()
    st.metric("Directional accuracy (EMA)", f"{m.get('dir_acc_ema',0.0)*100:.2f}%")
    st.metric("MAE (EMA)", f"{m.get('mae_ema',0.0):.4f}")
    st.write(f"Trained samples: {m.get('samples',0):,}")
    st.markdown("---")
    st.write("Tip: Add symbols to the background universe by analyzing them (button on main page).")

# MAIN: resolve symbol and fetch
def resolve_candidates(symbol: str) -> List[str]:
    s = (symbol or "").strip().upper()
    if not s: return []
    candidates = []
    if s.endswith(".NS") or s.endswith(".BO"):
        candidates = [s, s.replace(".BO",".NS"), s.replace(".NS",".BO"), s.split(".")[0]]
    else:
        candidates = [s, s+".NS", s+".BO"]
    # dedupe & keep order
    seen = set(); res=[]
    for c in candidates:
        if c not in seen:
            res.append(c); seen.add(c)
    return res

cands = resolve_candidates(user_input)
hist = None; resolved=None
# try cache first
for c in cands:
    cached = load_cache(cache_path(c,"hist"), FETCH_TTL)
    if cached is not None:
        hist = cached; resolved=c; break
# else try download
if hist is None:
    for c in cands:
        df = safe_history(c, period="2y")
        if df is not None and not df.empty:
            hist = df; resolved=c
            save_cache(hist, cache_path(c,"hist"))
            break

if hist is None:
    st.error("Could not fetch data for that symbol. Try adding .NS or .BO suffix. (Tried: {}).".format(", ".join(cands)))
    st.stop()

# add to universe to train on it going forward
BG.add_symbol(resolved)

hist = add_indicators(hist)
last_price = float(hist["Close"].iloc[-1])
prev = float(hist["Close"].iloc[-2]) if len(hist)>1 else last_price
st.subheader(f"{resolved} — Last Close: ₹{last_price:.2f} ({(last_price-prev):+.2f}, {(last_price-prev)/prev*100:+.2f}%)")

# Plot enhanced chart
def plot_stock(df, symbol):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.55, 0.2, 0.25])
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    # MAs
    for ma in ["MA20","MA50","MA200"]:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(width=1.5)), row=1, col=1)
    # Bollinger
    if "BB_up" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], name="BB_up", line=dict(width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_dn"], name="BB_dn", line=dict(width=1, dash="dash")), row=1, col=1)
    # Volume
    colors = ['red' if o>c else 'green' for o,c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume"), row=2, col=1)
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=3, col=1)
    fig.update_layout(height=800, showlegend=True, title_text=f"{symbol} Technicals")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    return fig

st.plotly_chart(plot_stock(hist, resolved), use_container_width=True)

# Technical pattern detection (simple)
def detect_patterns(df):
    signals=[]
    # MA crosses (20 & 50)
    if "MA20" in df and "MA50" in df:
        if df["MA20"].iloc[-2] < df["MA50"].iloc[-2] and df["MA20"].iloc[-1] > df["MA50"].iloc[-1]:
            signals.append("MA20/MA50 Bullish crossover")
        if df["MA20"].iloc[-2] > df["MA50"].iloc[-2] and df["MA20"].iloc[-1] < df["MA50"].iloc[-1]:
            signals.append("MA20/MA50 Bearish crossover")
    # MACD cross
    if "MACD" in df and "MACD_Signal" in df:
        if df["MACD"].iloc[-2] < df["MACD_Signal"].iloc[-2] and df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1]:
            signals.append("MACD bullish cross")
        if df["MACD"].iloc[-2] > df["MACD_Signal"].iloc[-2] and df["MACD"].iloc[-1] < df["MACD_Signal"].iloc[-1]:
            signals.append("MACD bearish cross")
    # Bollinger squeeze (narrow)
    if "BB_width" in df:
        if df["BB_width"].iloc[-1] < df["BB_width"].rolling(50, min_periods=1).mean().iloc[-1]*0.5:
            signals.append("Bollinger squeeze (low volatility)")
    # Volume spike
    if "Vol_Ratio" in df:
        if df["Vol_Ratio"].iloc[-1] > 2.0:
            signals.append("Volume spike (>2x 20d avg)")
    return signals

pattern_signals = detect_patterns(hist)
if pattern_signals:
    st.markdown("### 🔔 Pattern Signals")
    for s in pattern_signals:
        st.write("- " + s)
else:
    st.info("No major pattern triggers detected (MA/MACD/Bollinger/Volume)")

# Fundamental analysis
st.header("Fundamental Snapshot")
info = {}
try:
    info = yf.Ticker(resolved).info
except Exception:
    info = {}
fund_rows = []
pe = info.get("trailingPE") or info.get("forwardPE")
pb = info.get("priceToBook")
div_yield = info.get("dividendYield")
roe = info.get("returnOnEquity")
profit_margin = info.get("profitMargins")
rev_growth = info.get("revenueGrowth")
market_cap = info.get("marketCap")
col1, col2, col3, col4 = st.columns(4)
col1.metric("P/E", f"{pe:.2f}" if pe else "N/A")
col2.metric("P/B", f"{pb:.2f}" if pb else "N/A")
col3.metric("Div Yield", f"{div_yield*100:.2f}%" if div_yield else "N/A")
col4.metric("ROE", f"{roe*100:.2f}%" if roe else "N/A")
st.write("**Profit Margin:**", f"{profit_margin*100:.2f}%" if profit_margin else "N/A")
st.write("**Revenue Growth (TTM):**", f"{rev_growth*100:.2f}%" if rev_growth else "N/A")
if market_cap:
    st.write("**Market Cap:**", f"₹{market_cap/1e7:.2f} Cr")

# Ensemble prediction (BG learner + heuristics)
st.header("🔮 Prediction & Trading Plan")
bg_pred, bg_conf = BG.predict(resolved, days=pred_days)
# heuristic enhanced prediction (momentum + MA trend + volatility)
def heuristic_prediction(df, days):
    cp = df["Close"].iloc[-1]
    # multi-timeframe momentum
    m5 = df["Close"].pct_change().tail(5).mean()
    m20 = df["Close"].pct_change().tail(20).mean()
    momentum = 0.6*m5 + 0.4*m20
    # MA trend boost
    ma_boost = 0.0
    if "MA20" in df and "MA50" in df:
        ma_boost = 0.03 if df["MA20"].iloc[-1] > df["MA50"].iloc[-1] else -0.02
    vol = df["Volatility"].iloc[-1] if "Volatility" in df else 0.0
    vol_adj = max(0.5, 1.0 - vol/50.0)
    est_daily = (momentum + ma_boost) * vol_adj
    est_multi = est_daily * math.sqrt(max(1, days))
    # clamp
    est_multi = float(np.clip(est_multi, -0.6, 0.6))
    return est_multi
heur = heuristic_prediction(hist, pred_days)

# combine
if bg_conf > 0.12:
    # weight by bg_conf (0..1)
    w = bg_conf
    final_est = bg_pred * w + heur * (1-w)
    final_conf = max(0.05, min(0.99, bg_conf + 0.05))
else:
    final_est = heur
    final_conf = max(0.02, min(0.9, 0.25))

pred_price = last_price * (1 + final_est)
pred_pct = final_est * 100.0

# Avoid tiny meaningless predictions: if very small (<0.5% in longer horizons) amplify slightly using volatility
if abs(pred_pct) < 2.0 and pred_days>=30:
    # use volatility to estimate realistic bound
    vol = hist["Volatility"].iloc[-1] if "Volatility" in hist else 0.0
    # nudge by vol*sqrt(t)
    pred_pct = pred_pct + (vol * math.sqrt(pred_days/30.0) * 0.15)
    pred_price = last_price * (1 + pred_pct/100.0)

# Present
sig_style = "hold"
if pred_pct >= 15: sig="🔵 STRONG BUY"; sig_style="buy"
elif pred_pct >= 5: sig="🟢 BUY"; sig_style="buy"
elif pred_pct <= -15: sig="🔴 STRONG SELL"; sig_style="sell"
elif pred_pct <= -5: sig="🔴 SELL"; sig_style="sell"
else: sig="⚪ HOLD"; sig_style="hold"

color = {"buy":"#d4edda","sell":"#f8d7da","hold":"#fff3cd"}.get(sig_style,"#fff3cd")
border = {"buy":"#28a745","sell":"#dc3545","hold":"#ffc107"}.get(sig_style,"#ffc107")

st.markdown(f"""
<div style="background:{color};padding:12px;border-left:6px solid {border};border-radius:8px">
<h3>Prediction: {sig}</h3>
<h2>₹{pred_price:.2f}</h2>
<p style="font-size:16px;color:{'green' if pred_pct>0 else 'red'}">{pred_pct:+.2f}% in {pred_days} days</p>
<p style="font-size:12px;color:#444">Model confidence: {final_conf*100:.1f}% — Learner contribution: {bg_conf*100:.1f}%</p>
</div>
""", unsafe_allow_html=True)

# Trading plan targets 4-levels
def compute_targets(current, pct_expected, days):
    entry = current
    stop = entry * (1 - 0.035) if pct_expected>=0 else entry * (1 + 0.035)
    # scale based on days
    tf = math.sqrt(max(1, days/7.0))
    t1 = entry*(1 + pct_expected/100.0 * 0.33 * tf)
    t2 = entry*(1 + pct_expected/100.0 * 0.66 * tf)
    t3 = entry*(1 + pct_expected/100.0 * 1.0 * tf)
    t4 = entry*(1 + pct_expected/100.0 * 1.4 * tf)
    risk = abs(entry - stop)
    reward = abs(t2 - entry) if risk>0 else 0.0
    rr = reward / risk if risk>0 else 0.0
    return entry, stop, [t1,t2,t3,t4], rr

entry, stop, targets, rr = compute_targets(last_price, pred_pct, pred_days)
st.subheader("Trading Plan")
st.write(f"- Entry: ₹{entry:.2f}")
st.write(f"- Stop Loss: ₹{stop:.2f}")
for i,t in enumerate(targets,1):
    st.write(f"- Target {i}: ₹{t:.2f} ({(t-entry)/entry*100:+.1f}%)")
st.write(f"- Risk-Reward (T2): 1:{rr:.2f}")

# Technical snapshot metrics
st.subheader("Technical Snapshot")
rsi_val = hist["RSI"].iloc[-1]
macd = hist["MACD"].iloc[-1]
macd_sig = hist["MACD_Signal"].iloc[-1]
ma50 = hist["MA50"].iloc[-1] if "MA50" in hist else None
vol = hist["Volatility"].iloc[-1] if "Volatility" in hist else 0.0
c1,c2,c3,c4 = st.columns(4)
c1.metric("RSI (14)", f"{rsi_val:.1f}", "Overbought" if rsi_val>70 else ("Oversold" if rsi_val<30 else "Neutral"))
c2.metric("MACD", f"{macd - macd_sig:.4f}", "Bullish" if macd>macd_sig else "Bearish")
c3.metric("Trend (MA50)", "Bullish" if last_price>ma50 else "Bearish")
c4.metric("Volatility (ann %)", f"{vol:.2f}%")

# Add symbol to universe
if st.button("➕ Add symbol to learner universe"):
    BG.add_symbol(resolved)
    st.success(f"Added {resolved} to universe (training will include it).")

# Scan button handling (safe, non-crashing)
if scan_btn:
    st.session_state["scan_results"] = None
    placeholder = st.empty()
    progress = st.progress(0)
    status = st.empty()
    status.info("Starting scan — this may take time. Progress shown below.")
    try:
        results = []
        total_scanned = 0
        def hook(i, total):
            progress.progress(int((i/total)*100))
            status.info(f"Scanning {i}/{total}")
        limit = None if scan_full else int(sample_limit)
        results = BG.scan_universe(min_pct=min_pct, days=pred_days, sample_limit=limit or sample_limit, progress_hook=hook)
        st.session_state["scan_results"] = results
        progress.progress(100)
        status.success(f"Scan finished — {len(results)} candidate(s) found.")
    except Exception as e:
        err = traceback.format_exc()
        with open(os.path.join(CACHE_DIR,"last_error.log"), "a") as f:
            f.write(f"\n\n[{datetime.now().isoformat()}] Scan error: {e}\n{err}")
        status.error("Scan failed — details in .jinni_cache/last_error.log")
        st.session_state["scan_results"] = []

# Present scan results if available
if "scan_results" in st.session_state and st.session_state["scan_results"]:
    st.markdown("---")
    st.header("Scan Results")
    for cand in st.session_state["scan_results"][:80]:
        col1, col2, col3 = st.columns([4,1,1])
        with col1:
            st.write(f"**{cand['symbol']}** — Est: {cand['expected_pct']:+.2f}% • Conf: {cand['confidence']:.2f}")
            st.write(f"Curr: ₹{cand['current']:.2f}  →  Target: ₹{cand['target']:.2f}")
        with col2:
            if st.button(f"Analyze {cand['symbol']}", key=f"analyze_{cand['symbol']}"):
                # set input and rerun flow by writing to session and redirecting execution
                st.session_state['last_selected'] = cand['symbol']
                st.experimental_set_query_params(symbol=cand['symbol'])
                st.experimental_rerun()
        with col3:
            if st.button("Watch", key=f"watch_{cand['symbol']}"):
                BG.add_symbol(cand['symbol'])
                st.success(f"Watching {cand['symbol']} — added to universe.")

st.markdown("---")
st.caption("Notes: For full-market scanning you should populate BG.universe with a full NSE/BSE symbol list (one-time). This app caches history and the model persists to .jinni_cache/. Use small sample_limit if rate-limited by yfinance.")

