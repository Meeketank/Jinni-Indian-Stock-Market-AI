# streamlit_app.py
"""
JINNI — Indian Stock Market AI (Fixed number_input max-value issue)
- Ensures number_input default <= max_value
- Defensive guards for small/empty universe
- Retains background learner and scan UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time, threading, random, os, pickle, traceback
from typing import Tuple, List, Dict

# ----------------- Configuration -----------------
CACHE_DIR = ".jinni_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
MODEL_FILE = os.path.join(CACHE_DIR, "jinni_model.pkl")
HIST_TTL_SECONDS = 60 * 60 * 6  # 6 hours
FETCH_TTL_SECONDS = 60 * 20  # 20 minutes
RATE_LIMIT_MIN_INTERVAL = 0.6
BG_INTERVAL = 30
BG_BATCH_SIZE = 40
N_LAGS = 6
MAX_UNIVERSE = 5000

SEED_UNIVERSE = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","SBIN.NS"
]

# ----------------- Utilities -----------------
def now_ts(): return time.time()
def cache_path(symbol: str, kind: str="hist") -> str:
    safe = symbol.replace("/", "_").replace(":", "_").replace(" ", "_")
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
def throttle():
    global _last_fetch
    with _fetch_lock:
        elapsed = time.time() - _last_fetch
        if elapsed < RATE_LIMIT_MIN_INTERVAL:
            time.sleep(RATE_LIMIT_MIN_INTERVAL - elapsed + random.random()*0.1)
        _last_fetch = time.time()

def safe_yf_history(symbol: str, period="1y"):
    attempts = 3
    for i in range(attempts):
        try:
            throttle()
            t = yf.Ticker(symbol)
            df = t.history(period=period, interval="1d", auto_adjust=True)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            time.sleep(0.5 + i*0.5)
    return None

# ----------------- Features & Simple Online Model -----------------
def build_features(df: pd.DataFrame, n_lags=N_LAGS):
    if df is None or len(df) < n_lags + 8:
        return np.empty((0, n_lags + 3)), np.empty((0,))
    close = df["Close"].values.astype(float)
    returns = (close[1:] - close[:-1]) / (close[:-1] + 1e-12)
    ma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    ma50 = pd.Series(close).rolling(50, min_periods=1).mean().values
    delta = pd.Series(close).diff()
    gain = delta.where(delta>0,0).rolling(14,min_periods=1).mean()
    loss = (-delta.where(delta<0,0)).rolling(14,min_periods=1).mean().replace(0, np.nan)
    rsi = (100 - (100/(1 + (gain/loss).fillna(1.0)))).values
    X, y = [], []
    for i in range(n_lags, len(returns)):
        lag = returns[i-n_lags:i]
        idx = i + 1
        price = close[idx] if idx < len(close) else close[-1]
        ma_diff = (ma20[idx] - ma50[idx]) / (price if price != 0 else 1)
        rsi_n = rsi[idx] if idx < len(rsi) else 50.0
        vol = float(np.std(returns[max(0, i-20):i+1])) if i >= 1 else 0.0
        feat = np.concatenate([lag, [ma_diff, rsi_n, vol]])
        X.append(feat)
        y.append(returns[i])
    return np.array(X), np.array(y)

class OnlineReg:
    def __init__(self, dim):
        self.w = np.zeros(dim)
        self.b = 0.0
        self.lr = 0.004
    def predict(self, X):
        X = np.atleast_2d(X)
        return X.dot(self.w) + self.b
    def partial_fit(self, X, y):
        if getattr(X, "size", 0) == 0:
            return
        preds = self.predict(X)
        errs = preds - y
        grad_w = (errs[:,None] * X).mean(axis=0)
        grad_b = errs.mean()
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b
    def state(self): return {"w": self.w, "b": self.b}
    def load_state(self, s):
        if not s: return
        self.w = s.get("w", self.w)
        self.b = s.get("b", self.b)

# ----------------- Background Learner -----------------
class BackgroundLearner:
    def __init__(self, model_file=MODEL_FILE):
        self.model_file = model_file
        self.dim = N_LAGS + 3
        self.reg = OnlineReg(self.dim)
        self.universe = SEED_UNIVERSE.copy()
        self.metrics = {"dir_acc_ema": 0.5, "mae_ema": 0.5, "samples": 0}
        self.ema_alpha = 0.04
        self._stop = threading.Event()
        self._thread = None
        self.lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, "rb") as f:
                    obj = pickle.load(f)
                self.reg.load_state(obj.get("reg", {}))
                self.universe = obj.get("universe", self.universe)[:MAX_UNIVERSE]
                self.metrics.update(obj.get("metrics", {}))
            except Exception:
                pass

    def _persist(self):
        try:
            with open(self.model_file, "wb") as f:
                pickle.dump({"reg": self.reg.state(), "universe": self.universe, "metrics": self.metrics}, f)
        except Exception:
            pass

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._persist()

    def add_to_universe(self, sym: str):
        if not sym: return
        s = sym.strip().upper()
        if s not in self.universe:
            self.universe.insert(0, s)
            self.universe = self.universe[:MAX_UNIVERSE]

    def _loop(self):
        while not self._stop.is_set():
            try:
                if not self.universe:
                    time.sleep(1); continue
                batch = random.sample(self.universe, min(BG_BATCH_SIZE, len(self.universe)))
                batch_dir, batch_mae, batch_samples = [], [], 0
                for s in batch:
                    try:
                        hist = load_cache(cache_path(s,"hist"), HIST_TTL_SECONDS)
                        if hist is None:
                            hist = safe_yf_history(s, period="400d")
                            if hist is None: continue
                            save_cache(hist, cache_path(s,"hist"))
                        X, y = build_features(hist, N_LAGS)
                        if X.size == 0: continue
                        preds = self.reg.predict(X)
                        mae = float(np.mean(np.abs(preds - y)))
                        dir_acc = float(np.mean(np.sign(preds) == np.sign(y)))
                        self.reg.partial_fit(X, y)
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
                time.sleep(BG_INTERVAL + random.random()*3)
            except Exception:
                time.sleep(BG_INTERVAL + random.random()*3)

    def get_metrics(self):
        with self.lock:
            return dict(self.metrics)

    def predict_for(self, symbol: str, days: int = 7) -> Tuple[float, float]:
        try:
            if not symbol or not isinstance(symbol, str):
                return 0.0, 0.0
            s = symbol.strip().upper()
            hist = load_cache(cache_path(s,"hist"), HIST_TTL_SECONDS)
            if hist is None:
                hist = safe_yf_history(s, period="300d")
                if hist is None: return 0.0, 0.0
                save_cache(hist, cache_path(s,"hist"))
            X, _ = build_features(hist, N_LAGS)
            if X.size == 0:
                return 0.0, 0.0
            latest = X[-1].reshape(1, -1)
            est = float(self.reg.predict(latest)[0])
            recent = (hist["Close"].values[1:] - hist["Close"].values[:-1]) / (hist["Close"].values[:-1] + 1e-12)
            vol = float(np.std(recent[-60:])) if len(recent) >= 1 else 0.0
            data_factor = min(0.99, 0.05 + min(1.0, len(hist)/250.0))
            conf = max(0.01, min(0.99, data_factor * (1.0 - vol * 3.0)))
            est = float(np.clip(est, -0.8, 0.8))
            return est, conf
        except Exception:
            return 0.0, 0.0

    def scan_high_potential(self, min_pct: float = 10.0, days: int = 7, sample_limit: int = None, progress_callback=None):
        results = []
        try:
            pool = list(self.universe)
            if not pool:
                return results
            if sample_limit is None or sample_limit <= 0:
                sample_limit = len(pool)
            sample_limit = min(sample_limit, len(pool))
            random.shuffle(pool)
            pool = pool[:sample_limit]
            total = len(pool)
            for i, s in enumerate(pool, 1):
                try:
                    hist = load_cache(cache_path(s,"hist"), HIST_TTL_SECONDS)
                    if hist is None:
                        hist = safe_yf_history(s, period="180d")
                        if hist is None:
                            if progress_callback: progress_callback(i, total)
                            continue
                        save_cache(hist, cache_path(s,"hist"))
                    est_daily, conf = self.predict_for(s, days=days)
                    est_pct = est_daily * (days ** 0.5) * 100
                    if abs(est_pct) >= abs(min_pct) and conf > 0.07:
                        cur = float(hist["Close"].iloc[-1])
                        tgt = cur * (1 + est_daily * (days ** 0.5))
                        results.append({"symbol": s, "expected_pct": est_pct, "confidence": conf, "current": cur, "target": tgt})
                except Exception:
                    pass
                if progress_callback:
                    try:
                        progress_callback(i, total)
                    except Exception:
                        pass
            results.sort(key=lambda r: abs(r["expected_pct"]), reverse=True)
            return results
        except Exception:
            return results

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="JINNI — Indian Stock Market AI", page_icon="🧞", layout="wide")
st.title("🧞 JINNI — Indian Stock Market AI")
st.caption("Background training runs automatically. ⚠️ Educational only — not financial advice.")

if "bg" not in st.session_state:
    st.session_state.bg = BackgroundLearner()
    st.session_state.bg.start()
BG = st.session_state.bg

# Sidebar: controls
with st.sidebar:
    st.header("Controls")
    user_symbol = st.text_input("Enter stock symbol (e.g., RELIANCE or RELIANCE.NS)", value="RELIANCE")
    pred_days = st.selectbox("Prediction horizon (days)", options=[1,7,14,30,60,180,365,1825], index=1)
    st.markdown("---")
    st.subheader("Recommendations")
    st.write("Scan the model's universe for high potential stocks (>= threshold).")
    min_pct = st.number_input("Minimum expected move (%)", value=10.0, step=1.0)
    scan_all = st.checkbox("Scan full universe (may take long)", value=False)

    # --------- FIX: compute universe-based max_value safely ----------
    uni_len = len(BG.universe) if BG.universe else 0
    # define sensible minimum and maximum for the input widget
    widget_min = 10
    widget_default = 400
    # If universe empty or small, set a safe max to avoid StreamlitValueAboveMaxError
    if uni_len <= 0:
        widget_max = max(100, widget_default)  # allow user to set even if universe empty (but will scan fewer)
    else:
        widget_max = max(widget_min, uni_len)

    # Ensure default value <= widget_max
    widget_default = min(widget_default, widget_max)

    # Provide UI with safe bounds
    sample_limit = st.number_input(
        "Max symbols to scan now",
        min_value=widget_min,
        max_value=widget_max,
        value=widget_default,
        step=50,
        help="If the universe is small the max will be lowered automatically."
    )
    # ----------------------------------------------------------------

    if st.button("🔍 Find opportunities"):
        st.session_state["scan_results"] = None
        placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.info("Starting scan...")
        # determine sample_limit for scan call (allow full universe if requested)
        actual_limit = None if scan_all else int(min(sample_limit, len(BG.universe) if BG.universe else sample_limit))
        def progress_cb(i, total):
            frac = int((i/total)*100) if total>0 else 0
            progress_bar.progress(min(100, frac))
            status_text.info(f"Scanning {i}/{total} ({frac}%)")
        try:
            results = BG.scan_high_potential(min_pct=min_pct, days=pred_days, sample_limit=actual_limit, progress_callback=progress_cb)
            st.session_state["scan_results"] = results
            progress_bar.progress(100)
            status_text.success(f"Scan finished — {len(results)} candidate(s) found.")
        except Exception as e:
            tb = traceback.format_exc()
            with open(os.path.join(CACHE_DIR, "last_error.log"), "a") as f:
                f.write(f"\n\n[{datetime.now().isoformat()}] Error during scan: {str(e)}\n{tb}")
            status_text.error("Scan failed — see logs (.jinni_cache/last_error.log).")
            st.session_state["scan_results"] = []
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("Background Model")
    m = BG.get_metrics()
    st.metric("Directional accuracy (EMA)", f"{m.get('dir_acc_ema',0.0)*100:.2f}%")
    st.metric("MAE (EMA)", f"{m.get('mae_ema',0.0):.4f}")
    st.write(f"Trained samples: {m.get('samples',0):,}")
    st.markdown("---")
    st.info("Tip: Add symbols to training universe using 'Add to universe' on the main page.")

# ----------------- Main analysis area -----------------
st.header("Stock Analysis")

def resolve_and_fetch(symbol: str, period="2y"):
    s = (symbol or "").strip()
    if not s:
        return None, None
    s_up = s.upper()
    candidates = []
    if s_up.endswith(".NS") or s_up.endswith(".BO"):
        candidates = [s_up, s_up.replace(".BO", ".NS"), s_up.replace(".NS", ".BO"), s_up.split(".")[0]]
    else:
        candidates = [s_up, s_up + ".NS", s_up + ".BO", s_up]
    for c in candidates:
        cached = load_cache(cache_path(c,"hist"), FETCH_TTL_SECONDS)
        if cached is not None:
            return cached, c + " (cache)"
    for c in candidates:
        df = safe_yf_history(c, period=period)
        if df is not None and not df.empty:
            save_cache(df, cache_path(c,"hist"))
            return df, c + " (download)"
    return None, "Not found"

with st.spinner("Resolving symbol and fetching data (cache aware)..."):
    df, resolved_label = resolve_and_fetch(user_symbol, period="2y")

if df is None:
    st.error(f"Could not resolve the symbol or fetch market data. Tried candidates. Tip: add .NS for NSE or .BO for BSE. (Tried label: {resolved_label})")
    st.stop()

hist = df.copy()
symbol_display = resolved_label.split()[0]
last_price = float(hist["Close"].iloc[-1])
prev = float(hist["Close"].iloc[-2]) if len(hist)>1 else last_price
st.subheader(f"{symbol_display}  —  {resolved_label}")
st.metric("Last Close", f"₹{last_price:.2f}", f"{(last_price-prev):+.2f} ({(last_price-prev)/prev*100:+.2f}%)")

# Plot
def plot_chart(df):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6,0.2,0.2])
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
    # RSI quick calc
    delta = df["Close"].diff()
    gain = delta.where(delta>0,0).rolling(14,min_periods=1).mean()
    loss = (-delta.where(delta<0,0)).rolling(14,min_periods=1).mean().replace(0, np.nan)
    rsi = (100 - (100/(1 + (gain/loss).fillna(1.0))))
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI"), row=3, col=1)
    fig.update_layout(height=700, showlegend=True)
    return fig

st.plotly_chart(plot_chart(hist), use_container_width=True)

# add to BG universe (silently)
try:
    BG.add_to_universe(symbol_display)
except Exception:
    pass

# prediction call
try:
    est_daily, conf = BG.predict_for(symbol_display, days=pred_days)
    conf_pct = conf * 100.0
except Exception as e:
    est_daily, conf = 0.0, 0.0
    conf_pct = 0.0
    with open(os.path.join(CACHE_DIR,"last_error.log"), "a") as f:
        f.write(f"\n\n[{datetime.now().isoformat()}] Error in predict_for: {e}\n{traceback.format_exc()}")

pred_price = last_price * (1 + est_daily * (pred_days ** 0.5))
pred_pct = (pred_price - last_price) / last_price * 100.0

# fallback if model confidence very low
if conf < 0.08:
    fallback_daily = hist["Close"].pct_change().tail(5).mean() if len(hist) >= 5 else 0.0
    fallback_price = last_price * (1 + fallback_daily * (pred_days ** 0.5))
    pred_price = 0.6 * pred_price + 0.4 * fallback_price
    pred_pct = (pred_price - last_price) / last_price * 100.0

# show card
def show_prediction_card():
    if pred_pct >= 15:
        sig_style = "background:#d4edda;padding:12px;border-left:6px solid #28a745;border-radius:8px"
        sig_label = "🔵 STRONG BUY"
    elif pred_pct >= 5:
        sig_style = "background:#d4edda;padding:12px;border-left:6px solid #28a745;border-radius:8px"
        sig_label = "🟢 BUY"
    elif pred_pct <= -15:
        sig_style = "background:#f8d7da;padding:12px;border-left:6px solid #dc3545;border-radius:8px"
        sig_label = "🔴 STRONG SELL"
    elif pred_pct <= -5:
        sig_style = "background:#f8d7da;padding:12px;border-left:6px solid #dc3545;border-radius:8px"
        sig_label = "🔴 SELL"
    else:
        sig_style = "background:#fff3cd;padding:12px;border-left:6px solid #ffc107;border-radius:8px"
        sig_label = "⚪ HOLD"
    st.markdown(f"<div style='{sig_style}'><h3>Prediction: {sig_label}</h3><h2>₹{pred_price:.2f}</h2><p>{pred_pct:+.2f}% in {pred_days} days</p><p>Model confidence: {conf_pct:.1f}%</p></div>", unsafe_allow_html=True)

show_prediction_card()

# trading plan
entry = last_price
stop = entry*(1-0.03) if pred_pct>=0 else entry*(1+0.03)
targets = [entry*(1 + (pred_pct/100.0) * mult * (pred_days ** 0.5)) for mult in (0.35,0.7,1.05,1.4)]
risk = abs(entry - stop)
reward = abs(targets[1] - entry) if len(targets) > 1 else 0.0
rr = (reward / risk) if risk > 0 else 0.0

st.subheader("Trading Plan")
st.write(f"- Entry: ₹{entry:.2f}")
st.write(f"- Stop Loss: ₹{stop:.2f}")
for i, t in enumerate(targets, 1):
    st.write(f"- Target {i}: ₹{t:.2f} ({(t-entry)/entry*100:+.1f}%)")
st.write(f"- Est risk-reward (T2): 1:{rr:.2f}")

# technical snapshot
st.subheader("Technical Snapshot")
delta = hist["Close"].diff()
gain = delta.where(delta>0,0).rolling(14,min_periods=1).mean()
loss = (-delta.where(delta<0,0)).rolling(14,min_periods=1).mean().replace(0,np.nan)
rsi_val = float((100 - (100/(1 + (gain/loss).fillna(1.0)))) .iloc[-1]) if len(hist)>0 else np.nan
macd_val = float(hist["Close"].ewm(span=12).mean().iloc[-1] - hist["Close"].ewm(span=26).mean().iloc[-1])
ma50_val = float(hist["Close"].rolling(50, min_periods=1).mean().iloc[-1])
vol = float(hist["Close"].pct_change().rolling(20).std().iloc[-1])*np.sqrt(252)*100 if len(hist)>1 else 0.0
c1, c2, c3, c4 = st.columns(4)
c1.metric("RSI (14)", f"{rsi_val:.1f}", "Overbought" if rsi_val>70 else ("Oversold" if rsi_val<30 else "Neutral"))
c2.metric("MACD", f"{macd_val:.3f}", "Bullish" if macd_val>0 else "Bearish")
c3.metric("Trend (MA50)", "Bullish" if last_price>ma50_val else "Bearish")
c4.metric("Volatility (ann %)", f"{vol:.2f}%")

# add symbol to universe button
if st.button("➕ Add this symbol to background training universe"):
    try:
        BG.add_to_universe(symbol_display)
        st.success(f"Added {symbol_display} to training universe.")
    except Exception:
        st.error("Failed to add symbol to universe — see logs.")

# show scan results
if "scan_results" in st.session_state and st.session_state.scan_results:
    st.markdown("---")
    st.header("Scan Results (last)")
    for rec in st.session_state.scan_results[:30]:
        col1, col2 = st.columns([3,1])
        with col1:
            st.write(f"**{rec['symbol']}**  •  Est {rec['expected_pct']:+.1f}%  •  Conf {rec['confidence']:.2f}")
            st.write(f"Current: ₹{rec['current']:.2f}  →  Target: ₹{rec['target']:.2f}")
        with col2:
            if st.button(f"Analyze {rec['symbol']}", key=f"an_{rec['symbol']}"):
                st.experimental_set_query_params(symbol=rec['symbol'])
                st.experimental_rerun()

st.markdown("---")
st.caption("Notes: Scanning the entire Indian market requires a complete universe list. Add symbols to the universe for them to be included. Logs: .jinni_cache/last_error.log")
