# streamlit_app.py
"""
JINNI — Indian Stock Market AI (robust, cache-first, safe predict)
Improved error handling around BG.predict_for to avoid TypeError crash.
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time, threading, random, os, pickle, traceback
from typing import Tuple, Dict, List

# ---------------- Config ----------------
CACHE_DIR = ".jinni_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
MODEL_FILE = os.path.join(CACHE_DIR, "jinni_model.pkl")
FETCH_TTL_SECONDS = 60 * 30    # 30 minutes
HIST_TTL_SECONDS = 60 * 60 * 6 # 6 hours for background use
RATE_LIMIT_MIN_INTERVAL = 1.2
BG_INTERVAL = 40
BG_BATCH_SIZE = 30
N_LAGS = 6
MAX_UNIVERSE = 2500

DEFAULT_UNIVERSE = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"]

# ---------------- Utilities ----------------
def now_ts() -> float:
    return time.time()

def cache_path_for(symbol: str, kind: str = "hist") -> str:
    safe = symbol.replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe}__{kind}.pkl")

_last_fetch_time = 0.0
_last_fetch_lock = threading.Lock()

def throttle_min_interval():
    global _last_fetch_time
    with _last_fetch_lock:
        elapsed = time.time() - _last_fetch_time
        if elapsed < RATE_LIMIT_MIN_INTERVAL:
            to_sleep = RATE_LIMIT_MIN_INTERVAL - elapsed + random.random()*0.2
            time.sleep(to_sleep)
        _last_fetch_time = time.time()

def fetch_with_backoff(fetch_fn, attempts=3, base_sleep=1.0, max_sleep=8.0):
    last_exc = None
    for i in range(attempts):
        try:
            throttle_min_interval()
            return fetch_fn()
        except Exception as e:
            last_exc = e
            wait = min(max_sleep, base_sleep * (2 ** i)) + random.random()*0.5
            time.sleep(wait)
    raise last_exc if last_exc is not None else RuntimeError("Fetch failed")

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
        ts = d.get("ts", 0)
        if now_ts() - ts > ttl:
            return None
        return d.get("data")
    except Exception:
        return None

# ---------------- Fetch helpers ----------------
def fetch_history_yf_download(symbols: List[str], period="1y", interval="1d") -> Dict[str, pd.DataFrame]:
    tickers_str = " ".join(symbols)
    def _call():
        return yf.download(tickers=tickers_str, period=period, interval=interval, group_by='ticker', progress=False, threads=True)
    raw = fetch_with_backoff(_call, attempts=2)
    results = {}
    # normalize output
    try:
        if isinstance(raw, pd.DataFrame) and len(symbols) == 1:
            results[symbols[0]] = raw
            return results
        for sym in symbols:
            try:
                if sym in raw.columns.levels[0]:
                    sub = raw[sym].dropna(how='all')
                    results[sym] = sub if not sub.empty else None
                else:
                    results[sym] = None
            except Exception:
                results[sym] = None
    except Exception:
        # unexpected shape
        for sym in symbols:
            results[sym] = None
    return results

def safe_fetch_history(symbol: str, period="1y", use_cache_ttl=FETCH_TTL_SECONDS) -> Tuple[pd.DataFrame, str]:
    """
    Resolve user input to likely tickers and fetch history with caching/backoff.
    Returns: (df or None, resolved_label)
    """
    s = (symbol or "").strip()
    if not s:
        return None, "Empty symbol"
    s_up = s.upper()
    candidates = []
    if s_up.endswith(".NS") or s_up.endswith(".BO"):
        candidates = [s_up, s_up.replace(".BO", ".NS"), s_up.replace(".NS", ".BO"), s_up.split(".")[0]]
    else:
        candidates = [s_up, s_up + ".NS", s_up + ".BO"]
    # cache-first
    for cand in candidates:
        path = cache_path_for(cand, "hist")
        cached = load_cache(path, use_cache_ttl)
        if cached is not None:
            return cached, cand + " (cache)"
    # try batch download
    try:
        batch = fetch_history_yf_download(candidates, period=period)
        for cand in candidates:
            df = batch.get(cand)
            if df is not None and not df.empty:
                save_cache(df, cache_path_for(cand, "hist"))
                return df, cand + " (download)"
    except Exception:
        pass
    # fallback single
    last_exc = None
    for cand in candidates:
        try:
            def _call():
                return yf.Ticker(cand).history(period=period, interval="1d", auto_adjust=True)
            df = fetch_with_backoff(_call, attempts=2)
            if df is not None and not df.empty:
                save_cache(df, cache_path_for(cand, "hist"))
                return df, cand + " (single)"
        except Exception as e:
            last_exc = e
            continue
    # nothing
    return None, f"No data (attempted {candidates})"

# ---------------- Feature builder & model ----------------
def build_features(df: pd.DataFrame, n_lags=N_LAGS):
    if df is None or len(df) < n_lags + 8:
        return np.empty((0, n_lags+3)), np.empty((0,))
    close = df["Close"].values
    returns = (close[1:] - close[:-1]) / (close[:-1] + 1e-12)
    ma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    ma50 = pd.Series(close).rolling(50, min_periods=1).mean().values
    # RSI simple
    delta = pd.Series(close).diff()
    gain = delta.where(delta>0, 0).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14, min_periods=1).mean().replace(0, np.nan)
    rsi = (100 - (100/(1 + (gain/loss).fillna(1.0)))).values
    X, y = [], []
    for i in range(n_lags, len(returns)):
        lag = returns[i-n_lags:i]
        idx = i+1
        price = close[idx] if idx < len(close) else close[-1]
        ma_diff = (ma20[idx] - ma50[idx]) / (price if price != 0 else 1)
        rsi_n = rsi[idx] if idx < len(rsi) else 50.0
        vol = float(np.std(returns[max(0,i-20):i+1])) if i>=1 else 0.0
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
    def state(self):
        return {"w": self.w, "b": self.b}
    def load_state(self, s):
        if not s:
            return
        self.w = s.get("w", self.w)
        self.b = s.get("b", self.b)

# ---------------- Background learner ----------------
class BackgroundLearner:
    def __init__(self, model_file=MODEL_FILE):
        self.model_file = model_file
        self.dim = N_LAGS + 3
        self.reg = OnlineReg(self.dim)
        self.universe = DEFAULT_UNIVERSE.copy()
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
    def add_to_universe(self, symbol: str):
        if not symbol:
            return
        sym = symbol.strip().upper()
        if sym not in self.universe:
            self.universe.insert(0, sym)
            self.universe = self.universe[:MAX_UNIVERSE]
    def _loop(self):
        while not self._stop.is_set():
            try:
                if not self.universe:
                    time.sleep(5); continue
                batch = random.sample(self.universe, min(BG_BATCH_SIZE, len(self.universe)))
                batch_dir_accs, batch_maes, batch_samples = [], [], 0
                for s in batch:
                    # prefer cached history
                    hist = load_cache(cache_path_for(s, "hist"), HIST_TTL_SECONDS)
                    if hist is None:
                        try:
                            hist, _ = safe_fetch_history(s, period="400d")
                            if hist is None: continue
                        except Exception:
                            continue
                    X, y = build_features(hist, N_LAGS)
                    if X.size == 0:
                        continue
                    preds = self.reg.predict(X)
                    mae = float(np.mean(np.abs(preds - y)))
                    dir_acc = float(np.mean(np.sign(preds) == np.sign(y)))
                    self.reg.partial_fit(X, y)
                    batch_dir_accs.append(dir_acc); batch_maes.append(mae)
                    batch_samples += len(y)
                    time.sleep(0.02)
                if batch_samples > 0:
                    with self.lock:
                        a = self.ema_alpha
                        if batch_dir_accs:
                            batch_dir = float(np.mean(batch_dir_accs))
                            self.metrics["dir_acc_ema"] = a * batch_dir + (1 - a) * self.metrics["dir_acc_ema"]
                        if batch_maes:
                            batch_mae = float(np.mean(batch_maes))
                            self.metrics["mae_ema"] = a * batch_mae + (1 - a) * self.metrics["mae_ema"]
                        self.metrics["samples"] += batch_samples
                    self._persist()
                time.sleep(BG_INTERVAL + random.random()*5)
            except Exception:
                time.sleep(BG_INTERVAL + random.random()*3)
    def get_metrics(self):
        with self.lock:
            return dict(self.metrics)
    def predict_for(self, symbol: str, days: int = 7) -> Tuple[float, float]:
        """
        Returns (estimated_daily_return, confidence 0..1)
        Wrapped safely in the app (we still defend here)
        """
        if not symbol or not isinstance(symbol, str):
            return 0.0, 0.0
        # normalized symbol
        sym = symbol.strip().upper()
        hist = load_cache(cache_path_for(sym, "hist"), HIST_TTL_SECONDS)
        if hist is None:
            hist, resolved = safe_fetch_history(sym, period="300d")
            if hist is None:
                return 0.0, 0.0
            # store cache
            save_cache(hist, cache_path_for(resolved.split()[0], "hist"))
            sym = resolved.split()[0]
        X, _ = build_features(hist, N_LAGS)
        if X.size == 0:
            return 0.0, 0.0
        latest = X[-1].reshape(1, -1)
        try:
            est = float(self.reg.predict(latest)[0])
        except Exception:
            est = 0.0
        recent_returns = (hist["Close"].values[1:] - hist["Close"].values[:-1]) / (hist["Close"].values[:-1] + 1e-12)
        vol = float(np.std(recent_returns[-60:])) if len(recent_returns) >= 1 else 0.0
        data_factor = min(0.99, 0.1 + min(1.0, len(hist)/250.0))
        conf = max(0.01, min(0.99, data_factor * (1.0 - vol * 3.0)))
        est = float(np.clip(est, -0.6, 0.6))
        return est, conf
    def scan_high_potential(self, min_pct=10.0, days=7, sample_limit=400):
        results = []
        if not self.universe:
            return results
        pool = random.sample(self.universe, min(sample_limit, len(self.universe)))
        for s in pool:
            try:
                est_daily, conf = self.predict_for(s, days=days)
                est_pct = est_daily * (days ** 0.5) * 100
                if abs(est_pct) >= abs(min_pct) and conf > 0.12:
                    hist = load_cache(cache_path_for(s, "hist"), HIST_TTL_SECONDS)
                    if hist is None:
                        hist, _ = safe_fetch_history(s, period="120d")
                        if hist is None: continue
                    cur = float(hist["Close"].iloc[-1])
                    tgt = cur * (1 + est_daily * (days ** 0.5))
                    results.append({"symbol": s, "expected_pct": est_pct, "confidence": conf, "current": cur, "target": tgt})
            except Exception:
                continue
        results.sort(key=lambda r: abs(r["expected_pct"]), reverse=True)
        return results[:50]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="JINNI — Indian Stock Market AI", page_icon="🧞", layout="wide")
st.title("🧞 JINNI — Indian Stock Market AI")
st.caption("Background training runs automatically. ⚠️ Educational only — not financial advice.")

# create or reuse background learner in session_state
if "bg" not in st.session_state:
    st.session_state.bg = BackgroundLearner()
    st.session_state.bg.start()
BG = st.session_state.bg

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    user_symbol = st.text_input("Enter stock symbol (e.g., RELIANCE or RELIANCE.NS)", value="RELIANCE")
    pred_days = st.slider("Prediction horizon (days)", min_value=1, max_value=14, value=7)
    st.markdown("---")
    st.subheader("Recommendations")
    if st.button("Find 10%+ Weekly Opportunities"):
        with st.spinner("Scanning (uses cached history when available, may take time)..."):
            recs = BG.scan_high_potential(min_pct=10.0, days=7, sample_limit=400)
            st.session_state["last_recs"] = recs
            st.success(f"Scan finished — {len(recs)} candidates found.")
    st.markdown("---")
    st.subheader("Background Model")
    m = BG.get_metrics()
    st.metric("Directional accuracy (EMA)", f"{m['dir_acc_ema']*100:.2f}%")
    st.metric("MAE (EMA)", f"{m['mae_ema']:.4f}")
    st.write(f"Trained samples: {m['samples']:,}")

# Main area: Resolve and fetch history (robust)
st.header("Stock Analysis")
with st.spinner("Resolving symbol and fetching data (cache-aware)..."):
    df, resolved = safe_fetch_history(user_symbol, period="2y")
if df is None:
    st.error("Could not resolve the symbol or fetch market data. Try adding .NS (NSE) or .BO (BSE).")
    st.stop()

hist = df.copy()
# store long-term cache for background learner
try:
    save_cache(hist, cache_path_for(resolved.split()[0], "hist"))
except Exception:
    pass

# Basic indicators
def add_basic_tech(df):
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta>0,0).rolling(14,min_periods=1).mean()
    loss = (-delta.where(delta<0,0)).rolling(14,min_periods=1).mean().replace(0,np.nan)
    df["RSI"] = (100 - (100/(1 + (gain / loss).fillna(1.0))))
    df["RSI"].fillna(50, inplace=True)
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    return df

hist = add_basic_tech(hist)
company = resolved.split()[0]
last_price = float(hist["Close"].iloc[-1])
prev = float(hist["Close"].iloc[-2]) if len(hist)>1 else last_price
st.subheader(f"{company}")
st.metric("Last Close", f"₹{last_price:.2f}", f"{(last_price-prev):+.2f} ({(last_price-prev)/prev*100:+.2f}%)")

# Chart
def plot_chart(df):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6,0.2,0.2])
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    if "MA20" in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"), row=1, col=1)
    if "MA50" in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
    if "RSI" in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=3, col=1)
    fig.update_layout(height=700, showlegend=True)
    return fig

st.plotly_chart(plot_chart(hist), use_container_width=True)

# ---------- SAFE call to BG.predict_for ----------
est_daily = 0.0
conf = 0.0
try:
    symbol_to_pass = resolved.split()[0] if resolved else user_symbol.strip().upper()
    # defensive check
    if not isinstance(symbol_to_pass, str) or symbol_to_pass.strip() == "":
        raise ValueError("Invalid resolved symbol")
    # call in try/except to avoid TypeError crash
    est_daily, conf = BG.predict_for(symbol_to_pass, days=pred_days)
    if est_daily is None: est_daily = 0.0
    if conf is None: conf = 0.0
except Exception as e:
    # log full traceback to file for debugging
    tb = traceback.format_exc()
    try:
        with open(os.path.join(CACHE_DIR, "last_error.log"), "a") as f:
            f.write(f"\n\n[{datetime.now().isoformat()}] Error calling BG.predict_for for '{user_symbol}':\n")
            f.write(tb)
    except Exception:
        pass
    # fallback defaults
    est_daily, conf = 0.0, 0.0

# compute predicted price for the horizon
pred_price = last_price * (1 + est_daily * (pred_days ** 0.5))
pred_pct = (pred_price - last_price) / last_price * 100

# fallback if low confidence
if conf < 0.12:
    fallback_daily = hist["Close"].pct_change().tail(5).mean() if len(hist) >= 5 else 0.0
    fallback_price = last_price * (1 + fallback_daily * (pred_days ** 0.5))
    pred_price = 0.55 * pred_price + 0.45 * fallback_price
    pred_pct = (pred_price - last_price) / last_price * 100

# signal & trading plan
if pred_pct >= 15:
    sig = "🔵 STRONG BUY"; style = "background:#d4edda;padding:12px;border-left:4px solid #28a745;border-radius:8px"
elif pred_pct >= 5:
    sig = "🟢 BUY"; style = "background:#d4edda;padding:12px;border-left:4px solid #28a745;border-radius:8px"
elif pred_pct <= -15:
    sig = "🔴 STRONG SELL"; style = "background:#f8d7da;padding:12px;border-left:4px solid #dc3545;border-radius:8px"
elif pred_pct <= -5:
    sig = "🔴 SELL"; style = "background:#f8d7da;padding:12px;border-left:4px solid #dc3545;border-radius:8px"
else:
    sig = "⚪ HOLD"; style = "background:#fff3cd;padding:12px;border-left:4px solid #ffc107;border-radius:8px"

st.markdown(f"<div style='{style}'><h3>Prediction: {sig}</h3><h2>₹{pred_price:.2f}</h2><p>{pred_pct:+.2f}% in {pred_days} days</p><p>Model confidence: {conf*100:.1f}%</p></div>", unsafe_allow_html=True)

# Trading targets
entry = last_price
stop = entry*(1-0.03) if pred_pct>=0 else entry*(1+0.03)
targets = [entry*(1 + (pred_pct/100.0) * mult * (pred_days ** 0.5)) for mult in (0.4,0.8,1.2,1.6)]
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
rsi_val = float(hist["RSI"].iloc[-1]) if "RSI" in hist.columns else np.nan
macd_val = float(hist["MACD"].iloc[-1]) if "MACD" in hist.columns else np.nan
ma50_val = float(hist["MA50"].iloc[-1]) if "MA50" in hist.columns else last_price
vol = float(hist["Close"].pct_change().rolling(20).std().iloc[-1])*np.sqrt(252)*100 if len(hist)>1 else 0.0
c1, c2, c3, c4 = st.columns(4)
c1.metric("RSI (14)", f"{rsi_val:.1f}", "Overbought" if rsi_val>70 else ("Oversold" if rsi_val<30 else "Neutral"))
c2.metric("MACD", f"{macd_val:.3f}", "Bullish" if macd_val>0 else "Bearish")
c3.metric("Trend (MA50)", "Bullish" if last_price>ma50_val else "Bearish")
c4.metric("Volatility (ann %)", f"{vol:.2f}%")

# add to universe
if st.button("Add this symbol to background training universe"):
    try:
        BG.add_to_universe(resolved.split()[0])
        st.success(f"Added {resolved.split()[0]} to training universe (will be sampled by background learner).")
    except Exception:
        st.error("Failed to add symbol to universe (see logs).")

# last scan results
if "last_recs" in st.session_state and st.session_state.last_recs:
    st.markdown("---")
    st.subheader("Last scan results")
    for r in st.session_state.last_recs[:12]:
        with st.expander(f"{r['symbol']} — {r['expected_pct']:+.1f}% | Conf {r['confidence']:.2f}"):
            st.write(f"Current: ₹{r['current']:.2f} | Target: ₹{r['target']:.2f}")
            if st.button(f"Analyze {r['symbol']}", key=f"an_{r['symbol']}"):
                st.experimental_set_query_params(symbol=r['symbol'])
                st.experimental_rerun()

st.markdown("---")
st.caption("Notes: Logs written to .jinni_cache/last_error.log for debugging.")

