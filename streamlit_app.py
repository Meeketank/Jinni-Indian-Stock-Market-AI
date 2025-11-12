# streamlit_app.py
"""
JINNI - Indian Stock Market AI (updated)
Fixes symbol resolution & improves fetch reliability (tries .NS, .BO, raw).
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading, time, random, os, pickle
from typing import Tuple, List, Dict

# ---------- Config ----------
MODEL_FILE = "jinni_model.pkl"
BG_INTERVAL = 20
BG_BATCH_SIZE = 20
N_LAGS = 6
MAX_UNIVERSE = 1200
DEFAULT_UNIVERSE = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","SBIN.NS"
]
PERSIST_EVERY = 60
DISCLAIMER = "⚠️ Disclaimer: Educational only. Not financial advice."

# ---------- Helpers ----------
def candidate_variants(symbol: str) -> List[str]:
    s = symbol.strip().upper()
    if not s:
        return []
    # If user included suffix, try that first
    if s.endswith(".NS") or s.endswith(".BO"):
        return [s, s.replace(".BO", ".NS"), s.replace(".NS", ".BO"), s.rstrip(".NS").rstrip(".BO"), s.rstrip(".NS").rstrip(".BO")]
    # else try NSE then BSE then raw
    return [s + ".NS", s + ".BO", s]

def safe_history_fetch(symbol: str, period="1y", attempts: int = 2, timeout_sec: int = 8):
    """
    Try to fetch history via yfinance, with a simple retry loop and explanatory message.
    Returns (hist_df or None, info dict or {}, error_message or None)
    """
    last_err = None
    for attempt in range(attempts):
        try:
            ticker = yf.Ticker(symbol)
            # Use auto_adjust True for adjusted prices; you can change to False if you want raw OHLC.
            hist = ticker.history(period=period, auto_adjust=True, timeout=timeout_sec)
            if hist is None or hist.empty:
                last_err = f"No data returned for {symbol}"
                continue
            info = {}
            try:
                info = ticker.info or {}
            except Exception:
                info = {}
            return hist.sort_index(), info, None
        except Exception as e:
            last_err = str(e)
            time.sleep(0.3)
    return None, {}, last_err

def resolve_user_symbol(user_symbol: str, period="1y") -> Tuple[str, pd.DataFrame, dict, List[str]]:
    """
    Try candidate variants for the user's symbol and return the first successful one.
    Returns (resolved_symbol or "", hist or None, info dict, list_of_attempts_and_results)
    """
    attempts_log = []
    for cand in candidate_variants(user_symbol):
        if not cand:
            continue
        hist, info, err = safe_history_fetch(cand, period=period, attempts=2)
        if hist is not None:
            attempts_log.append((cand, "OK"))
            return cand, hist, info, attempts_log
        else:
            attempts_log.append((cand, f"FAIL: {err}"))
    # none worked
    return "", None, {}, attempts_log

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def add_basic_tech(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for p in (5, 10, 20, 50):
        df[f"MA{p}"] = df["Close"].rolling(p, min_periods=1).mean()
    df["RSI"] = calc_rsi(df["Close"])
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Volatility"] = df["Close"].pct_change().rolling(20, min_periods=1).std() * np.sqrt(252) * 100
    return df

# ---------- Simple online models (same pattern as before) ----------
class OnlineReg:
    def __init__(self, n_features, lr=0.005):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
    def predict(self, X):
        X = np.atleast_2d(X)
        return X.dot(self.w) + self.b
    def partial_fit(self, X, y):
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
        self.w = s.get("w", self.w)
        self.b = s.get("b", self.b)

class OnlineClf:
    def __init__(self, n_features, lr=0.01):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
    def _sigmoid(self, z):
        z = np.clip(z, -50, 50)
        return 1.0/(1.0+np.exp(-z))
    def predict_proba(self, X):
        z = X.dot(self.w) + self.b
        return self._sigmoid(z)
    def partial_fit(self, X, y):
        if X.size == 0:
            return
        p = self.predict_proba(X)
        errs = p - y
        grad_w = (errs[:,None] * X).mean(axis=0)
        grad_b = errs.mean()
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b
    def state(self):
        return {"w": self.w, "b": self.b}
    def load_state(self, s):
        self.w = s.get("w", self.w)
        self.b = s.get("b", self.b)

def build_features(df, n_lags=N_LAGS):
    if df is None or len(df) < n_lags + 8:
        return np.empty((0, n_lags+3)), np.empty((0,))
    close = df["Close"].values
    returns = (close[1:] - close[:-1])/ (close[:-1] + 1e-12)
    ma20 = df["Close"].rolling(20, min_periods=1).mean().values
    ma50 = df["Close"].rolling(50, min_periods=1).mean().values
    rsi = calc_rsi(df["Close"]).values
    X, y = [], []
    for i in range(n_lags, len(returns)):
        lag = returns[i-n_lags:i]
        idx = i+1
        price = close[idx] if idx < len(close) else close[-1]
        ma_diff = (ma20[idx] - ma50[idx]) / (price if price != 0 else 1)
        rsi_n = (rsi[idx]/100.0) if idx < len(rsi) else 0.5
        vol = float(np.std(returns[max(0, i-20): i+1])) if i>=1 else 0.0
        feat = np.concatenate([lag, [ma_diff, rsi_n, vol]])
        X.append(feat)
        y.append(returns[i])
    return np.array(X), np.array(y)

# ---------- Background learner ----------
class BackgroundLearner:
    def __init__(self, path=MODEL_FILE):
        self.path = path
        self.n_lags = N_LAGS
        self.reg = OnlineReg(self.n_lags+3)
        self.clf = OnlineClf(self.n_lags+3)
        self.universe = DEFAULT_UNIVERSE.copy()
        self.metrics = {"dir_acc_ema": 0.5, "mae_ema": 0.5, "samples": 0}
        self.ema_alpha = 0.05
        self._stop = threading.Event()
        self._thread = None
        self.lock = threading.Lock()
        self._load()
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    d = pickle.load(f)
                self.reg.load_state(d.get("reg", {}))
                self.clf.load_state(d.get("clf", {}))
                self.universe = d.get("universe", self.universe)[:MAX_UNIVERSE]
                self.metrics.update(d.get("metrics", {}))
            except Exception:
                pass
    def _persist(self):
        try:
            with open(self.path, "wb") as f:
                pickle.dump({"reg": self.reg.state(), "clf": self.clf.state(), "universe": self.universe, "metrics": self.metrics}, f)
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
    def add_to_universe(self, symbol):
        if symbol not in self.universe:
            self.universe.append(symbol)
            self.universe = self.universe[:MAX_UNIVERSE]
    def _loop(self):
        while not self._stop.is_set():
            try:
                if not self.universe:
                    time.sleep(10); continue
                batch = random.sample(self.universe, min(BG_BATCH_SIZE, len(self.universe)))
                dir_scores = []
                maes = []
                sample_count = 0
                for s in batch:
                    df = safe_history_fetch(s, period="400d")[0]
                    if df is None or len(df) < self.n_lags + 8:
                        continue
                    X, y = build_features(df, self.n_lags)
                    if X.size == 0:
                        continue
                    preds = self.reg.predict(X)
                    mae = float(np.mean(np.abs(preds - y)))
                    dir_acc = float(np.mean(np.sign(preds) == np.sign(y)))
                    self.clf.partial_fit(X, (y > 0).astype(int))
                    self.reg.partial_fit(X, y)
                    dir_scores.append(dir_acc); maes.append(mae)
                    sample_count += len(y)
                    time.sleep(0.02)
                if sample_count > 0:
                    a = self.ema_alpha
                    batch_dir = float(np.mean(dir_scores)) if dir_scores else 0.0
                    batch_mae = float(np.mean(maes)) if maes else 0.0
                    with self.lock:
                        self.metrics["dir_acc_ema"] = a * batch_dir + (1-a) * self.metrics["dir_acc_ema"]
                        self.metrics["mae_ema"] = a * batch_mae + (1-a) * self.metrics["mae_ema"]
                        self.metrics["samples"] += sample_count
                    self._persist()
                time.sleep(BG_INTERVAL)
            except Exception:
                time.sleep(BG_INTERVAL)
    def get_metrics(self):
        with self.lock:
            return dict(self.metrics)
    def predict_for(self, symbol):
        # returns daily_return_est, confidence (0..1)
        df = safe_history_fetch(symbol, period="300d")[0]
        if df is None or len(df) < self.n_lags + 8:
            return 0.0, 0.0
        X, _ = build_features(df, self.n_lags)
        if X.size == 0:
            return 0.0, 0.0
        latest = X[-1].reshape(1, -1)
        pred = float(self.reg.predict(latest)[0])
        prob_up = float(self.clf.predict_proba(latest)[0])
        recent_returns = (df["Close"].values[1:] - df["Close"].values[:-1]) / (df["Close"].values[:-1] + 1e-12)
        vol = float(np.std(recent_returns[-60:])) if len(recent_returns) >= 1 else 0.0
        conf = max(0.0, min(0.99, prob_up * (1.0 - min(0.8, vol * 4.0))))
        signed = pred * (prob_up * 2 - 1)
        signed = float(np.clip(signed, -0.9, 0.9))
        return signed, conf
    def scan_high_potential(self, min_pct=10.0, days=7, samples=200):
        res = []
        if not self.universe:
            return res
        samp = random.sample(self.universe, min(samples, len(self.universe)))
        for s in samp:
            try:
                daily, conf = self.predict_for(s)
                expected_pct = daily * np.sqrt(max(1, days)) * 100
                if abs(expected_pct) >= abs(min_pct) and conf > 0.08:
                    hist = safe_history_fetch(s, period="120d")[0]
                    if hist is None:
                        continue
                    cur = float(hist["Close"].iloc[-1])
                    tgt = cur * (1 + daily * np.sqrt(max(1, days)))
                    res.append({"symbol": s, "expected_pct": expected_pct, "confidence": conf, "current": cur, "target": tgt})
            except Exception:
                continue
        res.sort(key=lambda r: abs(r["expected_pct"]), reverse=True)
        return res[:30]

# ---------- trading plan helper ----------
def compute_targets(current_price, predicted_return_frac, days):
    entry = current_price
    sl_pct = 0.03
    stop = entry * (1 - sl_pct) if predicted_return_frac >= 0 else entry * (1 + sl_pct)
    time_factor = np.sqrt(max(1, days/7))
    t1 = entry * (1 + predicted_return_frac * 0.4 * time_factor)
    t2 = entry * (1 + predicted_return_frac * 0.8 * time_factor)
    t3 = entry * (1 + predicted_return_frac * 1.2 * time_factor)
    t4 = entry * (1 + predicted_return_frac * 1.6 * time_factor)
    risk = abs(entry - stop)
    reward = abs(t2 - entry) if abs(t2 - entry) > 0 else 0.0
    rr = (reward / risk) if risk > 0 else 0.0
    return entry, stop, [t1, t2, t3, t4], rr

# ---------- Streamlit UI ----------
st.set_page_config(page_title="JINNI - Indian Stock AI", page_icon="🧞", layout="wide")
st.title("🧞 JINNI — Indian Stock Market AI")
st.markdown(DISCLAIMER)

# instantiate BG in session_state
if "bg" not in st.session_state:
    st.session_state.bg = BackgroundLearner()
    st.session_state.bg.start()
BG = st.session_state.bg

# Sidebar
with st.sidebar:
    st.header("Controls")
    user_symbol = st.text_input("Enter stock symbol (e.g., RELIANCE or RELIANCE.NS)", value="IDBI")
    pred_days = st.slider("Prediction horizon (days)", 1, 14, 7)
    st.markdown("---")
    st.subheader("Recommendations")
    if st.button("Find 10%+ Weekly Opportunities"):
        with st.spinner("Scanning (this samples BG universe)..."):
            recs = BG.scan_high_potential(min_pct=10.0, days=7, samples=350)
            st.session_state["last_recs"] = recs
            st.success(f"Scan finished — {len(recs)} candidates found (top results stored).")
    st.markdown("---")
    st.subheader("Model (background)")
    m = BG.get_metrics()
    st.metric("Directional accuracy (EMA)", f"{m['dir_acc_ema']*100:.2f}%")
    st.metric("MAE (EMA)", f"{m['mae_ema']:.4f}")
    st.markdown(f"Trained samples: {m['samples']:,}")

# Main area
st.header("Stock Analysis")
resolved, hist, info, attempts = resolve_user_symbol(user_symbol, period="2y")
if resolved:
    st.success(f"Resolved symbol: {resolved}")
    if attempts:
        st.write("Attempts log (tried candidates):")
        for a, r in attempts:
            st.write(f"- {a}: {r}")
    hist = add_basic_tech(hist)
    company = info.get("longName") or info.get("shortName") or resolved
    st.subheader(f"{company} — {resolved}")
    last = float(hist["Close"].iloc[-1])
    prev = float(hist["Close"].iloc[-2]) if len(hist)>1 else last
    st.metric("Last Close", f"₹{last:.2f}", f"{(last-prev):+.2f} ({(last-prev)/prev*100:+.2f}%)")
    # plot
    def plot_chart(df):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6,0.2,0.2])
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
        if "MA20" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=3, col=1)
        fig.update_layout(height=700, showlegend=True)
        return fig
    st.plotly_chart(plot_chart(hist), use_container_width=True)

    # model prediction
    daily, conf = BG.predict_for(resolved)
    pred_price = last * (1 + daily * np.sqrt(max(1, pred_days)))
    pred_pct = (pred_price - last)/last*100
    # fallback if conf low
    if conf < 0.12:
        fallback = hist["Close"].pct_change().tail(5).mean() * np.sqrt(max(1, pred_days)) * 100
        pred_price = 0.6 * pred_price + 0.4 * last * (1 + fallback/100)
        pred_pct = (pred_price - last)/last*100

    # signal
    if pred_pct >= 15:
        sig = "STRONG BUY"; box_style="background:#d4edda;padding:12px;border-left:4px solid #28a745;border-radius:8px"
    elif pred_pct >=5:
        sig="BUY"; box_style="background:#d4edda;padding:12px;border-left:4px solid #28a745;border-radius:8px"
    elif pred_pct <= -15:
        sig="STRONG SELL"; box_style="background:#f8d7da;padding:12px;border-left:4px solid #dc3545;border-radius:8px"
    elif pred_pct <= -5:
        sig="SELL"; box_style="background:#f8d7da;padding:12px;border-left:4px solid #dc3545;border-radius:8px"
    else:
        sig="HOLD"; box_style="background:#fff3cd;padding:12px;border-left:4px solid #ffc107;border-radius:8px"

    st.markdown(f"<div style='{box_style}'><h3>🔮 AI Prediction: {sig}</h3><h2>₹{pred_price:.2f}</h2><p>{pred_pct:+.2f}% in {pred_days} days</p><p>Model confidence: {conf*100:.1f}%</p></div>", unsafe_allow_html=True)

    entry, sl, targets, rr = compute_targets(last, pred_pct/100.0, pred_days)
    st.subheader("Trading Plan")
    st.write(f"- Entry: ₹{entry:.2f}")
    st.write(f"- Stop Loss: ₹{sl:.2f}")
    for i,t in enumerate(targets,1):
        st.write(f"- Target {i}: ₹{t:.2f} ({(t-entry)/entry*100:+.1f}%)")
    st.write(f"- Est Risk-Reward (Target 2): 1:{rr:.2f}")

    st.subheader("Technical Snapshot")
    rsi_val = float(hist["RSI"].iloc[-1])
    macd = float(hist["MACD"].iloc[-1])
    macd_sig = float(hist["MACD_Signal"].iloc[-1])
    ma50 = float(hist["MA50"].iloc[-1]) if "MA50" in hist.columns else last
    vol = float(hist["Volatility"].iloc[-1])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RSI (14)", f"{rsi_val:.1f}", "Overbought" if rsi_val>70 else ("Oversold" if rsi_val<30 else "Neutral"))
    c2.metric("MACD", f"{macd:.3f}", "Bullish" if macd>macd_sig else "Bearish")
    c3.metric("Trend (vs MA50)", "Bullish" if last>ma50 else "Bearish")
    c4.metric("Volatility (ann %)", f"{vol:.2f}%")

    st.subheader("Fundamentals (yfinance)")
    pe = info.get("trailingPE")
    pb = info.get("priceToBook")
    mc = info.get("marketCap")
    f1, f2, f3 = st.columns(3)
    f1.metric("P/E", f"{pe:.2f}" if pe else "N/A")
    f2.metric("P/B", f"{pb:.2f}" if pb else "N/A")
    f3.metric("Market Cap (Cr)", f"₹{mc/1e7:.2f} Cr" if mc else "N/A")

    st.subheader("Model metrics & reasoning")
    mm = BG.get_metrics()
    st.write(f"- Directional accuracy (EMA): {mm['dir_acc_ema']*100:.2f}%")
    st.write(f"- MAE (EMA): {mm['mae_ema']:.4f}")
    st.write(f"- Samples trained: {mm['samples']:,}")
    st.write("**Why prediction:**")
    st.write(f"- Recent 5-day mean return: {hist['Close'].pct_change().tail(5).mean()*100:+.2f}%")
    st.write(f"- RSI: {rsi_val:.1f}")
    st.write(f"- MA50 position: {'above' if last>ma50 else 'below'} MA50")

    # allow adding to training universe explicitly
    if st.button("Add this resolved symbol to background training universe"):
        BG.add_to_universe(resolved)
        st.success(f"Added {resolved} to training universe.")

else:
    st.error("Could not resolve the symbol or fetch market data. Attempts:")
    st.write("Tried candidates (symbol: result):")
    _, _, _, attempts = resolve_user_symbol(user_symbol, period="2y")
    for a, r in attempts:
        st.write(f"- {a}: {r}")
    st.info("Tip: Try adding .NS for NSE symbols (e.g., RELIANCE.NS). Some BSE tickers may not be present on Yahoo Finance.")

# show last scan results if present
if "last_recs" in st.session_state and st.session_state["last_recs"]:
    st.markdown("---")
    st.header("Last Recommendations (from scan)")
    for r in st.session_state["last_recs"][:12]:
        with st.expander(f"{r['symbol']} — Est {r['expected_pct']:+.1f}% | Conf {r['confidence']:.2f}"):
            st.write(f"Current: ₹{r['current']:.2f} | Target: ₹{r['target']:.2f} | Conf: {r['confidence']:.2f}")
            if st.button(f"Analyze {r['symbol']}", key=f"an_{r['symbol']}"):
                st.experimental_set_query_params(symbol=r['symbol'])
                st.experimental_rerun()

st.markdown("---")
st.markdown("<small>Background learner trains automatically. If a symbol cannot be resolved it likely isn't available on Yahoo Finance under that ticker. Try NSE (.NS) or BSE (.BO) variants.</small>", unsafe_allow_html=True)
