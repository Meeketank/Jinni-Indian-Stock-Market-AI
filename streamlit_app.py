# streamlit_app.py
"""
JINNI - AI-Powered Indian Stock Market Analysis (single-file)
- Accepts any user-typed Indian stock symbol.
- Background online training runs automatically (daemon thread).
- Sidebar "Find 10%+ Weekly Opportunities" scans the learner's internal universe;
  each result has "Check more" to run full analysis for that stock.
- Lightweight online regressor/classifier; metrics persist to disk.
- NOT financial advice.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading, time, random, os, pickle
from typing import Tuple, List, Dict

# -----------------------------
# Config
# -----------------------------
MODEL_FILE = "jinni_model.pkl"
UNIVERSE_CACHE = "jinni_universe.pkl"
BG_INTERVAL = 20           # seconds between training batches
BG_BATCH_SIZE = 20         # tickers processed per training loop
N_LAGS = 6                 # lag features for returns
MAX_UNIVERSE = 1200        # cap universe size to be practical with yfinance limits
DEFAULT_UNIVERSE = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","SBIN.NS",
    "AXISBANK.NS","ITC.NS","MARUTI.NS","ONGC.NS","BAJFINANCE.NS",
    "ASIANPAINT.NS","HEROMOTOCO.NS","NESTLEIND.NS","ULTRACEMCO.NS","SUNPHARMA.NS"
]
PERSIST_EVERY = 60         # persist model every N seconds (approx)
DISCLAIMER = "⚠️ Disclaimer: Educational only. Not financial advice."

# -----------------------------
# Utility functions
# -----------------------------
def try_resolve_symbol(s: str) -> List[str]:
    """
    Return a prioritized list of symbols to try in yfinance for the user's input.
    If user supplies suffix (.NS or .BO) that will be tried first.
    Otherwise try .NS then .BO then raw.
    """
    s = s.strip().upper()
    if not s:
        return []
    if s.endswith(".NS") or s.endswith(".BO"):
        return [s]
    # try common variants: prefer NSE (.NS)
    return [s + ".NS", s + ".BO", s]

def fetch_history(ticker: str, period: str = "1y"):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            return None, {}
        # protect types
        hist = hist.sort_index()
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        return hist, info
    except Exception:
        return None, {}

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_prev = (df["High"] - df["Close"].shift(1)).abs()
    low_prev = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr.fillna(0.0)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for p in (5, 10, 20, 50, 100, 200):
        df[f"MA{p}"] = df["Close"].rolling(p, min_periods=1).mean()
    df["RSI"] = calc_rsi(df["Close"])
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Volatility"] = df["Close"].pct_change().rolling(20, min_periods=1).std() * np.sqrt(252) * 100
    df["ATR"] = calc_atr(df)
    df["Vol_Ratio"] = df["Volume"] / (df["Volume"].rolling(20, min_periods=1).mean().replace(0,1))
    return df

def build_features(df: pd.DataFrame, n_lags: int = N_LAGS):
    """
    Build features: last n_lags daily returns + ma_diff + rsi_norm + atr_norm + vol
    Returns X (m x f) and y (m,)
    """
    if df is None or len(df) < n_lags + 10:
        return np.empty((0, n_lags + 4)), np.empty((0,))
    close = df["Close"].values
    returns = (close[1:] - close[:-1]) / (close[:-1] + 1e-12)  # length L-1
    ma20 = df["Close"].rolling(20, min_periods=1).mean().values
    ma50 = df["Close"].rolling(50, min_periods=1).mean().values
    rsi = calc_rsi(df["Close"]).values
    atr = calc_atr(df).values
    X, y = [], []
    for i in range(n_lags, len(returns)):
        lag = returns[i - n_lags: i]
        idx = i + 1  # day of price that corresponds to this sample
        price = close[idx] if idx < len(close) else close[-1]
        ma_diff = (ma20[idx] - ma50[idx]) / (price if price != 0 else 1)
        rsi_n = (rsi[idx] / 100.0) if idx < len(rsi) else 0.5
        atr_n = (atr[idx] / price) if price != 0 else 0.0
        vol = float(np.std(returns[max(0, i - 20): i + 1])) if i >= 1 else 0.0
        feat = np.concatenate([lag, [ma_diff, rsi_n, atr_n, vol]])
        X.append(feat)
        y.append(returns[i])
    return np.array(X), np.array(y)

# -----------------------------
# Lightweight online models
# -----------------------------
class OnlineRegressor:
    def __init__(self, n_features: int, lr: float = 0.005, l2: float = 1e-4):
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
    def predict(self, X):
        if X.ndim == 1:
            return float(X.dot(self.w) + self.b)
        return X.dot(self.w) + self.b
    def partial_fit(self, X, y):
        if X.size == 0:
            return
        preds = self.predict(X)
        errs = preds - y
        grad_w = (errs[:, None] * X).mean(axis=0) + self.l2 * self.w
        grad_b = errs.mean()
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b
    def state(self):
        return {"w": self.w, "b": self.b}
    def load_state(self, d):
        self.w = d.get("w", self.w)
        self.b = d.get("b", self.b)

class OnlineClassifier:
    def __init__(self, n_features: int, lr: float = 0.01, l2: float = 1e-4):
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
    def _sigmoid(self, z):
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))
    def predict_proba(self, X):
        z = X.dot(self.w) + self.b
        return self._sigmoid(z)
    def partial_fit(self, X, y):
        if X.size == 0:
            return
        p = self.predict_proba(X)
        errs = p - y
        grad_w = (errs[:, None] * X).mean(axis=0) + self.l2 * self.w
        grad_b = errs.mean()
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b
    def state(self):
        return {"w": self.w, "b": self.b}
    def load_state(self, d):
        self.w = d.get("w", self.w)
        self.b = d.get("b", self.b)

# -----------------------------
# Background learner (daemon)
# -----------------------------
class BackgroundLearner:
    def __init__(self,
                 model_path: str = MODEL_FILE,
                 universe: List[str] = None,
                 n_lags: int = N_LAGS,
                 interval: int = BG_INTERVAL,
                 batch_size: int = BG_BATCH_SIZE):
        self.model_path = model_path
        self.n_lags = n_lags
        self.interval = max(5, interval)
        self.batch_size = max(1, batch_size)
        self.universe = universe or DEFAULT_UNIVERSE.copy()
        self.universe = list(dict.fromkeys(self.universe))[:MAX_UNIVERSE]
        self.reg = OnlineRegressor(n_features=self.n_lags + 4)
        self.clf = OnlineClassifier(n_features=self.n_lags + 4)
        # metrics: EMA smoothing for stability
        self.metrics = {"dir_acc_ema": 0.5, "mae_ema": 0.5, "samples": 0}
        self.ema_alpha = 0.05
        self._stop = threading.Event()
        self._thread = None
        self.lock = threading.Lock()
        self._load()
        self._last_persist = time.time()

    def _load(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                self.reg.load_state(data.get("reg", {}))
                self.clf.load_state(data.get("clf", {}))
                m = data.get("metrics", {})
                self.metrics.update(m)
                u = data.get("universe", [])
                if u:
                    self.universe = u[:MAX_UNIVERSE]
            except Exception:
                pass

    def _persist(self):
        try:
            d = {"reg": self.reg.state(), "clf": self.clf.state(), "metrics": self.metrics, "universe": self.universe}
            with open(self.model_path, "wb") as f:
                pickle.dump(d, f)
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

    def add_to_universe(self, symbols: List[str]):
        # add user-provided or external tickers to the training universe (avoid duplicates)
        for s in symbols:
            if s not in self.universe:
                self.universe.append(s)
        # cap
        self.universe = self.universe[:MAX_UNIVERSE]

    def _fetch_hist(self, ticker: str, days: int = 400):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=f"{days}d", interval="1d", auto_adjust=False)
            if df is None or df.empty:
                return None
            return df
        except Exception:
            return None

    def _update_metrics(self, batch_dir: float, batch_mae: float, n_new: int):
        a = self.ema_alpha
        with self.lock:
            self.metrics["dir_acc_ema"] = a * batch_dir + (1 - a) * self.metrics["dir_acc_ema"]
            self.metrics["mae_ema"] = a * batch_mae + (1 - a) * self.metrics["mae_ema"]
            self.metrics["samples"] += n_new

    def _loop(self):
        while not self._stop.is_set():
            try:
                if not self.universe:
                    time.sleep(self.interval)
                    continue
                batch = random.sample(self.universe, min(self.batch_size, len(self.universe)))
                batch_dir_scores = []
                batch_maes = []
                total_new = 0
                for sym in batch:
                    if self._stop.is_set():
                        break
                    df = self._fetch_hist(sym, days=400)
                    if df is None or len(df) < self.n_lags + 10:
                        continue
                    X, y = build_features(df, n_lags=self.n_lags)
                    if X.size == 0:
                        continue
                    # evaluate
                    preds = self.reg.predict(X)
                    mae = float(np.mean(np.abs(preds - y)))
                    dir_acc = float(np.mean(np.sign(preds) == np.sign(y)))
                    # train online
                    self.clf.partial_fit(X, (y > 0).astype(int))
                    self.reg.partial_fit(X, y)
                    batch_dir_scores.append(dir_acc)
                    batch_maes.append(mae)
                    total_new += len(y)
                    # light pause between tickers to avoid throttling
                    time.sleep(0.05)
                if total_new > 0:
                    self._update_metrics(np.mean(batch_dir_scores), np.mean(batch_maes), total_new)
                # persist periodically
                if time.time() - self._last_persist > PERSIST_EVERY:
                    self._persist()
                    self._last_persist = time.time()
                time.sleep(self.interval)
            except Exception:
                time.sleep(self.interval)

    def get_metrics(self) -> Dict:
        with self.lock:
            return dict(self.metrics)

    def predict_for(self, ticker: str) -> Tuple[float, float]:
        """
        Predict signed next-day return and confidence. Returns (daily_return, confidence 0..1).
        """
        df = self._fetch_hist(ticker, days=300)
        if df is None or len(df) < self.n_lags + 8:
            return 0.0, 0.0
        X, _ = build_features(df, n_lags=self.n_lags)
        if X.size == 0:
            return 0.0, 0.0
        latest = X[-1].reshape(1, -1)
        pred = float(self.reg.predict(latest)[0])  # daily return estimate
        prob_up = float(self.clf.predict_proba(latest)[0])
        recent_returns = (df["Close"].values[1:] - df["Close"].values[:-1]) / (df["Close"].values[:-1] + 1e-12)
        vol = float(np.std(recent_returns[-60:])) if len(recent_returns) >= 1 else 0.0
        # confidence heuristic: classifier confidence adjusted by volatility
        conf = max(0.0, min(0.99, prob_up * (1.0 - min(0.8, vol * 4.0))))
        # signed daily biased by prob_up
        signed = pred * (prob_up * 2 - 1)
        signed = float(np.clip(signed, -0.9, 0.9))
        return signed, conf

    def scan_high_potential(self, min_pct: float = 10.0, days: int = 7, sample_count: int = 300) -> List[Dict]:
        """
        Scan a random sample of the universe for candidates with absolute predicted return >= min_pct over 'days'.
        Returns top candidates sorted by absolute predicted return.
        """
        results = []
        if not self.universe:
            return results
        sample = random.sample(self.universe, min(sample_count, len(self.universe)))
        for sym in sample:
            try:
                daily, conf = self.predict_for(sym)
                # convert daily to horizon (conservative) using sqrt scaling
                expected_pct = daily * np.sqrt(max(1, days)) * 100
                if abs(expected_pct) >= abs(min_pct) and conf > 0.10:
                    hist = self._fetch_hist(sym, days=120)
                    if hist is None or hist.empty:
                        continue
                    current = float(hist["Close"].iloc[-1])
                    target_price = current * (1 + daily * np.sqrt(max(1, days)))
                    results.append({
                        "symbol": sym,
                        "expected_pct": expected_pct,
                        "confidence": conf,
                        "current": current,
                        "target": target_price
                    })
            except Exception:
                continue
        results.sort(key=lambda r: abs(r["expected_pct"]), reverse=True)
        return results[:30]

# -----------------------------
# Helpers: prediction combine / trading plan
# -----------------------------
def compute_targets(current_price: float, predicted_return_frac: float, days: int):
    """
    Return entry, stop_loss, targets[1..4], risk_reward_est
    predicted_return_frac is fractional e.g. 0.12 for +12%
    """
    entry = current_price
    # stop loss based on predicted direction and volatility
    sl_pct = 0.03  # base 3%
    stop_loss = entry * (1 - sl_pct) if predicted_return_frac >= 0 else entry * (1 + sl_pct)
    time_factor = np.sqrt(max(1, days / 7))
    # targets scaled conservatively
    t1 = entry * (1 + predicted_return_frac * 0.4 * time_factor)
    t2 = entry * (1 + predicted_return_frac * 0.8 * time_factor)
    t3 = entry * (1 + predicted_return_frac * 1.2 * time_factor)
    t4 = entry * (1 + predicted_return_frac * 1.6 * time_factor)
    risk = abs(entry - stop_loss)
    reward = abs(t2 - entry) if abs(t2 - entry) > 0 else 0.0
    rr = (reward / risk) if risk > 0 else 0.0
    return entry, stop_loss, [t1, t2, t3, t4], rr

# -----------------------------
# UI: Streamlit app
# -----------------------------
st.set_page_config(page_title="JINNI - Indian Stock AI", page_icon="🧞", layout="wide")
st.markdown("<h1 style='text-align:center'>🧞 JINNI — Indian Stock Market AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray'>Background training runs automatically. Enter any Indian stock symbol below.</p>", unsafe_allow_html=True)
st.markdown(DISCLAIMER)

# instantiate and start learner globally
if "bg" not in st.session_state:
    st.session_state.bg = BackgroundLearner()
    st.session_state.bg.start()

BG = st.session_state.bg

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    user_input = st.text_input("Enter stock symbol (e.g., RELIANCE or RELIANCE.NS)", value="RELIANCE")
    pred_days = st.slider("Prediction horizon (days)", 1, 14, 7)
    st.markdown("---")
    st.subheader("Recommendations")
    if st.button("Find 10%+ Weekly Opportunities"):
        with st.spinner("Scanning for high-potential candidates..."):
            recs = BG.scan_high_potential(min_pct=10.0, days=7, sample_count=350)
            if not recs:
                st.info("No good candidates found in this sampling. Try again later.")
            else:
                # store recs in session for "check more" interactions
                st.session_state["last_recs"] = recs
                st.success(f"Found {len(recs)} candidates (showing top {min(10, len(recs))}).")
    st.markdown("---")
    st.subheader("Model (background)")
    metrics = BG.get_metrics()
    st.metric("Directional accuracy (EMA)", f"{metrics['dir_acc_ema']*100:.2f}%")
    st.metric("MAE (EMA)", f"{metrics['mae_ema']:.4f}")
    st.markdown(f"Trained samples: {metrics['samples']:,}")
    st.markdown("""
    <small>Model is an online regressor+classifier trained continuously on sampled market history.
    Accuracy increases as training samples accumulate.</small>
    """, unsafe_allow_html=True)

# Main area: if we have recommendations show brief list
col_main, col_side = st.columns([3, 1])

with col_side:
    st.markdown("### Quick Actions")
    if st.button("Add current symbol to training universe"):
        # resolve symbol variants and add working yfinance variant to universe
        variants = try_resolve_symbol(user_input)
        added = []
        for v in variants:
            hist, _ = fetch_history(v, period="1y")
            if hist is not None:
                BG.add_to_universe([v])
                added.append(v)
                break
        if added:
            st.success(f"Added {added[0]} to training universe.")
        else:
            st.error("Could not resolve symbol to market data; not added.")
    if st.button("Force persist model now"):
        BG._persist()
        st.success("Model persisted to disk.")

with col_main:
    st.header("Stock analysis")
    # Resolve user symbol preferences
    resolved = None
    variants = try_resolve_symbol(user_input)
    for v in variants:
        hist, info = fetch_history(v, period="2y")
        if hist is not None:
            resolved = v
            break
    if resolved is None:
        st.error("Could not fetch data for that symbol. Try adding .NS or .BO suffix.")
    else:
        hist = add_technical_indicators(hist)
        company = info.get("longName") or info.get("shortName") or resolved
        st.subheader(f"{company} — {resolved}")
        last_close = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last_close
        st.metric("Last Close", f"₹{last_close:.2f}", f"{(last_close-prev_close):+.2f} ({(last_close-prev_close)/prev_close*100:+.2f}%)")

        # plot
        def plot_tech(df):
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
            if "MA20" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=3, col=1)
            fig.update_layout(height=700, showlegend=True)
            return fig
        st.plotly_chart(plot_tech(hist), use_container_width=True)

        # AI Prediction from BG learner
        daily_pred, conf = BG.predict_for(resolved)
        # scale daily_pred to pred_days (conservative sqrt)
        pred_price = last_close * (1 + daily_pred * np.sqrt(max(1, pred_days)))
        pred_pct = (pred_price - last_close) / last_close * 100

        # fallback simple momentum prediction if model low confidence
        if conf < 0.12:
            fallback_pct = hist["Close"].pct_change().tail(5).mean() * np.sqrt(max(1, pred_days)) * 100
            # combine with small weight towards fallback
            pred_price = 0.6 * pred_price + 0.4 * last_close * (1 + fallback_pct/100)
            pred_pct = (pred_price - last_close) / last_close * 100

        # Signal thresholds
        if pred_pct >= 15:
            signal = "STRONG BUY"
            css = "background-color:#d4edda; border-left:4px solid #28a745; padding:10px; border-radius:8px;"
        elif pred_pct >= 5:
            signal = "BUY"
            css = "background-color:#d4edda; border-left:4px solid #28a745; padding:10px; border-radius:8px;"
        elif pred_pct <= -15:
            signal = "STRONG SELL"
            css = "background-color:#f8d7da; border-left:4px solid #dc3545; padding:10px; border-radius:8px;"
        elif pred_pct <= -5:
            signal = "SELL"
            css = "background-color:#f8d7da; border-left:4px solid #dc3545; padding:10px; border-radius:8px;"
        else:
            signal = "HOLD"
            css = "background-color:#fff3cd; border-left:4px solid #ffc107; padding:10px; border-radius:8px;"

        st.markdown(f"<div style='{css}'>", unsafe_allow_html=True)
        st.markdown(f"### 🔮 AI Prediction: {signal}")
        st.markdown(f"**Predicted price in {pred_days} days:** ₹{pred_price:.2f} ({pred_pct:+.2f}%)")
        st.markdown(f"**Model confidence:** {conf*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        # Trading plan
        entry, stop_loss, targets, rr = compute_targets(last_close, pred_pct/100.0, pred_days)
        st.subheader("Trading Plan")
        st.write(f"- Entry: ₹{entry:.2f}")
        st.write(f"- Stop Loss: ₹{stop_loss:.2f}")
        for i, t in enumerate(targets, 1):
            st.write(f"- Target {i}: ₹{t:.2f}  ({(t-entry)/entry*100:+.1f}%)")
        st.write(f"- Estimated Risk-Reward (using Target 2): 1:{rr:.2f}")

        # Technical snapshot
        st.subheader("Technical Snapshot")
        rsi_val = float(hist["RSI"].iloc[-1])
        macd_val = float(hist["MACD"].iloc[-1])
        macd_sig = float(hist["MACD_Signal"].iloc[-1])
        ma50 = float(hist["MA50"].iloc[-1]) if "MA50" in hist.columns else None
        vol = float(hist["Volatility"].iloc[-1])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RSI (14)", f"{rsi_val:.1f}", "Overbought" if rsi_val>70 else ("Oversold" if rsi_val<30 else "Neutral"))
        c2.metric("MACD", f"{macd_val:.3f}", "Bullish" if macd_val>macd_sig else "Bearish")
        c3.metric("Trend (vs MA50)", "Bullish" if last_close>ma50 else "Bearish")
        c4.metric("Volatility (ann %)", f"{vol:.2f}%")

        # Fundamentals
        st.subheader("Fundamental Metrics (from yfinance)")
        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        mc = info.get("marketCap")
        div_y = info.get("dividendYield")
        f1, f2, f3 = st.columns(3)
        f1.metric("P/E", f"{pe:.2f}" if pe else "N/A")
        f2.metric("P/B", f"{pb:.2f}" if pb else "N/A")
        f3.metric("Market Cap (Cr)", f"₹{mc/1e7:.2f} Cr" if mc else "N/A")
        if div_y:
            st.write(f"- Dividend Yield: {div_y*100:.2f}%")

        # show model metrics and a short reasoning block
        st.subheader("Model Reasoning & Metrics")
        mm = BG.get_metrics()
        st.write(f"- Directional accuracy (EMA): {mm['dir_acc_ema']*100:.2f}%")
        st.write(f"- MAE (EMA): {mm['mae_ema']:.4f}")
        st.write(f"- Training samples: {mm['samples']:,}")
        st.markdown("**Why this prediction?**")
        reasons = []
        reasons.append(f"1. Momentum: Recent 5-day mean return = {hist['Close'].pct_change().tail(5).mean()*100:+.2f}%")
        reasons.append(f"2. RSI: {rsi_val:.1f} ({'overbought' if rsi_val>70 else ('oversold' if rsi_val<30 else 'neutral')})")
        reasons.append(f"3. MA50 position: {'above' if last_close>ma50 else 'below'} the 50-day MA")
        reasons.append(f"4. Model confidence: {conf*100:.1f}% (classifier+regressor ensemble)")
        st.write("\n".join(reasons))

# If there are recommendations stored show them below main area
if "last_recs" in st.session_state:
    st.markdown("---")
    st.header("Recommended candidates (from last scan)")
    recs = st.session_state["last_recs"]
    for r in recs[:12]:
        with st.expander(f"{r['symbol']} — Est {r['expected_pct']:+.1f}% | Conf {r['confidence']:.2f}"):
            # Provide "Check more" button which triggers full analysis for the selected candidate
            if st.button(f"Check more: {r['symbol']}", key=f"check_{r['symbol']}"):
                # reuse main analysis area logic by redirecting user_input and re-running
                # (here we'll just display the same details inline)
                hist_c, info_c = fetch_history(r['symbol'], period="2y")
                if hist_c is None:
                    st.error("Could not fetch details for candidate.")
                else:
                    hist_c = add_technical_indicators(hist_c)
                    last = float(hist_c["Close"].iloc[-1])
                    st.write(f"Current price: ₹{last:.2f}")
                    # AI prediction via BG
                    daily_c, conf_c = BG.predict_for(r['symbol'])
                    pred_price_c = last * (1 + daily_c * np.sqrt(max(1, 7)))
                    st.write(f"Predicted (7d): ₹{pred_price_c:.2f} ({(pred_price_c-last)/last*100:+.2f}%)")
                    st.write(f"Confidence: {conf_c:.2f}")
                    # Trading plan
                    entry_c, sl_c, targets_c, rr_c = compute_targets(last, (pred_price_c-last)/last, 7)
                    st.write(f"Entry: ₹{entry_c:.2f}")
                    st.write(f"Stop Loss: ₹{sl_c:.2f}")
                    for i,t in enumerate(targets_c,1):
                        st.write(f"Target {i}: ₹{t:.2f} ({(t-entry_c)/entry_c*100:+.1f}%)")
                    st.plotly_chart(plot_tech(hist_c), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<small style='color:gray'>Background learner runs in the background and improves as more historical samples are trained. For stronger performance replace the online models with robust ML models (XGBoost, LightGBM, Transformers) and add alternative data (news, orderflow, options).</small>", unsafe_allow_html=True)
