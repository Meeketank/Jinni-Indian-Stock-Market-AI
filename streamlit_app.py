# streamlit_app.py
"""
JINNI - AI-Powered Indian Stock Market Analysis (Streamlit)
Updated: Universe auto-fetcher for NSE/BSE (tries to build full exchange lists and trains on them).
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, threading, time, os, pickle, random, requests, io
from typing import List, Tuple, Dict

warnings.filterwarnings("ignore")

# -----------------------
# Config
# -----------------------
MODEL_PATH = "jinni_bg_full_universe.pkl"
UNIVERSE_REFRESH_INTERVAL = 24 * 3600  # refresh ticker list once a day
DEFAULT_FALLBACK_TICKERS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","HINDUNILVR.NS","BHARTIARTL.NS",
    "KOTAKBANK.NS","LT.NS","SBIN.NS","AXISBANK.NS","ITC.NS","MARUTI.NS","ONGC.NS","BAJFINANCE.NS"
]

# -----------------------
# Helper: fetch exchange tickers
# -----------------------
def fetch_nse_list() -> List[str]:
    """
    Attempt to fetch official NSE list (EQUITY_L.csv) from NSE archives.
    Returns list of NSE symbols without suffix (we will append .NS later).
    """
    urls = [
        # Common archive path; frequently available (may require headers)
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        # Another mirror used in some setups
        "https://www1.nseindia.com/content/equities/EQUITY_L.csv",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; JINNIBot/1.0; +https://example.com)",
        "Accept": "text/csv, */*; q=0.01",
        "Referer": "https://www.nseindia.com"
    }
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and resp.text.strip():
                df = pd.read_csv(io.StringIO(resp.text))
                # often column 'SYMBOL' contains tickers
                if 'SYMBOL' in df.columns:
                    syms = df['SYMBOL'].astype(str).str.strip().tolist()
                    return [s for s in syms if s and not s.startswith(" ")]
                # fallback try first column
                first_col = df.columns[0]
                syms = df[first_col].astype(str).str.strip().tolist()
                return syms
        except Exception:
            continue
    return []

def fetch_github_nse_list() -> List[str]:
    """
    Fallback: try to retrieve lists published on public GitHub repos (raw csv)
    Note: reliability depends on external repo availability.
    """
    candidates = [
        "https://raw.githubusercontent.com/neerajkumarorg/indian-stock-data/master/nse-listed.csv",
        "https://raw.githubusercontent.com/datasets/india-nifty50/master/data/nifty50.csv",
        "https://raw.githubusercontent.com/ozlerhakan/mongodb-json-files/master/datasets/nse-stock-symbols.csv",
    ]
    for url in candidates:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.text.strip():
                try:
                    df = pd.read_csv(io.StringIO(resp.text))
                    # try common columns
                    for col in ['symbol','Symbol','SYMBOL','code','Code']:
                        if col in df.columns:
                            syms = df[col].astype(str).str.strip().tolist()
                            return syms
                    # if single-column CSV
                    if df.shape[1] == 1:
                        syms = df.iloc[:,0].astype(str).str.strip().tolist()
                        return syms
                except Exception:
                    # fallback: parse raw lines
                    lines = [l.strip() for l in resp.text.splitlines() if l.strip()]
                    # ignore header if contains alphabetic
                    if len(lines) > 2:
                        # try splitting first line to detect separator
                        return [l.split(",")[0].strip() for l in lines[1:]]
        except Exception:
            continue
    return []

def fetch_bse_list() -> List[str]:
    """
    Attempt to fetch BSE ticker list. BSE has no simple static CSV endpoint for crawling;
    try known mirrors / GitHub fallbacks for BSE symbol lists.
    """
    candidates = [
        "https://raw.githubusercontent.com/varunpant/indian-stock-list/master/bse.csv",
        "https://raw.githubusercontent.com/datasets/india-nifty50/master/data/bse500.csv",
        "https://raw.githubusercontent.com/shubh26/indian-stock-list/master/BSE-Symbols.csv",
    ]
    for url in candidates:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.text.strip():
                try:
                    df = pd.read_csv(io.StringIO(resp.text))
                    for col in ['Symbol','SYMBOL','CODE','code','symbol']:
                        if col in df.columns:
                            syms = df[col].astype(str).str.strip().tolist()
                            return syms
                    if df.shape[1] == 1:
                        return df.iloc[:,0].astype(str).str.strip().tolist()
                except Exception:
                    lines = [l.strip() for l in resp.text.splitlines() if l.strip()]
                    if len(lines) > 2:
                        return [l.split(",")[0].strip() for l in lines[1:]]
        except Exception:
            continue
    return []

def fetch_all_exchange_tickers(force_refresh: bool = False) -> List[str]:
    """
    Build the universe by trying multiple sources for NSE & BSE.
    Caches the universe in 'tickers_cache.pkl' with timestamp to avoid repeated downloads.
    Returns list of yfinance-ready tickers (with .NS or .BO).
    """
    cache_file = "tickers_cache.pkl"
    now = time.time()
    # load cache
    try:
        if os.path.exists(cache_file) and not force_refresh:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            ts = cache.get("ts", 0)
            if now - ts < UNIVERSE_REFRESH_INTERVAL:
                return cache.get("tickers", DEFAULT_FALLBACK_TICKERS.copy())
    except Exception:
        pass

    # Attempt NSE official list
    nse = []
    try:
        nse = fetch_nse_list()
        if not nse:
            nse = fetch_github_nse_list()
    except Exception:
        nse = []

    # Attempt BSE list
    bse = []
    try:
        bse = fetch_bse_list()
    except Exception:
        bse = []

    # normalize: remove non-alphanumeric entries & duplicates
    def clean_list(lst):
        cleaned = []
        for s in lst:
            if not isinstance(s, str):
                continue
            s2 = s.strip().upper()
            s2 = s2.replace(" ", "")
            if s2 and len(s2) <= 12:
                cleaned.append(s2)
        # remove obviously non-stock tokens
        cleaned = [c for c in cleaned if not any(x in c for x in ["NIFTY","SENSEX","INDEX","ETF"])]
        return list(dict.fromkeys(cleaned))

    nse_clean = clean_list(nse)
    bse_clean = clean_list(bse)

    # build yfinance tickers: NSE -> .NS, BSE -> .BO
    yf_nse = [s + ".NS" for s in nse_clean if s.isalpha() or any(ch.isalnum() for ch in s)]
    yf_bse = [s + ".BO" for s in bse_clean if s.isalpha() or any(ch.isalnum() for ch in s)]

    # merge, dedupe
    universe = []
    # include NSE first (most active) then BSE
    for t in yf_nse + yf_bse:
        if t not in universe:
            universe.append(t)

    # fallback: if universe too small, use default list
    if not universe or len(universe) < 50:
        universe = DEFAULT_FALLBACK_TICKERS.copy()

    # save cache
    try:
        with open(cache_file, "wb") as f:
            pickle.dump({"ts": now, "tickers": universe}, f)
    except Exception:
        pass

    return universe

# -----------------------
# Small feature / model utilities (same as earlier)
# -----------------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_prev = np.abs(df['High'] - df['Close'].shift(1))
    low_prev = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr.fillna(method='ffill').fillna(0.0)

def build_features(df: pd.DataFrame, n_lags: int = 6):
    if df is None or len(df) < n_lags + 8:
        return np.empty((0, n_lags + 4)), np.empty((0,))
    close = df['Close'].values
    returns = (close[1:] - close[:-1]) / close[:-1]
    ma20 = df['Close'].rolling(20, min_periods=1).mean().values
    ma50 = df['Close'].rolling(50, min_periods=1).mean().values
    rsi = compute_rsi(df['Close']).values
    atr = compute_atr(df).values
    X, y = [], []
    for i in range(n_lags, len(returns)):
        lag = returns[i - n_lags: i]
        idx = i + 1
        price = close[idx] if idx < len(close) else close[-1]
        ma_diff = (ma20[idx] - ma50[idx]) / (price if price != 0 else 1)
        rsi_val = (rsi[idx] / 100.0) if idx < len(rsi) else 0.5
        atr_norm = (atr[idx] / price) if price != 0 else 0.0
        vol = float(np.std(returns[max(0, i - 20): i + 1])) if i >= 1 else 0.0
        feat = np.concatenate([lag, [ma_diff, rsi_val, atr_norm, vol]])
        X.append(feat)
        y.append(returns[i])
    return np.array(X), np.array(y)

# -----------------------
# Online learners (small)
# -----------------------
class OnlineRegressor:
    def __init__(self, n_features: int, lr: float = 0.004, l2: float = 1e-4):
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
    def to_dict(self): return {"w_reg": self.w, "b_reg": self.b}
    def from_dict(self, d): 
        self.w = d.get("w_reg", self.w); self.b = d.get("b_reg", self.b)

class OnlineClassifier:
    def __init__(self, n_features: int, lr: float = 0.008, l2: float = 1e-4):
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
    def _sigmoid(self, x): return 1.0/(1.0+np.exp(-np.clip(x, -50, 50)))
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
    def to_dict(self): return {"w_clf": self.w, "b_clf": self.b}
    def from_dict(self, d): self.w = d.get("w_clf", self.w); self.b = d.get("b_clf", self.b)

# -----------------------
# Background Learner (uses dynamic universe)
# -----------------------
class BackgroundLearner:
    def __init__(self,
                 model_path: str = MODEL_PATH,
                 n_lags: int = 6,
                 interval_sec: int = 20,
                 batch_size: int = 20):
        self.model_path = model_path
        self.n_lags = n_lags
        self.interval_sec = max(5, interval_sec)
        self.batch_size = max(1, batch_size)
        self.universe = []  # will be populated by fetch_all_exchange_tickers
        self.reg = OnlineRegressor(n_features=self.n_lags + 4, lr=0.004)
        self.clf = OnlineClassifier(n_features=self.n_lags + 4, lr=0.008)
        self.metrics = {"dir_acc_ema": 0.5, "mae_ema": 0.5, "samples": 0}
        self.ema_alpha = 0.06
        self._stop = threading.Event()
        self._thread = None
        self.lock = threading.Lock()
        self._load()
        # try to populate universe now
        self.refresh_universe()

    def _load(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    d = pickle.load(f)
                self.reg.from_dict(d)
                self.clf.from_dict(d)
                if "metrics" in d:
                    self.metrics.update(d["metrics"])
            except Exception:
                pass

    def _save(self):
        try:
            with open(self.model_path, "wb") as f:
                out = {}
                out.update(self.reg.to_dict())
                out.update(self.clf.to_dict())
                out["metrics"] = self.metrics
                pickle.dump(out, f)
        except Exception:
            pass

    def refresh_universe(self, force: bool = False):
        try:
            tickers = fetch_all_exchange_tickers(force_refresh=force)
            # small random shuffle for training variety
            random.shuffle(tickers)
            # limit to reasonable cap to protect rate-limits (if you want all, remove cap)
            # We'll keep up to 1500 tickers for training by default (practical).
            cap = 3000  # user can increase; keep large to include most exchange symbols
            self.universe = tickers[:cap]
        except Exception:
            # fallback
            self.universe = DEFAULT_FALLBACK_TICKERS.copy()

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

    def _fetch_history(self, ticker: str, days: int = 400):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=f"{days}d", interval="1d", auto_adjust=False)
            if df is None or df.empty:
                return None
            return df
        except Exception:
            return None

    def _update_ema(self, batch_dir: float, batch_mae: float, n_new: int):
        with self.lock:
            a = self.ema_alpha
            self.metrics["dir_acc_ema"] = a * batch_dir + (1 - a) * self.metrics["dir_acc_ema"]
            self.metrics["mae_ema"] = a * batch_mae + (1 - a) * self.metrics["mae_ema"]
            self.metrics["samples"] += n_new

    def _loop(self):
        while not self._stop.is_set():
            try:
                if not self.universe:
                    time.sleep(self.interval_sec)
                    continue
                # pick random batch of tickers across universe
                batch = random.sample(self.universe, min(self.batch_size, len(self.universe)))
                dir_accs, maes, n_total = [], [], 0
                for sym in batch:
                    if self._stop.is_set():
                        break
                    df = self._fetch_history(sym, days=400)
                    if df is None or len(df) < self.n_lags + 8:
                        continue
                    X, y = build_features(df, n_lags=self.n_lags)
                    if X.size == 0:
                        continue
                    preds = self.reg.predict(X)
                    mae = float(np.mean(np.abs(preds - y)))
                    dir_acc = float(np.mean(np.sign(preds) == np.sign(y)))
                    # train classifier and regressor online
                    self.clf.partial_fit(X, (y > 0).astype(int))
                    self.reg.partial_fit(X, y)
                    dir_accs.append(dir_acc)
                    maes.append(mae)
                    n_total += len(y)
                    time.sleep(0.06)  # small pause to reduce burst
                if n_total > 0:
                    batch_dir = float(np.mean(dir_accs)) if dir_accs else 0.5
                    batch_mae = float(np.mean(maes)) if maes else 0.5
                    self._update_ema(batch_dir, batch_mae, n_total)
                    # occasionally persist
                    if random.random() < 0.25:
                        self._save()
                # occasionally refresh universe (daily)
                if random.random() < 0.02:
                    self.refresh_universe()
                time.sleep(self.interval_sec)
            except Exception:
                time.sleep(self.interval_sec)

    def get_performance_report(self) -> Dict:
        with self.lock:
            return {
                "directional_accuracy": float(self.metrics["dir_acc_ema"]),
                "mae": float(self.metrics["mae_ema"]),
                "total_samples": int(self.metrics["samples"])
            }

    def predict_for_ticker(self, ticker: str) -> Tuple[float, float]:
        df = self._fetch_history(ticker, days=300)
        if df is None or len(df) < self.n_lags + 5:
            return 0.0, 0.0
        X, _ = build_features(df, n_lags=self.n_lags)
        if X.size == 0:
            return 0.0, 0.0
        latest = X[-1].reshape(1, -1)
        pred_daily = float(self.reg.predict(latest)[0])
        prob_up = float(self.clf.predict_proba(latest)[0])
        recent_returns = (df['Close'].values[1:] - df['Close'].values[:-1]) / df['Close'].values[:-1]
        vol = float(np.std(recent_returns[-60:])) if len(recent_returns) >= 1 else 0.0
        confidence = float(max(0.0, min(0.99, prob_up * (1.0 - min(0.85, vol * 4.0)))))
        # signed daily return biased by classifier
        signed = pred_daily * (prob_up * 2 - 1)
        signed = float(np.clip(signed, -0.8, 0.8))
        return signed, confidence

    def get_high_return_recommendations(self, min_return: float = 10.0, days: int = 7) -> List[Dict]:
        recs = []
        if not self.universe:
            return recs
        # sample subset for speed but cover randomly across full universe
        sample_universe = random.sample(self.universe, min(400, len(self.universe)))
        for sym in sample_universe:
            try:
                daily, conf = self.predict_for_ticker(sym)
                expected_pct = daily * np.sqrt(max(1, days)) * 100
                if abs(expected_pct) < abs(min_return):
                    continue
                if conf < 0.12:
                    continue
                df = self._fetch_history(sym, days=100)
                if df is None or df.empty:
                    continue
                cp = float(df['Close'].iloc[-1])
                tp = cp * (1 + daily * np.sqrt(max(1, days)))
                recs.append({
                    "symbol": sym,
                    "current_price": cp,
                    "target_price": tp,
                    "predicted_return_pct": expected_pct,
                    "confidence": conf,
                    "signal": "BUY" if expected_pct > 0 else "SELL"
                })
            except Exception:
                continue
        recs.sort(key=lambda x: abs(x['predicted_return_pct']), reverse=True)
        return recs[:12]

    def is_running(self):
        return bool(self._thread and self._thread.is_alive())

    def add_to_universe(self, list_of_symbols: List[str]):
        for s in list_of_symbols:
            if s not in self.universe:
                self.universe.append(s)

    def get_universe_sample(self, n=20):
        return random.sample(self.universe, min(n, len(self.universe)))

# -----------------------
# Instantiate learner and start
# -----------------------
BG = BackgroundLearner(model_path=MODEL_PATH, n_lags=6, interval_sec=18, batch_size=25)
BG.start()

# -----------------------
# Streamlit UI (light)
# -----------------------
st.set_page_config(page_title="JINNI - AI Stock Analysis (Full Exchange Universe)", page_icon="🧞", layout="wide")
st.markdown("<h1 style='text-align:center'>🧞 JINNI — AI Stock Analysis (NSE + BSE Universe)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666'>Learner trains across the full exchange tickers (fetched automatically). Refreshes daily.</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Controls")
    symbol = st.text_input("Stock (NSE/BSE)", value="RELIANCE.NS", help="Enter ticker with .NS or .BO (or plain symbol)")
    pred_days = st.slider("Prediction horizon (days)", 1, 90, 7)
    if st.button("Refresh universe now"):
        with st.spinner("Refreshing universe..."):
            BG.refresh_universe(force=True)
        st.success("Universe refreshed (may take a moment).")
    if st.button("Show sample universe"):
        sample = BG.get_universe_sample(30)
        st.write(sample)
    st.divider()
    perf = BG.get_performance_report()
    st.subheader("Learner metrics (EMA)")
    st.metric("Directional Acc (EMA)", f"{perf['directional_accuracy']*100:.2f}%")
    st.metric("MAE (EMA)", f"{perf['mae']:.4f}")
    st.metric("Trained samples", f"{perf['total_samples']:,}")

# small caching wrapper for fetch
@st.cache_data(ttl=300)
def fetch_hist(symbol: str, period: str = "6mo"):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period, auto_adjust=True)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        if hist is None or hist.empty:
            return None, {}
        return hist, info
    except Exception:
        return None, {}

def add_tech(df):
    df = df.copy()
    for p in [5,10,20,50,100,200]:
        df[f"MA{p}"] = df["Close"].rolling(p, min_periods=1).mean()
    df["RSI"] = compute_rsi(df["Close"])
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Volatility"] = df["Close"].pct_change().rolling(20, min_periods=1).std() * np.sqrt(252) * 100
    return df

def chart(df, symbol):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.6,0.2,0.2])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='price'), row=1, col=1)
    if 'MA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20'), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='vol'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=3, col=1)
    fig.update_layout(height=760, showlegend=True)
    return fig

# Main analysis
if symbol:
    # normalize input
    s = symbol.strip().upper()
    if not s.endswith(".NS") and not s.endswith(".BO"):
        # try both NSE and BSE if user didn't specify; prefer NSE
        try_symbols = [s + ".NS", s + ".BO", s]
    else:
        try_symbols = [s]
    hist, info = None, {}
    for ts in try_symbols:
        hist, info = fetch_hist(ts, period="1y")
        if hist is not None:
            symbol = ts
            break
    if hist is None:
        st.error("No data found for the symbol you entered. Try adding .NS or .BO or check ticker.")
    else:
        df = add_tech(hist)
        st.subheader(f"{info.get('longName', symbol)} — {symbol}")
        price = float(df['Close'].iloc[-1])
        prev = float(df['Close'].iloc[-2]) if len(df)>1 else price
        st.metric("Price", f"₹{price:.2f}", f"{(price-prev):+.2f} ({(price-prev)/prev*100:+.2f}%)")
        st.plotly_chart(chart(df, symbol), use_container_width=True)

        # prediction from BG learner
        bg_daily, bg_conf = BG.predict_for_ticker(symbol)
        # scale to horizon conservatively
        pred_price_bg = price * (1 + bg_daily * np.sqrt(max(1, pred_days)))
        # heuristic fallback (short momentum)
        heuristic_price = price * (1 + (df['Close'].pct_change().tail(5).mean() or 0) * np.sqrt(max(1, pred_days/7)))
        final_price = pred_price_bg * bg_conf + heuristic_price * (1 - bg_conf) if bg_conf>0 else heuristic_price
        pred_pct = (final_price - price)/price*100

        # display
        if pred_pct >= 15:
            css = "buy-signal"; rec = "STRONG BUY"
        elif pred_pct >= 5:
            css = "buy-signal"; rec = "BUY"
        elif pred_pct <= -15:
            css = "sell-signal"; rec = "STRONG SELL"
        elif pred_pct <= -5:
            css = "sell-signal"; rec = "SELL"
        else:
            css = "hold-signal"; rec = "HOLD"

        st.markdown(f"<div class='prediction-box {css}'>", unsafe_allow_html=True)
        st.markdown(f"### {rec}")
        st.markdown(f"<h2>₹{final_price:.2f}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p>{pred_pct:+.2f}% in {pred_days} days — model conf {bg_conf:.2f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # compute targets
        entry = price
        base_sl = 0.03
        sl = entry*(1-base_sl) if pred_pct>=0 else entry*(1+base_sl)
        time_factor = np.sqrt(max(1, pred_days/7))
        targets = [entry + (pred_pct/100)*entry*mult*time_factor for mult in [0.4,0.8,1.2,1.6]]
        st.subheader("Trading Plan")
        st.write(f"Entry: ₹{entry:.2f}")
        st.write(f"Stop Loss: ₹{sl:.2f}")
        for i,t in enumerate(targets,1):
            st.write(f"Target {i}: ₹{t:.2f} ({(t-entry)/entry*100:+.1f}%)")
        perf = BG.get_performance_report()
        st.subheader("Learner Live Metrics")
        st.write(f"- Directional Acc (EMA): {perf['directional_accuracy']*100:.2f}%")
        st.write(f"- MAE (EMA): {perf['mae']:.4f}")
        st.write(f"- Samples trained: {perf['total_samples']:,}")

# show a scan button
if st.button("Scan sample high-potential (10%+)"):
    with st.spinner("Scanning big random subset..."):
        recs = BG.get_high_return_recommendations(min_return=10, days=7)
        if not recs:
            st.info("No high-confidence opportunities found in this sample.")
        else:
            for r in recs:
                st.markdown(f"**{r['symbol']}** — Est: {r['predicted_return_pct']:+.2f}% | Conf: {r['confidence']:.2f}")
                st.write(f"Entry: ₹{r['current_price']:.2f}  Target: ₹{r['target_price']:.2f}")

st.markdown("<hr><small style='color:#666'>Universe fetch attempts NSE official -> GitHub fallbacks -> BSE fallbacks. If internet sources fail, the app uses a small fallback list. To include every single listed security reliably in production you may: (1) host a daily-updated CSV of tickers, or (2) provide exchange-licensed APIs.</small>", unsafe_allow_html=True)
