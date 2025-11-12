# streamlit_app.py
"""
JINNI - AI-Powered Indian Stock Market Analysis System (single-file)
Full app with:
 - robust symbol resolver + caching
 - continuous background learner (online SGD)
 - scanning/recommendations with progress UI
 - technical & basic fundamental analysis
 - multi-horizon predictions & targets
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading, time, os, pickle, random, math, json, requests
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

# ---- CONFIG ----
APP_DIR = os.path.abspath(".")
CACHE_DIR = os.path.join(APP_DIR, ".jinni_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(CACHE_DIR, "background_model.pkl")
SEARCH_CACHE = os.path.join(CACHE_DIR, "yahoo_search_cache.pkl")
SYMBOLS_CACHE = os.path.join(CACHE_DIR, "universe_symbols.pkl")

# Universe source (small fallback list if download fails). You can replace with your own list.
FALLBACK_UNIVERSE = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","SBIN.NS",
    "AXISBANK.NS","ITC.NS","WIPRO.NS","ULTRACEMCO.NS","ASIANPAINT.NS"
]

YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"

# Rate limit control (seconds between external yfinance/search calls)
MIN_FETCH_INTERVAL = 0.4
_last_fetch_time = 0.0

def throttle_fetch():
    global _last_fetch_time
    now = time.time()
    delta = now - _last_fetch_time
    if delta < MIN_FETCH_INTERVAL:
        time.sleep(MIN_FETCH_INTERVAL - delta)
    _last_fetch_time = time.time()

# ---- UTILS: caching helpers ----
def save_pickle(obj, path):
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except Exception:
        pass

def load_pickle(path):
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None

def log_error(msg):
    try:
        with open(os.path.join(CACHE_DIR, "last_error.log"), "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass

# ---- SYMBOL RESOLVER (robust) ----
_search_cache = load_pickle(SEARCH_CACHE) or {}

def save_search_cache():
    save_pickle(_search_cache, SEARCH_CACHE)

def yahoo_search_symbol(query: str) -> List[str]:
    q = (query or "").strip()
    if not q:
        return []
    qk = q.upper()
    if qk in _search_cache:
        return _search_cache[qk]
    try:
        throttle_fetch()
        r = requests.get(YAHOO_SEARCH_URL, params={"q": qk, "quotesCount": 10, "newsCount": 0}, timeout=6)
        candidates = []
        if r.status_code == 200:
            j = r.json()
            for itm in j.get("quotes", []):
                sym = itm.get("symbol")
                if sym:
                    candidates.append(sym.upper())
        else:
            candidates = []
    except Exception as e:
        log_error(f"yahoo_search_symbol error for {q}: {e}")
        candidates = []
    if not candidates:
        # Add common variants
        candidates = [qk + suf for suf in ["", ".NS", ".BO", ".NSE", ".BSE"]]
    # dedupe
    seen = set(); out = []
    for s in candidates:
        if s not in seen:
            out.append(s); seen.add(s)
    _search_cache[qk] = out
    save_search_cache()
    return out

def resolve_symbol(user_input: str, period="1y") -> Tuple[pd.DataFrame,str]:
    """Resolve a user-entered stock name/symbol to a ticker and return history dataframe (or (None,None))."""
    if not user_input:
        return None, None
    s = user_input.strip()
    candidates = []
    # If looks like "RELIANCE.NS" try directly first
    if "." in s:
        candidates.append(s.upper())
    # add naive variants
    s_up = s.upper()
    for v in [s_up, s_up + ".NS", s_up + ".BO", s_up + ".NSE", s_up + ".BSE"]:
        if v not in candidates:
            candidates.append(v)
    # add Yahoo suggestions (prefer them first)
    try:
        ycs = yahoo_search_symbol(s)
        for yc in reversed(ycs):  # insert in front
            if yc not in candidates:
                candidates.insert(0, yc)
    except Exception:
        pass

    last_err = None
    for c in candidates:
        try:
            # try cached history first
            cache_path = os.path.join(CACHE_DIR, f"hist_{c}.pkl")
            hist = load_pickle(cache_path)
            if hist is not None and not hist.empty:
                return hist, c
            throttle_fetch()
            tk = yf.Ticker(c)
            df = tk.history(period=period, interval="1d", auto_adjust=True)
            if df is not None and not df.empty:
                save_pickle(df, cache_path)
                return df, c
            else:
                last_err = f"No data for {c}"
        except Exception as e:
            last_err = str(e)
            time.sleep(0.2)
            continue
    log_error(f"resolve_symbol failed for {user_input}. Tried {candidates}. Last err: {last_err}")
    return None, None

# ---- UNIVERSE LOADER ----
def load_universe(force_refresh=False) -> List[str]:
    """Load NSE+BSE symbol universe. Tries cached file, then remote CSV (recommend replacing URL), else fallback list."""
    if not force_refresh:
        cached = load_pickle(SYMBOLS_CACHE)
        if cached and isinstance(cached, list) and len(cached) > 10:
            return cached
    # try to download a prepared symbol list (you may replace this URL with your own)
    try:
        throttle_fetch()
        # Sample GitHub raw file that users often host — replace with real source if you have one.
        url = "https://raw.githubusercontent.com/Meeket/sample-data/main/indian_tickers_sample.csv"
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            txt = r.text
            syms = []
            for line in txt.splitlines():
                line = line.strip()
                if not line: continue
                s = line.split(",")[0].strip()
                if not s.endswith(".NS") and not s.endswith(".BO"):
                    # try normalize to .NS as primary
                    s = s + ".NS"
                syms.append(s.upper())
            if len(syms) > 10:
                save_pickle(syms, SYMBOLS_CACHE)
                return syms
    except Exception:
        pass
    # fallback
    save_pickle(FALLBACK_UNIVERSE, SYMBOLS_CACHE)
    return FALLBACK_UNIVERSE

# ---- ONLINE MODEL (simple SGD linear) ----
class OnlineLinearModel:
    def __init__(self, n_features=5, lr=0.003):
        self.n = n_features
        self.lr = lr
        self.w = np.zeros(self.n, dtype=float)
        self.b = 0.0
        self.initialized = False

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.array([])
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
        self.initialized = True

    def save(self, path):
        try:
            with open(path, "wb") as f:
                pickle.dump({"w": self.w, "b": self.b}, f)
        except Exception:
            pass

    def load(self, path):
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    self.w = data.get("w", self.w)
                    self.b = data.get("b", self.b)
                    self.initialized = True
        except Exception:
            pass

# ---- BACKGROUND LEARNER ----
class BackgroundLearner:
    """
    - Maintains universe list (can be expanded).
    - Background thread samples tickers, builds lag-return features and trains model.
    - Keeps EMA metrics for directional accuracy and MAE.
    """
    def __init__(self, model_path=MODEL_PATH, n_lags=5, interval_sec=20):
        self.model_path = model_path
        self.n_lags = n_lags
        self.interval_sec = max(5, interval_sec)
        self.universe = load_universe()
        self.model = OnlineLinearModel(n_features=self.n_lags, lr=0.003)
        self.model.load(self.model_path)
        # EMA metrics
        self.ema_alpha = 0.02  # slow EMA for stability
        self.metrics = {
            "directional_accuracy_ema": 0.0,
            "mae_ema": np.nan,
            "trained_samples": 0
        }
        self._stop = threading.Event()
        self._thread = None
        self.lock = threading.Lock()

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

    def is_running(self):
        return self._thread and self._thread.is_alive()

    def add_symbol(self, symbol: str):
        s = symbol.upper().strip()
        if s and s not in self.universe:
            self.universe.append(s)
            save_pickle(self.universe, SYMBOLS_CACHE)

    def _fetch_history(self, ticker: str, days=120):
        try:
            throttle_fetch()
            tk = yf.Ticker(ticker)
            df = tk.history(period=f"{days}d", interval="1d", auto_adjust=True)
            if df is None or df.empty:
                return None
            return df
        except Exception as e:
            log_error(f"_fetch_history {ticker}: {e}")
            return None

    def _build_features(self, df: pd.DataFrame):
        """returns X (m,n_lags) and y (m,) where y is next-day return"""
        if df is None or len(df) <= self.n_lags + 1:
            return np.empty((0, self.n_lags)), np.empty((0,))
        closes = df['Close'].values
        rets = (closes[1:] - closes[:-1]) / closes[:-1]
        X, y = [], []
        for i in range(self.n_lags, len(rets)):
            X.append(rets[i-self.n_lags:i])
            y.append(rets[i])
        return np.array(X), np.array(y)

    def _update_ema_metrics(self, preds, truths):
        if preds.size == 0:
            return
        mae = np.mean(np.abs(preds - truths))
        dir_acc = float(np.mean(np.sign(preds) == np.sign(truths)))
        with self.lock:
            # EMA update
            prev_da = self.metrics["directional_accuracy_ema"]
            prev_mae = self.metrics["mae_ema"]
            a = self.ema_alpha
            self.metrics["directional_accuracy_ema"] = prev_da * (1 - a) + dir_acc * a if not math.isnan(prev_da) else dir_acc
            self.metrics["mae_ema"] = prev_mae * (1 - a) + mae * a if not math.isnan(prev_mae) else mae
            self.metrics["trained_samples"] += len(truths)

    def _loop(self):
        while not self._stop.is_set():
            try:
                # sample a small batch randomly from universe to avoid huge load
                if not self.universe:
                    time.sleep(self.interval_sec)
                    continue
                sample_count = min(6, max(1, len(self.universe)//10 or 1))
                batch = random.sample(self.universe, sample_count)
                for sym in batch:
                    if self._stop.is_set(): break
                    df = self._fetch_history(sym, days=180)
                    X, y = self._build_features(df)
                    if X.size == 0: 
                        continue
                    preds = self.model.predict(X)
                    self._update_ema_metrics(preds, y)
                    # incremental update with small mini-batches
                    # shuffle for SGD diversity
                    idx = np.arange(len(y))
                    np.random.shuffle(idx)
                    split = max(1, len(y)//4)
                    for start in range(0, len(y), split):
                        end = min(len(y), start+split)
                        self.model.partial_fit(X[idx[start:end]], y[idx[start:end]])
                    # sometimes persist
                    if random.random() < 0.15:
                        self.model.save(self.model_path)
                    time.sleep(0.2)
                time.sleep(self.interval_sec)
            except Exception as e:
                log_error(f"BackgroundLearner loop error: {e}")
                time.sleep(self.interval_sec)

    def predict_for(self, ticker: str, n_lags=None) -> Tuple[float, float]:
        """Return predicted next-day return (float) and confidence (0-100)"""
        n = n_lags or self.n_lags
        df = self._fetch_history(ticker, days=90)
        if df is None or len(df) <= n:
            return 0.0, 0.0
        closes = df['Close'].values
        rets = (closes[1:] - closes[:-1]) / closes[:-1]
        latest = rets[-n:]
        if len(latest) < n:
            latest = np.concatenate([np.zeros(n - len(latest)), latest])
        pred = float(self.model.predict(latest.reshape(1,-1))[0])
        vol = float(np.std(rets[-20:])) if len(rets) >= 6 else 0.0
        conf = max(0.0, min(100.0, (1.0 - vol * 8.0) * 100.0))
        return pred, conf

    def scan_high_potential(self, min_pct: float=7.0, days:int=7, sample_limit:int=200, progress_cb=None) -> List[Dict]:
        """
        Scan up to sample_limit tickers from universe and return items with predicted_return >= min_pct (abs).
        progress_cb: function(i, total) called during loop (for streamlit progress)
        """
        results = []
        tickers = list(self.universe)
        if not tickers:
            return results
        random.shuffle(tickers)
        tickers = tickers[:sample_limit]
        total = len(tickers)
        for i, t in enumerate(tickers, start=1):
            if progress_cb:
                progress_cb(i, total)
            try:
                pred_ret, conf = self.predict_for(t)
                # convert next-day predicted return to days horizon
                est_pct = pred_ret * days * 100.0
                # only return high-confidence results (conf > 30) to reduce spam
                if abs(est_pct) >= min_pct and conf > 25:
                    # fetch last close
                    throttle_fetch()
                    tk = yf.Ticker(t)
                    df = tk.history(period="5d", interval="1d", auto_adjust=True)
                    if df is None or df.empty:
                        continue
                    last = float(df['Close'].iloc[-1])
                    results.append({
                        "symbol": t,
                        "est_pct": est_pct,
                        "confidence": conf,
                        "last_close": last
                    })
            except Exception as e:
                log_error(f"scan_high_potential error {t}: {e}")
                continue
        # sort by descending estimated pct
        results.sort(key=lambda x: abs(x['est_pct']), reverse=True)
        return results

    def get_metrics(self):
        with self.lock:
            return dict(self.metrics)

# ---- TECHNICAL INDICATORS & PREDICTION UTILITIES ----
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for p in [5,10,20,50,100,200]:
        df[f"MA{p}"] = df['Close'].rolling(window=p, min_periods=1).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta>0,0).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta<0,0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100/(1+rs))
    df['RSI'].fillna(50, inplace=True)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_M'] = df['Close'].rolling(20, min_periods=1).mean()
    bb_std = df['Close'].rolling(20, min_periods=1).std().fillna(0)
    df['BB_U'] = df['BB_M'] + 2*bb_std
    df['BB_L'] = df['BB_M'] - 2*bb_std
    df['Vol_MA'] = df['Volume'].rolling(20, min_periods=1).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA'].replace(0,1)
    df['Volatility'] = df['Close'].pct_change().rolling(20, min_periods=1).std()*np.sqrt(252)*100
    return df

def enhanced_model_prediction(df: pd.DataFrame, days:int=7) -> float:
    """A rule-based strong-enough forecast to combine with BG learner. Returns predicted price (absolute)."""
    if df is None or df.empty:
        return 0.0
    df = calculate_indicators(df)
    current = float(df['Close'].iloc[-1])
    recent_5 = df['Close'].pct_change().tail(5).mean()
    recent_20 = df['Close'].pct_change().tail(20).mean()
    ma20 = df['MA20'].iloc[-1] if 'MA20' in df else current
    ma50 = df['MA50'].iloc[-1] if 'MA50' in df else current
    trend = 0.05 if ma20 > ma50 else -0.03
    momentum = recent_5*0.6 + recent_20*0.4
    vol = df['Volatility'].iloc[-1] / 100.0 if 'Volatility' in df else 0.2
    # Combine
    change = (momentum + trend) * math.sqrt(days/7) - vol*0.05
    change = max(-0.6, min(0.6, change))
    return current * (1 + change)

def compute_targets(current_price:float, predicted_price:float, days:int) -> Tuple[float,float,List[float],float]:
    entry = current_price
    pred_return = (predicted_price - current_price)/current_price
    # stop loss 3% for buys, 3% for shorts (flip)
    stop_loss = entry*(1 - 0.03) if pred_return >= 0 else entry*(1 + 0.03)
    # create 4 targets conservatively scaling with pred_return and duration
    mult_time = math.sqrt(max(1, days)/7)
    targets = []
    for factor in [0.3,0.6,1.0,1.5]:
        targets.append(entry + pred_return*entry*factor*mult_time)
    reward = abs(targets[1]-entry)
    risk = abs(entry-stop_loss)
    rr = (reward/risk) if risk>0 else 0.0
    return entry, stop_loss, targets, rr

# ---- PLOTTING ----
def create_chart(df: pd.DataFrame, symbol:str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.55,0.2,0.25])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    # add moving averages
    for ma in ['MA20','MA50','MA200']:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(width=1.5)), row=1, col=1)
    # volume
    colors = ['green' if r['Close']>=r['Open'] else 'red' for _,r in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
    # RSI + MACD
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash='dash', line_color='red', row=3, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='green', row=3, col=1)
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'), row=3, col=1)
    fig.update_layout(title=f"{symbol} - Price & Indicators", height=820, xaxis_rangeslider_visible=False)
    return fig

# ---- FUNDAMENTAL ANALYZER (basic from yfinance info) ----
def extract_fundamentals(info:dict) -> Dict:
    out = {}
    if not info:
        return out
    out['trailingPE'] = info.get('trailingPE')
    out['forwardPE'] = info.get('forwardPE')
    out['priceToBook'] = info.get('priceToBook')
    out['marketCap'] = info.get('marketCap')
    out['dividendYield'] = info.get('dividendYield')
    out['beta'] = info.get('beta')
    return out

# ---- APP UI ----
st.set_page_config(page_title="JINNI - Indian Stock Market AI", page_icon="🧞", layout="wide")
st.markdown("<h1 style='text-align:center'>🧞 JINNI — Indian Stock Market AI</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#666'>Background training runs automatically. Educational only — not financial advice.</div>", unsafe_allow_html=True)
st.write("---")

# instantiate background learner once
if "BG" not in st.session_state:
    st.session_state.BG = BackgroundLearner(interval_sec=15)
    st.session_state.BG.start()

BG = st.session_state.BG

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    input_symbol = st.text_input("Enter stock symbol (e.g., RELIANCE or RELIANCE.NS)", value="RELIANCE")
    horizon_days = st.selectbox("Prediction horizon (days)", options=[1,3,7,14,30,60,90,180,365,1825], index=2)
    st.markdown("**Recommendations**")
    min_expected = st.number_input("Minimum expected move (%)", value=7.0, min_value=1.0, step=1.0)
    max_scan = st.number_input("Max symbols to scan now", min_value=50, max_value=len(BG.universe) if BG.universe else 500, value=min(300, max(50, len(BG.universe) or 100)), step=25)
    st.write("---")
    st.subheader("Model (background)")
    metrics = BG.get_metrics()
    st.metric("Directional accuracy (EMA)", f"{metrics.get('directional_accuracy_ema',0.0)*100:.2f}%")
    st.metric("MAE (EMA)", f"{metrics.get('mae_ema',float('nan')):.4f}")
    st.write(f"Trained samples: {metrics.get('trained_samples',0):,}")
    st.write("Model: online regressor (fast SGD). Improves as more data is trained.")
    st.write("---")
    if st.button("Populate Universe (download symbols)"):
        st.info("Fetching symbol list — this may take a few seconds.")
        syms = load_universe(force_refresh=True)
        st.success(f"Universe loaded with {len(syms):,} symbols.")

# Main controls area
col1, col2 = st.columns([1,2])

with col1:
    st.markdown("### Quick Actions")
    if st.button("Scan & Recommend (use model)"):
        st.session_state.scan_results = None
        st.session_state.scan_status = "running"
        # run scanning in this request with progress bar
        progress = st.progress(0)
        placeholder = st.empty()
        try:
            results = []
            def progress_cb(i,total):
                progress.progress(min(100, int((i/total)*100)))
            results = BG.scan_high_potential(min_pct=min_expected, days=horizon_days, sample_limit=int(max_scan), progress_cb=progress_cb)
            st.session_state.scan_results = results
            st.session_state.scan_status = "done"
            if not results:
                st.info("No high-confidence opportunities found in this scan.")
            else:
                st.success(f"Found {len(results)} candidate(s).")
        except Exception as e:
            st.error("Scan failed — see logs.")
            log_error(f"scan_button_error: {e}")
            st.session_state.scan_status = "error"

    if st.session_state.get("scan_results"):
        st.markdown("#### Scanner results (click to expand)")
        for r in st.session_state["scan_results"][:20]:
            with st.expander(f"{r['symbol']} — Est: {r['est_pct']:+.2f}% | Conf: {r['confidence']:.1f}%"):
                st.write(f"Last close: ₹{r['last_close']:.2f}")
                if st.button(f"Analyze {r['symbol']}", key=f"an_{r['symbol']}"):
                    # put symbol into input and run analysis below
                    input_symbol = r['symbol']
                    st.session_state['analysis_symbol'] = input_symbol

with col2:
    st.markdown("### Stock analysis")
    symbol_to_analyze = st.session_state.get('analysis_symbol') or input_symbol
    st.write(f"Symbol (raw input): {symbol_to_analyze}")
    # resolve symbol
    period_map = {
        1:"5d", 3:"5d", 7:"1mo", 14:"3mo", 30:"3mo", 60:"6mo", 90:"1y",180:"2y",365:"5y",1825:"10y"
    }
    period = period_map.get(horizon_days, "1y")
    hist, resolved = resolve_symbol(symbol_to_analyze, period=period)
    if hist is None:
        st.error("Could not resolve the symbol or fetch market data. Try adding .NS or .BO suffix.")
    else:
        st.success(f"Resolved to: {resolved}")
        hist = calculate_indicators(hist)
        # header metrics
        last_close = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist)>1 else last_close
        delta = last_close - prev_close
        delta_pct = (delta/prev_close*100) if prev_close!=0 else 0.0
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Last Close", f"₹{last_close:.2f}", f"{delta:+.2f} ({delta_pct:+.2f}%)")
        c2.metric("Day High", f"₹{float(hist['High'].iloc[-1]):.2f}")
        c3.metric("Day Low", f"₹{float(hist['Low'].iloc[-1]):.2f}")
        c4.metric("Volume", f"{int(hist['Volume'].iloc[-1]):,}")
        yf_info = None
        try:
            throttle_fetch()
            tk = yf.Ticker(resolved)
            yf_info = tk.info
        except Exception:
            yf_info = {}
        if yf_info and yf_info.get('marketCap'):
            c5.metric("Market Cap (Cr)", f"₹{yf_info.get('marketCap')/1e7:.2f}")
        else:
            c5.metric("Volatility", f"{hist['Volatility'].iloc[-1]:.2f}%")
        # chart
        st.plotly_chart(create_chart(hist, resolved), use_container_width=True)
        # predictions
        with st.spinner("Generating predictions..."):
            # BG prediction
            est_daily_ret, conf = BG.predict_for(resolved)
            # combine with enhanced model
            enhanced_price = enhanced_model_prediction(hist, days=horizon_days)
            # convert BG next-day return to days-horizon predicted price
            pred_price_bg = last_close*(1 + est_daily_ret*horizon_days)
            # combine weights: BG confidence -> scale
            bg_weight = min(0.9, conf/100.0)
            final_pred_price = pred_price_bg*bg_weight + enhanced_price*(1-bg_weight)
            pred_pct = (final_pred_price - last_close)/last_close*100.0
            # compute targets
            entry, stop_loss, targets, rr = compute_targets(last_close, final_pred_price, horizon_days)
            # fundamentals
            fundamentals = extract_fundamentals(yf_info)
        # Recommendation block
        if pred_pct >= 15:
            rec = "🔵 STRONG BUY"
            style = "buy"
        elif pred_pct >= 5:
            rec = "🟢 BUY"
            style = "buy"
        elif pred_pct <= -15:
            rec = "🔴 STRONG SELL"
            style = "sell"
        elif pred_pct <= -5:
            rec = "🔴 SELL"
            style = "sell"
        else:
            rec = "⚪ HOLD"
            style = "hold"
        st.markdown(f"### Recommendation: **{rec}**")
        st.markdown(f"**Predicted price in {horizon_days} days:** ₹{final_pred_price:.2f} ({pred_pct:+.2f}%)")
        st.markdown(f"**Model confidence:** {conf:.1f}%  |  **BG Accuracy (EMA):** {metrics.get('directional_accuracy_ema',0.0)*100:.2f}%")
        st.markdown("**Trading Plan (targets & stop-loss)**")
        st.write(f"- Entry: ₹{entry:.2f}")
        st.write(f"- Stop-loss: ₹{stop_loss:.2f}")
        for i,t in enumerate(targets,1):
            st.write(f"- Target {i}: ₹{t:.2f}  ({(t-entry)/entry*100:+.2f}%)")
        st.write(f"- Est. Risk-Reward (T2): 1:{rr:.2f}")
        # Technical summary
        st.markdown("### Technical Summary")
        st.write(f"- RSI (14): {hist['RSI'].iloc[-1]:.2f}")
        st.write(f"- MACD: {hist['MACD'].iloc[-1]:.4f} (Signal: {hist['MACD_Signal'].iloc[-1]:.4f})")
        st.write(f"- MA20: {hist['MA20'].iloc[-1]:.2f}  MA50: {hist['MA50'].iloc[-1]:.2f}  MA200: {hist['MA200'].iloc[-1]:.2f}")
        st.write(f"- Volatility (ann): {hist['Volatility'].iloc[-1]:.2f}%")
        # Fundamentals
        st.markdown("### Fundamental Snapshot (from Yahoo if available)")
        if fundamentals:
            for k,v in fundamentals.items():
                st.write(f"- {k}: {v}")
        else:
            st.info("No fundamental metrics found in Yahoo info for this ticker.")
        # Add symbol to universe for training if not present
        if resolved and resolved not in BG.universe:
            if st.button(f"Add {resolved} to training universe"):
                BG.add_symbol(resolved)
                st.success(f"{resolved} added to universe for background training.")

st.write("---")
st.markdown("<small>Tip: populate universe then click 'Scan & Recommend'. Background learner will continue training automatically.</small>", unsafe_allow_html=True)
