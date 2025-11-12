"""
JINNI - AI-Powered Indian Stock Market Analysis System (Streamlit)
Full app with upgraded BackgroundLearnerV2 (ensemble online models, EMA-smoothed metrics,
feature engineering, ATR stops, multi-targets, confidence-weighted scoring).
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import threading
import time
import os
import pickle
import random
from typing import List
warnings.filterwarnings("ignore")

# ----------------------------
# Lightweight Online Models
# ----------------------------
class OnlineRegressor:
    def __init__(self, n_features, lr=0.004, l2=1e-4):
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

    def save_to_dict(self):
        return {"w_reg": self.w, "b_reg": self.b}

    def load_from_dict(self, data):
        self.w = data.get("w_reg", self.w)
        self.b = data.get("b_reg", self.b)

class OnlineLogistic:
    def __init__(self, n_features, lr=0.008, l2=1e-4):
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def predict_proba(self, X):
        z = X.dot(self.w) + self.b
        return self._sigmoid(z)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def partial_fit(self, X, y):
        if X.size == 0:
            return
        p = self.predict_proba(X)
        errs = p - y
        grad_w = (errs[:, None] * X).mean(axis=0) + self.l2 * self.w
        grad_b = errs.mean()
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b

    def save_to_dict(self):
        return {"w_log": self.w, "b_log": self.b}

    def load_from_dict(self, data):
        self.w = data.get("w_log", self.w)
        self.b = data.get("b_log", self.b)

# ----------------------------
# Feature helpers
# ----------------------------
def compute_rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_atr_series(df: pd.DataFrame, period=14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_pc = np.abs(df['High'] - df['Close'].shift(1))
    low_pc = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr.fillna(method='ffill').fillna(0.0)

def build_features_from_df(df: pd.DataFrame, n_lags=6):
    """
    Features:
    - n_lags previous daily returns
    - (MA20 - MA50) / price
    - RSI / 100
    - ATR / price
    - recent vol (std of returns)
    Returns X, y_reg (next-day returns), y_dir (binary)
    """
    if df is None or len(df) < n_lags + 5:
        return np.empty((0, n_lags + 4)), np.empty((0,)), np.empty((0,))
    close = df['Close'].values
    returns = (close[1:] - close[:-1]) / close[:-1]  # length N-1
    ma20 = df['Close'].rolling(20, min_periods=1).mean().values
    ma50 = df['Close'].rolling(50, min_periods=1).mean().values
    rsi = compute_rsi(df['Close']).values
    atr = compute_atr_series(df).values
    X, y_reg, y_dir = [], [], []
    # we align such that returns[i] is return from day i -> i+1
    for i in range(n_lags, len(returns)):
        lag = returns[i - n_lags: i]
        # index mapping: returns index i corresponds to df index i+1
        idx = i + 1
        price = close[idx] if idx < len(close) else close[-1]
        ma_diff = (ma20[idx] - ma50[idx]) / (price if price != 0 else 1)
        rsi_val = rsi[idx] / 100.0
        atr_val = atr[idx] / (price if price != 0 else 1)
        vol = float(np.std(returns[max(0, i - 20): i + 1])) if i >= 1 else 0.0
        feat = np.concatenate([lag, [ma_diff, rsi_val, atr_val, vol]])
        X.append(feat)
        y_reg.append(returns[i])
        y_dir.append(1 if returns[i] > 0 else 0)
    return np.array(X), np.array(y_reg), np.array(y_dir)

# ----------------------------
# Improved BackgroundLearnerV2
# ----------------------------
class BackgroundLearnerV2:
    def __init__(self, universe: List[str] = None, n_lags=6, interval_sec=12, batch_size=12, model_path="bg_v2.pkl"):
        # provide your 114 tickers list here or pass it into constructor
        default_universe = [
            "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","HINDUNILVR.NS",
            "LT.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS"
        ]
        self.universe = universe or default_universe
        self.n_lags = n_lags
        self.batch_size = batch_size
        self.interval_sec = max(5, interval_sec)
        self.model_path = model_path
        n_features = n_lags + 4
        self.regressor = OnlineRegressor(n_features=n_features, lr=0.004, l2=1e-4)
        self.classifier = OnlineLogistic(n_features=n_features, lr=0.008, l2=1e-4)
        self.ema_alpha = 0.06
        self.metrics = {"dir_acc_ema": 0.5, "mae_ema": 0.5, "samples": 0}
        self._stop = threading.Event()
        self._thread = None
        self.lock = threading.Lock()
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                self.regressor.load_from_dict(data)
                self.classifier.load_from_dict(data)
                if "metrics" in data:
                    self.metrics.update(data["metrics"])
            except Exception:
                pass

    def _persist(self):
        try:
            with open(self.model_path, "wb") as f:
                pdump = {}
                pdump.update(self.regressor.save_to_dict())
                pdump.update(self.classifier.save_to_dict())
                pdump["metrics"] = self.metrics
                pickle.dump(pdump, f)
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

    def _fetch_history_safe(self, ticker, days=220):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=f"{days}d", interval="1d", auto_adjust=False)
            if df is None or df.empty:
                return None
            return df
        except Exception:
            return None

    def _update_ema(self, dir_acc_batch, mae_batch, n_new):
        with self.lock:
            alpha = self.ema_alpha
            self.metrics["dir_acc_ema"] = alpha * dir_acc_batch + (1 - alpha) * self.metrics["dir_acc_ema"]
            self.metrics["mae_ema"] = alpha * mae_batch + (1 - alpha) * self.metrics["mae_ema"]
            self.metrics["samples"] += n_new

    def _loop(self):
        while not self._stop.is_set():
            try:
                batch = random.sample(self.universe, min(self.batch_size, len(self.universe)))
                dir_accs, maes, n_total = [], [], 0
                for tk in batch:
                    if self._stop.is_set():
                        break
                    df = self._fetch_history_safe(tk, days=300)
                    if df is None or len(df) < self.n_lags + 10:
                        continue
                    X, y_reg, y_dir = build_features_from_df(df, n_lags=self.n_lags)
                    if X.size == 0:
                        continue
                    preds_reg = self.regressor.predict(X)
                    preds_dir_p = self.classifier.predict_proba(X)
                    mae = float(np.mean(np.abs(preds_reg - y_reg)))
                    dir_acc = float(np.mean((preds_dir_p >= 0.5) == (y_dir == 1)))
                    # update models
                    try:
                        self.classifier.partial_fit(X, y_dir)
                        self.regressor.partial_fit(X, y_reg)
                    except Exception:
                        pass
                    dir_accs.append(dir_acc)
                    maes.append(mae)
                    n_total += len(y_reg)
                    time.sleep(0.12)
                if n_total > 0:
                    batch_dir = float(np.mean(dir_accs)) if dir_accs else 0.5
                    batch_mae = float(np.mean(maes)) if maes else 0.5
                    self._update_ema(batch_dir, batch_mae, n_total)
                    self._persist()
                time.sleep(self.interval_sec)
            except Exception:
                time.sleep(self.interval_sec)

    def get_performance_report(self):
        with self.lock:
            return {
                "directional_accuracy": float(self.metrics["dir_acc_ema"]),
                "mae": float(self.metrics["mae_ema"]),
                "samples": int(self.metrics["samples"])
            }

    def predict_for_ticker(self, ticker):
        df = self._fetch_history_safe(ticker, days=300)
        if df is None or len(df) < self.n_lags + 5:
            return 0.0, 0.0
        X, _, _ = build_features_from_df(df, n_lags=self.n_lags)
        if X.size == 0:
            return 0.0, 0.0
        latest = X[-1].reshape(1, -1)
        pred_ret = float(self.regressor.predict(latest)[0])
        prob_up = float(self.classifier.predict_proba(latest)[0])
        # volatility heuristic
        recent_returns = (df['Close'].values[1:] - df['Close'].values[:-1]) / df['Close'].values[:-1]
        vol = float(np.std(recent_returns[-60:])) if len(recent_returns) >= 1 else 0.0
        confidence = float(max(0.0, min(0.99, prob_up * (1.0 - min(0.85, vol * 4.0)))))
        # calibrate signed return by classifier (push toward sign-probability)
        signed = pred_ret * (prob_up * 2 - 1)
        # damp extremes
        signed = max(-0.6, min(0.6, signed))
        return float(signed), confidence

# instantiate BG_LEARNER (use your full universe list here if you want)
BG_LEARNER = BackgroundLearnerV2(
    universe=None,      # put your 114 tickers list here if you have it
    n_lags=6,
    interval_sec=12,
    batch_size=12,
    model_path="bg_v2.pkl"
)
BG_LEARNER.start()

# ----------------------------
# Streamlit UI & analysis logic
# ----------------------------
st.set_page_config(page_title="JINNI - Indian Stock Market AI", page_icon="🧞", layout="wide")

# CSS
st.markdown("""
    <style>
    .main-header { font-size:48px; font-weight:bold; text-align:center; color:#1E88E5; text-shadow:2px 2px 4px rgba(0,0,0,0.1); }
    .sub-header { font-size:20px; text-align:center; color:#424242; margin-bottom:18px; }
    .prediction-box { background:#E3F2FD; padding:16px; border-radius:10px; border-left:5px solid #1E88E5; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧞 JINNI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Indian Stock Market Analysis & Prediction System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Genie/3D/genie_3d.png", width=130)
    st.title("⚙️ Controls")
    stock_symbol = st.text_input("Enter NSE/BSE Stock Symbol", value="RELIANCE.NS", help="Add .NS for NSE stocks, .BO for BSE stocks")
    timeframe = st.selectbox("Select Timeframe", ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"])
    pred_days = st.slider("Prediction Horizon (Days)", 1, 90, 30)
    analyze_btn = st.button("🔮 Analyze Stock", use_container_width=True)
    st.divider()
    st.info("💡 Background learner runs automatically and updates performance metrics (EMA-smoothed).")

timeframe_map = {"1 Month":"1mo","3 Months":"3mo","6 Months":"6mo","1 Year":"1y","2 Years":"2y","5 Years":"5y"}

def fetch_stock_data(symbol, period):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, auto_adjust=False)
        info = ticker.info if hasattr(ticker, "info") else {}
        return hist, info, ticker
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None, None, None

def calculate_technical_indicators(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["RSI"] = compute_rsi(df["Close"])
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["BB_Middle"] = df["Close"].rolling(window=20, min_periods=1).mean()
    bb_std = df["Close"].rolling(window=20, min_periods=1).std().fillna(0)
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
    df["ATR"] = compute_atr_series(df)
    return df

# improved simple prediction (momentum + volatility + trend_boost)
def simple_prediction(df, days):
    if df is None or df.empty:
        return 0.0
    current_price = df["Close"].iloc[-1]
    recent_5d = df["Close"].pct_change().tail(5).mean()
    recent_20d = df["Close"].pct_change().tail(20).mean()
    recent_60d = df["Close"].pct_change().tail(60).mean()
    momentum = recent_5d * 0.5 + recent_20d * 0.3 + recent_60d * 0.2
    time_multiplier = np.sqrt(days / 30)
    volatility = df["Close"].pct_change().std() * np.sqrt(252)
    volatility_boost = volatility * 0.25
    trend_boost = 0.0
    if len(df) >= 50:
        ma20 = df["Close"].rolling(20).mean().iloc[-1]
        ma50 = df["Close"].rolling(50).mean().iloc[-1]
        trend_boost = 0.05 if ma20 > ma50 else -0.05 if ma20 < ma50 else 0.0
    total_change = (momentum * days * time_multiplier) + volatility_boost + trend_boost
    if abs(total_change) < 0.03 and days >= 30:
        total_change = 0.05 if momentum > 0 else -0.03
    total_change = max(-0.5, min(0.5, total_change))
    predicted_price = current_price * (1 + total_change)
    return float(predicted_price)

def create_price_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))
    for col in ["MA5","MA20","MA50"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)))
    fig.update_layout(title="Price Chart with Technical Indicators", yaxis_title="Price (₹)", xaxis_title="Date", height=600, template="plotly_white")
    return fig

def compute_targets_from_prediction(current_price: float, pred_return: float, atr: float = None):
    entry = current_price
    if atr is None or atr <= 0:
        atr = current_price * 0.02  # fallback 2% as ATR
    # ATR-based stop loss: 1.5 ATR for buy, symmetric for sell
    if pred_return >= 0:
        stop_loss = entry - 1.5 * atr
    else:
        stop_loss = entry + 1.5 * atr
    # build conservative targets scaled to predicted return and ATR multipliers
    t1 = entry + np.sign(pred_return) * max(abs(pred_return) * 0.5 * entry, 0.8 * atr)
    t2 = entry + np.sign(pred_return) * max(abs(pred_return) * 1.0 * entry, 1.2 * atr)
    t3 = entry + np.sign(pred_return) * max(abs(pred_return) * 1.5 * entry, 2.0 * atr)
    t4 = entry + np.sign(pred_return) * max(abs(pred_return) * 2.0 * entry, 3.0 * atr)
    # ensure targets are sensible (not equal to entry)
    targets = [max(0.01, x) for x in [t1, t2, t3, t4]]
    risk = abs(entry - stop_loss) if stop_loss != entry else entry * 0.01
    reward = abs(t2 - entry) if t2 != entry else 0.0
    rr = (reward / risk) if risk > 0 else 0.0
    return entry, stop_loss, targets, rr

# ----------------------------
# Main analysis routine
# ----------------------------
def run_analysis():
    period = timeframe_map.get(timeframe, "6mo")
    hist, info, ticker = fetch_stock_data(stock_symbol, period)
    if hist is None or hist.empty:
        st.error("Could not fetch stock data. Check symbol and try again.")
        return
    hist = calculate_technical_indicators(hist)
    company_name = info.get("longName") or info.get("shortName") or stock_symbol
    st.header(f"📊 {company_name}")

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    current_price = float(hist["Close"].iloc[-1])
    prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current_price
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0.0
    market_cap = info.get("marketCap", None)

    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    with col2:
        st.metric("Day High", f"₹{float(hist['High'].iloc[-1]):.2f}")
    with col3:
        st.metric("Day Low", f"₹{float(hist['Low'].iloc[-1]):.2f}")
    with col4:
        st.metric("Volume", f"{int(hist['Volume'].iloc[-1]):,}")
    with col5:
        if market_cap:
            st.metric("Market Cap (Cr)", f"₹{market_cap / 1e7:.2f}")

    st.plotly_chart(create_price_chart(hist), use_container_width=True)

    # Use BG_LEARNER to predict (signed daily return) + confidence
    st.header("🔮 AI Predictions")
    bg_pred_daily, bg_conf = BG_LEARNER.predict_for_ticker(stock_symbol)
    # scale to pred_days: conservative scaling using sqrt(days)
    pred_price_bg = current_price * (1 + bg_pred_daily * np.sqrt(max(1, pred_days)))
    pred_change_bg = ((pred_price_bg - current_price) / current_price) * 100 if current_price != 0 else 0.0

    # fallback:
    fallback_price = simple_prediction(hist, pred_days)
    pred_price = pred_price_bg if bg_conf >= 0.10 else 0.65 * fallback_price + 0.35 * pred_price_bg
    pred_change = ((pred_price - current_price) / current_price) * 100 if current_price != 0 else 0.0

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="prediction-box">
                <h3>Target Price ({pred_days} days)</h3>
                <h2>₹{pred_price:.2f}</h2>
                <p style="font-size:16px; color:{'green' if pred_change>0 else 'red'}">{pred_change:+.2f}% from current price</p>
                <p style="font-size:12px; color:#666;">Model confidence: {bg_conf:.2f} (directional)</p>
            </div>
        """, unsafe_allow_html=True)

    # Recommendation rules (sharper thresholds)
    rsi = hist['RSI'].iloc[-1] if 'RSI' in hist.columns else 50.0
    ma_signal = "Bullish" if hist['Close'].iloc[-1] > hist['MA20'].iloc[-1] else "Bearish"

    # Interpret predicted percent (over pred_days)
    if pred_change >= 15 and bg_conf >= 0.45:
        recommendation = "🔵 STRONG BUY"
    elif pred_change >= 7 and bg_conf >= 0.35:
        recommendation = "🟢 BUY"
    elif pred_change >= 3 and bg_conf >= 0.25:
        recommendation = "🟡 BUY (Tactical)"
    elif pred_change <= -15 and bg_conf >= 0.45:
        recommendation = "🔴 STRONG SELL"
    elif pred_change <= -7 and bg_conf >= 0.35:
        recommendation = "🔴 SELL"
    elif pred_change <= -3 and bg_conf >= 0.25:
        recommendation = "🟠 SELL (Tactical)"
    else:
        recommendation = "⚪ HOLD"

    reason = f"Pred {pred_change:+.2f}% over {pred_days} days | RSI {rsi:.1f} | MA trend {ma_signal}"
    if rsi > 75 and "BUY" in recommendation:
        reason += f" | ⚠️ RSI overbought ({rsi:.1f})"
    if rsi < 25 and "SELL" in recommendation:
        reason += f" | ⚠️ RSI oversold ({rsi:.1f})"

    # Trading plan using ATR
    atr = float(hist["ATR"].iloc[-1]) if "ATR" in hist.columns else None
    entry, stop_loss, targets, rr = compute_targets_from_prediction(current_price, bg_pred_daily, atr=atr)
    confidence_level = round(BG_LEARNER.get_performance_report()["directional_accuracy"] * 100, 2)

    with col2:
        st.markdown(f"""
            <div class="prediction-box">
                <h3>AI Recommendation</h3>
                <h2>{recommendation}</h2>
                <p style="font-size:14px;">{reason}</p>
                <p style="font-size:12px; color:#666;">Model directional accuracy (EMA): {confidence_level:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

    # Trading plan display
    if "BUY" in recommendation:
        st.markdown(f"""
            <div class="prediction-box" style="background-color:#dff0d8; border-left:4px solid #28a745;">
                <h3>💰 Trading Plan</h3>
                <p><strong>Entry:</strong> ₹{entry:.2f}</p>
                <p><strong>Stop Loss:</strong> ₹{stop_loss:.2f}</p>
                <p><strong>Target 1:</strong> ₹{targets[0]:.2f}</p>
                <p><strong>Target 2:</strong> ₹{targets[1]:.2f}</p>
                <p><strong>Target 3:</strong> ₹{targets[2]:.2f}</p>
                <p><strong>Target 4:</strong> ₹{targets[3]:.2f}</p>
                <p><strong>Risk-Reward (est):</strong> 1:{rr:.2f}</p>
                <p><strong>Model Confidence (directional):</strong> {confidence_level:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
    elif "SELL" in recommendation:
        st.markdown(f"""
            <div class="prediction-box" style="background-color:#f8d7da; border-left:4px solid #dc3545;">
                <h3>💰 Trading Plan (SHORT)</h3>
                <p><strong>Entry (short):</strong> ₹{entry:.2f}</p>
                <p><strong>Stop Loss:</strong> ₹{stop_loss:.2f}</p>
                <p><strong>Target 1:</strong> ₹{targets[0]:.2f}</p>
                <p><strong>Target 2:</strong> ₹{targets[1]:.2f}</p>
                <p><strong>Target 3:</strong> ₹{targets[2]:.2f}</p>
                <p><strong>Target 4:</strong> ₹{targets[3]:.2f}</p>
                <p><strong>Model Confidence (directional):</strong> {confidence_level:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-box" style="background-color:#fff3cd; border-left:4px solid #ffc107;">
                <h3>⏸️ Watch & Wait</h3>
                <p>Neutral outlook. Observe watch levels or wait for a clearer signal.</p>
                <p><strong>Model directional accuracy (EMA):</strong> {confidence_level:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

    # Technical indicators panel
    st.header("📈 Technical Analysis")
    col1, col2, col3, col4 = st.columns(4)
    rsi_val = float(hist["RSI"].iloc[-1])
    macd_val = float(hist["MACD"].iloc[-1])
    ma_sig = "Bullish" if hist["Close"].iloc[-1] > hist["MA20"].iloc[-1] else "Bearish"
    volatility = hist["Close"].pct_change().std() * np.sqrt(252) * 100

    with col1:
        st.metric("RSI (14)", f"{rsi_val:.2f}")
    with col2:
        st.metric("MACD", f"{macd_val:.4f}")
    with col3:
        st.metric("MA Signal", ma_sig)
    with col4:
        st.metric("Volatility (annualized)", f"{volatility:.2f}%")

    # Reasoning
    st.header("🧠 AI Analysis Reasoning")
    reason_lines = [
        f"1. Price Action: Current price ₹{current_price:.2f} is {'above' if current_price > hist['MA20'].iloc[-1] else 'below'} the 20-day MA (₹{hist['MA20'].iloc[-1]:.2f}).",
        f"2. RSI Analysis: RSI at {rsi_val:.2f} indicates " + ("oversold - potential buy." if rsi_val < 30 else ("overbought - consider taking profits." if rsi_val > 70 else "neutral momentum.")),
        f"3. Trend: {'Uptrend' if hist['Close'].iloc[-1] > hist['MA50'].iloc[-1] else 'Downtrend'}.",
        f"4. Prediction Confidence: Model directional accuracy (EMA) {confidence_level:.2f}%.",
        f"5. Risk Level: {'High' if volatility > 40 else 'Moderate' if volatility > 25 else 'Low'} (Volatility: {volatility:.2f}%)."
    ]
    st.markdown("<br/>".join(reason_lines), unsafe_allow_html=True)

    # Live model performance
    st.header("🎯 Model Performance (Live)")
    perf = BG_LEARNER.get_performance_report()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Directional Accuracy (EMA)", f"{perf['directional_accuracy']*100:.2f}%")
    with col2:
        st.metric("MAE (EMA)", f"{perf['mae']:.4f}")
    with col3:
        st.metric("Samples Trained", f"{perf['samples']:,}")

    st.success("✅ Analysis complete. Background learner continues to train in background.")

# Sidebar: show learner metrics and recommendations scan
with st.sidebar:
    st.divider()
    st.subheader("🧪 Background Learner (auto)")
    perf = BG_LEARNER.get_performance_report()
    st.metric("Directional Accuracy (EMA)", f"{perf['directional_accuracy']*100:.2f}%")
    st.metric("MAE (EMA)", f"{perf['mae']:.4f}")
    st.caption(f"Samples trained: {perf['samples']:,}")

    st.divider()
    st.subheader("🔍 Scan Top Stocks (model-driven)")
    if st.button("Scan & Recommend (use model)"):
        recs = []
        universe = BG_LEARNER.universe
        for tk in universe:
            try:
                hist_tmp, info_tmp, _ = fetch_stock_data(tk, "1y")
                if hist_tmp is None or hist_tmp.empty:
                    continue
                cur_p = float(hist_tmp["Close"].iloc[-1])
                pred_r, conf = BG_LEARNER.predict_for_ticker(tk)
                # scale to 7-day expected percent (rough)
                pred_pct_7d = pred_r * np.sqrt(7) * 100
                if abs(pred_pct_7d) < 2.0 or conf < 0.15:
                    # filter out tiny picks and low confidence
                    continue
                entry, stop_loss, targets, rr = compute_targets_from_prediction(cur_p, pred_r, atr=compute_atr_series(hist_tmp).iloc[-1] if "ATR" not in hist_tmp else hist_tmp["ATR"].iloc[-1])
                recs.append({
                    "ticker": tk,
                    "pred_pct_7d": pred_pct_7d,
                    "entry": entry, "stop_loss": stop_loss, "targets": targets, "rr": rr, "conf": conf
                })
            except Exception:
                continue
        recs_sorted = sorted(recs, key=lambda x: x["pred_pct_7d"], reverse=True)[:12]
        if not recs_sorted:
            st.info("No high-confidence recommendations found. Try widening universe or increasing batch_size in learner.")
        else:
            for i, r in enumerate(recs_sorted, 1):
                st.markdown(f"**{i}. {r['ticker']}**  \n- Est (7d): {r['pred_pct_7d']:+.2f}%  \n- Entry: ₹{r['entry']:.2f}  \n- Targets: ₹{r['targets'][0]:.2f}, ₹{r['targets'][1]:.2f}, ₹{r['targets'][2]:.2f}, ₹{r['targets'][3]:.2f}  \n- Stop: ₹{r['stop_loss']:.2f}  \n- Est RR: 1:{r['rr']:.2f}  \n- Model conf: {r['conf']:.2f}")
                st.divider()

# Footer
st.divider()
st.markdown("""
<div style="text-align:center;color:#666;">
    <p>JINNI - AI Stock Analysis System for Indian Markets</p>
    <p>BackgroundLearnerV2: ensemble online models (persisted to bg_v2.pkl). Tune hyperparams and expand universe for better results.</p>
    <p>⚠️ Disclaimer: Educational purposes only. Not financial advice. Backtest before trading real capital.</p>
</div>
""", unsafe_allow_html=True)

# Trigger analysis (auto-run once, or on button)
if analyze_btn:
    run_analysis()
elif "auto_run_done" not in st.session_state:
    st.session_state["auto_run_done"] = True
    run_analysis()
