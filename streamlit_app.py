"""
JINNI - AI-Powered Indian Stock Market Analysis System (Streamlit)
Includes a self-learning BackgroundLearner and live model performance metrics.
NOTE: This implements a lightweight online linear model for speed & portability.
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
from typing import List, Dict

warnings.filterwarnings("ignore")

# Optional analytics modules (wrap imports so missing modules don't break the UI)
try:
    from analytics.trading_signals import TradingSignals
except Exception:
    TradingSignals = None

try:
    from analytics.stock_recommender import StockRecommender
except Exception:
    StockRecommender = None

# --- NEW: BackgroundLearner Implementation (self-contained, online linear model) ---
class OnlineLinearModel:
    """
    Very small online linear regressor that predicts next-day return from last N lag returns.
    Uses simple SGD updates and persists weights to disk.
    """
    def __init__(self, n_features=5, lr=0.001):
        self.n = n_features
        self.lr = lr
        self.w = np.zeros(self.n)  # weights
        self.b = 0.0
        self.initialized = False

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X shape (m, n)
        if X.size == 0:
            return np.array([])
        return X.dot(self.w) + self.b

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        # SGD update for each sample
        if X.size == 0:
            return
        preds = self.predict(X)
        errs = preds - y
        # gradient step
        grad_w = (errs[:, None] * X).mean(axis=0)
        grad_b = errs.mean()
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b
        self.initialized = True

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"w": self.w, "b": self.b}, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.w = data.get("w", self.w)
                self.b = data.get("b", self.b)
                self.initialized = True

class BackgroundLearner:
    """
    Background learning loop:
    - Periodically samples tickers from a universe
    - Builds lag-return features and next-day target
    - Trains OnlineLinearModel incrementally
    - Persists model and performance metrics
    """
    def __init__(self,
                 model_path="background_model.pkl",
                 universe: List[str] = None,
                 n_lags: int = 5,
                 interval_sec: int = 60):
        self.model_path = model_path
        self.n_lags = n_lags
        self.interval_sec = max(10, interval_sec)  # min 10s
        self.universe = universe or self._default_universe()
        self.model = OnlineLinearModel(n_features=self.n_lags, lr=0.003)
        self.model.load(self.model_path)  # load existing if present
        self.metrics = {
            "directional_accuracy": 0.0,
            "mae": float("inf"),
            "total_samples": 0,
            "history": []
        }
        self._stop_event = threading.Event()
        self._thread = None
        self.lock = threading.Lock()

    def _default_universe(self) -> List[str]:
        # A reasonable short default universe (NSE tickers with .NS)
        # You can expand this list as needed
        return [return [
            # Nifty 50
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
            "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
            "TITAN.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS",
            "HCLTECH.NS", "TECHM.NS", "M&M.NS", "ONGC.NS", "NTPC.NS",
            "POWERGRID.NS", "TATASTEEL.NS", "ADANIENT.NS", "BAJAJFINSV.NS", "COALINDIA.NS",
            "TATAMOTORS.NS", "HINDALCO.NS", "JSWSTEEL.NS", "INDUSINDBK.NS", "BPCL.NS",
            "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HEROMOTOCO.NS",
            "DIVISLAB.NS", "APOLLOHOSP.NS", "BRITANNIA.NS", "SHRIRAMFIN.NS", "ADANIPORTS.NS",
            "TATACONSUM.NS", "SBILIFE.NS", "BAJAJ-AUTO.NS", "LTIM.NS", "TRENT.NS",
            # Nifty Next 50
            "ADANIPOWER.NS", "AMBUJACEM.NS", "ACC.NS", "GODREJCP.NS", "HAVELLS.NS",
            "MOTHERSON.NS", "SIEMENS.NS", "DLF.NS", "PIDILITIND.NS", "GAIL.NS",
            "BOSCHLTD.NS", "INDIGO.NS", "VEDL.NS", "BANKBARODA.NS", "PNB.NS",
            "COLPAL.NS", "DABUR.NS", "TORNTPHARM.NS", "LUPIN.NS", "BIOCON.NS",
            "GODREJPROP.NS", "BERGEPAINT.NS", "MARICO.NS", "NMDC.NS", "HAL.NS",
            "BEL.NS", "IDEA.NS", "SAIL.NS", "UPL.NS", "SHREECEM.NS",
            # Midcap & Others (100+ more)
            "ZOMATO.NS", "PAYTM.NS", "NYKAA.NS", "IRCTC.NS", "IRFC.NS",
            "RVNL.NS", "NBCC.NS", "PFC.NS", "RECLTD.NS", "HUDCO.NS",
            "CANBK.NS", "UNIONBANK.NS", "INDIANB.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS",
            "BANDHANBNK.NS", "RBLBANK.NS", "AUBANK.NS", "YESBANK.NS", "JUBLFOOD.NS",
            "DIXON.NS", "AFFLE.NS", "COFORGE.NS", "PERSISTENT.NS", "LTTS.NS",
            "MPHASIS.NS", "MINDTREE.NS", "CYIENT.NS", "HAPPSTMNDS.NS", "ROUTE.NS",
            "POLYCAB.NS", "KEI.NS", "ASTRAL.NS", "SUPREME IND.NS", "RELAXO.NS",
            "BATA.NS", "VBL.NS", "TATAPOWER.NS", "JSW ENERGY.NS", "ADANIGREEN.NS",
            "TORNTPOWER.NS", "ABB.NS", "CUMMINSIND.NS", "VOLTAS.NS", "CROMPTON.NS",
            "WHIRLPOOL.NS", "SYMPHONY.NS", "BAJAJHLDNG.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS",
            "LICH SGFIN.NS", "PEL.NS", "DMART.NS", "TATAELXSI.NS", "MCDOWELL-N.NS",
            "PIIND.NS", "PAGEIND.NS", "ALKEM.NS", "LAURUSLABS.NS", "GLENMARK.NS",
            "SUNPHARMA.NS", "LALPATHLAB.NS", "METROPOLIS.NS", "FORTIS.NS", "MAXHEALTH.NS",
            "JKCEMENT.NS", "RAMCOCEM.NS", "HEIDELBERG.NS", "STAR CEMENT.NS", "ORIENT CEMENT.NS",
            "ASHOKLEY.NS", "ESCORTS.NS", "TVSMOT OR.NS", "BAJAJ-AUTO.NS", "EXIDEIND.NS",
            "AMARAJABAT.NS", "MRF.NS", "APOLLOTYRE.NS", "CEAT.NS", "BALKRISIND.NS",
            "MCX.NS", "CDSL.NS", "CAMS.NS", "FACT.NS", "DEEPAKNTR.NS",
            "AARTI IND.NS", "GNFC.NS", "CHAMBLFERT.NS", "COROMANDEL.NS", "PIIND.NS",
            "MANYAVAR.NS", "V-MART.NS", "ABFRL.NS", "RAYMOND.NS", "GOKEX.NS",
            "IEX.NS", "ADANIGAS.NS", "GUJGASLTD.NS", "IGL.NS", "MGL.NS",
            "GMRINFRA.NS", "CONCOR.NS", "AEGISCHEM.NS", "CLEAN SCIENCE.NS", "FINE.NS"
        ]

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._learn_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _fetch_history(self, ticker: str, days=90):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=f"{days}d", interval="1d", auto_adjust=False)
            if df is None or df.empty:
                return None
            return df
        except Exception:
            return None

    def _build_samples(self, df: pd.DataFrame):
        """
        Build (X, y) where X are last n_lags returns and y is next-day return.
        Returns arrays; if insufficient data return empty arrays.
        """
        if df is None or len(df) <= self.n_lags + 1:
            return np.empty((0, self.n_lags)), np.empty((0,))
        close = df["Close"].values
        # compute daily returns
        returns = (close[1:] - close[:-1]) / close[:-1]
        # returns index aligns with days 1..N-1 relative to close
        X = []
        y = []
        for i in range(self.n_lags, len(returns)):
            x = returns[i - self.n_lags : i]
            target = returns[i]  # next day (relative)
            X.append(x)
            y.append(target)
        return (np.array(X), np.array(y))

    def _update_metrics(self, preds: np.ndarray, truths: np.ndarray):
        if preds.size == 0:
            return
        mae = np.mean(np.abs(preds - truths))
        dir_correct = np.mean(np.sign(preds) == np.sign(truths))
        with self.lock:
            total_before = self.metrics["total_samples"]
            # incremental averaging for directional accuracy and mae
            prev_acc = self.metrics["directional_accuracy"]
            prev_mae = self.metrics["mae"] if self.metrics["mae"] != float("inf") else 0.0
            n_new = len(truths)
            if total_before == 0:
                new_acc = dir_correct
                new_mae = mae
            else:
                new_acc = (prev_acc * total_before + dir_correct * n_new) / (total_before + n_new)
                new_mae = (prev_mae * total_before + mae * n_new) / (total_before + n_new)
            self.metrics["directional_accuracy"] = float(new_acc)
            self.metrics["mae"] = float(new_mae)
            self.metrics["total_samples"] = total_before + n_new
            # store recent history for plotting / improvement-rate calc
            self.metrics["history"].append({
                "ts": time.time(),
                "dir_acc": float(dir_correct),
                "mae": float(mae),
                "samples": n_new
            })
            # cap history length
            if len(self.metrics["history"]) > 500:
                self.metrics["history"].pop(0)

    def _learn_loop(self):
        while not self._stop_event.is_set():
            try:
                # sample tickers (random small batch)
                sample_count = min(4, max(1, len(self.universe)))
                tickers = random.sample(self.universe, sample_count)
                for tk in tickers:
                    if self._stop_event.is_set():
                        break
                    df = self._fetch_history(tk, days=120)
                    X, y = self._build_samples(df)
                    if X.size == 0:
                        continue
                    # make predictions on this batch for metric calc
                    preds = self.model.predict(X)
                    self._update_metrics(preds, y)
                    # update model incrementally using this batch
                    self.model.partial_fit(X, y)
                    # persist model occasionally
                    if random.random() < 0.2:
                        try:
                            self.model.save(self.model_path)
                        except Exception:
                            pass
                    # small delay between tickers to avoid hammering
                    time.sleep(0.5)
                # after batch, short sleep before next round
                time.sleep(self.interval_sec)
            except Exception:
                # swallow errors to keep background thread alive
                time.sleep(self.interval_sec)

    def get_performance_report(self):
        with self.lock:
            report = {
                "directional_accuracy": self.metrics["directional_accuracy"],
                "mae": self.metrics["mae"],
                "total_samples": self.metrics["total_samples"],
                "history": list(self.metrics["history"])
            }
        return report

    def predict_for_ticker(self, ticker: str, n_lags: int = None):
        """
        Returns predicted next-day return (float) and a small confidence estimate.
        """
        n_lags = n_lags or self.n_lags
        df = self._fetch_history(ticker, days=90)
        if df is None or len(df) <= n_lags:
            return 0.0, 0.0
        close = df["Close"].values
        returns = (close[1:] - close[:-1]) / close[:-1]
        latest = returns[-n_lags:]
        if len(latest) < n_lags:
            # pad with zeros
            latest = np.concatenate([np.zeros(n_lags - len(latest)), latest])
        pred = float(self.model.predict(latest.reshape(1, -1))[0])
        # confidence heuristic: more recent volatility lowers confidence
        vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        confidence = max(0.0, min(0.99, 1.0 - vol * 5))  # heuristic
        return pred, confidence

# Instantiate global background learner (module-level)
BG_LEARNER = BackgroundLearner(interval_sec=30)  # adjust interval to taste
BG_LEARNER.start()

# --- End BackgroundLearner implementation ---

# Page Configuration
st.set_page_config(
    page_title="JINNI - Indian Stock Market AI",
    page_icon="🧞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS (same as before)
st.markdown(
    """
    <style>
    .main-header { font-size:48px; font-weight:bold; text-align:center; color:#1E88E5; text-shadow:2px 2px 4px rgba(0,0,0,0.1); }
    .sub-header { font-size:20px; text-align:center; color:#424242; margin-bottom:18px; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding:20px; border-radius:10px; color:white; text-align:center; }
    .prediction-box { background:#E3F2FD; padding:16px; border-radius:10px; border-left:5px solid #1E88E5; }
    </style>
    """, unsafe_allow_html=True
)

# Header
st.markdown('<div class="main-header">🧞 JINNI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Indian Stock Market Analysis & Prediction System</div>', unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.image("https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Genie/3D/genie_3d.png", width=130)
    st.title("⚙️ Controls")
    stock_symbol = st.text_input("Enter NSE/BSE Stock Symbol", value="RELIANCE.NS", help="Add .NS for NSE stocks, .BO for BSE stocks")
    timeframe = st.selectbox("Select Timeframe", ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"])
    pred_days = st.slider("Prediction Horizon (Days)", 1, 90, 30)
    analyze_btn = st.button("🔮 Analyze Stock", use_container_width=True)
    st.divider()
    st.info("💡 Demo: The learner runs automatically in background and updates model metrics.")

# Timeframe mapping
timeframe_map = {"1 Month":"1mo","3 Months":"3mo","6 Months":"6mo","1 Year":"1y","2 Years":"2y","5 Years":"5y"}

# (re-used helper functions for fetching, indicators, plotting)
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
    delta = df["Close"].diff()
    gain = delta.where(delta>0, 0)
    loss = -delta.where(delta<0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"].fillna(50, inplace=True)
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["BB_Middle"] = df["Close"].rolling(window=20, min_periods=1).mean()
    bb_std = df["Close"].rolling(window=20, min_periods=1).std().fillna(0)
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
    return df

def simple_prediction(df, days):
        # IMPROVED PREDICTION MODEL - More realistic price targets
    if df is None or df.empty:
        return 0.0
    
    current_price = df["Close"].iloc[-1]
    
    # Calculate multiple trend indicators
    recent_5d_change = df["Close"].pct_change().tail(5).mean()
    recent_20d_change = df["Close"].pct_change().tail(20).mean()
    recent_60d_change = df["Close"].pct_change().tail(60).mean()
    
    # Momentum-based prediction (weighted recent changes)
    momentum_factor = (recent_5d_change * 10 + recent_20d_change * 5 + recent_60d_change * 0.2)
    
    # Amplify the prediction based on days horizon (longer = more movement expected)
    time_multiplier = np.sqrt(days / 30)  # Square root scaling for realistic growth
    
    # Add volatility component for realistic movement
    volatility = df["Close"].pct_change().std() * np.sqrt(252)  # Annualized volatility
    volatility_boost = volatility * 0.3  # Add 30% of volatility to prediction
    
    # Calculate trend strength from moving averages
    if len(df) >= 50:
        ma20 = df["Close"].rolling(20).mean().iloc[-1]
        ma50 = df["Close"].rolling(50).mean().iloc[-1]
        if ma20 > ma50:
            trend_boost = 0.05  # 5% bullish boost
        elif ma20 < ma50:
            trend_boost = -0.05  # 5% bearish boost
        else:
            trend_boost = 0
    else:
        trend_boost = 0
    
    # Combine all factors for final prediction
    total_change = (momentum_factor * days * time_multiplier) + volatility_boost + trend_boost
    
    # Ensure minimum meaningful prediction (at least 3-5% move over 90 days)
    if abs(total_change) recent_5d_change * 10 + recent_20d_change * 5< 0.10 and days >= 30:
        total_change =  = 0.15 if momentum_factor_factor > 0 else -0.10
    
    # Cap extreme predictions (max ±40% to stay realistic)
    total_change = max(-0.40, min(0.40, total_change))
    
    predicted_price = current_price * (1 + total_change)
    return float(predicted_price)
def create_price_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    for col in ["MA5","MA20","MA50"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)))
    fig.update_layout(title="Price Chart with Technical Indicators", yaxis_title="Price (₹)", xaxis_title="Date", height=600, template="plotly_white")
    return fig

# Utility for computing targets from a predicted return
def compute_targets_from_prediction(current_price: float, pred_return: float):
    """
    Compute a sensible set of targets based on predicted return.
    Targets are multiplicative steps of predicted return with safety margins.
    Returns entry, stop_loss, targets[1..4], risk_reward_ratio_estimate
    """
    entry = current_price
    # tiny buffer for SL
    if pred_return >= 0:
        stop_loss = entry * (1 - 0.03)  # 3% stop by default for buys
    else:
        stop_loss = entry * (1 + 0.03)  # for sells
    # compute targets: scale predicted return conservatively
    t1 = entry * (1 + pred_return * 0.5)
    t2 = entry * (1 + pred_return * 1.0)
    t3 = entry * (1 + pred_return * 1.5)
    t4 = entry * (1 + pred_return * 2.0)
    # risk_reward (approx from t2)
    risk = abs(entry - stop_loss)
    reward = abs(t2 - entry) if t2 != entry else 0.0
    rr = (reward / risk) if risk > 0 else 0.0
    return entry, stop_loss, [t1, t2, t3, t4], rr

# Main analysis logic (uses BG_LEARNER for predictions)
def run_analysis():
    period = timeframe_map.get(timeframe, "6mo")
    hist, info, ticker = fetch_stock_data(stock_symbol, period)
    if hist is None or hist.empty:
        st.error("Could not fetch stock data. Please check the symbol and try again.")
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

    # Predictions: use background learner to get next-day return & confidence
    st.header("🔮 AI Predictions")
    bg_pred_return, bg_conf = BG_LEARNER.predict_for_ticker(stock_symbol)
    # convert next-day return to n-day horizon roughly by scaling
    pred_price_bg = current_price * (1 + bg_pred_return * pred_days)
    pred_change_bg = ((pred_price_bg - current_price) / current_price) * 100 if current_price != 0 else 0.0

    # fallback simple_prediction if BG_LEARNER gives zero or low confidence
    fallback_price = simple_prediction(hist, pred_days)
    if abs(bg_conf) < 0.05:  # low confidence -> combine fallback
        pred_price = 0.6 * fallback_price + 0.4 * pred_price_bg
    else:
        pred_price = pred_price_bg

    pred_change = ((pred_price - current_price) / current_price) * 100 if current_price != 0 else 0.0

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="prediction-box">
                <h3>Target Price ({pred_days} days)</h3>
                <h2>₹{pred_price:.2f}</h2>
                <p style="font-size:16px; color:{'green' if pred_change>0 else 'red'}">{pred_change:+.2f}% from current price</p>
                <p style="font-size:12px; color:#666;">Model confidence: {bg_conf:.2f}</p>
            </div>
        """, unsafe_allow_html=True)

    # Intelligent recommendation using predicted return and technicals
    rsi = hist['RSI'].iloc[-1] if 'RSI' in hist.columns else 50
    ma_signal = "Bullish" if hist['Close'].iloc[-1] > hist['MA20'].iloc[-1] else "Bearish"

    # Use predicted percent (pred_change) thresholds
    if pred_change >= 20:
        recommendation = "🔵 STRONG BUY"
    elif pred_change >= 8:
        recommendation = "🟢 BUY"
    elif pred_change >= 2:
        recommendation = "🟡 BUY (Moderate)"
    elif pred_change <= -20:
        recommendation = "🔴 STRONG SELL"
    elif pred_change <= -8:
        recommendation = "🔴 SELL"
    elif pred_change <= -2:
        recommendation = "🟠 SELL (Moderate)"
    else:
        recommendation = "⚪ HOLD"

    reason = f"Prediction: {pred_change:+.2f}% over {pred_days} days | RSI: {rsi:.1f} | MA trend: {ma_signal}"
    if rsi > 75 and "BUY" in recommendation:
        reason += f" | ⚠️ RSI overbought ({rsi:.1f})"
    if rsi < 25 and "SELL" in recommendation:
        reason += f" | ⚠️ RSI oversold ({rsi:.1f})"

    # Compute trading plan (targets 1..4)
    entry, stop_loss, targets, rr = compute_targets_from_prediction(current_price, bg_pred_return)
    confidence_level = round(float(BG_LEARNER.get_performance_report()["directional_accuracy"]) * 100, 2)

    with col2:
        st.markdown(f"""
            <div class="prediction-box">
                <h3>AI Recommendation</h3>
                <h2>{recommendation}</h2>
                <p style="font-size:14px;">{reason}</p>
                <p style="font-size:12px; color:#666;">Model directional accuracy (live): {confidence_level:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

    # Trading plan display (targets 1..4)
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
                <p>Neutral outlook. Consider waiting or observing watch levels defined by your strategy.</p>
                <p><strong>Model directional accuracy:</strong> {confidence_level:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

    # Technical indicators panel
    st.header("📈 Technical Analysis")
    col1, col2, col3, col4 = st.columns(4)
    rsi_val = float(hist["RSI"].iloc[-1])
    macd_val = float(hist["MACD"].iloc[-1])
    ma_signal = "Bullish" if hist["Close"].iloc[-1] > hist["MA20"].iloc[-1] else "Bearish"
    volatility = hist["Close"].pct_change().std() * np.sqrt(252) * 100

    with col1:
        st.metric("RSI (14)", f"{rsi_val:.2f}")
    with col2:
        st.metric("MACD", f"{macd_val:.4f}")
    with col3:
        st.metric("MA Signal", ma_signal)
    with col4:
        st.metric("Volatility (annualized)", f"{volatility:.2f}%")

    # Reasoning
    st.header("🧠 AI Analysis Reasoning")
    reason_lines = [
        f"1. Price Action: Current price ₹{current_price:.2f} is {'above' if current_price > hist['MA20'].iloc[-1] else 'below'} the 20-day MA (₹{hist['MA20'].iloc[-1]:.2f}).",
        f"2. RSI Analysis: RSI at {rsi_val:.2f} indicates " + ("oversold - potential buy." if rsi_val < 30 else ("overbought - consider taking profits." if rsi_val > 70 else "neutral momentum.")),
        f"3. Trend: {'Uptrend' if hist['Close'].iloc[-1] > hist['MA50'].iloc[-1] else 'Downtrend'}.",
        f"4. Prediction Confidence: Model directional accuracy {confidence_level:.2f}%.",
        f"5. Risk Level: {'High' if volatility > 40 else 'Moderate' if volatility > 25 else 'Low'} (Volatility: {volatility:.2f}%)."
    ]
    st.markdown("<br/>".join(reason_lines), unsafe_allow_html=True)

    st.header("🎯 Model Performance (Live)")
    perf = BG_LEARNER.get_performance_report()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Directional Accuracy", f"{perf['directional_accuracy']*100:.2f}%", delta=None)
    with col2:
        st.metric("MAE (recent)", f"{perf['mae']:.4f}", delta=None)
    with col3:
        st.metric("Samples Trained", f"{perf['total_samples']:,}", delta=None)

    st.success("✅ Analysis complete. Background learner continues to train and improve over time.")

# Sidebar: show BG learner metrics and optional recommender usage
with st.sidebar:
    st.divider()
    st.subheader("🧪 Background Learner (auto)")
    perf = BG_LEARNER.get_performance_report()
    st.metric("Directional Accuracy", f"{perf['directional_accuracy']*100:.2f}%")
    st.metric("MAE", f"{perf['mae']:.4f}")
    st.caption(f"Samples trained: {perf['total_samples']:,}")

    st.divider()
    st.subheader("🔍 Scan Top Stocks (from learner predictions)")
    if st.button("Scan & Recommend (use model)"):
        # generate recommendations from the universe using BG_LEARNER
        recs = []
        for tk in BG_LEARNER.universe:
            try:
                hist_tmp, info_tmp, _ = fetch_stock_data(tk, "1y")
                if hist_tmp is None or hist_tmp.empty:
                    continue
                current_p = float(hist_tmp["Close"].iloc[-1])
                pred_r, conf = BG_LEARNER.predict_for_ticker(tk)
                pred_pct_7d = pred_r * 7 * 100  # rough scale
                entry, stop_loss, targets, rr = compute_targets_from_prediction(current_p, pred_r)
                recs.append({
                    "ticker": tk,
                    "pred_pct_7d": pred_pct_7d,
                    "entry": entry, "stop_loss": stop_loss, "targets": targets, "rr": rr, "conf": conf
                })
            except Exception:
                continue
        # sort by highest predicted pct
        recs_sorted = sorted(recs, key=lambda x: x["pred_pct_7d"], reverse=True)[:8]
        if not recs_sorted:
            st.info("No recommendations available (insufficient data).")
        else:
            for i, r in enumerate(recs_sorted, 1):
                st.markdown(f"**{i}. {r['ticker']}**  \n- Pred est (7d): {r['pred_pct_7d']:+.2f}%  \n- Entry: ₹{r['entry']:.2f}  \n- Targets: ₹{r['targets'][0]:.2f}, ₹{r['targets'][1]:.2f}, ₹{r['targets'][2]:.2f}, ₹{r['targets'][3]:.2f}  \n- Stop: ₹{r['stop_loss']:.2f}  \n- Est RR: 1:{r['rr']:.2f}  \n- Model conf: {r['conf']:.2f}")
                st.divider()

# Footer
st.divider()
st.markdown("""
<div style="text-align:center;color:#666;">
    <p>JINNI - AI Stock Analysis System for Indian Markets</p>
    <p>BackgroundLearner: lightweight online model (persisted to background_model.pkl). Replace with XGBoost/Transformer for stronger performance.</p>
    <p>⚠️ Disclaimer: For educational purposes only. Not financial advice. Trade at your own risk.</p>
</div>
""", unsafe_allow_html=True)

# Trigger analysis
if analyze_btn:
    run_analysis()
# Auto-analyze first load for convenience
elif "auto_run_done" not in st.session_state:
    st.session_state["auto_run_done"] = True
    run_analysis()
