"""
JINNI - AI-Powered Indian Stock Market Analysis System (Streamlit)
Enhanced version with proper error handling, improved UI, and real-time learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import threading
import time
import os
import pickle
import random
from typing import List, Dict, Tuple
import requests
from bs4 import BeautifulSoup

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

# --- Enhanced BackgroundLearner Implementation ---
class OnlineLinearModel:
    """
    Online linear regressor that predicts next-day return from last N lag returns.
    Uses SGD updates and persists weights to disk.
    """
    def __init__(self, n_features=5, lr=0.001):
        self.n = n_features
        self.lr = lr
        self.w = np.zeros(self.n)  # weights
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
    Background learning system that continuously learns from market data
    """
    def __init__(self, model_path="background_model.pkl", universe: List[str] = None, 
                 n_lags: int = 5, interval_sec: int = 60):
        self.model_path = model_path
        self.n_lags = n_lags
        self.interval_sec = max(10, interval_sec)
        self.universe = universe or self._default_universe()
        self.model = OnlineLinearModel(n_features=self.n_lags, lr=0.003)
        self.model.load(self.model_path)
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
        """Return empty list - uses only user-provided stock"""
        return []
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
            df = t.history(period=f"{days}d", interval="1d")
            return df if not df.empty else None
        except Exception:
            return None

    def _build_samples(self, df: pd.DataFrame):
        if df is None or len(df) <= self.n_lags + 1:
            return np.empty((0, self.n_lags)), np.empty((0,))
        
        close = df["Close"].values
        returns = (close[1:] - close[:-1]) / close[:-1]
        
        X, y = [], []
        for i in range(self.n_lags, len(returns)):
            x = returns[i - self.n_lags : i]
            target = returns[i]
            X.append(x)
            y.append(target)
        
        return np.array(X), np.array(y)

    def _update_metrics(self, preds: np.ndarray, truths: np.ndarray):
        if preds.size == 0:
            return
        
        mae = np.mean(np.abs(preds - truths))
        dir_correct = np.mean(np.sign(preds) == np.sign(truths))
        
        with self.lock:
            total_before = self.metrics["total_samples"]
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
            
            self.metrics["history"].append({
                "ts": time.time(),
                "dir_acc": float(dir_correct),
                "mae": float(mae),
                "samples": n_new
            })
            
            if len(self.metrics["history"]) > 500:
                self.metrics["history"].pop(0)

    def _learn_loop(self):
        while not self._stop_event.is_set():
            try:
                sample_count = min(4, max(1, len(self.universe)))
                tickers = random.sample(self.universe, sample_count)
                
                for tk in tickers:
                    if self._stop_event.is_set():
                        break
                    
                    df = self._fetch_history(tk, days=120)
                    X, y = self._build_samples(df)
                    
                    if X.size == 0:
                        continue
                    
                    preds = self.model.predict(X)
                    self._update_metrics(preds, y)
                    self.model.partial_fit(X, y)
                    
                    if random.random() < 0.2:
                        try:
                            self.model.save(self.model_path)
                        except Exception:
                            pass
                    
                    time.sleep(0.5)
                
                time.sleep(self.interval_sec)
            except Exception:
                time.sleep(self.interval_sec)

    def get_performance_report(self):
        with self.lock:
            return {
                "directional_accuracy": self.metrics["directional_accuracy"],
                "mae": self.metrics["mae"],
                "total_samples": self.metrics["total_samples"],
                "history": list(self.metrics["history"])
            }

    def predict_for_ticker(self, ticker: str, n_lags: int = None):
        n_lags = n_lags or self.n_lags
        df = self._fetch_history(ticker, days=90)
        
        if df is None or len(df) <= n_lags:
            return 0.0, 0.0
        
        close = df["Close"].values
        returns = (close[1:] - close[:-1]) / close[:-1]
        latest = returns[-n_lags:]
        
        if len(latest) < n_lags:
            latest = np.concatenate([np.zeros(n_lags - len(latest)), latest])
        
        pred = float(self.model.predict(latest.reshape(1, -1))[0])
        vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        confidence = max(0.0, min(0.99, 1.0 - vol * 5))
        
        return pred, confidence

    def get_high_return_recommendations(self, min_return: float = 10.0, days: int = 7) -> List[Dict]:
        recommendations = []
        
        for symbol in self.universe[:30]:  # Check first 30 for speed
            try:
                df = self._fetch_history(symbol, days=100)
                if df is None or len(df) < 60:
                    continue
                
                predicted_price, confidence = self.predict_for_ticker(symbol)
                current_price = df['Close'].iloc[-1]
                predicted_return = ((predicted_price - current_price) / current_price) * 100
                
                if abs(predicted_return) >= min_return and confidence > 60:
                    signal = "BUY" if predicted_return > 0 else "SELL"
                    recommendations.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'target_price': predicted_price,
                        'predicted_return': predicted_return,
                        'confidence': confidence,
                        'signal': signal
                    })
                    
            except Exception:
                continue
        
        recommendations.sort(key=lambda x: abs(x['predicted_return']), reverse=True)
        return recommendations[:10]

# Instantiate global background learner
BG_LEARNER = BackgroundLearner(interval_sec=30)
BG_LEARNER.start()

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="JINNI - Indian Stock Market AI",
    page_icon="🧞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    .prediction-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .buy-signal { 
        border-left-color: #28a745 !important; 
        background: #d4edda !important; 
    }
    .sell-signal { 
        border-left-color: #dc3545 !important; 
        background: #f8d7da !important; 
    }
    .hold-signal { 
        border-left-color: #ffc107 !important; 
        background: #fff3cd !important; 
    }
    .stock-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧞 JINNI AI STOCK ANALYSIS</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Real-time Indian Stock Market Analysis & Prediction System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Genie/3D/genie_3d.png", width=120)
    st.title("🎯 Analysis Controls")
    
    # Stock selection
    stock_symbol = st.text_input("📈 Stock Symbol", value="RELIANCE.NS", 
                               help="Enter NSE stock symbol (e.g., RELIANCE.NS, TCS.NS)")
    
    # Analysis timeframe
    timeframe = st.selectbox("📅 Timeframe", 
                           ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"])
    
    # Prediction horizon
    pred_days = st.slider("🎯 Prediction Horizon (Days)", 1, 90, 7)
    
    # Analysis type
    analysis_type = st.multiselect(
        "🔍 Analysis Types",
        ["Technical", "Fundamental", "Risk", "Momentum"],
        default=["Technical", "Fundamental"]
    )
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🚀 Analyze", use_container_width=True, type="primary")
    with col2:
        scan_btn = st.button("🔍 Scan Stocks", use_container_width=True)
    
    st.divider()
    
    # Model performance
    st.subheader("🧠 AI Model Performance")
    perf = BG_LEARNER.get_performance_report()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{perf['directional_accuracy']*100:.1f}%")
        st.metric("Samples", f"{perf['total_samples']:,}")
    with col2:
        st.metric("MAE", f"{perf['mae']:.4f}")
        st.metric("Stocks", f"{len(BG_LEARNER.universe)}")
    
    st.divider()
    
    # Quick recommendations
    st.subheader("💎 Weekly High-Potential")
    if st.button("Find 10%+ Opportunities"):
        with st.spinner("Scanning for high-return stocks..."):
            recommendations = BG_LEARNER.get_high_return_recommendations(min_return=10, days=7)
            if recommendations:
                for i, rec in enumerate(recommendations[:3], 1):
                    with st.expander(f"{i}. {rec['symbol']} - Est: {rec['predicted_return']:.1f}%"):
                        st.write(f"**Current:** ₹{rec['current_price']:.2f}")
                        st.write(f"**Target:** ₹{rec['target_price']:.2f}")
                        st.write(f"**Confidence:** {rec['confidence']:.1f}%")
                        st.write(f"**Signal:** {rec['signal']}")
            else:
                st.info("No high-confidence opportunities found")

# Helper functions
@st.cache_data(ttl=300)
def fetch_stock_data(symbol: str, period: str) -> Tuple[pd.DataFrame, Dict, yf.Ticker]:
    """Fetch stock data with enhanced error handling"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, auto_adjust=True)
        
        if hist.empty:
            return None, {}, None
            
        info = ticker.info
        return hist, info, ticker
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, {}, None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators"""
    if df is None or df.empty:
        return df
        
    df = df.copy()
    
    # Moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'MA{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'].fillna(50, inplace=True)
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(20, min_periods=1).mean()
    bb_std = df['Close'].rolling(20, min_periods=1).std().fillna(0)
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(20, min_periods=1).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(20, min_periods=1).std() * np.sqrt(252) * 100
    
    return df

def create_enhanced_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create comprehensive stock chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price with Indicators', 'Volume', 'RSI & MACD'),
        row_heights=[0.5, 0.2, 0.3]
    )
    
    # Price chart
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], 
        low=df['Low'], close=df['Close'], name='Price'
    ), row=1, col=1)
    
    # Moving averages
    for ma in ['MA20', 'MA50']:
        if ma in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[ma], name=ma, line=dict(width=2)
            ), row=1, col=1)
    
    # Volume
    colors = ['red' if row['Open'] > row['Close'] else 'green' 
             for _, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name='Volume',
        marker_color=colors, opacity=0.7
    ), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="grey", row=3, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'), row=3, col=1)
    
    fig.update_layout(
        title=f'{symbol} - Technical Analysis',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def compute_targets_from_prediction(current_price: float, pred_return: float, days: int):
    """Compute realistic trading targets"""
    entry = current_price
    
    # Dynamic stop loss based on volatility and time horizon
    base_sl = 0.03  # 3% base stop loss
    stop_loss = entry * (1 - base_sl) if pred_return >= 0 else entry * (1 + base_sl)
    
    # Scale targets based on prediction confidence and time horizon
    time_factor = np.sqrt(days / 7)  # Scale with square root of time
    
    targets = []
    for multiplier in [0.3, 0.6, 0.9, 1.2]:
        target_price = entry + (pred_return * entry * multiplier * time_factor)
        targets.append(max(0.01, target_price))
    
    # Risk-reward ratio
    risk = abs(entry - stop_loss)
    reward = abs(targets[1] - entry) if len(targets) > 1 else 0
    rr = (reward / risk) if risk > 0 else 0
    
    return entry, stop_loss, targets, rr

def enhanced_prediction(df: pd.DataFrame, days: int) -> float:
    """Enhanced prediction with multiple factors"""
    if df is None or df.empty:
        return 0.0
    
    current_price = df["Close"].iloc[-1]
    
    # Multiple timeframe analysis
    recent_5d = df["Close"].pct_change().tail(5).mean()
    recent_20d = df["Close"].pct_change().tail(20).mean()
    recent_60d = df["Close"].pct_change().tail(60).mean()
    
    # Weighted momentum
    momentum = (recent_5d * 0.5 + recent_20d * 0.3 + recent_60d * 0.2)
    
    # Technical confirmation
    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
    rsi_factor = 1.0
    if rsi < 30:  # Oversold
        rsi_factor = 1.2
    elif rsi > 70:  # Overbought
        rsi_factor = 0.8
    
    # Volume confirmation
    volume_trend = df['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df.columns else 1.0
    volume_factor = min(2.0, max(0.5, volume_trend))
    
    # Time scaling
    time_multiplier = np.sqrt(days / 30)
    
    # Combine factors
    total_change = momentum * days * time_multiplier * rsi_factor * volume_factor
    
    # Realistic bounds
    total_change = max(-0.40, min(0.40, total_change))
    
    return current_price * (1 + total_change)

# Main analysis function
def run_comprehensive_analysis():
    """Run comprehensive stock analysis"""
    timeframe_map = {
        "1 Week": "5d", "1 Month": "1mo", "3 Months": "3mo",
        "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"
    }
    
    period = timeframe_map.get(timeframe, "6mo")
    
    # Fetch data
    with st.spinner("🔄 Fetching market data..."):
        hist, info, ticker = fetch_stock_data(stock_symbol, period)
        
    if hist is None or hist.empty:
        st.error("❌ Could not fetch stock data. Please check the symbol and try again.")
        return
    
    # Calculate indicators
    hist = calculate_technical_indicators(hist)
    company_name = info.get('longName', info.get('shortName', stock_symbol))
    
    # Company overview
    st.header(f"🏢 {company_name}")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        st.metric("Current Price", f"₹{current_price:.2f}", 
                 f"{change:+.2f} ({change_pct:+.2f}%)")
    
    with col2:
        day_high = hist['High'].iloc[-1]
        st.metric("Day High", f"₹{day_high:.2f}")
    
    with col3:
        day_low = hist['Low'].iloc[-1]
        st.metric("Day Low", f"₹{day_low:.2f}")
    
    with col4:
        volume = hist['Volume'].iloc[-1]
        st.metric("Volume", f"{volume:,.0f}")
    
    with col5:
        market_cap = info.get('marketCap')
        if market_cap:
            st.metric("Market Cap", f"₹{market_cap/1e7:.0f} Cr")
        else:
            st.metric("Volatility", f"{hist['Volatility'].iloc[-1]:.1f}%")
    
    # Technical chart
    st.plotly_chart(create_enhanced_chart(hist, stock_symbol), use_container_width=True)
    
    # AI Prediction Section
    st.header("🤖 AI Prediction & Recommendations")
    
    # Get predictions from both models
    with st.spinner("🧠 Generating AI predictions..."):
        bg_pred_return, bg_conf = BG_LEARNER.predict_for_ticker(stock_symbol)
        enhanced_pred_price = enhanced_prediction(hist, pred_days)
        
        # Combine predictions
        if bg_conf > 0.3:
            # Weighted combination based on confidence
            final_pred_price = (bg_pred_return * bg_conf + enhanced_pred_price * (1 - bg_conf))
        else:
            final_pred_price = enhanced_pred_price
        
        current_price = hist['Close'].iloc[-1]
        pred_change = ((final_pred_price - current_price) / current_price) * 100
    
    # Display predictions
    col1, col2 = st.columns(2)
    
    with col1:
        # Determine signal and styling
        if pred_change >= 15:
            signal, signal_class = "🔵 STRONG BUY", "buy-signal"
        elif pred_change >= 5:
            signal, signal_class = "🟢 BUY", "buy-signal"
        elif pred_change <= -15:
            signal, signal_class = "🔴 STRONG SELL", "sell-signal"
        elif pred_change <= -5:
            signal, signal_class = "🔴 SELL", "sell-signal"
        else:
            signal, signal_class = "⚪ HOLD", "hold-signal"
        
        st.markdown(f"""
        <div class="prediction-box {signal_class}">
            <h3>🎯 AI Recommendation: {signal}</h3>
            <h2>₹{final_pred_price:.2f}</h2>
            <p style="font-size:16px; color:{'green' if pred_change > 0 else 'red'}">
                {pred_change:+.2f}% in {pred_days} days
            </p>
            <p style="font-size:14px;">Model Confidence: {max(bg_conf * 100, 60):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Trading plan
        entry, stop_loss, targets, rr = compute_targets_from_prediction(current_price, pred_change/100, pred_days)
        
        st.markdown("""
        <div class="prediction-box">
            <h3>💰 Trading Plan</h3>
        """, unsafe_allow_html=True)
        
        st.write(f"**Entry Price:** ₹{entry:.2f}")
        st.write(f"**Stop Loss:** ₹{stop_loss:.2f}")
        st.write("---")
        st.write("**Profit Targets:**")
        for i, target in enumerate(targets[:4], 1):
            target_return = ((target - entry) / entry) * 100
            st.write(f"Target {i}: ₹{target:.2f} ({target_return:+.1f}%)")
        
        st.write(f"**Risk-Reward:** 1:{rr:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Technical Analysis
    if "Technical" in analysis_type:
        st.header("📈 Technical Analysis")
        
        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
        
        with tech_col1:
            rsi = hist['RSI'].iloc[-1]
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            st.metric("RSI (14)", f"{rsi:.1f}", rsi_status)
        
        with tech_col2:
            macd = hist['MACD'].iloc[-1]
            macd_signal = hist['MACD_Signal'].iloc[-1]
            macd_trend = "Bullish" if macd > macd_signal else "Bearish"
            st.metric("MACD", f"{macd:.3f}", macd_trend)
        
        with tech_col3:
            ma_trend = "Bullish" if hist['Close'].iloc[-1] > hist['MA50'].iloc[-1] else "Bearish"
            st.metric("Trend (MA50)", ma_trend)
        
        with tech_col4:
            volatility = hist['Volatility'].iloc[-1]
            vol_level = "High" if volatility > 40 else "Medium" if volatility > 20 else "Low"
            st.metric("Volatility", f"{volatility:.1f}%", vol_level)
    
    # Fundamental Analysis
    if "Fundamental" in analysis_type:
        st.header("📊 Fundamental Analysis")
        
        fund_col1, fund_col2, fund_col3 = st.columns(3)
        
        with fund_col1:
            pe_ratio = info.get('trailingPE')
            if pe_ratio:
                pe_status = "Low" if pe_ratio < 15 else "High" if pe_ratio > 25 else "Reasonable"
                st.metric("P/E Ratio", f"{pe_ratio:.2f}", pe_status)
            else:
                st.metric("P/E Ratio", "N/A")
        
        with fund_col2:
            pb_ratio = info.get('priceToBook')
            if pb_ratio:
                pb_status = "Low" if pb_ratio < 1.5 else "High" if pb_ratio > 3 else "Reasonable"
                st.metric("P/B Ratio", f"{pb_ratio:.2f}", pb_status)
            else:
                st.metric("P/B Ratio", "N/A")
        
        with fund_col3:
            market_cap = info.get('marketCap')
            if market_cap:
                cap_category = "Large" if market_cap > 2e12 else "Mid" if market_cap > 5e11 else "Small"
                st.metric("Market Cap", f"₹{market_cap/1e7:.0f} Cr", cap_category)
            else:
                st.metric("Market Cap", "N/A")
    
    # Risk Analysis
    if "Risk" in analysis_type:
        st.header("⚠️ Risk Analysis")
        
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            # Simple risk score based on volatility and drawdown
            volatility = hist['Volatility'].iloc[-1]
            max_drawdown = (hist['Close'] / hist['Close'].expanding().max() - 1).min() * 100
            risk_score = min(10, (volatility / 10) + abs(max_drawdown))
            
            st.metric("Risk Score", f"{risk_score:.1f}/10")
            
            if risk_score > 7:
                st.error("High Risk - Consider careful position sizing")
            elif risk_score > 5:
                st.warning("Medium Risk - Monitor closely")
            else:
                st.success("Low Risk - Favorable conditions")
        
        with risk_col2:
            st.write("**Risk Factors:**")
            st.write(f"- Volatility: {volatility:.1f}%")
            st.write(f"- Max Drawdown: {max_drawdown:.1f}%")
            st.write("- Market Cap: " + ("Large" if (info.get('marketCap', 0) > 2e12) else "Mid/Small"))
    
    # Final Recommendation
    st.header("🎯 Investment Conclusion")
    
    conclusion_col1, conclusion_col2 = st.columns([2, 1])
        
        with conclusion_col1:
            # Display comprehensive recommendation
            recommendation = "STRONG BUY" if pred_return > 20 else "BUY" if pred_return > 10 else "HOLD" if pred_return > -5 else "SELL"
            
            if recommendation == "STRONG BUY":
                st.success(f"🎯 **Recommendation: {recommendation}**")
                st.write("High conviction opportunity with strong technical and fundamental alignment.")
            elif recommendation == "BUY":
                st.info(f"📈 **Recommendation: {recommendation}**")
                st.write("Positive outlook with good risk-reward setup.")
            elif recommendation == "HOLD":
                st.warning(f"⏸️ **Recommendation: {recommendation}**")
                st.write("Neutral outlook - wait for better entry or confirmation.")
            else:
                st.error(f"⚠️ **Recommendation: {recommendation}**")
                st.write("Negative outlook - consider reducing exposure.")
        
        with conclusion_col2:
            st.metric("Confidence", f"{min(abs(pred_return) * 3, 95):.0f}%")
            st.metric("Risk Level", "Low" if risk_score < 5 else "Medium" if risk_score < 7 else "High")
    
    except Exception as e:
        st.error(f"❌ Analysis Error: {str(e)}")
        st.info("💡 Try a different stock symbol or check your internet connection.")

# CRITICAL: Button trigger logic - This is what makes the Analyze button actually work!
if analyze_btn:
    run_comprehensive_analysis()

elif scan_btn:
    with st.spinner("🔍 Scanning for high-opportunity stocks..."):
        try:
            recommendations = BG_LEARNER.get_high_return_recommendations(min_return=10, days=7)
            
            st.header("📊 Stock Scanner Results")
            
            if recommendations:
                st.success(f"Found {len(recommendations)} high-potential opportunities!")
                
                # Create results table
                scan_df = pd.DataFrame(recommendations)
                scan_df = scan_df.sort_values('expected_return', ascending=False)
                
                # Display top opportunities
                for idx, row in scan_df.head(10).iterrows():
                    with st.expander(f"🎯 {row['symbol']} - Expected Return: {row['expected_return']:.1f}%"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"₹{row['current_price']:.2f}")
                        with col2:
                            st.metric("Target Price", f"₹{row['target_price']:.2f}")
                        with col3:
                            st.metric("Timeframe", f"{row['days']} days")
            else:
                st.warning("No high-confidence opportunities found at this time.")
                st.info("Try adjusting the scanner parameters or check back later.")
        
        except Exception as e:
            st.error(f"Scanner Error: {str(e)}")

# Auto-refresh mechanism (every 60 seconds)
if 'last_refresh' not in st.session_state:
    st.session_state['last_refresh'] = time.time()

current_time = time.time()
if current_time - st.session_state['last_refresh'] > 60:
    st.session_state['last_refresh'] = current_time
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational and informational purposes only.</p>
    <p>Not financial advice. Always do your own research and consult with a qualified financial advisor.</p>
    <p>Past performance does not guarantee future results. Stock markets involve risk.</p>
    <p style='margin-top: 10px; font-size: 12px;'>JINNI AI © 2025 | Powered by Advanced ML Algorithms</p>
</div>
""", unsafe_allow_html=True)
    
