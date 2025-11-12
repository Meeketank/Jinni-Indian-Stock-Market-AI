# main.py - Enhanced Main Application
"""
JINNI - AI-Powered Indian Stock Market Analysis System
Advanced version with real-time learning, improved accuracy, and comprehensive analysis
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
import json
from typing import List, Dict, Tuple
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# Import enhanced modules
from enhanced_learner import EnhancedBackgroundLearner
from fundamental_analyzer import FundamentalAnalyzer
from realtime_price import RealTimePriceTracker

# Initialize core components
BG_LEARNER = EnhancedBackgroundLearner()
FUNDAMENTAL_ANALYZER = FundamentalAnalyzer()
PRICE_TRACKER = RealTimePriceTracker()

# Page Configuration
st.set_page_config(
    page_title="JINNI - AI Stock Analysis",
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
    .buy-signal { border-left-color: #28a745 !important; background: #d4edda !important; }
    .sell-signal { border-left-color: #dc3545 !important; background: #f8d7da !important; }
    .hold-signal { border-left-color: #ffc107 !important; background: #fff3cd !important; }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    .stock-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
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
    pred_days = st.slider("🎯 Prediction Horizon (Days)", 1, 90, 7, 
                         help="Number of days for price prediction")
    
    # Analysis type
    analysis_type = st.multiselect(
        "🔍 Analysis Types",
        ["Technical", "Fundamental", "Sentiment", "Risk", "Momentum"],
        default=["Technical", "Fundamental"]
    )
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🚀 Analyze", use_container_width=True)
    with col2:
        realtime_btn = st.button("📊 Live Data", use_container_width=True)
    
    st.divider()
    
    # Model performance
    st.subheader("🧠 AI Model Performance")
    perf = BG_LEARNER.get_performance_report()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{perf['accuracy']:.1f}%")
        st.metric("Samples", f"{perf['total_samples']:,}")
    with col2:
        st.metric("Precision", f"{perf['precision']:.1f}%")
        st.metric("Recall", f"{perf['recall']:.1f}%")
    
    st.divider()
    
    # Quick recommendations
    st.subheader("💎 Top Picks (Weekly 10%+)")
    if st.button("Scan High-Potential Stocks"):
        with st.spinner("Scanning for high-return opportunities..."):
            recommendations = BG_LEARNER.get_high_return_recommendations(min_return=10, days=7)
            for i, rec in enumerate(recommendations[:5], 1):
                with st.expander(f"{i}. {rec['symbol']} - Est: {rec['predicted_return']:.1f}%"):
                    st.write(f"**Current:** ₹{rec['current_price']:.2f}")
                    st.write(f"**Target:** ₹{rec['target_price']:.2f}")
                    st.write(f"**Confidence:** {rec['confidence']:.1f}%")
                    st.write(f"**Signal:** {rec['signal']}")

# Enhanced data fetching with caching
@st.cache_data(ttl=300)
def fetch_stock_data(symbol: str, period: str) -> Tuple[pd.DataFrame, Dict, yf.Ticker]:
    """Fetch stock data with enhanced error handling"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, auto_adjust=True)
        
        if hist.empty:
            st.error(f"No data found for {symbol}")
            return None, {}, None
            
        info = ticker.info
        return hist, info, ticker
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, {}, None

# Enhanced technical indicators
def calculate_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators"""
    if df is None or df.empty:
        return df
        
    df = df.copy()
    
    # Moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'MA{period}'] = df['Close'].rolling(window=period).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Support and Resistance
    df['Resistance'] = df['High'].rolling(20).max()
    df['Support'] = df['Low'].rolling(20).min()
    
    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
    
    return df

# Enhanced chart creation
def create_enhanced_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create comprehensive stock chart"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price with Indicators', 'Volume', 'RSI', 'MACD'),
        row_heights=[0.4, 0.15, 0.15, 0.15]
    )
    
    # Price chart
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], 
        low=df['Low'], close=df['Close'], name='Price'
    ), row=1, col=1)
    
    # Moving averages
    for ma in ['MA20', 'MA50', 'MA200']:
        if ma in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[ma], name=ma, line=dict(width=1)
            ), row=1, col=1)
    
    # Bollinger Bands
    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Upper'], name='BB Upper',
            line=dict(color='rgba(255,0,0,0.3)', dash='dash')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Lower'], name='BB Lower',
            line=dict(color='rgba(0,255,0,0.3)', dash='dash'),
            fill='tonexty'
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
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'), row=4, col=1)
    
    fig.update_layout(
        title=f'{symbol} - Comprehensive Technical Analysis',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Enhanced prediction engine
class EnhancedPredictionEngine:
    def __init__(self):
        self.learner = BG_LEARNER
        
    def generate_comprehensive_prediction(self, symbol: str, df: pd.DataFrame, days: int) -> Dict:
        """Generate comprehensive prediction with multiple models"""
        try:
            # Get AI prediction
            ai_pred, ai_confidence = self.learner.predict_stock(symbol, df, days)
            
            # Technical prediction
            tech_pred = self._technical_prediction(df, days)
            
            # Momentum prediction
            momentum_pred = self._momentum_prediction(df, days)
            
            # Combine predictions
            current_price = df['Close'].iloc[-1]
            final_pred = self._combine_predictions(
                ai_pred, tech_pred, momentum_pred, 
                ai_confidence, current_price, days
            )
            
            # Generate targets and levels
            targets = self._calculate_targets(current_price, final_pred, df)
            support_resistance = self._calculate_support_resistance(df)
            
            return {
                'current_price': current_price,
                'predicted_price': final_pred,
                'predicted_return': ((final_pred - current_price) / current_price) * 100,
                'targets': targets,
                'support_resistance': support_resistance,
                'confidence': ai_confidence,
                'signal': self._generate_signal(final_pred, current_price, df),
                'time_horizon': f"{days} days"
            }
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def _technical_prediction(self, df: pd.DataFrame, days: int) -> float:
        """Technical analysis based prediction"""
        current_price = df['Close'].iloc[-1]
        
        # Trend analysis
        trend_strength = self._calculate_trend_strength(df)
        
        # Momentum analysis
        momentum = self._calculate_momentum(df)
        
        # Volatility adjustment
        volatility = df['Close'].pct_change().std() * np.sqrt(252)
        
        predicted_change = (trend_strength * 0.6 + momentum * 0.4) * np.sqrt(days/30)
        predicted_change = max(-0.3, min(0.3, predicted_change))  # Cap at ±30%
        
        return current_price * (1 + predicted_change)
    
    def _momentum_prediction(self, df: pd.DataFrame, days: int) -> float:
        """Momentum-based prediction"""
        current_price = df['Close'].iloc[-1]
        
        # Short-term momentum (5 days)
        mom_5d = (current_price - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
        
        # Medium-term momentum (20 days)
        mom_20d = (current_price - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
        
        # Volume confirmation
        volume_trend = df['Volume'].iloc[-5:].mean() / df['Volume'].iloc[-20:].mean()
        
        momentum_score = (mom_5d * 0.6 + mom_20d * 0.4) * min(volume_trend, 2.0)
        predicted_change = momentum_score * np.sqrt(days/7)
        
        return current_price * (1 + predicted_change)
    
    def _combine_predictions(self, ai_pred: float, tech_pred: float, 
                           momentum_pred: float, confidence: float, 
                           current_price: float, days: int) -> float:
        """Intelligently combine different prediction methods"""
        # Weight predictions based on confidence and market conditions
        ai_weight = confidence / 100.0
        tech_weight = 0.3 * (1 - ai_weight)
        momentum_weight = 0.2 * (1 - ai_weight)
        
        # Normalize weights
        total_weight = ai_weight + tech_weight + momentum_weight
        ai_weight /= total_weight
        tech_weight /= total_weight
        momentum_weight /= total_weight
        
        combined = (ai_pred * ai_weight + 
                   tech_pred * tech_weight + 
                   momentum_pred * momentum_weight)
        
        return combined
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate overall trend strength"""
        if len(df) < 50:
            return 0.0
            
        # Multiple timeframe analysis
        ma_ratios = []
        for short, long in [(5, 20), (10, 50), (20, 100)]:
            if f'MA{short}' in df.columns and f'MA{long}' in df.columns:
                ratio = (df[f'MA{short}'].iloc[-1] / df[f'MA{long}'].iloc[-1]) - 1
                ma_ratios.append(ratio)
        
        return np.mean(ma_ratios) if ma_ratios else 0.0
    
    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate momentum score"""
        if len(df) < 20:
            return 0.0
            
        # RSI momentum
        rsi_momentum = (df['RSI'].iloc[-1] - 50) / 50 if 'RSI' in df.columns else 0
        
        # Price momentum
        returns = df['Close'].pct_change()
        price_momentum = returns.tail(10).mean()
        
        # MACD momentum
        macd_momentum = (df['MACD'].iloc[-1] - df['MACD'].iloc[-5]).mean() if 'MACD' in df.columns else 0
        
        return (rsi_momentum * 0.3 + price_momentum * 0.5 + macd_momentum * 0.2)
    
    def _calculate_targets(self, current_price: float, predicted_price: float, df: pd.DataFrame) -> Dict:
        """Calculate realistic price targets"""
        volatility = df['Close'].pct_change().std() * np.sqrt(252)
        price_change = (predicted_price - current_price) / current_price
        
        # Conservative target scaling based on volatility
        if volatility > 0.4:  # High volatility
            multipliers = [0.3, 0.6, 0.8, 1.0]
        elif volatility > 0.2:  # Medium volatility
            multipliers = [0.4, 0.7, 0.9, 1.0]
        else:  # Low volatility
            multipliers = [0.5, 0.8, 0.95, 1.0]
        
        targets = {}
        for i, mult in enumerate(multipliers, 1):
            target_price = current_price + (predicted_price - current_price) * mult
            targets[f'target_{i}'] = {
                'price': round(target_price, 2),
                'return': round(((target_price - current_price) / current_price) * 100, 2)
            }
        
        return targets
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate dynamic support and resistance levels"""
        if len(df) < 20:
            return {}
            
        recent_high = df['High'].tail(20).max()
        recent_low = df['Low'].tail(20).min()
        current_price = df['Close'].iloc[-1]
        
        # Fibonacci levels
        fib_236 = recent_high - (recent_high - recent_low) * 0.236
        fib_382 = recent_high - (recent_high - recent_low) * 0.382
        fib_500 = recent_high - (recent_high - recent_low) * 0.5
        fib_618 = recent_high - (recent_high - recent_low) * 0.618
        
        return {
            'resistance_1': round(recent_high, 2),
            'resistance_2': round(fib_236, 2),
            'support_1': round(recent_low, 2),
            'support_2': round(fib_618, 2),
            'pivot': round((recent_high + recent_low + current_price) / 3, 2)
        }
    
    def _generate_signal(self, predicted_price: float, current_price: float, df: pd.DataFrame) -> str:
        """Generate trading signal"""
        predicted_return = (predicted_price - current_price) / current_price * 100
        
        # Technical confirmation
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        ma_signal = "BULLISH" if current_price > df['MA20'].iloc[-1] else "BEARISH"
        
        if predicted_return >= 15 and rsi < 70 and ma_signal == "BULLISH":
            return "STRONG BUY"
        elif predicted_return >= 8:
            return "BUY"
        elif predicted_return <= -15 and rsi > 30 and ma_signal == "BEARISH":
            return "STRONG SELL"
        elif predicted_return <= -8:
            return "SELL"
        elif abs(predicted_return) < 3:
            return "HOLD"
        else:
            return "NEUTRAL"

# Main analysis function
def run_comprehensive_analysis():
    """Run comprehensive stock analysis"""
    # Map timeframe
    timeframe_map = {
        "1 Week": "5d", "1 Month": "1mo", "3 Months": "3mo",
        "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"
    }
    
    period = timeframe_map.get(timeframe, "6mo")
    
    # Fetch data
    with st.spinner("🔄 Fetching real-time data..."):
        hist, info, ticker = fetch_stock_data(stock_symbol, period)
        
    if hist is None or hist.empty:
        st.error("❌ Could not fetch stock data. Please check the symbol and try again.")
        return
    
    # Calculate indicators
    hist = calculate_enhanced_indicators(hist)
    company_name = info.get('longName', info.get('shortName', stock_symbol))
    
    # Display real-time price
    current_price = PRICE_TRACKER.get_current_price(stock_symbol)
    if current_price:
        st.success(f"📊 **Real-time Price for {company_name}: ₹{current_price:.2f}**")
    
    # Company overview
    st.header(f"🏢 {company_name} Analysis")
    
    # Key metrics in columns
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
    
    prediction_engine = EnhancedPredictionEngine()
    with st.spinner("🧠 Generating AI predictions..."):
        prediction = prediction_engine.generate_comprehensive_prediction(
            stock_symbol, hist, pred_days
        )
    
    if prediction:
        # Prediction results
        col1, col2 = st.columns(2)
        
        with col1:
            signal_class = {
                "STRONG BUY": "buy-signal", "BUY": "buy-signal",
                "STRONG SELL": "sell-signal", "SELL": "sell-signal",
                "HOLD": "hold-signal", "NEUTRAL": "hold-signal"
            }.get(prediction['signal'], 'prediction-box')
            
            st.markdown(f"""
            <div class="prediction-box {signal_class}">
                <h3>🎯 AI Recommendation: {prediction['signal']}</h3>
                <h2>₹{prediction['predicted_price']:.2f}</h2>
                <p style="font-size:16px; color:{'green' if prediction['predicted_return'] > 0 else 'red'}">
                    {prediction['predicted_return']:+.2f}% in {prediction['time_horizon']}
                </p>
                <p style="font-size:14px;">Confidence: {prediction['confidence']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Trading plan
            st.markdown("""
            <div class="prediction-box">
                <h3>💰 Trading Plan</h3>
            """, unsafe_allow_html=True)
            
            for i, target in enumerate(prediction['targets'].values(), 1):
                st.write(f"**Target {i}:** ₹{target['price']:.2f} ({target['return']:+.1f}%)")
            
            st.write("---")
            st.write("**Support Levels:**")
            for key, value in list(prediction['support_resistance'].items())[2:]:
                st.write(f"- {key.replace('_', ' ').title()}: ₹{value:.2f}")
            
            st.write("**Resistance Levels:**")
            for key, value in list(prediction['support_resistance'].items())[:2]:
                st.write(f"- {key.replace('_', ' ').title()}: ₹{value:.2f}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Fundamental Analysis
    if "Fundamental" in analysis_type:
        st.header("📊 Fundamental Analysis")
        fundamental_data = FUNDAMENTAL_ANALYZER.analyze(stock_symbol, info)
        
        if fundamental_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Valuation")
                for metric, value in fundamental_data.get('valuation', {}).items():
                    st.metric(metric, value)
            
            with col2:
                st.subheader("Profitability")
                for metric, value in fundamental_data.get('profitability', {}).items():
                    st.metric(metric, value)
            
            with col3:
                st.subheader("Financial Health")
                for metric, value in fundamental_data.get('health', {}).items():
                    st.metric(metric, value)
    
    # Technical Analysis Details
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
    
    # Risk Analysis
    if "Risk" in analysis_type:
        st.header("⚠️ Risk Analysis")
        
        risk_score = BG_LEARNER.calculate_risk_score(stock_symbol, hist)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Overall Risk Score", f"{risk_score:.1f}/10")
            
            # Risk factors
            st.write("**Risk Factors:**")
            if risk_score > 7:
                st.error("High Risk - Consider careful position sizing")
            elif risk_score > 5:
                st.warning("Medium Risk - Monitor closely")
            else:
                st.success("Low Risk - Favorable conditions")
        
        with col2:
            st.write("**Risk Mitigation:**")
            st.write("- Use appropriate stop-loss")
            st.write("- Diversify portfolio")
            st.write("- Monitor key support levels")
    
    # Final Recommendation
    st.header("🎯 Final Conclusion")
    
    conclusion_col1, conclusion_col2 = st.columns([2, 1])
    
    with conclusion_col1:
        if prediction:
            st.write(f"**AI Analysis Summary for {company_name}:**")
            st.write(f"- **Predicted Movement:** {prediction['predicted_return']:+.2f}% in {prediction['time_horizon']}")
            st.write(f"- **Confidence Level:** {prediction['confidence']:.1f}%")
            st.write(f"- **Recommended Action:** {prediction['signal']}")
            st.write(f"- **Key Support:** ₹{prediction['support_resistance']['support_1']:.2f}")
            st.write(f"- **Key Resistance:** ₹{prediction['support_resistance']['resistance_1']:.2f}")
    
    with conclusion_col2:
        # Quick sentiment indicator
        sentiment_score = BG_LEARNER.get_sentiment_score(stock_symbol)
        st.metric("Market Sentiment", f"{sentiment_score:.1f}/10")
        
        if sentiment_score > 7:
            st.success("Bullish Sentiment")
        elif sentiment_score > 4:
            st.info("Neutral Sentiment")
        else:
            st.warning("Bearish Sentiment")

# Real-time data display
def show_realtime_data():
    """Display real-time stock data"""
    st.header("📊 Live Market Data")
    
    current_price = PRICE_TRACKER.get_current_price(stock_symbol)
    if current_price:
        st.success(f"**Live Price for {stock_symbol}: ₹{current_price:.2f}**")
    
    # Placeholder for real-time chart updates
    st.info("Real-time chart updates would be implemented here with WebSocket connections")

# Main application logic
def main():
    """Main application controller"""
    
    # Start background learner if not already running
    if not BG_LEARNER.is_running():
        BG_LEARNER.start()
    
    # Handle button actions
    if realtime_btn:
        show_realtime_data()
    elif analyze_btn or "auto_analyzed" not in st.session_state:
        st.session_state.auto_analyzed = True
        run_comprehensive_analysis()
    else:
        # Default view
        st.info("👆 Click 'Analyze' to start comprehensive stock analysis or 'Live Data' for real-time updates")
        
        # Show recent high-performing stocks
        st.header("🚀 Recent High-Performers")
        try:
            performers = BG_LEARNER.get_recent_performers()
            for perf in performers[:3]:
                st.write(f"**{perf['symbol']}**: {perf['return']:.1f}% return | Confidence: {perf['confidence']:.1f}%")
        except:
            st.write("Performance data loading...")

# Footer
st.divider()
st.markdown("""
<div style="text-align:center;color:#666;font-size:0.9rem;">
    <p><strong>JINNI AI Stock Analysis System</strong> - Advanced Machine Learning for Indian Markets</p>
    <p>⚠️ <em>Disclaimer: This is for educational purposes only. Not financial advice. Always do your own research.</em></p>
    <p>🔄 <em>Real-time learning active. Model accuracy improves continuously.</em></p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
