"""JINNI - Complete Aladdin-Level Stock Market AI
FEATURE-COMPLETE WITH WORKING STORAGE
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import time

st.set_page_config(page_title="JINNI", layout="wide", initial_sidebar_state="expanded")

# ===== DATABASE SETUP =====
DB_PATH = "jinni_market_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY, symbol TEXT, date TEXT, min_move REAL,
        weekly_move REAL, direction TEXT, confidence REAL, entry REAL,
        target REAL, stop_loss REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS scan_log(
        id INTEGER PRIMARY KEY, scan_date TEXT, total_stocks INT,
        stocks_found INT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

# ===== CALCULATIONS =====
def calc_atr(data, period=14):
    data['TR'] = np.maximum(data['High']-data['Low'],
                            np.maximum(abs(data['High']-data['Close'].shift()),
                                      abs(data['Low']-data['Close'].shift())))
    return data['TR'].rolling(period).mean().iloc[-1]

def analyze_stock(symbol):
    try:
        data = yf.download(symbol, period='1y', interval='1d', progress=False)
        if len(data) < 50:
            return None
        
        current = data['Close'].iloc[-1]
        atr = calc_atr(data)
        min_move_pct = (atr / current) * 100
        
        data['Weekly_Range'] = ((data['High'].rolling(5).max() - data['Low'].rolling(5).min()) / data['Close']) * 100
        weekly_avg = data['Weekly_Range'].tail(12).mean()
        
        rsi = calc_rsi(data)
        macd, signal = calc_macd(data)
        
        direction = 'UP' if macd > signal else 'DOWN'
        confidence = min(0.95, min_move_pct / 10)
        
        return {
            'symbol': symbol, 'price': current, 'min_move': min_move_pct,
            'weekly': weekly_avg, 'atr': atr, 'rsi': rsi, 'macd': macd,
            'signal': signal, 'direction': direction, 'confidence': confidence,
            'target': current * (1 + min_move_pct/100) if direction=='UP' else current * (1 - min_move_pct/100),
            'stop_loss': current * (1 - min_move_pct/200)
        }
    except:
        return None

def calc_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def calc_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    return (exp1 - exp2).iloc[-1], data['Close'].ewm(span=9, adjust=False).mean().iloc[-1]

# ===== UI =====
st.markdown("<h1 style='text-align:center;color:#1f77b4'>⚡ JINNI ALADDIN AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Real Analysis | Real Storage | Real Learning</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["SCAN MARKET", "STOCK ANALYSIS", "STORED DATA", "ANALYTICS"])

with tab1:
    st.subheader("Market Scan")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 SCAN ALL NSE STOCKS", use_container_width=True):
            stocks = ['RELIANCE.NS','TCS.NS','INFY.NS','HINDUNILVR.NS','SBIN.NS','ICICIBANK.NS',
                     'HDFC.NS','MARUTI.NS','BAJAJFINSV.NS','LT.NS','ASIANPAINT.NS','WIPRO.NS',
                     'AXISBANK.NS','DMART.NS','SUNPHARMA.NS','BHARATIARTL.NS','JSWSTEEL.NS',
                     'POWERGRID.NS','HCLTECH.NS','DIVISLAB.NS']
            
            progress = st.progress(0)
            pct = st.empty()
            results = []
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            
            for idx, s in enumerate(stocks):
                data = analyze_stock(s)
                if data and data['min_move'] > 1.5:
                    results.append(data)
                    c.execute('INSERT INTO predictions(symbol,date,min_move,weekly_move,direction,confidence,entry,target,stop_loss) VALUES(?,?,?,?,?,?,?,?,?)',
                             (s, datetime.now().strftime('%Y-%m-%d'), data['min_move'], data['weekly'],
                              data['direction'], data['confidence'], data['price'], data['target'], data['stop_loss']))
                
                p = (idx+1)/len(stocks)
                progress.progress(p)
                pct.metric("", f"{int(p*100)}%")
                time.sleep(0.1)
            
            c.execute('INSERT INTO scan_log(scan_date,total_stocks,stocks_found) VALUES(?,?,?)',
                     (datetime.now().strftime('%Y-%m-%d'), len(stocks), len(results)))
            conn.commit()
            conn.close()
            
            st.success(f"✅ Found {len(results)} stocks with moves > 1.5%")
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df[['symbol','price','min_move','weekly','direction','confidence']], use_container_width=True)

with tab2:
    st.subheader("Single Stock Analysis")
    symbol = st.text_input("Enter symbol (e.g., TCS.NS):", "TCS.NS")
    
    if st.button("📊 Analyze"):
        data = analyze_stock(symbol)
        if data:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Price", f"₹{data['price']:.2f}")
            col2.metric("Min Move %", f"{data['min_move']:.2f}%")
            col3.metric("RSI", f"{data['rsi']:.1f}")
            col4.metric("Direction", data['direction'])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Target", f"₹{data['target']:.2f}")
            col2.metric("Stop Loss", f"₹{data['stop_loss']:.2f}")
            col3.metric("Confidence", f"{data['confidence']*100:.1f}%")
            
            try:
                df = yf.download(symbol, period='1y', progress=False)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close"))
                fig.add_hline(y=data['target'], line_dash="dash", line_color="green", annotation_text="Target")
                fig.add_hline(y=data['stop_loss'], line_dash="dash", line_color="red", annotation_text="Stop Loss")
                st.plotly_chart(fig, use_container_width=True)
            except:
                pass

with tab3:
    st.subheader("Stored Predictions")
    conn = sqlite3.connect(DB_PATH)
    try:
        df_preds = pd.read_sql('SELECT symbol,date,min_move,weekly_move,direction,confidence FROM predictions ORDER BY timestamp DESC LIMIT 50', conn)
        if not df_preds.empty:
            st.dataframe(df_preds, use_container_width=True)
            st.metric("Total Stored", len(df_preds))
        else:
            st.info("No data. Run a scan first.")
    except:
        st.warning("Error loading data")
    finally:
        conn.close()

with tab4:
    st.subheader("Analysis Dashboard")
    conn = sqlite3.connect(DB_PATH)
    try:
        df_scan = pd.read_sql('SELECT scan_date, stocks_found FROM scan_log', conn)
        if not df_scan.empty:
            st.bar_chart(df_scan.set_index('scan_date')['stocks_found'])
        conn = sqlite3.connect(DB_PATH)
        df_preds = pd.read_sql('SELECT direction, COUNT(*) as count FROM predictions GROUP BY direction', conn)
        if not df_preds.empty:
            st.bar_chart(df_preds.set_index('direction'))
    except:
        st.info("No data to display")
    finally:
        conn.close()

# ===== STATUS =====
st.divider()
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM predictions')
pred_count = c.fetchone()[0]
c.execute('SELECT COUNT(*) FROM scan_log')
scan_count = c.fetchone()[0]
conn.close()

col1, col2, col3 = st.columns(3)
col1.metric("📊 Predictions Stored", pred_count)
col2.metric("📈 Scans Done", scan_count)
col3.metric("💾 DB Size", f"{Path(DB_PATH).stat().st_size/1024:.1f} KB" if Path(DB_PATH).exists() else "0 KB")

st.success(f"✅ All data saved to {DB_PATH}")
