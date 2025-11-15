import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

DB_PATH = 'jinni_market_data.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analysis(
        id INTEGER PRIMARY KEY,
        symbol TEXT,
        date TEXT,
        price REAL,
        rsi REAL,
        macd REAL,
        ma20 REAL,
        ma50 REAL,
        target REAL,
        stop_loss REAL,
        direction TEXT,
        confidence REAL,
        fundamental_score REAL,
        technical_score REAL,
        momentum_score REAL,
        recommendation TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS scan_log(
        scan_date TEXT,
        total_scanned INT,
        bullish INT,
        bearish INT,
        neutral INT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

def calc_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed>=0].sum()/period
    down = -seed[seed<0].sum()/period
    rs = up/down if down != 0 else 1
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1.+rs)
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period
        rs = up/down if down != 0 else 1
        rsi[i] = 100. - 100./(1.+rs)
    return rsi

def calc_macd(prices):
    ema12 = pd.Series(prices).ewm(span=12).mean().values
    ema26 = pd.Series(prices).ewm(span=26).mean().values
    macd = ema12 - ema26
    return macd

def analyze_stock(symbol):
    try:
        ticker = yf.Ticker(symbol if '.NS' in symbol or '.BO' in symbol else symbol+'.NS')
        hist = ticker.history(period='1y')
        if hist.empty:
            return None
        
        close = hist['Close'].values
        high = hist['High'].values
        low = hist['Low'].values
        volume = hist['Volume'].values
        
        current_price = close[-1]
        ma20 = pd.Series(close).rolling(20).mean().values[-1]
        ma50 = pd.Series(close).rolling(50).mean().values[-1]
        rsi_val = calc_rsi(close)[-1]
        macd_val = calc_macd(close)[-1]
        
        technical_score = 0
        if current_price > ma20: technical_score += 30
        if ma20 > ma50: technical_score += 20
        if rsi_val < 70 and rsi_val > 30: technical_score += 25
        if macd_val > 0: technical_score += 25
        
        momentum = ((close[-1] - close[-5])/close[-5]*100) if close[-5] != 0 else 0
        momentum_score = min(100, max(0, 50 + momentum*5))
        
        try:
            info = ticker.info
            pe = info.get('trailingPE', 0) or 0
            pb = info.get('priceToBook', 0) or 0
            dividend = info.get('dividendYield', 0) or 0
            fundamental_score = 100 - (min(pe/50*30, 30) + min(pb/10*30, 30) + (0 if dividend > 0 else -10))
            fundamental_score = max(0, min(100, fundamental_score))
        except:
            fundamental_score = 50
        
        avg_score = (technical_score * 0.4 + momentum_score * 0.3 + fundamental_score * 0.3)
        
        if current_price > ma50 and rsi_val < 70 and momentum > 0:
            direction = 'UP'
            confidence = min(95, 50 + (avg_score-50)*0.9)
        elif current_price < ma50 and rsi_val > 30 and momentum < 0:
            direction = 'DOWN'
            confidence = min(95, 50 + (avg_score-50)*0.9)
        else:
            direction = 'HOLD'
            confidence = 30 + (avg_score-50)*0.2
        
        atr = np.mean(np.abs(high[:-1] - low[:-1]))
        move_pct = (atr / current_price) * 100
        
        target = current_price * (1 + move_pct/100) if direction == 'UP' else current_price * (1 - move_pct/100)
        stop_loss = current_price * (1 - 0.03) if direction == 'UP' else current_price * (1 + 0.03)
        
        if avg_score >= 75:
            recommendation = f'STRONG {direction}'
        elif avg_score >= 60:
            recommendation = direction
        elif avg_score >= 45:
            recommendation = 'HOLD'
        else:
            recommendation = f'WEAK {direction}'
        
        return {
            'symbol': symbol,
            'price': current_price,
            'ma20': ma20,
            'ma50': ma50,
            'rsi': rsi_val,
            'macd': macd_val,
            'technical_score': technical_score,
            'momentum_score': momentum_score,
            'fundamental_score': fundamental_score,
            'avg_score': avg_score,
            'direction': direction,
            'confidence': confidence,
            'target': target,
            'stop_loss': stop_loss,
            'recommendation': recommendation,
            'momentum': momentum
        }
    except Exception as e:
        return None

st.set_page_config(page_title='JINNI', layout='wide')
st.markdown('# ⚡ JINNI - ALADDIN-LEVEL AI STOCK ANALYSIS')
st.markdown('### Real Storage | Real Analysis | Real Learning')

tabs = st.tabs(['🔍 SCAN MARKET', '📊 SINGLE STOCK', '📈 PORTFOLIO'])

with tabs[0]:
    st.header('Market Scan - NSE Universe')
    if st.button('🔍 SCAN ALL NSE STOCKS'):
        nse_stocks = ['RELIANCE.NS','TCS.NS','INFY.NS','HINDUNILVR.NS','SBIN.NS','ICICIBANK.NS','HDFC.NS','MARUTI.NS','BAJAJFINSV.NS','LT.NS',
                     'ASIANPAINT.NS','WIPRO.NS','AXISBANK.NS','DMART.NS','SUNPHARMA.NS','BHARATIARTL.NS','JSWSTEEL.NS','POWERGRID.NS','HCLTECH.NS','DIVISLAB.NS']
        
        progress_bar = st.progress(0)
        status = st.empty()
        results = []
        
        for i, stock in enumerate(nse_stocks):
            progress_bar.progress((i+1)/len(nse_stocks))
            status.info(f'Scanning {i+1}/{len(nse_stocks)}: {stock}')
            
            analysis = analyze_stock(stock)
            if analysis:
                results.append(analysis)
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute('''INSERT INTO analysis VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (stock, datetime.now().strftime('%Y-%m-%d'),analysis['price'],analysis['rsi'],analysis['macd'],
                     analysis['ma20'],analysis['ma50'],analysis['target'],analysis['stop_loss'],
                     analysis['direction'],analysis['confidence'],analysis['fundamental_score'],
                     analysis['technical_score'],analysis['momentum_score'],analysis['recommendation']))
                conn.commit()
                conn.close()
            time.sleep(0.2)
        
        bullish = sum(1 for r in results if r['direction']=='UP')
        bearish = sum(1 for r in results if r['direction']=='DOWN')
        neutral = len(results) - bullish - bearish
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO scan_log VALUES(?,?,?,?,?)',
                 (datetime.now().strftime('%Y-%m-%d'),len(nse_stocks),bullish,bearish,neutral))
        conn.commit()
        conn.close()
        
        status.success(f'✅ Scan Complete! {len(results)} stocks analyzed')
        
        df = pd.DataFrame(results).sort_values('avg_score', ascending=False)
        st.dataframe(df[['symbol','price','rsi','macd','technical_score','recommendation','confidence']],use_container_width=True)
        
        col1,col2,col3 = st.columns(3)
        col1.metric('📈 Bullish',bullish)
        col2.metric('📉 Bearish',bearish)
        col3.metric('⏸️ Neutral',neutral)

with tabs[1]:
    st.header('Deep Analysis - Single Stock')
    symbol = st.text_input('Enter Symbol','TCS.NS')
    if st.button('📊 Analyze'):
        result = analyze_stock(symbol)
        if result:
            col1,col2,col3,col4,col5 = st.columns(5)
            col1.metric('Price',f'₹{result["price"]:.2f}')
            col2.metric('RSI',f'{result["rsi"]:.1f}')
            col3.metric('MACD',f'{result["macd"]:.3f}')
            col4.metric('Tech Score',f'{result["technical_score"]:.0f}%')
            col5.metric('Fund Score',f'{result["fundamental_score"]:.0f}%')
            
            st.write(f'### 🎯 RECOMMENDATION: {result["recommendation"]}')
            st.write(f'**Technical Score**: {result["technical_score"]:.0f}% | **Momentum**: {result["momentum"]:.2f}% | **Fundamental**: {result["fundamental_score"]:.0f}%')
            st.write(f'**Target**: ₹{result["target"]:.2f} | **Stop Loss**: ₹{result["stop_loss"]:.2f} | **Confidence**: {result["confidence"]:.1f}%')
            st.write(f'**Direction**: {result["direction"]} | **Overall Score**: {result["avg_score"]:.1f}/100')
        else:
            st.error('Stock not found')

with tabs[2]:
    st.header('Portfolio History')
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM analysis ORDER BY timestamp DESC LIMIT 100',conn)
    conn.close()
    if not df.empty:
        st.dataframe(df[['symbol','date','price','direction','recommendation','confidence']],use_container_width=True)
    else:
        st.info('No data yet. Run a scan first!')
