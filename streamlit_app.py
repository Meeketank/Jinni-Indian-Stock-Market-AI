# streamlit_jinni_improved.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
from datetime import datetime
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

DB_PATH = 'jinni_market_data.db'

# ---------------------------
# Database initialization
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
    c = conn.cursor()
    # Create analysis table (explicit columns, timestamp default)
    c.execute('''
    CREATE TABLE IF NOT EXISTS analysis(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        momentum REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS scan_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_date TEXT,
        total_scanned INT,
        bullish INT,
        bearish INT,
        neutral INT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

init_db()

# ---------------------------
# Indicator utilities
# ---------------------------
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # Standard RSI using Wilder's smoothing via ewm
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # Use ewm with adjust=False to replicate Wilder
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calc_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # True range
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().fillna(tr.rolling(window=period, min_periods=1).mean())
    return atr

# ---------------------------
# yfinance fetch with caching
# ---------------------------
@st.cache_data(show_spinner=False, persist=True)
def fetch_history(symbol: str, period: str = '1y', interval: str = '1d'):
    # normalize symbol for NSE if missing extension
    if not (symbol.endswith('.NS') or symbol.endswith('.BO') or '.' in symbol):
        symbol_query = symbol + '.NS'
    else:
        symbol_query = symbol
    ticker = yf.Ticker(symbol_query)
    hist = ticker.history(period=period, interval=interval)
    info = {}
    try:
        info = ticker.info
    except Exception:
        info = {}
    return symbol_query, hist, info

# ---------------------------
# Core analysis function
# ---------------------------
def analyze_stock(symbol: str):
    try:
        sym, hist, info = fetch_history(symbol, period='1y', interval='1d')

        if hist is None or hist.empty or len(hist) < 20:
            return None

        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        volume = hist.get('Volume', pd.Series(np.zeros(len(close)), index=close.index))

        current_price = float(close.iloc[-1])

        # indicators
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float(close.rolling(20).mean().iloc[-1])
        rsi_series = calc_rsi(close, period=14)
        rsi_val = float(rsi_series.iloc[-1])
        macd_line, macd_signal, macd_hist = calc_macd(close)
        macd_val = float(macd_hist.iloc[-1])

        atr_series = calc_atr(hist, period=14)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else float((high - low).mean())

        # momentum (5-day)
        if len(close) >= 6:
            momentum = float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100)
        else:
            momentum = float((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100)

        # Technical scoring (modular)
        technical_score = 0
        # price vs MA
        technical_score += 25 if current_price > ma20 else 0
        technical_score += 20 if ma20 > ma50 else 0
        technical_score += 20 if rsi_val > 50 and rsi_val < 70 else 0
        technical_score += 20 if macd_line.iloc[-1] > macd_signal.iloc[-1] else 0
        technical_score += min(15, max(0, min(15, (volume.iloc[-1] / (volume.iloc[-20:].mean() + 1)) * 5)))

        technical_score = float(np.clip(technical_score, 0, 100))

        # Momentum score
        momentum_score = float(np.clip(50 + momentum * 1.5, 0, 100))

        # Fundamental scoring - resilient and defensive
        pe = info.get('trailingPE') or info.get('forwardPE') or np.nan
        pb = info.get('priceToBook') or np.nan
        dividend = info.get('dividendYield') or 0.0
        # Normalize and penalize extreme values; when no data, be neutral (50)
        if np.isfinite(pe):
            pe_score = max(0, min(30, (25 - min(pe, 100))/25 * 30))  # lower PE -> higher score (capped)
        else:
            pe_score = 15  # neutral

        if np.isfinite(pb):
            pb_score = max(0, min(30, (3 - min(pb, 20))/3 * 30))  # pb near 1-3 desirable
        else:
            pb_score = 15

        div_score = 10 if dividend and dividend > 0 else 0

        # Optional quality metrics if available
        market_cap = info.get('marketCap') or 0
        mc_score = 10 if market_cap and market_cap > 1e10 else 5

        fundamental_score = float(np.clip(pe_score + pb_score + div_score + mc_score, 0, 100))

        # Aggregate
        avg_score = technical_score * 0.45 + momentum_score * 0.25 + fundamental_score * 0.30

        # Direction & Confidence (explicit rules)
        if current_price > ma50 and rsi_val < 70 and momentum > 0 and macd_hist.iloc[-1] > 0:
            direction = 'UP'
        elif current_price < ma50 and rsi_val > 30 and momentum < 0 and macd_hist.iloc[-1] < 0:
            direction = 'DOWN'
        else:
            direction = 'HOLD'

        # Confidence scaled off avg_score
        confidence = float(np.clip(40 + (avg_score - 50) * 0.9, 10, 95))

        # target and stop calculation using ATR for realistic bands
        move_pct = (atr / current_price) if current_price > 0 else 0.03
        if direction == 'UP':
            target = current_price * (1 + move_pct * 2)   # two ATR as conservative target
            stop_loss = current_price - atr * 1.5
        elif direction == 'DOWN':
            target = current_price * (1 - move_pct * 2)
            stop_loss = current_price + atr * 1.5
        else:
            target = current_price * (1 + move_pct)   # narrower expectations
            stop_loss = current_price * (1 - move_pct)

        # Human-readable recommendation
        if avg_score >= 75:
            recommendation = f'STRONG {direction}'
        elif avg_score >= 60:
            recommendation = f'{direction}'
        elif avg_score >= 45:
            recommendation = 'HOLD'
        else:
            recommendation = f'WEAK {direction}'

        # Final textual summary
        final_statement = (
            f"{sym}: Price ₹{current_price:.2f}. Direction: {direction} (Confidence {confidence:.1f}%). "
            f"Technical: {technical_score:.1f}/100, Momentum: {momentum_score:.1f}/100, Fundamental: {fundamental_score:.1f}/100. "
            f"Target ₹{target:.2f} | Stop ₹{stop_loss:.2f}. Recommendation: {recommendation}."
        )

        return {
            'symbol': sym,
            'price': current_price,
            'ma20': ma20,
            'ma50': ma50,
            'rsi': rsi_val,
            'macd': float(macd_hist.iloc[-1]),
            'technical_score': technical_score,
            'momentum_score': momentum_score,
            'fundamental_score': fundamental_score,
            'avg_score': avg_score,
            'direction': direction,
            'confidence': confidence,
            'target': target,
            'stop_loss': stop_loss,
            'recommendation': recommendation,
            'momentum': momentum,
            'final_statement': final_statement
        }
    except Exception as e:
        # For debugging locally you can st.error(str(e)) but don't expose in production
        return None

# ---------------------------
# DB write helpers
# ---------------------------
def insert_analysis_row(row: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    columns = ('symbol','date','price','rsi','macd','ma20','ma50','target','stop_loss',
               'direction','confidence','fundamental_score','technical_score','momentum_score',
               'recommendation','momentum')
    placeholders = ','.join(['?'] * len(columns))
    values = (
        row['symbol'],
        datetime.now().strftime('%Y-%m-%d'),
        row['price'],
        row['rsi'],
        row['macd'],
        row['ma20'],
        row['ma50'],
        row['target'],
        row['stop_loss'],
        row['direction'],
        row['confidence'],
        row['fundamental_score'],
        row['technical_score'],
        row['momentum_score'],
        row['recommendation'],
        row['momentum']
    )
    c.execute(f'INSERT INTO analysis ({",".join(columns)}) VALUES ({placeholders})', values)
    conn.commit()
    conn.close()

def insert_scan_log(scan_date, total, bullish, bearish, neutral):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO scan_log (scan_date, total_scanned, bullish, bearish, neutral) VALUES (?,?,?,?,?)',
              (scan_date, total, bullish, bearish, neutral))
    conn.commit()
    conn.close()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title='JINNI', layout='wide')
st.markdown('# ⚡ JINNI - Improved AI Stock Analysis')
st.markdown('Real Storage | Better Analysis | Clear Recommendations')

tabs = st.tabs(['🔍 SCAN MARKET', '📊 SINGLE STOCK', '📈 PORTFOLIO'])

with tabs[0]:
    st.header('Market Scan - small NSE universe (example)')
    if st.button('🔍 SCAN SAMPLE NSE LIST'):
        nse_stocks = [
            'RELIANCE.NS','TCS.NS','INFY.NS','HINDUNILVR.NS','SBIN.NS','ICICIBANK.NS','HDFC.NS',
            'MARUTI.NS','BAJAJFINSV.NS','LT.NS','ASIANPAINT.NS','WIPRO.NS','AXISBANK.NS','DMART.NS'
        ]
        progress_bar = st.progress(0)
        status = st.empty()
        results = []

        for i, stock in enumerate(nse_stocks):
            progress_bar.progress((i+1) / len(nse_stocks))
            status.info(f'Scanning {i+1}/{len(nse_stocks)}: {stock}')
            analysis = analyze_stock(stock)
            if analysis:
                results.append(analysis)
                try:
                    insert_analysis_row(analysis)
                except Exception as e:
                    st.warning(f'DB insert failed for {stock}: {e}')
            time.sleep(0.15)

        bullish = sum(1 for r in results if r['direction'] == 'UP')
        bearish = sum(1 for r in results if r['direction'] == 'DOWN')
        neutral = len(results) - bullish - bearish

        insert_scan_log(datetime.now().strftime('%Y-%m-%d'), len(nse_stocks), bullish, bearish, neutral)

        status.success(f'✅ Scan Complete! {len(results)} stocks analyzed')
        if results:
            df = pd.DataFrame(results).sort_values('avg_score', ascending=False)
            st.dataframe(df[['symbol','price','rsi','macd','technical_score','fundamental_score','momentum_score','recommendation','confidence']].round(2),
                         use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric('📈 Bullish', bullish)
            col2.metric('📉 Bearish', bearish)
            col3.metric('⏸️ Neutral', neutral)
        else:
            st.info('No valid results returned. Check connectivity or symbol list.')

with tabs[1]:
    st.header('Deep Analysis - Single Stock')
    symbol = st.text_input('Enter Symbol', 'TCS.NS')
    if st.button('📊 Analyze'):
        result = analyze_stock(symbol)
        if result:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric('Price', f'₹{result["price"]:.2f}')
            col2.metric('RSI', f'{result["rsi"]:.1f}')
            col3.metric('MACD', f'{result["macd"]:.4f}')
            col4.metric('Tech Score', f'{result["technical_score"]:.0f}%')
            col5.metric('Fund Score', f'{result["fundamental_score"]:.0f}%')

            st.write(f'### 🎯 RECOMMENDATION: {result["recommendation"]}')
            st.write(f'**Technical**: {result["technical_score"]:.1f}/100 | **Momentum**: {result["momentum_score"]:.1f}/100 | **Fundamental**: {result["fundamental_score"]:.1f}/100')
            st.write(f'**Target**: ₹{result["target"]:.2f} | **Stop Loss**: ₹{result["stop_loss"]:.2f} | **Confidence**: {result["confidence"]:.1f}%')
            st.write(f'**Direction**: {result["direction"]} | **Overall Score**: {result["avg_score"]:.1f}/100')
            st.write('---')
            st.write('**Final summary:**')
            st.info(result['final_statement'])
            # store single analysis
            try:
                insert_analysis_row(result)
            except Exception as e:
                st.warning(f'Could not save to DB: {e}')
        else:
            st.error('Stock not found or insufficient history (need >=20 bars).')

with tabs[2]:
    st.header('Portfolio / History (last 200 entries)')
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM analysis ORDER BY timestamp DESC LIMIT 200', conn)
    conn.close()
    if not df.empty:
        st.dataframe(df[['symbol','date','price','direction','recommendation','confidence']].round(2), use_container_width=True)
    else:
        st.info('No data yet. Run a scan or analyze a symbol.')
