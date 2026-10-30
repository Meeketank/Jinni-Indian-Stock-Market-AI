# streamlit_jinni_full_with_learning.py
# Single-file Streamlit app:
# - indicators, analysis, plotting
# - scanner with expected-% filter
# - robust SQLite schema migration
# - optional Firestore save (set GOOGLE_APPLICATION_CREDENTIALS env var to ServiceAccount JSON)
# - simple trainer that builds rolling-window samples and trains a classifier
#
# Requirements:
# pip install streamlit yfinance pandas numpy plotly scikit-learn lightgbm google-cloud-firestore

import os
import time
import math
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# optional ML libs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# optional Firestore
USE_FIRESTORE = False
try:
    from google.cloud import firestore
    # use GOOGLE_APPLICATION_CREDENTIALS env var to point to service account JSON
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        db_fs = firestore.Client()
        USE_FIRESTORE = True
except Exception:
    USE_FIRESTORE = False

st.set_page_config(page_title='JINNI - Full (Learner)', layout='wide')
st.title('⚡ JINNI — All-In-One Deep Analysis + Learning')
st.markdown("Candles, indicators, fundamentals, pattern detection, multi-targets, scanning, training and saving (SQLite and optional Firestore).")

DB_PATH = 'jinni_market_data.db'
MODEL_PATH = 'jinni_model.pkl'

# ---------------- DB: migration helpers ----------------
REQUIRED_ANALYSIS_COLUMNS = {
    'symbol': 'TEXT', 'date': 'TEXT', 'price': 'REAL', 'rsi': 'REAL', 'macd': 'REAL',
    'ma20': 'REAL', 'ma50': 'REAL', 'ma200': 'REAL',
    'target1': 'REAL', 'target2': 'REAL', 'target3': 'REAL', 'target4': 'REAL',
    'stop_loss': 'REAL', 'direction': 'TEXT', 'confidence': 'REAL',
    'fundamental_score': 'REAL', 'technical_score': 'REAL', 'momentum_score': 'REAL',
    'recommendation': 'TEXT', 'pattern_summary': 'TEXT', 'final_statement': 'TEXT'
}

REQUIRED_SCANLOG_COLUMNS = {
    'scan_date':'TEXT','total_scanned':'INT','bullish':'INT','bearish':'INT','neutral':'INT',
    'min_expected_pct':'REAL','direction':'TEXT'
}

def get_table_columns(conn, table_name) -> Dict[str,str]:
    c = conn.cursor()
    try:
        c.execute(f"PRAGMA table_info({table_name})")
        rows = c.fetchall()
        return {r[1]: r[2] for r in rows}  # name -> type
    except Exception:
        return {}

def ensure_tables_and_columns():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # create analysis table if not exist (minimal)
    if 'analysis' not in [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
        col_defs = ", ".join([f"{k} {v}" for k,v in REQUIRED_ANALYSIS_COLUMNS.items()])
        # add id and timestamp
        sql = f"CREATE TABLE analysis (id INTEGER PRIMARY KEY AUTOINCREMENT, {col_defs}, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"
        c.execute(sql)
    else:
        # check and add missing columns
        existing = get_table_columns(conn, 'analysis')
        for col, coltype in REQUIRED_ANALYSIS_COLUMNS.items():
            if col not in existing:
                c.execute(f"ALTER TABLE analysis ADD COLUMN {col} {coltype}")
    # create scan_log
    if 'scan_log' not in [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
        col_defs = ", ".join([f"{k} {v}" for k,v in REQUIRED_SCANLOG_COLUMNS.items()])
        c.execute(f"CREATE TABLE scan_log (id INTEGER PRIMARY KEY AUTOINCREMENT, {col_defs}, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
    else:
        existing = get_table_columns(conn, 'scan_log')
        for col, coltype in REQUIRED_SCANLOG_COLUMNS.items():
            if col not in existing:
                c.execute(f"ALTER TABLE scan_log ADD COLUMN {col} {coltype}")
    conn.commit()
    conn.close()

ensure_tables_and_columns()

# ----------------- Indicators & utils -----------------
@st.cache_data(show_spinner=False)
def fetch_history(symbol: str, period: str='2y', interval: str='1d'):
    sym = symbol
    if not (symbol.upper().endswith('.NS') or symbol.upper().endswith('.BO') or '.' in symbol):
        sym = symbol + '.NS'
    try:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period=period, interval=interval)
        info = {}
        try:
            info = ticker.info
        except Exception:
            info = {}
        return sym, hist, info
    except Exception:
        return sym, pd.DataFrame(), {}

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
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

def calc_atr(df: pd.DataFrame, period: int = 14):
    high = df['High']; low = df['Low']; close = df['Close']; prev_close = close.shift(1)
    tr1 = high - low; tr2 = (high - prev_close).abs(); tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().fillna(tr.rolling(min_periods=1, window=period).mean())
    return atr

# pattern helpers (same as before)
def detect_golden_death(ma_short, ma_long):
    if len(ma_short) < 2 or len(ma_long) < 2:
        return "Insufficient data"
    prev = ma_short[-2] - ma_long[-2]; curr = ma_short[-1] - ma_long[-1]
    if prev < 0 and curr > 0: return "Golden Cross"
    if prev > 0 and curr < 0: return "Death Cross"
    return "No recent cross"

def detect_double_top_bottom(close, window=60):
    series = close[-window:]; 
    if len(series) < 20: return "Insufficient data"
    peaks = (series.shift(1) < series) & (series.shift(-1) < series)
    troughs= (series.shift(1) > series) & (series.shift(-1) > series)
    peak_dates = series.index[peaks].tolist(); trough_dates = series.index[troughs].tolist()
    if len(peak_dates)>=2:
        p = series.loc[peak_dates]; 
        if abs(p.iloc[-1]-p.iloc[-2])/p.iloc[-2] < 0.03: return "Possible Double Top"
    if len(trough_dates)>=2:
        t = series.loc[trough_dates];
        if abs(t.iloc[-1]-t.iloc[-2])/t.iloc[-2] < 0.03: return "Possible Double Bottom"
    return "No clear double top/bottom"

def detect_higher_highs_lows(close, lookback=20):
    s = close[-lookback:]; 
    if len(s)<6: return "Insufficient data"
    highs=s.rolling(5).max().dropna(); lows=s.rolling(5).min().dropna()
    if len(highs)<2 or len(lows)<2: return "Insufficient data"
    if highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]: return "Uptrend"
    if highs.iloc[-1] < highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2]: return "Downtrend"
    return "Range/Mixed"

def detect_volume_spike(volume, multiplier=3):
    if len(volume) < 21: return "Insufficient data"
    avg = volume[-21:-1].mean()
    if volume.iloc[-1] > avg * multiplier: return f"Volume spike: {volume.iloc[-1]:.0f} vs avg {avg:.0f}"
    return "No big volume spike"

def compute_targets_and_stops(price, atr, direction):
    if atr <= 0 or price <= 0: return [price]*4, price*0.98
    if direction == 'UP':
        t1 = price + atr*1.0; t2 = price + atr*2.0; t3 = price + atr*3.5; t4 = price + atr*6.0; sl = price - atr*1.25
    elif direction == 'DOWN':
        t1 = price - atr*1.0; t2 = price - atr*2.0; t3 = price - atr*3.5; t4 = price - atr*6.0; sl = price + atr*1.25
    else:
        t1 = price + atr*0.5; t2 = price + atr*1.0; t3 = price + atr*1.5; t4 = price + atr*2.5; sl = price - atr*0.5
    return [t1,t2,t3,t4], sl

# ----------------- Analysis engine -----------------
def analyze_full(symbol):
    sym, hist, info = fetch_history(symbol, period='2y', interval='1d')
    if hist is None or hist.empty or len(hist) < 30: return None
    df = hist.copy().dropna(subset=['Close'])
    price = float(df['Close'].iloc[-1])
    ma20 = df['Close'].rolling(20).mean(); ma50 = df['Close'].rolling(50).mean(); ma200 = df['Close'].rolling(200).mean()
    rsi = calc_rsi(df['Close'])
    macd_line, macd_signal, macd_hist = calc_macd(df['Close'])
    atr = calc_atr(df, period=14).iloc[-1]
    volume = df['Volume']
    momentum = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100 if len(df) > 6 else 0.0

    tech = 0
    tech += 25 if price > ma20.iloc[-1] else 0
    tech += 20 if ma20.iloc[-1] > ma50.iloc[-1] else 0
    tech += 15 if ma50.iloc[-1] > ma200.iloc[-1] else 0
    tech += 15 if macd_hist.iloc[-1] > 0 else 0
    tech += 15 if (rsi.iloc[-1] > 45 and rsi.iloc[-1] < 70) else 0
    tech = float(np.clip(tech,0,100))
    mom_score = float(np.clip(50 + momentum*1.5,0,100))

    pe = info.get('trailingPE') or info.get('forwardPE') or np.nan
    pb = info.get('priceToBook') or np.nan
    div_yield = info.get('dividendYield') or 0.0
    market_cap = info.get('marketCap') or np.nan
    sector = info.get('sector') or info.get('industry') or 'N/A'
    pe_score = 20 if (np.isfinite(pe) and pe < 30) else (10 if np.isfinite(pe) else 12)
    pb_score = 15 if (np.isfinite(pb) and pb < 5) else 8
    div_score = 10 if div_yield and div_yield > 0 else 0
    mc_score = 10 if np.isfinite(market_cap) and market_cap > 1e10 else 5
    fundamental_score = float(np.clip(pe_score + pb_score + div_score + mc_score, 0, 100))

    avg_score = tech*0.45 + mom_score*0.25 + fundamental_score*0.30

    if price > ma50.iloc[-1] and rsi.iloc[-1] < 70 and momentum > 0 and macd_hist.iloc[-1] > 0:
        direction = 'UP'
    elif price < ma50.iloc[-1] and rsi.iloc[-1] > 30 and momentum < 0 and macd_hist.iloc[-1] < 0:
        direction = 'DOWN'
    else:
        direction = 'HOLD'

    confidence = float(np.clip(40 + (avg_score - 50) * 0.9, 10, 95))
    targets, stop_loss = compute_targets_and_stops(price, atr, direction)

    gd_cross = detect_golden_death(ma50.values if len(ma50)>0 else np.array([]), ma200.values if len(ma200)>0 else np.array([]))
    double = detect_double_top_bottom(df['Close'], window=120)
    hhll = detect_higher_highs_lows(df['Close'], lookback=40)
    vol_spike = detect_volume_spike(df['Volume'])
    pattern_summary = f"{gd_cross} | {double} | {hhll} | {vol_spike}"

    if avg_score >= 75: recommendation = f"STRONG {direction}"
    elif avg_score >= 60: recommendation = f"{direction}"
    elif avg_score >= 45: recommendation = "HOLD"
    else: recommendation = f"WEAK {direction}"

    final_statement = (f"{sym}: Last ₹{price:.2f}. Dir {direction} (Conf {confidence:.1f}%). "
                       f"Tech {tech:.1f}/100 | Mom {mom_score:.1f}/100 | Fund {fundamental_score:.1f}/100. "
                       f"Targets: {', '.join([f'₹{x:.2f}' for x in targets])}. Stop ₹{stop_loss:.2f}. {pattern_summary}")

    fundamentals_table = {
        'symbol': sym, 'sector': sector, 'marketCap': market_cap,
        'trailingPE': pe, 'priceToBook': pb, 'dividendYield': div_yield,
        'longName': info.get('longName') or info.get('shortName') or sym
    }

    out = {
        'sym': sym, 'df': df, 'price': price, 'ma20': ma20, 'ma50': ma50, 'ma200': ma200,
        'rsi': rsi, 'macd_hist': macd_hist, 'macd_line': macd_line, 'macd_signal': macd_signal,
        'atr': atr, 'volume': volume, 'technical_score': tech, 'momentum_score': mom_score,
        'fundamental_score': fundamental_score, 'avg_score': avg_score, 'direction': direction,
        'confidence': confidence, 'targets': targets, 'stop_loss': stop_loss, 'recommendation': recommendation,
        'pattern_summary': pattern_summary, 'final_statement': final_statement, 'fundamentals_table': fundamentals_table
    }
    return out

# ----------------- Persistence -----------------
def save_analysis_to_sqlite(result):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cols = list(REQUIRED_ANALYSIS_COLUMNS.keys())
    placeholders = ",".join("?" for _ in cols)
    values = []
    # map keys
    for k in cols:
        if k == 'symbol':
            values.append(result.get('sym'))
        elif k == 'date':
            values.append(datetime.now().strftime('%Y-%m-%d'))
        elif k == 'price':
            values.append(float(result.get('price')))
        elif k == 'rsi':
            try: values.append(float(result['rsi'].iloc[-1]))
            except: values.append(None)
        elif k == 'macd':
            try: values.append(float(result['macd_hist'].iloc[-1]))
            except: values.append(None)
        elif k == 'ma20':
            try: values.append(float(result['ma20'].iloc[-1]))
            except: values.append(None)
        elif k == 'ma50':
            try: values.append(float(result['ma50'].iloc[-1]))
            except: values.append(None)
        elif k == 'ma200':
            try: values.append(float(result['ma200'].iloc[-1]))
            except: values.append(None)
        elif k.startswith('target'):
            idx = int(k[-1]) - 1
            try: values.append(float(result['targets'][idx]))
            except: values.append(None)
        elif k == 'stop_loss':
            values.append(float(result.get('stop_loss')))
        elif k in ['technical_score','momentum_score','fundamental_score','confidence']:
            values.append(float(result.get(k)))
        elif k in ['direction','recommendation','pattern_summary','final_statement']:
            values.append(result.get(k))
        else:
            values.append(None)
    try:
        c.execute(f"INSERT INTO analysis ({','.join(cols)}) VALUES ({placeholders})", tuple(values))
        conn.commit()
    except sqlite3.OperationalError as e:
        st.error(f"SQLite write failed: {e}")
    finally:
        conn.close()

def save_scan_log_sqlite(total, bullish, bearish, neutral, min_pct, direction):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO scan_log (scan_date,total_scanned,bullish,bearish,neutral,min_expected_pct,direction) VALUES (?,?,?,?,?,?,?)",
                  (datetime.now().strftime('%Y-%m-%d'), total, bullish, bearish, neutral, float(min_pct), direction))
        conn.commit()
    except sqlite3.OperationalError as e:
        st.error(f"Could not insert scan_log: {e}")
    finally:
        conn.close()

def save_to_firestore(result):
    if not USE_FIRESTORE:
        return {'ok':False, 'reason':'Firestore not enabled'}
    try:
        doc = {
            'symbol': result['sym'],
            'ts': datetime.utcnow(),
            'price': float(result['price']),
            'technical_score': float(result['technical_score']),
            'fundamental_score': float(result['fundamental_score']),
            'momentum_score': float(result['momentum_score']),
            'direction': result['direction'],
            'confidence': float(result['confidence']),
            'targets': result['targets'],
            'stop_loss': float(result['stop_loss']),
            'final_statement': result['final_statement']
        }
        db_fs.collection('analyses').add(doc)
        return {'ok':True}
    except Exception as e:
        return {'ok':False, 'reason':str(e)}

# ----------------- Scanner UI -----------------
st.markdown("## Scan / Recommender")
with st.expander("Scanner settings", expanded=False):
    min_expected_pct = st.number_input('Minimum expected % move (to Target1)', min_value=0.1, max_value=100.0, value=3.0, step=0.1)
    scan_direction = st.selectbox('Direction', ['UP','DOWN','EITHER'], index=2)
    symbols_text = st.text_area('Symbols (comma-separated). Default small sample (edit/paste full list)', value="RELIANCE.NS,TCS.NS,INFY.NS,HINDUNILVR.NS,SBIN.NS,ICICIBANK.NS,HDFC.NS,MARUTI.NS,BAJAJFINSV.NS,LT.NS,ASIANPAINT.NS,WIPRO.NS,AXISBANK.NS,DMART.NS")
    throttle_ms = st.number_input('Delay between symbols (ms)', value=150, min_value=50, max_value=5000, step=50)
    enable_firestore_write = st.checkbox('Also save each analysis to Firestore (requires GOOGLE_APPLICATION_CREDENTIALS env var)', value=False)
    if enable_firestore_write and not USE_FIRESTORE:
        st.warning("Firestore client not initialized. Set GOOGLE_APPLICATION_CREDENTIALS environment variable to ServiceAccount JSON path and restart app.")
scan_now = st.button('🔎 Run Scan')

if scan_now:
    ensure_tables_and_columns()
    raw = symbols_text.strip()
    if not raw:
        st.warning("Provide symbol list.")
    else:
        symbol_list = [s.strip().upper() for s in raw.split(',') if s.strip()]
        st.info(f"Scanning {len(symbol_list)} symbols — min {min_expected_pct}% — direction {scan_direction}")
        progress = st.progress(0)
        status = st.empty()
        results = []
        for i,sym in enumerate(symbol_list):
            status.info(f"Scanning {i+1}/{len(symbol_list)}: {sym}")
            res = analyze_full(sym)
            if res is None:
                time.sleep(throttle_ms/1000)
                progress.progress((i+1)/len(symbol_list))
                continue
            price = res['price']; target1 = res['targets'][0]
            if res['direction']=='UP':
                expected_pct = (target1-price)/price*100.0
            elif res['direction']=='DOWN':
                expected_pct = (price-target1)/price*100.0
            else:
                expected_pct = abs((target1-price)/price*100.0)
            qualifies=False
            if scan_direction=='EITHER':
                qualifies = expected_pct >= min_expected_pct
            elif scan_direction=='UP':
                qualifies = (res['direction']=='UP') and expected_pct >= min_expected_pct
            else:
                qualifies = (res['direction']=='DOWN') and expected_pct >= min_expected_pct
            # always save
            save_analysis_to_sqlite(res)
            if enable_firestore_write and USE_FIRESTORE:
                save_to_firestore(res)
            if qualifies:
                results.append({
                    'symbol':res['sym'],'price':price,'direction':res['direction'],
                    'expected_pct':round(expected_pct,3),'confidence':round(res['confidence'],2),
                    'tech':round(res['technical_score'],2),'fund':round(res['fundamental_score'],2),
                    'target1':res['targets'][0],'stop_loss':res['stop_loss']
                })
            time.sleep(throttle_ms/1000)
            progress.progress((i+1)/len(symbol_list))
        bullish = sum(1 for r in results if r['direction']=='UP')
        bearish = sum(1 for r in results if r['direction']=='DOWN')
        neutral = len(results)-bullish-bearish
        save_scan_log_sqlite(len(symbol_list), bullish, bearish, neutral, min_expected_pct, scan_direction)
        st.success(f"Scan complete. Found {len(results)} matching symbols.")
        if results:
            dfres = pd.DataFrame(results).sort_values(['expected_pct','confidence'],ascending=[False,False])
            st.dataframe(dfres, use_container_width=True)
            c1,c2,c3 = st.columns(3)
            c1.metric("📈 Bullish", bullish)
            c2.metric("📉 Bearish", bearish)
            c3.metric("⏸️ Neutral", neutral)

# ----------------- Single-symbol analysis UI -----------------
st.markdown("## Single Symbol Deep Analysis")
colA, colB = st.columns([3,1])
with colA:
    query_sym = st.text_input("Enter symbol (example TCS.NS)", value="TCS.NS")
with colB:
    run_btn = st.button("Analyze & Plot")
if run_btn:
    with st.spinner("Analyzing..."):
        r = analyze_full(query_sym)
    if r is None:
        st.error("Symbol not found or insufficient history.")
    else:
        df = r['df']
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6,0.2,0.2])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=r['ma20'], name='MA20'), row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=r['ma50'], name='MA50'), row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=r['ma200'], name='MA200'), row=1,col=1)
        bb_mid = df['Close'].rolling(20).mean(); bb_std = df['Close'].rolling(20).std()
        fig.add_trace(go.Scatter(x=df.index, y=bb_mid+2*bb_std, name='BB Up', opacity=0.4), row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb_mid-2*bb_std, name='BB Lo', opacity=0.4), row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=r['rsi'], name='RSI'), row=2,col=1)
        fig.add_hline(y=70, line_dash='dash', row=2,col=1); fig.add_hline(y=30, line_dash='dash', row=2,col=1)
        fig.add_trace(go.Bar(x=df.index, y=r['macd_hist'], name='MACD Hist'), row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=r['macd_line'], name='MACD Line'), row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=r['macd_signal'], name='MACD Signal'), row=3,col=1)
        fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Price", f"₹{r['price']:.2f}", delta=None)
        st.write("**Final statement:**"); st.info(r['final_statement'])
        st.write("**Targets:**"); st.write([f"₹{x:.2f}" for x in r['targets']])
        st.write("**Fundamentals:**")
        ft = r['fundamentals_table']
        st.table(pd.DataFrame({'Field':['Name','Sector','MarketCap','PE','PB','DivYield'], 'Value':[ft.get('longName'),ft.get('sector'),ft.get('marketCap'),ft.get('trailingPE'),ft.get('priceToBook'),ft.get('dividendYield')]}))
        # save
        save_analysis_to_sqlite(r)
        if enable_firestore_write and USE_FIRESTORE:
            save_to_firestore(r)
        st.success("Saved analysis locally.")

# ----------------- Model training (simple) -----------------
st.markdown("## Model: Train / Predict")
st.write("This trains a classifier that predicts whether next 5-day return >= threshold using rolling windows from historical prices.")
train_col1, train_col2 = st.columns([2,1])
with train_col1:
    label_threshold = st.number_input("Label threshold for 'up' (5-day return fraction)", min_value=0.0, max_value=0.2, value=0.03, step=0.005)
    symbols_for_training = st.text_area("Symbols to use for training (comma-separated). Leave blank to use last 200 saved analyses' symbols.", value="")
    train_button = st.button("Train model now")
with train_col2:
    st.write("Model artifact:", MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        st.write("Model exists. You can 'Predict' below.")
if train_button:
    # Build training dataset by downloading history per symbol and rolling
    st.info("Building training dataset. This may take time for many symbols.")
    # choose symbols
    if symbols_for_training.strip():
        train_symbols = [s.strip().upper() for s in symbols_for_training.split(',') if s.strip()]
    else:
        # take last 200 saved analyses
        conn = sqlite3.connect(DB_PATH)
        df_hist = pd.read_sql_query("SELECT symbol FROM analysis ORDER BY timestamp DESC LIMIT 200", conn)
        conn.close()
        train_symbols = df_hist['symbol'].unique().tolist()
    st.write(f"Training symbols: {len(train_symbols)}")
    rows = []
    bar = st.progress(0)
    for idx,sym in enumerate(train_symbols):
        _, hist, _ = fetch_history(sym, period='3y', interval='1d')
        if hist is None or hist.empty or len(hist) < 60:
            bar.progress((idx+1)/len(train_symbols))
            continue
        dfh = hist.copy().dropna(subset=['Close'])
        # build rolling features: for each date where forward 5 days exist
        for i in range(60, len(dfh)-6):
            window = dfh.iloc[i-60:i]  # 60-day lookback
            target_price = dfh['Close'].iloc[i+5]
            curr_price = dfh['Close'].iloc[i]
            # features: last values of indicators computed on window
            close = window['Close']
            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            rsi_val = calc_rsi(close).iloc[-1]
            macd_l, macd_s, macd_h = calc_macd(close)
            macd_hist_val = macd_h.iloc[-1]
            atr_val = calc_atr(window).iloc[-1]
            mom5 = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]
            tech_score = 0
            try:
                tech_score += 25 if close.iloc[-1] > ma20 else 0
                tech_score += 20 if ma20 > ma50 else 0
                tech_score += 15 if macd_hist_val > 0 else 0
                tech_score += 15 if (rsi_val > 45 and rsi_val < 70) else 0
            except:
                continue
            ret5 = (target_price - curr_price) / curr_price
            label = 1 if ret5 >= label_threshold else 0
            rows.append({'symbol':sym,'ma20':ma20,'ma50':ma50,'rsi':rsi_val,'macd':macd_hist_val,'atr':atr_val,'mom5':mom5,'tech':tech_score,'label':label})
        bar.progress((idx+1)/len(train_symbols))
    st.write(f"Built dataset with {len(rows)} rows.")
    if not rows:
        st.error("No training rows generated.")
    else:
        df_train = pd.DataFrame(rows).dropna()
        X = df_train[['ma20','ma50','rsi','macd','atr','mom5','tech']].fillna(0)
        y = df_train['label'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test); probs = clf.predict_proba(X_test)[:,1]
        metrics = {
            'accuracy': float(accuracy_score(y_test,preds)),
            'precision': float(precision_score(y_test,preds,zero_division=0)),
            'recall': float(recall_score(y_test,preds,zero_division=0)),
            'f1': float(f1_score(y_test,preds,zero_division=0)),
            'auc': float(roc_auc_score(y_test,probs))
        }
        joblib.dump(clf, MODEL_PATH)
        st.success("Model trained and saved.")
        st.write("Metrics:", metrics)
        # save model metadata to Firestore if available
        if USE_FIRESTORE:
            try:
                db_fs.collection('models').add({'ts':datetime.utcnow(),'metrics':metrics,'model_path':MODEL_PATH})
                st.success("Model metadata saved to Firestore.")
            except Exception as e:
                st.warning(f"Could not save model metadata to Firestore: {e}")

# ----------------- Predict UI -----------------
st.markdown("### Predict with current model")
predict_sym = st.text_input("Symbol for prediction (uses latest model features)", value="TCS.NS", key="predsym")
predict_button = st.button("Predict")
if predict_button:
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Train first.")
    else:
        model = joblib.load(MODEL_PATH)
        r = analyze_full(predict_sym)
        if r is None:
            st.error("Could not analyze symbol.")
        else:
            # construct features same as training
            try:
                ma20 = float(r['ma20'].iloc[-1])
                ma50 = float(r['ma50'].iloc[-1])
                rsi_v = float(r['rsi'].iloc[-1])
                macd_v = float(r['macd_hist'].iloc[-1])
                atr_v = float(r['atr'])
                mom5 = float((r['df']['Close'].iloc[-1] - r['df']['Close'].iloc[-6]) / r['df']['Close'].iloc[-6])
                tech = float(r['technical_score'])
                X = pd.DataFrame([{'ma20':ma20,'ma50':ma50,'rsi':rsi_v,'macd':macd_v,'atr':atr_v,'mom5':mom5,'tech':tech}])
                prob = model.predict_proba(X)[0,1]; lab = int(prob>=0.5)
                st.write(f"Predicted prob of 5-day return >= threshold: {prob:.3f} → label {lab}")
                # log to Firestore predictions
                if USE_FIRESTORE:
                    try:
                        db_fs.collection('predictions').add({'symbol':r['sym'],'ts':datetime.utcnow(),'prob':float(prob),'label':int(lab)})
                    except Exception as e:
                        st.warning(f"Could not save prediction to Firestore: {e}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ----------------- History & admin -----------------
st.sidebar.header("History & Admin")
if st.sidebar.button("Show recent analyses (local)"):
    conn = sqlite3.connect(DB_PATH)
    df_hist = pd.read_sql_query("SELECT symbol,date,price,direction,recommendation,confidence,timestamp FROM analysis ORDER BY timestamp DESC LIMIT 300", conn)
    conn.close()
    if df_hist.empty:
        st.sidebar.info("No saved analyses yet.")
    else:
        st.sidebar.dataframe(df_hist)
st.sidebar.markdown("---")
st.sidebar.write("Tips:")
st.sidebar.write("- For Firestore enablement, set GOOGLE_APPLICATION_CREDENTIALS to service account JSON path before launching app.")
st.sidebar.write("- Training uses rolling windows scraped from yfinance; training many symbols will be slow. Start small.")
