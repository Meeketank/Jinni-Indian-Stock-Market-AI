# streamlit_jinni_all_in_one.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
from datetime import datetime
import time
import warnings
from plotly.subplots import make_subplots
import plotly.graph_objects as go

warnings.filterwarnings('ignore')
DB_PATH = 'jinni_market_data.db'

# ----------------- DB init -----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
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
        ma200 REAL,
        target1 REAL,
        target2 REAL,
        target3 REAL,
        target4 REAL,
        stop_loss REAL,
        direction TEXT,
        confidence REAL,
        fundamental_score REAL,
        technical_score REAL,
        momentum_score REAL,
        recommendation TEXT,
        pattern_summary TEXT,
        final_statement TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # scan_log will be created dynamically if needed but create here as well
    c.execute('''
        CREATE TABLE IF NOT EXISTS scan_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT,
            total_scanned INT,
            bullish INT,
            bearish INT,
            neutral INT,
            min_expected_pct REAL,
            direction TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ----------------- Helpers / Indicators -----------------
@st.cache_data(show_spinner=False, persist=True)
def fetch_history(symbol: str, period: str = '2y', interval: str = '1d'):
    sym = symbol
    if not (symbol.upper().endswith('.NS') or symbol.upper().endswith('.BO') or '.' in symbol):
        sym = symbol + '.NS'
    ticker = yf.Ticker(sym)
    hist = ticker.history(period=period, interval=interval)
    info = {}
    try:
        info = ticker.info
    except Exception:
        info = {}
    return sym, hist, info

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
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().fillna(tr.rolling(min_periods=1, window=period).mean())
    return atr

# Basic pattern detectors
def detect_golden_death(ma_short, ma_long):
    if len(ma_short) < 2 or len(ma_long) < 2:
        return "Insufficient data"
    prev = ma_short[-2] - ma_long[-2]
    curr = ma_short[-1] - ma_long[-1]
    if prev < 0 and curr > 0:
        return "Golden Cross (short MA crossed above long MA)"
    if prev > 0 and curr < 0:
        return "Death Cross (short MA crossed below long MA)"
    return "No recent cross"

def detect_double_top_bottom(close, window=60):
    series = close[-window:]
    if len(series) < 20:
        return "Insufficient data"
    peaks = (series.shift(1) < series) & (series.shift(-1) < series)
    troughs = (series.shift(1) > series) & (series.shift(-1) > series)
    peak_dates = series.index[peaks].tolist()
    trough_dates = series.index[troughs].tolist()
    if len(peak_dates) >= 2:
        p = series.loc[peak_dates]
        if abs(p.iloc[-1] - p.iloc[-2]) / p.iloc[-2] < 0.03:
            return "Possible Double Top"
    if len(trough_dates) >= 2:
        t = series.loc[trough_dates]
        if abs(t.iloc[-1] - t.iloc[-2]) / t.iloc[-2] < 0.03:
            return "Possible Double Bottom"
    return "No clear double top/bottom"

def detect_higher_highs_lows(close, lookback=20):
    s = close[-lookback:]
    if len(s) < 6:
        return "Insufficient data"
    highs = s.rolling(5).max().dropna()
    lows = s.rolling(5).min().dropna()
    if len(highs) < 2 or len(lows) < 2:
        return "Insufficient data"
    if highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]:
        return "Uptrend (higher highs & higher lows)"
    if highs.iloc[-1] < highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2]:
        return "Downtrend (lower highs & lower lows)"
    return "Range / Mixed"

def detect_volume_spike(volume, multiplier=3):
    if len(volume) < 21:
        return "Insufficient data"
    avg = volume[-21:-1].mean()
    if volume.iloc[-1] > avg * multiplier:
        return f"Volume spike: {volume.iloc[-1]:.0f} vs avg {avg:.0f}"
    return "No big volume spike"

# ----------------- Analysis engine -----------------
def compute_targets_and_stops(price, atr, direction):
    if atr <= 0 or price <= 0:
        return [price, price, price, price], price * 0.98
    if direction == 'UP':
        t1 = price + atr * 1.0
        t2 = price + atr * 2.0
        t3 = price + atr * 3.5
        t4 = price + atr * 6.0
        sl = price - atr * 1.25
    elif direction == 'DOWN':
        t1 = price - atr * 1.0
        t2 = price - atr * 2.0
        t3 = price - atr * 3.5
        t4 = price - atr * 6.0
        sl = price + atr * 1.25
    else:
        t1 = price + atr * 0.5
        t2 = price + atr * 1.0
        t3 = price + atr * 1.5
        t4 = price + atr * 2.5
        sl = price - atr * 0.5
    return [t1, t2, t3, t4], sl

def analyze_full(symbol):
    sym, hist, info = fetch_history(symbol, period='2y', interval='1d')
    if hist is None or hist.empty or len(hist) < 30:
        return None
    df = hist.copy()
    df = df.dropna(subset=['Close'])
    price = float(df['Close'].iloc[-1])
    ma20 = df['Close'].rolling(20).mean()
    ma50 = df['Close'].rolling(50).mean()
    ma200 = df['Close'].rolling(200).mean()
    rsi = calc_rsi(df['Close'])
    macd_line, macd_signal, macd_hist = calc_macd(df['Close'])
    atr = calc_atr(df, period=14).iloc[-1]
    volume = df['Volume']
    momentum = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100 if len(df) > 6 else 0.0

    # Technical score (refined)
    tech = 0
    tech += 25 if price > ma20.iloc[-1] else 0
    tech += 20 if ma20.iloc[-1] > ma50.iloc[-1] else 0
    tech += 15 if ma50.iloc[-1] > ma200.iloc[-1] else 0
    tech += 15 if macd_hist.iloc[-1] > 0 else 0
    tech += 15 if (rsi.iloc[-1] > 45 and rsi.iloc[-1] < 70) else 0
    tech = float(np.clip(tech, 0, 100))

    mom_score = float(np.clip(50 + momentum * 1.5, 0, 100))

    # Fundamentals
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

    avg_score = tech * 0.45 + mom_score * 0.25 + fundamental_score * 0.30

    # Direction
    if price > ma50.iloc[-1] and rsi.iloc[-1] < 70 and momentum > 0 and macd_hist.iloc[-1] > 0:
        direction = 'UP'
    elif price < ma50.iloc[-1] and rsi.iloc[-1] > 30 and momentum < 0 and macd_hist.iloc[-1] < 0:
        direction = 'DOWN'
    else:
        direction = 'HOLD'

    confidence = float(np.clip(40 + (avg_score - 50) * 0.9, 10, 95))

    # Targets
    targets, stop_loss = compute_targets_and_stops(price, atr, direction)

    # Patterns
    gd_cross = detect_golden_death(ma50.values if len(ma50)>0 else np.array([]), ma200.values if len(ma200)>0 else np.array([]))
    double = detect_double_top_bottom(df['Close'], window=120)
    hhll = detect_higher_highs_lows(df['Close'], lookback=40)
    vol_spike = detect_volume_spike(df['Volume'])

    pattern_summary = f"{gd_cross} | {double} | {hhll} | {vol_spike}"

    # Final recommendation text
    if avg_score >= 75:
        recommendation = f"STRONG {direction}"
    elif avg_score >= 60:
        recommendation = f"{direction}"
    elif avg_score >= 45:
        recommendation = "HOLD"
    else:
        recommendation = f"WEAK {direction}"

    final_statement = (
        f"{sym}: Last ₹{price:.2f}. Direction {direction} (Confidence {confidence:.1f}%). "
        f"Tech {tech:.1f}/100 | Momentum {mom_score:.1f}/100 | Fundamental {fundamental_score:.1f}/100. "
        f"Targets (1-4): ₹{targets[0]:.2f}, ₹{targets[1]:.2f}, ₹{targets[2]:.2f}, ₹{targets[3]:.2f}. "
        f"Stop Loss: ₹{stop_loss:.2f}. Recommendation: {recommendation}. Patterns: {pattern_summary}"
    )

    fundamentals_table = {
        'symbol': sym,
        'sector': sector,
        'marketCap': market_cap,
        'trailingPE': pe,
        'priceToBook': pb,
        'dividendYield': div_yield,
        'longName': info.get('longName') or info.get('shortName') or sym
    }

    out = {
        'sym': sym,
        'df': df,
        'price': price,
        'ma20': ma20,
        'ma50': ma50,
        'ma200': ma200,
        'rsi': rsi,
        'macd_hist': macd_hist,
        'macd_line': macd_line,
        'macd_signal': macd_signal,
        'atr': atr,
        'volume': volume,
        'technical_score': tech,
        'momentum_score': mom_score,
        'fundamental_score': fundamental_score,
        'avg_score': avg_score,
        'direction': direction,
        'confidence': confidence,
        'targets': targets,
        'stop_loss': stop_loss,
        'recommendation': recommendation,
        'pattern_summary': pattern_summary,
        'final_statement': final_statement,
        'fundamentals_table': fundamentals_table
    }
    return out

def save_analysis_to_db(result):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    INSERT INTO analysis (symbol,date,price,rsi,macd,ma20,ma50,ma200,target1,target2,target3,target4,
                         stop_loss,direction,confidence,fundamental_score,technical_score,momentum_score,
                         recommendation,pattern_summary,final_statement)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        result['sym'],
        datetime.now().strftime('%Y-%m-%d'),
        float(result['price']),
        float(result['rsi'].iloc[-1]) if 'rsi' in result else None,
        float(result['macd_hist'].iloc[-1]) if 'macd_hist' in result else None,
        float(result['ma20'].iloc[-1]) if 'ma20' in result else None,
        float(result['ma50'].iloc[-1]) if 'ma50' in result else None,
        float(result['ma200'].iloc[-1]) if 'ma200' in result else None,
        float(result['targets'][0]),
        float(result['targets'][1]),
        float(result['targets'][2]),
        float(result['targets'][3]),
        float(result['stop_loss']),
        result['direction'],
        float(result['confidence']),
        float(result['fundamental_score']),
        float(result['technical_score']),
        float(result['momentum_score']),
        result['recommendation'],
        result['pattern_summary'],
        result['final_statement']
    ))
    conn.commit()
    conn.close()

# ----------------- UI -----------------
st.set_page_config(page_title='JINNI - Full Analysis', layout='wide')
st.title('⚡ JINNI - All-In-One Stock Deep Analysis')
st.write('Candles, indicators, fundamentals, pattern detection, multi-targets, and final summary.')

col1, col2 = st.columns([3,1])
with col1:
    symbol = st.text_input('Enter symbol (append .NS if needed)', 'TCS.NS')
with col2:
    run_analysis = st.button('📊 Analyze & Plot')

if run_analysis:
    with st.spinner('Fetching data and analyzing...'):
        res = analyze_full(symbol)
    if res is None:
        st.error('Insufficient data or symbol not found. Need >= 30 daily bars.')
    else:
        df = res['df']
        # ---------------- Plotly chart ----------------
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.2, 0.2],
                            specs=[[{"type":"xy"}],
                                   [{"type":"xy"}],
                                   [{"type":"xy"}]])
        # Candles
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        # MAs
        fig.add_trace(go.Scatter(x=df.index, y=res['ma20'], name='MA20', line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=res['ma50'], name='MA50', line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=res['ma200'], name='MA200', line=dict(width=1)), row=1, col=1)
        # Bollinger Bands
        bb_mid = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        fig.add_trace(go.Scatter(x=df.index, y=bb_upper, name='BB Upper', line=dict(width=1), opacity=0.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb_lower, name='BB Lower', line=dict(width=1), opacity=0.5), row=1, col=1)

        # RSI subplot
        fig.add_trace(go.Scatter(x=df.index, y=res['rsi'], name='RSI', line=dict(width=1)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", row=2, col=1)

        # MACD subplot
        fig.add_trace(go.Bar(x=df.index, y=res['macd_hist'], name='MACD Hist'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=res['macd_line'], name='MACD Line', line=dict(width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=res['macd_signal'], name='MACD Signal', line=dict(width=1)), row=3, col=1)

        fig.update_layout(height=900, showlegend=True, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Volume + ATR (small)
        vol_col, atr_col = st.columns(2)
        with vol_col:
            st.bar_chart(df['Volume'].tail(120))
        with atr_col:
            st.metric('ATR (14)', f'{res["atr"]:.2f}')

        # ----------- Fundamentals -----------
        st.subheader('Fundamentals & Company details')
        ft = res['fundamentals_table']
        fund_df = pd.DataFrame({
            'Field': ['Name','Sector/Industry','Market Cap','Trailing PE','Price to Book','Dividend Yield'],
            'Value': [ft.get('longName'), ft.get('sector'), f"{ft.get('marketCap'):,}" if pd.notna(ft.get('marketCap')) else 'N/A',
                      f"{ft.get('trailingPE'):.2f}" if pd.notna(ft.get('trailingPE')) else 'N/A',
                      f"{ft.get('priceToBook'):.2f}" if pd.notna(ft.get('priceToBook')) else 'N/A',
                      f"{ft.get('dividendYield'):.3f}" if ft.get('dividendYield') else '0.000']
        })
        st.table(fund_df)

        # Basic fundamental summary
        f_summary = []
        if pd.notna(ft.get('trailingPE')):
            if ft.get('trailingPE') < 15:
                f_summary.append('Cheap PE vs typical market (PE < 15)')
            elif ft.get('trailingPE') < 30:
                f_summary.append('Reasonable PE')
            else:
                f_summary.append('High PE - may be growth priced')
        else:
            f_summary.append('PE not available')

        if pd.notna(ft.get('priceToBook')):
            if ft.get('priceToBook') < 3:
                f_summary.append('PB reasonable (<3)')
            else:
                f_summary.append('High PB')
        else:
            f_summary.append('PB not available')

        if ft.get('dividendYield') and ft.get('dividendYield') > 0:
            f_summary.append('Pays dividend')
        else:
            f_summary.append('No dividend or dividend data not present')

        st.write('**Fundamental interpretation:** ', ' | '.join(f_summary))

        # ----------- Targets & Stops -----------
        st.subheader('Targets & Stop-loss (ATR based)')
        t = res['targets']
        st.write(f"Target 1 (conservative): ₹{t[0]:.2f}")
        st.write(f"Target 2 (moderate): ₹{t[1]:.2f}")
        st.write(f"Target 3 (ambitious): ₹{t[2]:.2f}")
        st.write(f"Target 4 (very ambitious): ₹{t[3]:.2f}")
        st.write(f"Suggested Stop Loss: ₹{res['stop_loss']:.2f}")

        # Show percentage moves
        percents = [((x - res['price'])/res['price']*100) for x in t]
        st.write('Expected % moves from current price:', ', '.join([f"{p:.2f}%" for p in percents]))

        # ----------- Pattern detection -----------
        st.subheader('Pattern detection & Technical Signals')
        st.write(res['pattern_summary'])
        st.write('Simple trend status (higher highs/lows):', detect_higher_highs_lows(df['Close'], lookback=40))

        # ----------- Scoring & Final -----------
        st.subheader('Scores & Final Recommendation')
        st.metric('Technical Score', f'{res["technical_score"]:.1f}/100')
        st.metric('Momentum Score', f'{res["momentum_score"]:.1f}/100')
        st.metric('Fundamental Score', f'{res["fundamental_score"]:.1f}/100')
        st.write('**Final analysis:**')
        st.info(res['final_statement'])

        # Save result
        try:
            save_analysis_to_db(res)
            st.success('Analysis saved to local DB.')
        except Exception as e:
            st.warning(f'Could not save to DB: {e}')

# ----------------- Market Recommender / Scanner -----------------
st.header('🔍 Market Recommender - Scan by expected % move')

with st.expander('Scanner settings (click to expand)', expanded=False):
    min_expected_pct = st.number_input('Minimum expected % move (from current price to Target1)', min_value=0.1, max_value=100.0, value=3.0, step=0.1, help="Show stocks whose Target1 is at least this % away from current price")
    scan_direction = st.selectbox('Direction to look for', options=['UP','DOWN','EITHER'], index=2)
    symbols_text = st.text_area('Symbols (comma-separated). Leave blank to use default sample NSE list.', 
                                value='RELIANCE.NS,TCS.NS,INFY.NS,HINDUNILVR.NS,SBIN.NS,ICICIBANK.NS,HDFC.NS,MARUTI.NS,BAJAJFINSV.NS,LT.NS,ASIANPAINT.NS,WIPRO.NS,AXISBANK.NS,DMART.NS')
    throttle_ms = st.number_input('Delay between requests (ms) to avoid rate limits', value=150, min_value=50, max_value=2000, step=50)

scan_button = st.button('🔎 Run Market Scan')

def insert_scan_log_row(total, bullish, bearish, neutral, min_pct, direction):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO scan_log (scan_date,total_scanned,bullish,bearish,neutral,min_expected_pct,direction) VALUES (?,?,?,?,?,?,?)',
              (datetime.now().strftime('%Y-%m-%d'), total, bullish, bearish, neutral, float(min_pct), direction))
    conn.commit()
    conn.close()

if scan_button:
    raw = symbols_text.strip()
    if raw == '':
        st.warning('No symbols provided. Please paste a list or use default sample.')
    else:
        symbol_list = [s.strip().upper() for s in raw.split(',') if s.strip()]
        if not symbol_list:
            st.warning('Symbol list empty after parsing.')
        else:
            st.info(f'Scanning {len(symbol_list)} symbols — min expected {min_expected_pct:.2f}% — direction {scan_direction}')
            progress = st.progress(0)
            status = st.empty()
            results = []
            for i, sym in enumerate(symbol_list):
                status.info(f'Scanning {i+1}/{len(symbol_list)}: {sym}')
                try:
                    res = analyze_full(sym)
                    if res is None:
                        time.sleep(throttle_ms/1000.0)
                        progress.progress((i+1)/len(symbol_list))
                        continue

                    price = res['price']
                    target1 = res['targets'][0]
                    if res['direction'] == 'UP':
                        expected_pct = (target1 - price) / price * 100.0
                    elif res['direction'] == 'DOWN':
                        expected_pct = (price - target1) / price * 100.0
                    else:
                        expected_pct = abs((target1 - price) / price * 100.0)

                    qualifies = False
                    if scan_direction == 'EITHER':
                        qualifies = expected_pct >= min_expected_pct
                    elif scan_direction == 'UP':
                        qualifies = (res['direction'] == 'UP') and (expected_pct >= min_expected_pct)
                    elif scan_direction == 'DOWN':
                        qualifies = (res['direction'] == 'DOWN') and (expected_pct >= min_expected_pct)

                    if qualifies:
                        row = {
                            'symbol': res['sym'],
                            'price': res['price'],
                            'direction': res['direction'],
                            'expected_pct': expected_pct,
                            'confidence': res['confidence'],
                            'technical_score': res['technical_score'],
                            'fundamental_score': res['fundamental_score'],
                            'momentum_score': res['momentum_score'],
                            'target1': res['targets'][0],
                            'target2': res['targets'][1],
                            'target3': res['targets'][2],
                            'target4': res['targets'][3],
                            'stop_loss': res['stop_loss'],
                            'final_statement': res['final_statement']
                        }
                        results.append(row)

                    try:
                        save_analysis_to_db(res)
                    except Exception as e:
                        st.warning(f'Could not save analysis for {sym}: {e}')

                except Exception as e:
                    st.error(f'Error scanning {sym}: {e}')
                time.sleep(throttle_ms/1000.0)
                progress.progress((i+1)/len(symbol_list))

            if results:
                df_results = pd.DataFrame(results).sort_values(['expected_pct','confidence'], ascending=[False, False]).reset_index(drop=True)
                st.success(f'Found {len(df_results)} symbols matching criteria (min {min_expected_pct}%).')
                st.dataframe(df_results[['symbol','price','direction','expected_pct','confidence','technical_score','fundamental_score','target1','stop_loss']].round(3), use_container_width=True)

                bullish = sum(1 for r in results if r['direction']=='UP')
                bearish = sum(1 for r in results if r['direction']=='DOWN')
                neutral = len(results) - bullish - bearish
                colA, colB, colC = st.columns(3)
                colA.metric('📈 Bullish', bullish)
                colB.metric('📉 Bearish', bearish)
                colC.metric('⏸️ Neutral', neutral)

                insert_scan_log_row(len(symbol_list), bullish, bearish, neutral, min_expected_pct, scan_direction)
            else:
                st.info('No symbols matched the criteria.')
                insert_scan_log_row(len(symbol_list), 0, 0, 0, min_expected_pct, scan_direction)

# ----------------- History / Portfolio -----------------
st.sidebar.header('History & Saved Analyses')
if st.sidebar.button('Show recent saved analyses'):
    conn = sqlite3.connect(DB_PATH)
    df_hist = pd.read_sql_query('SELECT symbol,date,price,direction,recommendation,confidence,timestamp FROM analysis ORDER BY timestamp DESC LIMIT 200', conn)
    conn.close()
    if df_hist.empty:
        st.sidebar.info('No saved analyses yet.')
    else:
        st.sidebar.dataframe(df_hist, use_container_width=True)

st.sidebar.markdown('---')
st.sidebar.write('Notes:')
st.sidebar.write('- Targets use ATR multipliers (conservative → aggressive).')
st.sidebar.write('- Pattern detectors are heuristic: for production you must backtest pattern signals.')
st.sidebar.write('- Keep your symbol list updated. If you want full NSE/BSE, replace the symbols_text default with your full list.')
