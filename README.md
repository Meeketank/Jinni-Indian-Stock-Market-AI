# JINNI · Indian Market Risk & Analytics Terminal

An Aladdin-style risk and analytics platform for NSE/BSE equities. It runs on **Vercel** as a Python API plus a
static, mobile-first web app (installable on a phone's home screen), and opens with an animated launch screen and a
market-pulse home page.

## Deploy on Vercel

1. Push this repo to GitHub and **Import Project** on [vercel.com/new](https://vercel.com/new).
2. Leave the framework preset on **Other** (or FastAPI); there is no build command and no output directory to set.
   Vercel detects `index.py` (FastAPI), installs `requirements.txt` on Python 3.12 (`.python-version`) and serves
   `public/` from its CDN. Settings come from `vercel.json`.
3. Deploy. No environment variables are required. Open the URL, then **Settings → Check live data** to confirm Yahoo
   Finance is reachable from Vercel.

What to know:

| | |
|---|---|
| **Bundle size** | ~215 MiB of Vercel's **225 MiB** uncompressed Python limit (measured with the real `vercel build`): only NumPy, pandas, SciPy, yfinance and FastAPI. Gradient boosting, isotonic calibration and Ledoit-Wolf live in `jinni/mlcore.py` (no scikit-learn); `scripts/vercel_prune.py` strips unused SciPy modules, and CI fails if the bundle grows past 220 MiB. |
| **Time limit** | `maxDuration` is 60 s (works on Hobby). Heavy work is split so no request nears it: the whole-market scan is driven by the browser in chunks of 100 stocks, and the deep analysis covers at most 70 stocks per request. |
| **No server state** | Holdings, watchlist, journal and settings live in the browser (`localStorage`). Export a backup in **Settings**. |
| **Sentiment & outlook** | Live news from Google News (9 themes), five publisher feeds, Yahoo and GDELT tone; an ML next-session/5-session direction estimate for NIFTY and any stock, validated walk-forward and shown with its out-of-sample AUC. News only tilts it by at most ±3 points. |
| **Light by default** | Opens in light mode like the original app; switch to dark with the sun/moon button or in **Settings → Appearance** (Light / Dark / Match device). |
| **Caching** | Read-only endpoints send `s-maxage`, so Vercel's CDN serves repeat requests instantly; the function also caches downloads in memory while warm. |
| **Data source** | Yahoo Finance via `yfinance`. Shared cloud IPs are sometimes rate-limited by Yahoo; JINNI then returns a clear "live data unavailable" message (never made-up numbers) and offers an explicit demo mode. |
| **Stock list** | Always live: NSE's EQUITY_L file, then Yahoo's screener (paged) if NSE is unreachable. NIFTY 50 membership comes from NSE's index file. Nothing is bundled or hard-coded; if no source answers, the app says so. |
| **Security** | Strict CSP (no inline scripts, no third-party scripts: Plotly is vendored in `public/vendor`), `X-Frame-Options: DENY`, no secrets, no cookies. |
| **Env vars** | `JINNI_DEMO=1` forces synthetic data (handy for preview deployments). |

Run it locally (API and front end on one port):

```bash
pip install -r requirements-dev.txt
python dev.py                # http://localhost:8000   (JINNI_DEMO=1 python dev.py for offline demo data)
pytest -q                    # 85 tests
```

or with the Vercel CLI: `vercel dev`.

### The original Streamlit UI (optional)

`streamlit_app.py` and `app/` are the first version of the interface. They cannot run on Vercel (Streamlit needs a
long-lived server), so they are not part of the deployment. To use them locally or in Codespaces:

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

## Workspaces

| Page | What it does |
|---|---|
| **Return Finder** | Pick a goal (+2/5/10/15%/custom within 1 week to 3 months, auto/fixed/no stop). Every stock gets P(target first), P(stop first), P(touch), expected return after costs, an A–D grade and a plain verdict, from three engines: a **Time Machine** (4,000 volatility-rescaled block-bootstrap paths of the stock's own daily bars, stop checked before target, gaps fill at the open), **history replay** (every past day and "days like today", shrunk for small samples) and a **calibrated ML model** that only gets blend weight in proportion to its walk-forward AUC. Deep dive: price cone, engine breakdown, setup history, odds grid, sizing that names the deciding limit, PASS/WARN/BLOCK risk gate, portfolio VaR impact, news and results-date warnings |
| **Analyzer: 360° view** | Past (1-year return, this month's history, drawdown), present (87-indicator summary, active patterns, news tone) and future (odds of +5% in a month, price range, events ahead) combined into one 360° score |
| **Analyzer: Chart** | 35 overlays and 52 indicator panels (every MA type, SuperTrend, PSAR, Ichimoku, Keltner, Donchian, StochRSI, TSI, KST, Fisher, Connors RSI, MFI, CMF, Klinger, volatility estimators…), Heikin-Ashi, zoom, and auto-drawn patterns |
| **Analyzer: Patterns** | 45 candlestick patterns; double/triple tops & bottoms, head & shoulders, triangles, wedges, channels, rectangles, broadening, flags, pennants, cup & handle, rounding tops/bottoms; Gartley, Bat, Butterfly, Crab, Shark, AB=CD; Elliott impulse (rule-checked); RSI/MACD divergences; classified gaps; trendlines; Fibonacci; S/R zones. Each with target and invalidation, plus how it performed on this stock historically (look-ahead-free replay) |
| **Sentiment & Events** | Market mood index (momentum, VIX, breadth, new highs/lows, equity vs gold), news tone per stock, an **Event lab** ("2nd Friday of every month", Budget, RBI, Fed, elections, or any pasted dates → did the market react, how often, could it be luck), calendar effects with multiple-testing correction, and upcoming events |
| **Whole-market scan + budget** | Return Finder can scan every NSE-listed stock (live list from NSE, bundled fallback, or uploaded CSV): stage 1 filters ~2,000 stocks on budget, price, liquidity and circuit locks; stage 2 runs the three engines on the best N. Enter a budget and get a whole-share plan with the basket simulated jointly: odds of reaching the target, chance of loss, and bad/good cases in rupees |
| **Journal** | Tracks predictions and paper trade plans, scores them against the real price path afterwards (target / stop / expired), and shows hit rate vs predicted, calibration and results by grade |
| **Home** | Index ticker tape, market regime (NIFTY trend), India VIX, breadth, advance/decline, sector-rotation heatmap, top movers |
| **Analyzer** | Candlestick chart with overlays (SMA/EMA/Bollinger/VWAP) and RSI/MACD/ADX panels, a 5-factor rating (Trend, Momentum, Timing, Volume, Risk + fundamentals), ATR/structure-based trade plan with position sizing, pivots and swing levels, and the **measured** forward returns after each past rating |
| **Screener** | Scores the NIFTY 50, your watchlist or a custom list. Filters, CSV export, momentum-vs-risk map, sector scores |
| **Risk** | Holdings editor (CSV import/export), live P&L, VaR/CVaR four ways (historical, Gaussian, Cornish-Fisher, Monte Carlo Student-t), Kupiec VaR backtest, Euler risk contribution, correlation, beta/idiosyncratic split, historical stress replay (COVID 2020, 2022 rate shock, Adani-Hindenburg, 2024 election day, …), beta shocks, drawdowns, rolling beta, liquidity / days-to-exit |
| **Optimizer** | Max-Sharpe, min-volatility, risk parity, max diversification. Ledoit-Wolf covariance, CAPM-implied or shrunk expected returns, per-stock and sector caps, efficient frontier, whole-share allocation and a rebalance trade list with Indian cost estimates |
| **Lab** | Backtests five rules net of costs (no look-ahead) and a gradient-boosting model validated **walk-forward with a purge gap**, reported against the base rate with AUC, Brier, quintile returns and feature importance |

Ideas adapted from [day-trader-skill](https://github.com/Jakeschincariol/day-trader-skill) for NSE daily bars: the Time Machine replay and its break-even verdict (a target S/X away must come first in S/(X+S) of decided cases), sizing to a max loss that names the deciding limit, a PASS/BLOCK risk gate, setup scans, news tagging, and "never round a coin flip up into an edge".

## Accuracy fixes vs v1

* **ML leakage removed.** v1 scored the model with a random train/test split on overlapping windows, which inflates accuracy. v2 uses purged walk-forward validation and scale-free features.
* **Wilder smoothing** for RSI/ATR/ADX, matching TradingView and Kite.
* **VaR**: real portfolio weights instead of equal weights; multi-day VaR from actual h-day returns, not √t scaling; fat-tail models; a backtest of the VaR itself.
* **Indian risk-free rate** (6.5% default, configurable) instead of 2%.
* **Tickers**: HDFC→HDFCBANK, ZOMATO→ETERNAL and TATAMOTORS→TMPV aliases; current NIFTY 50 list.
* The invented "confidence %" is replaced by transparent factor scores plus measured hit-rates.

## Layout

```
index.py              FastAPI entrypoint for Vercel (all /api routes)
server/               API layer: routes_*.py return JSON "blocks"; core.py (caching, errors); figs.py (Plotly dicts)
public/               static front end served by Vercel's CDN: index.html, css/, js/ (router, UI kit, pages), vendor/plotly
jinni/                the engine, free of web frameworks: data, indicators, ta, patterns, signals, forecast, risk, optimizer, ml…
vercel.json           function limits, bundle excludes, security headers
tests/                pytest suite (engine + API + deployment wiring)
dev.py                local server (API + public/ on one port)
streamlit_app.py, app/   optional original Streamlit UI (see above)
legacy/               v1 app and the original heavy requirements list
```

Older experimental modules (`main.py`, `ml_models/`, `online_learning/`, …) are left untouched. They are not used by the app,
and some import modules that do not exist in the repo.

## Disclaimer

For analytics and education only. Not investment advice; not a SEBI-registered adviser. Data from Yahoo Finance may be delayed.

---

<details><summary>Original README (v1)</summary>

# 🧞 Jinni - Indian Stock Market AI

**AI-Powered Real-Time Indian Stock Market Analysis & Prediction System**

Custom ML/DL with Online Learning for NSE/BSE stocks. Built from scratch with continuous accuracy improvement every second.

---

## 🌟 Key Features

### 🤖 Advanced ML/DL Models
- **LSTM (Long Short-Term Memory)**: Captures long-term dependencies in stock price patterns
- **GRU (Gated Recurrent Unit)**: Faster training with similar performance to LSTM
- **Transformer**: Attention-based architecture for complex pattern recognition
- **Ensemble Model**: Combines all three models with dynamic weight adjustment

### 📊 Continuous Online Learning
- **Real-time model updates** every few seconds
- **Accuracy improvement rate**: Tracks improvement per second
- **Incremental training**: Models learn from new data continuously
- **Adaptive ensemble weights**: Automatically adjusts based on performance

### 📈 Technical Analysis
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- ATR (Average True Range)
- ADX (Average Directional Index)
- OBV (On-Balance Volume)
- EMA & SMA (Moving Averages)

### 🎯 Accuracy Tracking
- **Live accuracy monitoring** with per-stock breakdowns
- **Performance metrics**: Error rates, prediction counts, improvement rates
- **Top/worst performing stocks** identification
- **JSON export** of detailed accuracy reports

### 📡 NSE/BSE Data Integration
- Custom data fetchers for Indian stock markets
- Real-time price updates
- Historical data retrieval
- Support for 15+ major stocks (RELIANCE, TCS, INFY, HDFCBANK, etc.)

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x
NumPy, Pandas
Streamlit (for dashboard)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Meeketank/Jinni-Indian-Stock-Market-AI.git
cd Jinni-Indian-Stock-Market-AI
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the system**
```bash
python run.py
```

This will:
- Load all ML models (LSTM, GRU, Transformer)
- Train the ensemble on historical data
- Start live prediction mode
- Display real-time accuracy metrics
- Show final comprehensive results

---

## 📂 Project Structure

```
Jinni-Indian-Stock-Market-AI/
├── ml_models/
│   ├── lstm_model.py          # LSTM neural network
│   ├── gru_model.py            # GRU neural network
│   ├── transformer_model.py    # Transformer with attention
│   └── ensemble_model.py       # Ensemble combining all models
├── data_scraper/
│   └── nse_bse_fetcher.py      # NSE/BSE data fetching
├── analytics/
│   └── technical_indicators.py # All technical indicators
├── online_learning/
│   ├── incremental_trainer.py  # Continuous learning system
│   └── accuracy_tracker.py     # Live accuracy monitoring
├── dashboard/
│   └── app.py                  # Streamlit real-time dashboard
├── config.py                   # Configuration settings
├── main.py                     # Application orchestrator
├── run.py                      # Main entry point (DEMO)
└── requirements.txt            # Dependencies
```

---

## 💻 Usage Examples

### Running the Complete Demo
```bash
python run.py
```

Output:
```
================================================================================
   ___  ___  _   _  _   _  ___
  |_  ||_  || \ | || \ | ||_ _|
   |_||___||  \| ||  \| | |_|

  INDIAN STOCK MARKET AI - Real-Time Analysis & Prediction System
  With Continuous Online Learning & Accuracy Improvement
================================================================================

[1/8] Loading ML Models...
✓ ML Models loaded successfully
...

TRAINING PHASE: Training Ensemble Models (LSTM + GRU + Transformer)
...

LIVE PREDICTION MODE: Continuous Learning & Accuracy Improvement

JINNI LIVE ACCURACY METRICS
Overall Accuracy: 94.35%
Total Predictions: 247
Improvement Rate: 0.0823%/second
...
```

### Running the Dashboard
```bash
cd dashboard
streamlit run app.py
```

---

## 🎯 System Performance

### Accuracy Metrics (Sample)
- **Overall Accuracy**: 92-96% (improves continuously)
- **Average Error**: <2%
- **Predictions Per Second**: ~2
- **Improvement Rate**: 0.05-0.15% per second

### Model Weights (Dynamic)
```
LSTM: 33-35%
GRU: 32-34%
Transformer: 31-35%
(Automatically adjusted based on performance)
```

---

## 🔬 Technical Details

### Online Learning Algorithm
1. Make prediction using ensemble
2. Compare with actual price
3. Calculate error and update accuracy
4. Every N predictions, retrain models incrementally
5. Update ensemble weights based on individual model performance
6. Repeat (continuous improvement)

### Ensemble Strategy
- Weighted average of model predictions
- Weights adjusted dynamically based on recent performance
- Inversely proportional to prediction errors

---

## 📊 Supported Stocks

**NSE/BSE Top Stocks:**
- RELIANCE
- TCS
- INFY
- HDFCBANK
- ICICIBANK
- SBIN
- BHARTIARTL
- ITC
- KOTAKBANK
- LT
- HINDUNILVR
- AXISBANK
- ASIANPAINT
- MARUTI
- WIPRO

*(Easily extensible to all NSE/BSE stocks)*

---

## 🛠️ Development

### Adding New Models
1. Create new model file in `ml_models/`
2. Implement `train()` and `predict()` methods
3. Add to ensemble in `ensemble_model.py`

### Adding New Technical Indicators
1. Add method to `TechnicalIndicators` class
2. Follow existing pattern for calculations

---

## 📝 Configuration

Edit `config.py` to customize:
- Stock symbols to track
- Model hyperparameters
- Training epochs and batch sizes
- Prediction intervals
- Accuracy thresholds

---

## 🚦 System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- 2GB free disk space

**Recommended:**
- Python 3.10+
- 16GB RAM
- GPU (for faster training)
- SSD storage

---

## 📄 License

MIT License - See LICENSE file

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

For questions or issues:
- Create an issue on GitHub
- Repository: https://github.com/Meeketank/Jinni-Indian-Stock-Market-AI

---

## 🎯 Roadmap

- [ ] Real NSE/BSE API integration
- [ ] More ML models (XGBoost, Prophet)
- [ ] Portfolio optimization
- [ ] Risk analysis module
- [ ] Mobile app
- [ ] Backtesting framework
- [ ] Deployment to cloud

---

## ⚠️ Disclaimer

**This system is for educational and research purposes only. Not financial advice. Always do your own research before making investment decisions.**

---

**Built with ❤️ for the Indian Stock Market**

</details>
