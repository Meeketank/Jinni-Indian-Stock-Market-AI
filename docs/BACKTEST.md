# Real-data backtest of the "chance of reaching my target" system

Run on 1 Nov 2026 with real Yahoo Finance daily history (5 years, split-adjusted) for 19 NSE stocks:
HDFCBANK, ICICIBANK, INFY, RELIANCE, SUNPHARMA, ITC, LT, MARUTI, BAJFINANCE, TRENT, TATASTEEL, HINDALCO, JSWSTEEL,
ADANIENT, TATAPOWER, IRCTC, HAL, DIXON, POLYCAB, with NIFTY 50 as the benchmark.
Method: every 7th trading day from Jan 2023 to Oct 2026 the model saw **only the past**, stated a probability, and the
following days decided the outcome (did the day's high touch the level?). Reproduce with `scripts/backtest_real.py`.

| Goal | Predictions | Actually reached | Model's average | AUC (ranking skill) | Brier vs always-base-rate |
|---|---|---|---|---|---|
| +2% in 5 days | 2,527 | 52% | 55% | 0.61 | 0.240 vs 0.249 |
| **+5% in 21 days** | 2,489 | **51%** | 53% | **0.60** | 0.244 vs 0.250 |
| +10% in 21 days | 2,489 | 22% | 23% | 0.67 | 0.161 vs 0.170 |

**Calibration (+5% in a month)**: when the model said 35% it happened 43%; 46% → 42%; 53% → 49%; 60% → 56%; 70% → 66%.
Slightly compressed at the extremes, close to honest in the middle.

## What this means for "find me stocks that give 5% in a month"
- A typical stock **touches +5% within a month about half the time**. Finding candidates is easy.
- Making money from it is not: with a target of +5%, an automatic volatility stop and 0.3% costs, the **average result per
  trade was −0.08%** across all samples and **+0.15%** for the best-ranked fifth (66% of those reached +5%). That is why the
  Finder's "Avg return / trade" column is small (often well under 1%) even when the chance of touching the target is high.
- For +10% in a month: average +0.33% per trade overall (t ≈ 2.6) and +1.32% for the best-ranked fifth.

## Which signals rank the winners (AUC for "will it reach +5% in 21 days")
| Signal | AUC | Top-20% hit rate | Bottom-20% hit rate |
|---|---|---|---|
| Time-Machine probability (the engine) | 0.60 | 66% | 43% |
| Volatility alone | 0.59 | 66% | 42% |
| Stock's own past hit rate | 0.59 | 61% | 39% |
| Technical score (87 indicators → 5 factors) | 0.50 | 55% | 53% |
| 3-month momentum | 0.51 | 58% | 55% |
| Relative strength vs NIFTY | 0.52 | 58% | 53% |

A gradient-boosted model trained on volatility + technical + momentum features did **not** beat the simulation on a
held-out final 45% of the data (AUC 0.55–0.66 vs 0.64–0.73 for the simulation). Conclusion: the probability engine
(mostly volatility, adjusted for the stock's own return distribution) is the only component with proven ranking power.
Technical scores and momentum are therefore shown as **watch-outs**, not used to rank.

## What could not be tested
- **Sentiment and fundamentals**: free sources have no point-in-time history, so no honest back-test exists. They are shown
  as watch-outs and (for the outlook) a capped ±3-point tilt.
- Only 19 mostly large, liquid stocks; overlapping windows reduce the effective sample; survivorship bias; "reached" uses the
  intraday high. Small/illiquid stocks behave differently (higher volatility, harder execution).

## Price-forecast ranges (Forecast tab) — run 11 Nov 2026, same 19 stocks
Each range was built with only the past and compared with the price h trading days later.

| Horizon | Forecasts | Inside the 9-in-10 range (target 90%) | Inside the half range (target 50%) | Ended above the middle |
|---|---|---|---|---|
| 1 week | 1,767 | **89.2%** | 52.4% | 49.9% |
| 1 month | 1,748 | **89.7%** | 51.5% | 49.5% |
| 3 months | 1,672 | **87.4%** | 52.1% | 49.9% |

The ranges are honest: outcomes fell outside about as often as stated, on both sides. They are also wide (about ±14% over a
month for a typical stock), which is the real uncertainty, not a flaw.

## Trend state as a predictor
Pooled over the 19 stocks, the next-month return by trend state did not hold up across time (2021–mid 2024: strong uptrends
+3.3%, strong downtrends +2.9%; mid 2024–2026: −0.6% and −0.3%). Trend describes the present; it did not predict.
Among the highest-chance picks, however, stocks with a technical score of 40 or more averaged +0.7% to +1.8% per trade on
monthly goals against −0.1% to +1.0% for weak-trend ones, which is why the Finder offers an "Uptrends first" ranking.

## Scenario tools
- **What-if shocks**: a one-factor (NIFTY) weekly model, fitted on the first 70% of three years and checked on the rest,
  explained on average **28%** of each stock's weekly moves out of sample (correlation 0.57; from 60% for HDFC Bank to
  none for Sun Pharma and Tata Power). The app shows this figure per stock next to every scenario.
- **Event branches** (NIFTY, 5 years): replaying RBI decisions (21 replays) and Fed decisions (34), the 1-week result landed
  between the "goes badly" and "goes well" levels 76% of the time and inside the full earlier range 86–94%. RBI and Fed days
  were only about 7–8% bigger than ordinary days; Budget days in this window were smaller. Elections have 3–5 occurrences,
  too few to back-test, and the app says so.
