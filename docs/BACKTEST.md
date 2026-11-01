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
