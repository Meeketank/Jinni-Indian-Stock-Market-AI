"""Walk-forward accuracy test on real price history stored as CSV files (columns: date, open, high, low, close, volume).

    python scripts/backtest_real.py DIR [--target 0.05 --horizon 21 --step 7]

DIR holds one CSV per stock named SYMBOL.csv plus IDX_NSEI.csv (the benchmark). Prints calibration, signal AUCs and the
trade-level result. The same test runs live inside the app (Lab -> System accuracy).
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jinni import accuracy as ac  # noqa: E402


def load(d: str):
    dfs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        x = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        x.columns = [c.title() for c in x.columns]
        dfs[os.path.basename(f)[:-4]] = x
    bench = dfs.pop("IDX_NSEI")["Close"]
    return dfs, bench


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--step", type=int, default=7)
    a = ap.parse_args()
    dfs, bench = load(a.dir)
    s = ac.collect(dfs, bench, a.target, a.horizon, a.step)
    print(ac.summary(s))
    print(ac.calibration(s["p_sim"], s["hit"]).round(3).to_string())
    print(ac.signal_table(s).round(3).to_string())
