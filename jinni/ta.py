"""Full technical-indicator library with a TradingView-style signal summary.

Every indicator is registered in ``CATALOG`` with its category, whether it
draws on the price chart (overlay) or in its own panel, and a rule that turns
the latest value into BUY (+1), NEUTRAL (0) or SELL (-1). ``summary`` runs all
of them; ``compute`` returns the series for charting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import indicators as base

# ------------------------------------------------------------ moving averages


def wma(s, n):
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def dema(s, n):
    e = base.ema(s, n)
    return 2 * e - base.ema(e, n)


def tema(s, n):
    e1 = base.ema(s, n)
    e2 = base.ema(e1, n)
    return 3 * e1 - 3 * e2 + base.ema(e2, n)


def hma(s, n):
    return wma(2 * wma(s, n // 2) - wma(s, n), int(np.sqrt(n)))


def zlema(s, n):
    lag = (n - 1) // 2
    return base.ema(s + (s - s.shift(lag)), n)


def kama(s, n=10, fast=2, slow=30):
    change = (s - s.shift(n)).abs()
    vol = s.diff().abs().rolling(n).sum()
    er = (change / vol).fillna(0)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    out = np.full(len(s), np.nan)
    vals = s.values
    start = n
    if len(s) <= n:
        return pd.Series(out, index=s.index)
    out[start] = vals[start]
    for i in range(start + 1, len(s)):
        out[i] = out[i - 1] + sc.iloc[i] * (vals[i] - out[i - 1])
    return pd.Series(out, index=s.index)


def alma(s, n=9, offset=0.85, sigma=6):
    m = offset * (n - 1)
    sd = n / sigma
    w = np.exp(-((np.arange(n) - m) ** 2) / (2 * sd * sd))
    w /= w.sum()
    return s.rolling(n).apply(lambda x: np.dot(x, w), raw=True)


def vwma(df, n=20):
    return (df["Close"] * df["Volume"]).rolling(n).sum() / df["Volume"].rolling(n).sum()


def smma(s, n):
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def mcginley(s, n=14):
    out = np.full(len(s), np.nan)
    v = s.values
    if len(v) == 0:
        return pd.Series(out, index=s.index)
    out[0] = v[0]
    for i in range(1, len(v)):
        prev = out[i - 1]
        out[i] = prev + (v[i] - prev) / (n * (v[i] / prev) ** 4) if prev else v[i]
    return pd.Series(out, index=s.index)


# --------------------------------------------------------------- trend tools


def supertrend(df, n=10, mult=3.0):
    a = base.atr(df, n)
    hl2 = (df["High"] + df["Low"]) / 2
    up = (hl2 - mult * a).values
    dn = (hl2 + mult * a).values
    c = df["Close"].values
    st = np.full(len(c), np.nan)
    dirn = np.ones(len(c))
    fu, fd = up.copy(), dn.copy()
    for i in range(1, len(c)):
        if np.isnan(up[i]):
            continue
        fu[i] = max(up[i], fu[i - 1]) if not np.isnan(fu[i - 1]) and c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if not np.isnan(fd[i - 1]) and c[i - 1] < fd[i - 1] else dn[i]
        if c[i] > fd[i - 1] if not np.isnan(fd[i - 1]) else False:
            dirn[i] = 1
        elif c[i] < fu[i - 1] if not np.isnan(fu[i - 1]) else False:
            dirn[i] = -1
        else:
            dirn[i] = dirn[i - 1]
        st[i] = fu[i] if dirn[i] == 1 else fd[i]
    return pd.DataFrame({"SuperTrend": st, "ST_dir": dirn}, index=df.index)


def psar(df, step=0.02, max_step=0.2):
    h, lo = df["High"].values, df["Low"].values
    n = len(h)
    sar = np.full(n, np.nan)
    if n < 3:
        return pd.Series(sar, index=df.index)
    bull = True
    af = step
    ep = h[0]
    sar[0] = lo[0]
    for i in range(1, n):
        prev = sar[i - 1]
        s = prev + af * (ep - prev)
        if bull:
            s = min(s, lo[i - 1], lo[i - 2] if i > 1 else lo[i - 1])
            if lo[i] < s:
                bull, s, ep, af = False, ep, lo[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_step)
        else:
            s = max(s, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > s:
                bull, s, ep, af = True, ep, h[i], step
            elif lo[i] < ep:
                ep, af = lo[i], min(af + step, max_step)
        sar[i] = s
    return pd.Series(sar, index=df.index)


def ichimoku(df):
    h, lo = df["High"], df["Low"]
    tenkan = (h.rolling(9).max() + lo.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    chikou = df["Close"].shift(-26)
    return pd.DataFrame({"Tenkan": tenkan, "Kijun": kijun, "Span A": span_a, "Span B": span_b, "Chikou": chikou})


def aroon(df, n=25):
    up = df["High"].rolling(n + 1).apply(lambda x: x.argmax() / n * 100, raw=True)
    dn = df["Low"].rolling(n + 1).apply(lambda x: x.argmin() / n * 100, raw=True)
    return pd.DataFrame({"Aroon Up": up, "Aroon Down": dn, "Aroon Osc": up - dn})


def vortex(df, n=14):
    tr = base.true_range(df).rolling(n).sum()
    vp = (df["High"] - df["Low"].shift(1)).abs().rolling(n).sum() / tr
    vm = (df["Low"] - df["High"].shift(1)).abs().rolling(n).sum() / tr
    return pd.DataFrame({"VI+": vp, "VI-": vm})


def linreg(s, n=20):
    x = np.arange(n)
    xm = x.mean()
    den = ((x - xm) ** 2).sum()

    def slope(y):
        return ((x - xm) * (y - y.mean())).sum() / den

    sl = s.rolling(n).apply(slope, raw=True)
    val = s.rolling(n).mean() + sl * (n - 1 - xm)
    return pd.DataFrame({"LinReg": val, "LinReg slope": sl})


def mass_index(df, n=25):
    r = df["High"] - df["Low"]
    e1 = base.ema(r, 9)
    return (e1 / base.ema(e1, 9)).rolling(n).sum()


def choppiness(df, n=14):
    tr = base.true_range(df).rolling(n).sum()
    rng = df["High"].rolling(n).max() - df["Low"].rolling(n).min()
    return 100 * np.log10(tr / rng) / np.log10(n)


# ----------------------------------------------------------------- momentum


def roc(s, n=12):
    return (s / s.shift(n) - 1) * 100


def cci(df, n=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - tp.rolling(n).mean()) / (0.015 * md)


def williams_r(df, n=14):
    hh, ll = df["High"].rolling(n).max(), df["Low"].rolling(n).min()
    return -100 * (hh - df["Close"]) / (hh - ll)


def stoch_rsi(s, n=14, k=3, d=3):
    r = base.rsi(s, n)
    lo, hi = r.rolling(n).min(), r.rolling(n).max()
    sr = (r - lo) / (hi - lo) * 100
    kk = sr.rolling(k).mean()
    return pd.DataFrame({"StochRSI K": kk, "StochRSI D": kk.rolling(d).mean()})


def tsi(s, r=25, sm=13):
    m = s.diff()
    num = base.ema(base.ema(m, r), sm)
    den = base.ema(base.ema(m.abs(), r), sm)
    t = 100 * num / den
    return pd.DataFrame({"TSI": t, "TSI signal": base.ema(t, 7)})


def ultimate_osc(df):
    pc = df["Close"].shift(1)
    bp = df["Close"] - pd.concat([df["Low"], pc], axis=1).min(axis=1)
    tr = pd.concat([df["High"], pc], axis=1).max(axis=1) - pd.concat([df["Low"], pc], axis=1).min(axis=1)
    a7 = bp.rolling(7).sum() / tr.rolling(7).sum()
    a14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    a28 = bp.rolling(28).sum() / tr.rolling(28).sum()
    return 100 * (4 * a7 + 2 * a14 + a28) / 7


def awesome(df):
    mp = (df["High"] + df["Low"]) / 2
    return base.sma(mp, 5) - base.sma(mp, 34)


def trix(s, n=15):
    e = base.ema(base.ema(base.ema(s, n), n), n)
    t = e.pct_change() * 100
    return pd.DataFrame({"TRIX": t, "TRIX signal": base.ema(t, 9)})


def kst(s):
    r = [roc(s, n) for n in (10, 15, 20, 30)]
    k = base.sma(r[0], 10) + 2 * base.sma(r[1], 10) + 3 * base.sma(r[2], 10) + 4 * base.sma(r[3], 15)
    return pd.DataFrame({"KST": k, "KST signal": base.sma(k, 9)})


def dpo(s, n=20):
    return s.shift(n // 2 + 1) - base.sma(s, n)


def ppo(s, f=12, sl=26, sig=9):
    p = (base.ema(s, f) - base.ema(s, sl)) / base.ema(s, sl) * 100
    return pd.DataFrame({"PPO": p, "PPO signal": base.ema(p, sig)})


def cmo(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).sum()
    dn = (-d.clip(upper=0)).rolling(n).sum()
    return 100 * (up - dn) / (up + dn)


def fisher(df, n=10):
    hl2 = (df["High"] + df["Low"]) / 2
    lo, hi = hl2.rolling(n).min(), hl2.rolling(n).max()
    v = (2 * ((hl2 - lo) / (hi - lo) - 0.5)).clip(-0.999, 0.999).fillna(0)
    v = v.ewm(alpha=0.33, adjust=False).mean().clip(-0.999, 0.999)
    f = 0.5 * np.log((1 + v) / (1 - v))
    f = f.ewm(alpha=0.5, adjust=False).mean()
    return pd.DataFrame({"Fisher": f, "Fisher trigger": f.shift(1)})


def elder_ray(df, n=13):
    e = base.ema(df["Close"], n)
    return pd.DataFrame({"Bull power": df["High"] - e, "Bear power": df["Low"] - e})


def coppock(s):
    return wma(roc(s, 14) + roc(s, 11), 10)


def momentum(s, n=10):
    return s - s.shift(n)


def rvi(df, n=10):
    num = (df["Close"] - df["Open"]).rolling(4).apply(lambda x: (x[3] + 2 * x[2] + 2 * x[1] + x[0]) / 6, raw=True)
    den = (df["High"] - df["Low"]).rolling(4).apply(lambda x: (x[3] + 2 * x[2] + 2 * x[1] + x[0]) / 6, raw=True)
    r = num.rolling(n).mean() / den.rolling(n).mean()
    return pd.DataFrame({"RVI": r, "RVI signal": r.rolling(4).apply(lambda x: (x[3] + 2 * x[2] + 2 * x[1] + x[0]) / 6, raw=True)})


def schaff(s, f=23, sl=50, cyc=10):
    m = base.ema(s, f) - base.ema(s, sl)
    lo, hi = m.rolling(cyc).min(), m.rolling(cyc).max()
    k = 100 * (m - lo) / (hi - lo)
    d = k.ewm(alpha=0.5, adjust=False).mean()
    lo2, hi2 = d.rolling(cyc).min(), d.rolling(cyc).max()
    k2 = 100 * (d - lo2) / (hi2 - lo2)
    return k2.ewm(alpha=0.5, adjust=False).mean()


def connors_rsi(s):
    r3 = base.rsi(s, 3)
    d = np.sign(s.diff()).fillna(0).values
    streak = np.zeros(len(d))
    for i in range(1, len(d)):
        streak[i] = (streak[i - 1] + 1 if streak[i - 1] >= 0 else 1) if d[i] > 0 else \
            (streak[i - 1] - 1 if streak[i - 1] <= 0 else -1) if d[i] < 0 else 0
    sr = base.rsi(pd.Series(streak, index=s.index), 2)
    pr = s.pct_change().rolling(100).apply(lambda x: (x[:-1] < x[-1]).mean() * 100, raw=True)
    return (r3 + sr + pr) / 3


# --------------------------------------------------------------- volatility


def keltner(df, n=20, mult=2.0):
    mid = base.ema(df["Close"], n)
    a = base.atr(df, 10)
    return pd.DataFrame({"KC upper": mid + mult * a, "KC mid": mid, "KC lower": mid - mult * a})


def donchian(df, n=20):
    up, lo = df["High"].rolling(n).max(), df["Low"].rolling(n).min()
    return pd.DataFrame({"DC upper": up, "DC mid": (up + lo) / 2, "DC lower": lo})


def bb_width(s, n=20):
    up, mid, lo = base.bollinger(s, n)
    return (up - lo) / mid * 100


def bb_pctb(s, n=20):
    up, _, lo = base.bollinger(s, n)
    return (s - lo) / (up - lo)


def natr(df, n=14):
    return base.atr(df, n) / df["Close"] * 100


def ulcer(s, n=14):
    dd = 100 * (s / s.rolling(n).max() - 1)
    return np.sqrt((dd ** 2).rolling(n).mean())


def chaikin_vol(df, n=10):
    e = base.ema(df["High"] - df["Low"], n)
    return (e / e.shift(n) - 1) * 100


def parkinson_vol(df, n=20):
    x = np.log(df["High"] / df["Low"]) ** 2
    return np.sqrt(x.rolling(n).mean() / (4 * np.log(2))) * np.sqrt(252)


def garman_klass(df, n=20):
    hl = np.log(df["High"] / df["Low"]) ** 2
    co = np.log(df["Close"] / df["Open"]) ** 2
    return np.sqrt((0.5 * hl - (2 * np.log(2) - 1) * co).rolling(n).mean()) * np.sqrt(252)


def zscore(s, n=20):
    return (s - s.rolling(n).mean()) / s.rolling(n).std()


def std_channel(s, n=50, k=2):
    lr = linreg(s, n)["LinReg"]
    sd = (s - lr).rolling(n).std()
    return pd.DataFrame({"Channel upper": lr + k * sd, "Channel mid": lr, "Channel lower": lr - k * sd})


def envelope(s, n=20, pct=0.025):
    m = base.sma(s, n)
    return pd.DataFrame({"Env upper": m * (1 + pct), "Env lower": m * (1 - pct)})


# ------------------------------------------------------------------- volume


def mfi(df, n=14):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    pos = mf.where(tp > tp.shift(1), 0).rolling(n).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def adl(df):
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    return (mfm.fillna(0) * df["Volume"]).cumsum()


def cmf(df, n=20):
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    return (mfm.fillna(0) * df["Volume"]).rolling(n).sum() / df["Volume"].rolling(n).sum()


def chaikin_osc(df):
    a = adl(df)
    return base.ema(a, 3) - base.ema(a, 10)


def force_index(df, n=13):
    return base.ema(df["Close"].diff() * df["Volume"], n)


def eom(df, n=14):
    dm = ((df["High"] + df["Low"]) / 2).diff()
    br = (df["Volume"] / 1e6) / (df["High"] - df["Low"]).replace(0, np.nan)
    return (dm / br).rolling(n).mean()


def vpt(df):
    return (df["Close"].pct_change().fillna(0) * df["Volume"]).cumsum()


def nvi(df):
    r = df["Close"].pct_change().fillna(0)
    up = df["Volume"] < df["Volume"].shift(1)
    return 1000 * (1 + r.where(up, 0)).cumprod()


def pvi(df):
    r = df["Close"].pct_change().fillna(0)
    up = df["Volume"] > df["Volume"].shift(1)
    return 1000 * (1 + r.where(up, 0)).cumprod()


def klinger(df):
    tp = df["High"] + df["Low"] + df["Close"]
    trend = np.where(tp > tp.shift(1), 1, -1)
    vf = df["Volume"] * trend
    k = base.ema(vf, 34) - base.ema(vf, 55)
    return pd.DataFrame({"Klinger": k, "Klinger signal": base.ema(k, 13)})


def volume_osc(df):
    v = df["Volume"]
    return (base.ema(v, 5) - base.ema(v, 10)) / base.ema(v, 10) * 100


def rel_volume(df, n=20):
    return df["Volume"] / df["Volume"].shift(1).rolling(n).mean()


def anchored_vwap(df, anchor_days=252):
    d = df.iloc[-anchor_days:]
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    v = (tp * d["Volume"]).cumsum() / d["Volume"].cumsum()
    return v.reindex(df.index)


def volume_profile(df, bins=24, lookback=252):
    d = df.iloc[-lookback:]
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    hist, edges = np.histogram(tp, bins=bins, weights=d["Volume"])
    centers = (edges[:-1] + edges[1:]) / 2
    poc = centers[hist.argmax()]
    order = np.argsort(-hist)
    cum = np.cumsum(hist[order]) / hist.sum()
    va = centers[order[: np.searchsorted(cum, 0.7) + 1]]
    return {"profile": pd.Series(hist, index=centers), "POC": poc, "VAH": va.max(), "VAL": va.min()}


# --------------------------------------------------------------- pivots etc


def pivots_all(df):
    last = df.iloc[-1]
    h, lo, c, o = last["High"], last["Low"], last["Close"], last["Open"]
    p = (h + lo + c) / 3
    r = h - lo
    return pd.DataFrame({
        "Classic": [p + r, 2 * p - lo, p, 2 * p - h, p - r],
        "Fibonacci": [p + 0.618 * r, p + 0.382 * r, p, p - 0.382 * r, p - 0.618 * r],
        "Camarilla": [c + r * 1.1 / 6, c + r * 1.1 / 12, p, c - r * 1.1 / 12, c - r * 1.1 / 6],
        "Woodie": [(h + lo + 2 * c) / 4 + r, 2 * (h + lo + 2 * c) / 4 - lo, (h + lo + 2 * c) / 4,
                   2 * (h + lo + 2 * c) / 4 - h, (h + lo + 2 * c) / 4 - r],
        "DeMark": [np.nan, (h + lo + c + (h if c < o else lo if c > o else c)) / 2 - lo, np.nan,
                   (h + lo + c + (h if c < o else lo if c > o else c)) / 2 - h, np.nan],
    }, index=["R2", "R1", "Pivot", "S1", "S2"])


def heikin_ashi(df):
    ha_c = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_o = np.empty(len(df))
    ha_o[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    hc = ha_c.values
    for i in range(1, len(df)):
        ha_o[i] = (ha_o[i - 1] + hc[i - 1]) / 2
    ha_o = pd.Series(ha_o, index=df.index)
    return pd.DataFrame({"Open": ha_o, "High": pd.concat([df["High"], ha_o, ha_c], axis=1).max(axis=1),
                         "Low": pd.concat([df["Low"], ha_o, ha_c], axis=1).min(axis=1), "Close": ha_c})


# ============================================================== the catalog


@dataclass
class Ind:
    name: str
    cat: str            # Moving averages | Trend | Momentum | Volatility | Volume
    kind: str           # overlay | panel
    fn: Callable        # df -> Series | DataFrame
    rule: Callable | None  # (df, out) -> -1/0/+1 on the last bar
    desc: str = ""


def _last(x):
    try:
        return float(x.iloc[-1])
    except Exception:
        return np.nan


def _ma_rule(df, out):
    c, m = df["Close"].iloc[-1], _last(out)
    return 0 if not np.isfinite(m) else (1 if c > m else -1)


def _band(lo, hi, rev=False):
    def r(df, out):
        v = _last(out if isinstance(out, pd.Series) else out.iloc[:, 0])
        if not np.isfinite(v):
            return 0
        s = 1 if v < lo else -1 if v > hi else 0  # oversold → buy, overbought → sell
        return -s if rev else s
    return r


def _cross(a, b):
    def r(df, out):
        x, y = _last(out[a]), _last(out[b])
        return 0 if not (np.isfinite(x) and np.isfinite(y)) else (1 if x > y else -1)
    return r


def _sign(df, out):
    v = _last(out if isinstance(out, pd.Series) else out.iloc[:, 0])
    return 0 if not np.isfinite(v) else (1 if v > 0 else -1 if v < 0 else 0)


def _rising(df, out):
    s = out if isinstance(out, pd.Series) else out.iloc[:, 0]
    s = s.dropna()
    if len(s) < 6:
        return 0
    return 1 if s.iloc[-1] > s.iloc[-6] else -1


C = lambda df: df["Close"]  # noqa: E731

CATALOG: dict[str, Ind] = {}


def _reg(*items):
    for i in items:
        CATALOG[i.name] = i


for n in (5, 10, 20, 30, 50, 100, 200):
    _reg(Ind(f"SMA {n}", "Moving averages", "overlay", lambda df, n=n: base.sma(C(df), n), _ma_rule,
             f"Simple moving average, {n} days"))
    _reg(Ind(f"EMA {n}", "Moving averages", "overlay", lambda df, n=n: base.ema(C(df), n), _ma_rule,
             f"Exponential moving average, {n} days"))
_reg(
    Ind("WMA 20", "Moving averages", "overlay", lambda df: wma(C(df), 20), _ma_rule, "Weighted MA"),
    Ind("DEMA 20", "Moving averages", "overlay", lambda df: dema(C(df), 20), _ma_rule, "Double EMA, less lag"),
    Ind("TEMA 20", "Moving averages", "overlay", lambda df: tema(C(df), 20), _ma_rule, "Triple EMA"),
    Ind("Hull MA 20", "Moving averages", "overlay", lambda df: hma(C(df), 20), _ma_rule, "Hull MA, very low lag"),
    Ind("ZLEMA 20", "Moving averages", "overlay", lambda df: zlema(C(df), 20), _ma_rule, "Zero-lag EMA"),
    Ind("KAMA 10", "Moving averages", "overlay", lambda df: kama(C(df)), _ma_rule, "Kaufman adaptive MA"),
    Ind("ALMA 9", "Moving averages", "overlay", lambda df: alma(C(df)), _ma_rule, "Arnaud Legoux MA"),
    Ind("VWMA 20", "Moving averages", "overlay", vwma, _ma_rule, "Volume-weighted MA"),
    Ind("SMMA 20", "Moving averages", "overlay", lambda df: smma(C(df), 20), _ma_rule, "Smoothed (Wilder) MA"),
    Ind("McGinley 14", "Moving averages", "overlay", lambda df: mcginley(C(df)), _ma_rule, "McGinley dynamic"),
    Ind("VWAP 20", "Moving averages", "overlay", lambda df: base.vwap(df, 20), _ma_rule, "Rolling 20-day VWAP"),
    Ind("Anchored VWAP (1y)", "Moving averages", "overlay", anchored_vwap, _ma_rule, "VWAP anchored 1 year ago"),
    # trend
    Ind("SuperTrend", "Trend", "overlay", lambda df: supertrend(df)["SuperTrend"],
        lambda df, out: _ma_rule(df, out), "ATR trailing trend line (10, 3)"),
    Ind("Parabolic SAR", "Trend", "overlay", psar, _ma_rule, "Stop-and-reverse dots"),
    Ind("Ichimoku", "Trend", "overlay", ichimoku,
        lambda df, out: (1 if df["Close"].iloc[-1] > max(_last(out["Span A"]), _last(out["Span B"])) and
                         _last(out["Tenkan"]) > _last(out["Kijun"]) else
                         -1 if df["Close"].iloc[-1] < min(_last(out["Span A"]), _last(out["Span B"])) else 0),
        "Cloud, Tenkan/Kijun"),
    Ind("ADX / DMI", "Trend", "panel", lambda df: pd.DataFrame(dict(zip(["ADX", "+DI", "-DI"], base.adx(df)))),
        lambda df, out: 0 if _last(out["ADX"]) < 20 else (1 if _last(out["+DI"]) > _last(out["-DI"]) else -1),
        "Trend strength (ADX > 25 = trending)"),
    Ind("Aroon", "Trend", "panel", aroon, lambda df, out: 1 if _last(out["Aroon Osc"]) > 50 else
        -1 if _last(out["Aroon Osc"]) < -50 else 0, "Time since highs/lows"),
    Ind("Vortex", "Trend", "panel", vortex, _cross("VI+", "VI-"), "VI+ vs VI-"),
    Ind("Linear regression", "Trend", "overlay", lambda df: linreg(C(df))["LinReg"], _ma_rule, "20-day regression line"),
    Ind("LinReg slope", "Trend", "panel", lambda df: linreg(C(df))["LinReg slope"], _sign, "Regression slope"),
    Ind("Mass index", "Trend", "panel", mass_index, lambda df, out: 0, "Reversal bulge > 27"),
    Ind("Choppiness", "Trend", "panel", choppiness, lambda df, out: 0, "> 61.8 choppy, < 38.2 trending"),
    Ind("Std-dev channel", "Trend", "overlay", lambda df: std_channel(C(df)), lambda df, out: 0,
        "Regression channel ±2σ"),
    # momentum
    Ind("RSI 14", "Momentum", "panel", lambda df: base.rsi(C(df)), _band(30, 70), "Relative strength index"),
    Ind("RSI 2", "Momentum", "panel", lambda df: base.rsi(C(df), 2), _band(10, 90), "Short-term RSI"),
    Ind("Stochastic", "Momentum", "panel", lambda df: pd.DataFrame(dict(zip(["%K", "%D"], base.stochastic(df)))),
        _band(20, 80), "%K / %D (14,3)"),
    Ind("Stochastic RSI", "Momentum", "panel", lambda df: stoch_rsi(C(df)), _band(20, 80), "Stochastic of RSI"),
    Ind("MACD", "Momentum", "panel", lambda df: pd.DataFrame(dict(zip(["MACD", "Signal", "Hist"], base.macd(C(df))))),
        _cross("MACD", "Signal"), "12/26/9"),
    Ind("PPO", "Momentum", "panel", lambda df: ppo(C(df)), _cross("PPO", "PPO signal"), "Percentage price oscillator"),
    Ind("CCI 20", "Momentum", "panel", cci, _band(-100, 100), "Commodity channel index"),
    Ind("Williams %R", "Momentum", "panel", williams_r, _band(-80, -20), "Williams percent range"),
    Ind("ROC 12", "Momentum", "panel", lambda df: roc(C(df)), _sign, "Rate of change"),
    Ind("Momentum 10", "Momentum", "panel", lambda df: momentum(C(df)), _sign, "Price minus price 10 days ago"),
    Ind("TSI", "Momentum", "panel", lambda df: tsi(C(df)), _cross("TSI", "TSI signal"), "True strength index"),
    Ind("Ultimate oscillator", "Momentum", "panel", ultimate_osc, _band(30, 70), "7/14/28 blended"),
    Ind("Awesome oscillator", "Momentum", "panel", awesome, _sign, "5 vs 34 median-price SMA"),
    Ind("TRIX", "Momentum", "panel", lambda df: trix(C(df)), _cross("TRIX", "TRIX signal"), "Triple-smoothed ROC"),
    Ind("KST", "Momentum", "panel", lambda df: kst(C(df)), _cross("KST", "KST signal"), "Know sure thing"),
    Ind("DPO", "Momentum", "panel", lambda df: dpo(C(df)), lambda df, out: 0, "Detrended price oscillator (cycles)"),
    Ind("CMO", "Momentum", "panel", lambda df: cmo(C(df)), _band(-50, 50), "Chande momentum oscillator"),
    Ind("Fisher transform", "Momentum", "panel", fisher, _cross("Fisher", "Fisher trigger"), "Gaussianised price"),
    Ind("Elder ray", "Momentum", "panel", elder_ray,
        lambda df, out: 1 if _last(out["Bear power"]) < 0 < _last(out["Bull power"]) and _rising(df, out["Bear power"]) > 0
        else -1 if _last(out["Bull power"]) < 0 else 0, "Bull / bear power"),
    Ind("Coppock curve", "Momentum", "panel", lambda df: coppock(C(df)), _rising, "Long-term bottom finder"),
    Ind("RVI", "Momentum", "panel", rvi, _cross("RVI", "RVI signal"), "Relative vigor index"),
    Ind("Schaff trend cycle", "Momentum", "panel", lambda df: schaff(C(df)), _band(25, 75), "MACD + stochastic cycle"),
    Ind("Connors RSI", "Momentum", "panel", lambda df: connors_rsi(C(df)), _band(10, 90), "RSI + streak + rank"),
    # volatility
    Ind("Bollinger Bands", "Volatility", "overlay",
        lambda df: pd.DataFrame(dict(zip(["BB upper", "BB mid", "BB lower"], base.bollinger(C(df))))),
        lambda df, out: 1 if df["Close"].iloc[-1] < _last(out["BB lower"]) else
        -1 if df["Close"].iloc[-1] > _last(out["BB upper"]) else 0, "20, 2σ"),
    Ind("Bollinger %B", "Volatility", "panel", lambda df: bb_pctb(C(df)), _band(0, 1), "Position inside the bands"),
    Ind("Bollinger width", "Volatility", "panel", lambda df: bb_width(C(df)), lambda df, out: 0, "Squeeze detector"),
    Ind("Keltner channel", "Volatility", "overlay", keltner,
        lambda df, out: 1 if df["Close"].iloc[-1] > _last(out["KC upper"]) else
        -1 if df["Close"].iloc[-1] < _last(out["KC lower"]) else 0, "EMA ± 2 ATR"),
    Ind("Donchian channel", "Volatility", "overlay", donchian,
        lambda df, out: 1 if df["Close"].iloc[-1] >= _last(out["DC upper"]) else
        -1 if df["Close"].iloc[-1] <= _last(out["DC lower"]) else 0, "20-day high/low"),
    Ind("Envelope", "Volatility", "overlay", lambda df: envelope(C(df)), lambda df, out: 0, "SMA ± 2.5%"),
    Ind("ATR 14", "Volatility", "panel", lambda df: base.atr(df), lambda df, out: 0, "Average true range"),
    Ind("NATR", "Volatility", "panel", natr, lambda df, out: 0, "ATR as % of price"),
    Ind("Historical vol 20", "Volatility", "panel", lambda df: base.realized_vol(C(df)) * 100, lambda df, out: 0,
        "Annualised close-to-close"),
    Ind("Parkinson vol", "Volatility", "panel", lambda df: parkinson_vol(df) * 100, lambda df, out: 0, "High-low"),
    Ind("Garman-Klass vol", "Volatility", "panel", lambda df: garman_klass(df) * 100, lambda df, out: 0, "OHLC"),
    Ind("Ulcer index", "Volatility", "panel", lambda df: ulcer(C(df)), lambda df, out: 0, "Drawdown depth"),
    Ind("Chaikin volatility", "Volatility", "panel", chaikin_vol, lambda df, out: 0, "Range expansion"),
    Ind("Z-score 20", "Volatility", "panel", lambda df: zscore(C(df)), _band(-2, 2), "Distance from mean in σ"),
    # volume
    Ind("OBV", "Volume", "panel", base.obv, _rising, "On-balance volume"),
    Ind("MFI 14", "Volume", "panel", mfi, _band(20, 80), "Money flow index"),
    Ind("Accumulation/Distribution", "Volume", "panel", adl, _rising, "A/D line"),
    Ind("Chaikin money flow", "Volume", "panel", cmf, lambda df, out: 1 if _last(out) > 0.05 else
        -1 if _last(out) < -0.05 else 0, "20-day CMF"),
    Ind("Chaikin oscillator", "Volume", "panel", chaikin_osc, _sign, "A/D momentum"),
    Ind("Force index", "Volume", "panel", force_index, _sign, "Price change × volume"),
    Ind("Ease of movement", "Volume", "panel", eom, _sign, "Price move per volume"),
    Ind("Volume price trend", "Volume", "panel", vpt, _rising, "VPT"),
    Ind("Negative volume index", "Volume", "panel", nvi, _rising, "Smart-money proxy"),
    Ind("Positive volume index", "Volume", "panel", pvi, _rising, "Crowd proxy"),
    Ind("Klinger oscillator", "Volume", "panel", klinger, _cross("Klinger", "Klinger signal"), "Volume force"),
    Ind("Volume oscillator", "Volume", "panel", volume_osc, lambda df, out: 0, "5 vs 10 EMA of volume"),
    Ind("Relative volume", "Volume", "panel", rel_volume, lambda df, out: 0, "Today vs 20-day average"),
)

CATEGORIES = ["Moving averages", "Trend", "Momentum", "Volatility", "Volume"]


def compute(df: pd.DataFrame, name: str) -> pd.Series | pd.DataFrame:
    return CATALOG[name].fn(df)


def label(score: float) -> str:
    return ("STRONG BUY" if score >= 0.5 else "BUY" if score >= 0.1 else "STRONG SELL" if score <= -0.5
            else "SELL" if score <= -0.1 else "NEUTRAL")


def summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Evaluate every indicator on the last bar.

    Returns (table, verdicts) where verdicts maps each category and 'Overall'
    to {'buy','neutral','sell','score','label'}. Indicators with no rule
    (pure measurements like ATR) are listed with value but don't vote."""
    rows = []
    for ind in CATALOG.values():
        try:
            out = ind.fn(df)
            val = out.iloc[-1] if isinstance(out, pd.Series) else out.iloc[-1, 0]
            sig = ind.rule(df, out) if ind.rule else 0
            votes = _votes(ind)
        except Exception:
            val, sig, votes = np.nan, 0, False
        rows.append({"Indicator": ind.name, "Category": ind.cat, "Value": float(val) if np.isfinite(val) else np.nan,
                     "Signal": {1: "BUY", -1: "SELL", 0: "NEUTRAL"}[int(sig)], "Votes": bool(votes),
                     "About": ind.desc})
    tbl = pd.DataFrame(rows)
    verdicts = {}
    for cat in CATEGORIES + ["Overall"]:
        t = tbl[tbl["Votes"]] if cat == "Overall" else tbl[(tbl["Category"] == cat) & tbl["Votes"]]
        b, s, n = (t["Signal"] == "BUY").sum(), (t["Signal"] == "SELL").sum(), (t["Signal"] == "NEUTRAL").sum()
        tot = max(b + s + n, 1)
        score = (b - s) / tot
        verdicts[cat] = {"buy": int(b), "sell": int(s), "neutral": int(n), "score": float(score), "label": label(score)}
    return tbl, verdicts


_NO_VOTE = {"Mass index", "Choppiness", "Std-dev channel", "DPO", "Bollinger width", "Envelope", "ATR 14", "NATR",
            "Historical vol 20", "Parkinson vol", "Garman-Klass vol", "Ulcer index", "Chaikin volatility",
            "Volume oscillator", "Relative volume"}


def _votes(ind: Ind) -> bool:
    return ind.name not in _NO_VOTE
