"""Automatic pattern recognition.

* 40 candlestick patterns (vectorised, with trend context)
* ZigZag swing pivots with their confirmation bar (no look-ahead)
* Chart patterns: double/triple tops & bottoms, head & shoulders (and
  inverse), ascending/descending/symmetrical triangles, rising/falling
  wedges, rectangles, up/down channels, broadening formations, bull/bear
  flags and pennants, cup & handle, rounding bottom/top
* Harmonics: Gartley, Bat, Butterfly, Crab, Cypher, AB=CD
* Elliott 5-wave impulse (rule-checked)
* RSI / MACD divergences (regular and hidden)
* Gaps (breakaway / runaway / exhaustion / common) with fill status
* Trendlines, Fibonacci retracements, support/resistance zones

Every detection carries the points and lines needed to draw it, a bias, a
measured-move target and an invalidation level where the textbook defines
one. ``pattern_history`` replays the chart-pattern detector bar by bar using
only pivots confirmed at that time, so reliability stats are look-ahead free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind

# =========================================================== candlesticks


def _ctx(df):
    o, h, lo, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    rng = (h - lo).replace(0, np.nan)
    up_sh = h - pd.concat([o, c], axis=1).max(axis=1)
    lo_sh = pd.concat([o, c], axis=1).min(axis=1) - lo
    avg_body = body.rolling(14).mean()
    sma10 = c.rolling(10).mean()
    down = (c.shift(1) < sma10.shift(1)) & (c.shift(1) < c.shift(6))
    up = (c.shift(1) > sma10.shift(1)) & (c.shift(1) > c.shift(6))
    bull = c > o
    bear = c < o
    return o, h, lo, c, body, rng, up_sh, lo_sh, avg_body, down, up, bull, bear


def candlestick_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame, one column per candlestick pattern."""
    o, h, lo, c, body, rng, ush, lsh, ab, down, up, bull, bear = _ctx(df)
    s = lambda x, k=1: x.shift(k)  # noqa: E731
    doji = body <= 0.1 * rng
    long_body = body >= 1.3 * ab
    small = body <= 0.5 * ab
    mid1 = (s(o) + s(c)) / 2
    P = {}
    P["Doji"] = doji & (ush > 0.1 * rng) & (lsh > 0.1 * rng)
    P["Dragonfly doji"] = doji & (lsh >= 0.6 * rng) & (ush <= 0.1 * rng)
    P["Gravestone doji"] = doji & (ush >= 0.6 * rng) & (lsh <= 0.1 * rng)
    P["Long-legged doji"] = doji & (ush >= 0.35 * rng) & (lsh >= 0.35 * rng) & (rng >= 1.5 * ab)
    hammer_shape = (lsh >= 2 * body) & (ush <= 0.25 * body + 0.05 * rng) & (body > 0.05 * rng)
    inv_shape = (ush >= 2 * body) & (lsh <= 0.25 * body + 0.05 * rng) & (body > 0.05 * rng)
    P["Hammer"] = hammer_shape & down
    P["Hanging man"] = hammer_shape & up
    P["Inverted hammer"] = inv_shape & down
    P["Shooting star"] = inv_shape & up
    P["Bullish marubozu"] = bull & long_body & (ush <= 0.05 * rng) & (lsh <= 0.05 * rng)
    P["Bearish marubozu"] = bear & long_body & (ush <= 0.05 * rng) & (lsh <= 0.05 * rng)
    P["Spinning top"] = small & ~doji & (ush > body) & (lsh > body)
    P["Bullish belt hold"] = bull & long_body & (lsh <= 0.03 * rng) & down
    P["Bearish belt hold"] = bear & long_body & (ush <= 0.03 * rng) & up
    # two-bar
    P["Bullish engulfing"] = bull & s(bear) & (c >= s(o)) & (o <= s(c)) & (body > s(body)) & down
    P["Bearish engulfing"] = bear & s(bull) & (o >= s(c)) & (c <= s(o)) & (body > s(body)) & up
    inside = (pd.concat([o, c], axis=1).max(axis=1) < pd.concat([s(o), s(c)], axis=1).max(axis=1)) & \
             (pd.concat([o, c], axis=1).min(axis=1) > pd.concat([s(o), s(c)], axis=1).min(axis=1))
    P["Bullish harami"] = s(bear) & s(long_body) & inside & bull & down
    P["Bearish harami"] = s(bull) & s(long_body) & inside & bear & up
    P["Harami cross"] = s(long_body) & inside & doji
    P["Piercing line"] = s(bear) & s(long_body) & bull & (o < s(lo)) & (c > mid1) & (c < s(o)) & down
    P["Dark cloud cover"] = s(bull) & s(long_body) & bear & (o > s(h)) & (c < mid1) & (c > s(o)) & up
    tol = 0.002 * c
    P["Tweezer bottom"] = ((lo - s(lo)).abs() <= tol) & s(bear) & bull & down
    P["Tweezer top"] = ((h - s(h)).abs() <= tol) & s(bull) & bear & up
    P["Bullish kicker"] = s(bear) & bull & (o > s(o)) & long_body
    P["Bearish kicker"] = s(bull) & bear & (o < s(o)) & long_body
    P["Matching low"] = s(bear) & bear & ((c - s(c)).abs() <= tol) & down
    P["Bullish counterattack"] = s(bear) & s(long_body) & bull & long_body & ((c - s(c)).abs() <= tol) & down
    P["Bearish counterattack"] = s(bull) & s(long_body) & bear & long_body & ((c - s(c)).abs() <= tol) & up
    # three-bar
    star_gap_dn = pd.concat([o, c], axis=1).max(axis=1).shift(1) < pd.concat([s(o, 2), s(c, 2)], axis=1).min(axis=1)
    star_gap_up = pd.concat([o, c], axis=1).min(axis=1).shift(1) > pd.concat([s(o, 2), s(c, 2)], axis=1).max(axis=1)
    mid2 = (s(o, 2) + s(c, 2)) / 2
    P["Morning star"] = s(bear, 2) & s(long_body, 2) & s(small) & star_gap_dn & bull & (c > mid2) & s(down)
    P["Evening star"] = s(bull, 2) & s(long_body, 2) & s(small) & star_gap_up & bear & (c < mid2) & s(up)
    P["Morning doji star"] = P["Morning star"] & s(doji)
    P["Evening doji star"] = P["Evening star"] & s(doji)
    P["Bullish abandoned baby"] = s(bear, 2) & s(doji) & (s(h) < s(lo, 2)) & bull & (lo > s(h)) & s(down)
    P["Bearish abandoned baby"] = s(bull, 2) & s(doji) & (s(lo) > s(h, 2)) & bear & (h < s(lo)) & s(up)
    sold = lambda k: bull.shift(k) & (c.shift(k) > c.shift(k + 1)) & (ush.shift(k) <= 0.3 * body.shift(k))  # noqa: E731
    P["Three white soldiers"] = sold(0) & sold(1) & sold(2) & (o > s(o)) & (o < s(c)) & (s(o) > s(o, 2)) & \
        (body >= 0.7 * ab) & (s(body) >= 0.7 * ab) & (s(body, 2) >= 0.7 * ab)
    crow = lambda k: bear.shift(k) & (c.shift(k) < c.shift(k + 1)) & (lsh.shift(k) <= 0.3 * body.shift(k))  # noqa: E731
    P["Three black crows"] = crow(0) & crow(1) & crow(2) & (o < s(o)) & (o > s(c)) & \
        (body >= 0.7 * ab) & (s(body) >= 0.7 * ab) & (s(body, 2) >= 0.7 * ab)
    P["Three inside up"] = P["Bullish harami"].shift(1, fill_value=False) & bull & (c > s(o, 2))
    P["Three inside down"] = P["Bearish harami"].shift(1, fill_value=False) & bear & (c < s(o, 2))
    P["Three outside up"] = P["Bullish engulfing"].shift(1, fill_value=False) & bull & (c > s(c))
    P["Three outside down"] = P["Bearish engulfing"].shift(1, fill_value=False) & bear & (c < s(c))
    P["Bullish three-line strike"] = s(bear, 3) & s(bear, 2) & s(bear) & (s(c, 2) < s(c, 3)) & (s(c) < s(c, 2)) & \
        bull & (o <= s(c)) & (c >= s(o, 3))
    P["Bearish three-line strike"] = s(bull, 3) & s(bull, 2) & s(bull) & (s(c, 2) > s(c, 3)) & (s(c) > s(c, 2)) & \
        bear & (o >= s(c)) & (c <= s(o, 3))
    P["Upside tasuki gap"] = s(bull, 2) & s(bull) & (s(lo) > s(h, 2)) & bear & (o < s(c)) & (o > s(o)) & \
        (c < s(o)) & (c > s(h, 2))
    P["Downside tasuki gap"] = s(bear, 2) & s(bear) & (s(h) < s(lo, 2)) & bull & (o > s(c)) & (o < s(o)) & \
        (c > s(o)) & (c < s(lo, 2))
    # five-bar
    rt = s(bull, 4) & s(long_body, 4) & s(small, 3) & s(small, 2) & s(small, 1) & \
        (pd.concat([s(h, k) for k in (1, 2, 3)], axis=1).max(axis=1) < s(h, 4)) & \
        (pd.concat([s(lo, k) for k in (1, 2, 3)], axis=1).min(axis=1) > s(lo, 4)) & bull & long_body & (c > s(c, 4))
    ft = s(bear, 4) & s(long_body, 4) & s(small, 3) & s(small, 2) & s(small, 1) & \
        (pd.concat([s(h, k) for k in (1, 2, 3)], axis=1).max(axis=1) < s(h, 4)) & \
        (pd.concat([s(lo, k) for k in (1, 2, 3)], axis=1).min(axis=1) > s(lo, 4)) & bear & long_body & (c < s(c, 4))
    P["Rising three methods"] = rt
    P["Falling three methods"] = ft
    return pd.DataFrame({k: v.fillna(False).astype(bool) for k, v in P.items()}, index=df.index)


CANDLE_BIAS = {
    "Doji": 0, "Dragonfly doji": 1, "Gravestone doji": -1, "Long-legged doji": 0, "Hammer": 1, "Hanging man": -1,
    "Inverted hammer": 1, "Shooting star": -1, "Bullish marubozu": 1, "Bearish marubozu": -1, "Spinning top": 0,
    "Bullish belt hold": 1, "Bearish belt hold": -1, "Bullish engulfing": 1, "Bearish engulfing": -1,
    "Bullish harami": 1, "Bearish harami": -1, "Harami cross": 0, "Piercing line": 1, "Dark cloud cover": -1,
    "Tweezer bottom": 1, "Tweezer top": -1, "Bullish kicker": 1, "Bearish kicker": -1, "Matching low": 1,
    "Bullish counterattack": 1, "Bearish counterattack": -1, "Morning star": 1, "Evening star": -1,
    "Morning doji star": 1, "Evening doji star": -1, "Bullish abandoned baby": 1, "Bearish abandoned baby": -1,
    "Three white soldiers": 1, "Three black crows": -1, "Three inside up": 1, "Three inside down": -1,
    "Three outside up": 1, "Three outside down": -1, "Bullish three-line strike": 1,
    "Bearish three-line strike": -1, "Upside tasuki gap": 1, "Downside tasuki gap": -1,
    "Rising three methods": 1, "Falling three methods": -1,
}


def candle_stats(df: pd.DataFrame, frame: pd.DataFrame | None = None, horizon: int = 10) -> pd.DataFrame:
    """How the stock moved ``horizon`` days after each candlestick pattern, vs its average."""
    frame = frame if frame is not None else candlestick_frame(df)
    fwd = df["Close"].shift(-horizon) / df["Close"] - 1
    base_mean = fwd.mean()
    base_hit = (fwd > 0).mean()
    rows = []
    for k in frame.columns:
        f = fwd[frame[k]].dropna()
        bias = CANDLE_BIAS.get(k, 0)
        rows.append({
            "Pattern": k, "Bias": {1: "Bullish", -1: "Bearish", 0: "Neutral"}[bias], "Times": len(f),
            f"Avg {horizon}d": f.mean() if len(f) else np.nan,
            "Up %": (f > 0).mean() if len(f) else np.nan,
            "Worked %": ((f > 0) if bias >= 0 else (f < 0)).mean() if len(f) else np.nan,
            "Edge vs avg": (f.mean() - base_mean) * (1 if bias >= 0 else -1) if len(f) else np.nan,
            "Last seen": frame.index[frame[k]][-1] if frame[k].any() else None,
        })
    out = pd.DataFrame(rows).set_index("Pattern")
    out.attrs["base_mean"], out.attrs["base_hit"] = base_mean, base_hit
    return out


# =============================================================== pivots


@dataclass
class Pivot:
    i: int          # bar index of the extreme
    price: float
    kind: str       # "H" or "L"
    confirmed: int  # bar index when the reversal confirmed it


def zigzag(df: pd.DataFrame, pct: float | None = None) -> list[Pivot]:
    """Swing pivots: a high/low becomes a pivot once price reverses by ``pct``
    (default: max(3%, 2×median ATR%)). Each pivot records when it was confirmed."""
    h, lo, c = df["High"].values, df["Low"].values, df["Close"].values
    if pct is None:
        a = (ind.atr(df) / df["Close"]).median()
        pct = float(np.clip(2 * a if np.isfinite(a) else 0.04, 0.03, 0.15))
    piv: list[Pivot] = []
    n = len(c)
    if n < 3:
        return piv
    trend = 0
    hi_i, hi_p, lo_i, lo_p = 0, h[0], 0, lo[0]
    ext_i, ext_p = 0, c[0]
    for i in range(1, n):
        if trend == 0:
            if h[i] > hi_p:
                hi_i, hi_p = i, h[i]
            if lo[i] < lo_p:
                lo_i, lo_p = i, lo[i]
            if hi_p >= lo_p * (1 + pct) and lo_i < hi_i:
                piv.append(Pivot(lo_i, float(lo_p), "L", i))
                trend, ext_i, ext_p = 1, hi_i, hi_p
            elif lo_p <= hi_p * (1 - pct) and hi_i < lo_i:
                piv.append(Pivot(hi_i, float(hi_p), "H", i))
                trend, ext_i, ext_p = -1, lo_i, lo_p
        elif trend == 1:
            if h[i] > ext_p:
                ext_i, ext_p = i, h[i]
            elif lo[i] <= ext_p * (1 - pct):
                piv.append(Pivot(ext_i, float(ext_p), "H", i))
                trend, ext_i, ext_p = -1, i, lo[i]
        else:
            if lo[i] < ext_p:
                ext_i, ext_p = i, lo[i]
            elif h[i] >= ext_p * (1 + pct):
                piv.append(Pivot(ext_i, float(ext_p), "L", i))
                trend, ext_i, ext_p = 1, i, h[i]
    return piv


# ======================================================== chart patterns


@dataclass
class Pattern:
    name: str
    kind: str                  # chart | harmonic | wave | divergence | gap | candle | level
    bias: int                  # +1 bullish, -1 bearish, 0 neutral
    start: pd.Timestamp
    end: pd.Timestamp
    points: list = field(default_factory=list)   # [(date, price, label)]
    lines: list = field(default_factory=list)    # [((d1,p1),(d2,p2), role)]
    status: str = "forming"    # forming | confirmed | failed
    target: float | None = None
    invalidation: float | None = None
    note: str = ""

    @property
    def bias_text(self):
        return {1: "Bullish", -1: "Bearish", 0: "Neutral"}[self.bias]


def _line_at(p1, p2, x):
    (x1, y1), (x2, y2) = p1, p2
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1) if x2 != x1 else y1


def _fit(xs, ys):
    if len(xs) < 2:
        return None
    k, b = np.polyfit(xs, ys, 1)
    return k, b


def _alt(pivs):
    """Collapse consecutive same-kind pivots, keeping the more extreme one."""
    out = []
    for p in pivs:
        if out and out[-1].kind == p.kind:
            if (p.kind == "H" and p.price > out[-1].price) or (p.kind == "L" and p.price < out[-1].price):
                out[-1] = p
        else:
            out.append(p)
    return out


def detect_chart_patterns(df: pd.DataFrame, pivs: list[Pivot], t: int | None = None,
                          tol: float | None = None) -> list[Pattern]:
    """Chart patterns using pivots confirmed by bar ``t`` (default: last bar)."""
    n = len(df)
    t = n - 1 if t is None else t
    P = _alt([p for p in pivs if p.confirmed <= t])
    if len(P) < 3:
        return []
    idx = df.index
    c = df["Close"].values
    price = c[t]
    atrp = float((ind.atr(df.iloc[: t + 1]) / df["Close"].iloc[: t + 1]).iloc[-1]) if t > 20 else 0.02
    tol = tol or float(np.clip(1.2 * atrp, 0.015, 0.05))
    D = lambda i: idx[min(i, n - 1)]  # noqa: E731
    out: list[Pattern] = []

    # --- double / triple tops & bottoms, head & shoulders -------------------
    a, b, c3 = P[-3], P[-2], P[-1]
    if a.kind == "H" and c3.kind == "H" and abs(a.price - c3.price) / a.price < tol and \
            (min(a.price, c3.price) - b.price) / b.price > 2 * tol and c3.i - a.i >= 8:
        conf = price < b.price
        out.append(Pattern("Double top", "chart", -1, D(a.i), D(t),
                           [(D(a.i), a.price, "Top 1"), (D(b.i), b.price, "Neckline"), (D(c3.i), c3.price, "Top 2")],
                           [((D(b.i), b.price), (D(t), b.price), "neckline")],
                           "confirmed" if conf else "forming", b.price - (max(a.price, c3.price) - b.price),
                           max(a.price, c3.price), "Breaks down on a close below the neckline."))
    if a.kind == "L" and c3.kind == "L" and abs(a.price - c3.price) / a.price < tol and \
            (b.price - max(a.price, c3.price)) / b.price > 2 * tol and c3.i - a.i >= 8:
        conf = price > b.price
        out.append(Pattern("Double bottom", "chart", 1, D(a.i), D(t),
                           [(D(a.i), a.price, "Bottom 1"), (D(b.i), b.price, "Neckline"), (D(c3.i), c3.price, "Bottom 2")],
                           [((D(b.i), b.price), (D(t), b.price), "neckline")],
                           "confirmed" if conf else "forming", b.price + (b.price - min(a.price, c3.price)),
                           min(a.price, c3.price), "Breaks out on a close above the neckline."))
    if len(P) >= 5:
        p1, p2, p3, p4, p5 = P[-5:]
        if p1.kind == "H" and p3.kind == "H" and p5.kind == "H":
            hs = [p1.price, p3.price, p5.price]
            if max(hs) / min(hs) - 1 < tol * 1.2:
                nk = min(p2.price, p4.price)
                out.append(Pattern("Triple top", "chart", -1, D(p1.i), D(t),
                                   [(D(p.i), p.price, "") for p in (p1, p2, p3, p4, p5)],
                                   [((D(p2.i), nk), (D(t), nk), "neckline")],
                                   "confirmed" if price < nk else "forming", nk - (max(hs) - nk), max(hs)))
            elif p3.price > p1.price * (1 + tol) and p3.price > p5.price * (1 + tol) and \
                    abs(p1.price - p5.price) / p1.price < 2 * tol:
                nl = _line_at((p2.i, p2.price), (p4.i, p4.price), t)
                head_h = p3.price - _line_at((p2.i, p2.price), (p4.i, p4.price), p3.i)
                out.append(Pattern("Head & shoulders", "chart", -1, D(p1.i), D(t),
                                   [(D(p1.i), p1.price, "L shoulder"), (D(p3.i), p3.price, "Head"),
                                    (D(p5.i), p5.price, "R shoulder")],
                                   [((D(p2.i), p2.price), (D(t), nl), "neckline")],
                                   "confirmed" if price < nl else "forming", nl - head_h, p5.price,
                                   "Classic top reversal; confirmed on a close below the neckline."))
        if p1.kind == "L" and p3.kind == "L" and p5.kind == "L":
            ls = [p1.price, p3.price, p5.price]
            if max(ls) / min(ls) - 1 < tol * 1.2:
                nk = max(p2.price, p4.price)
                out.append(Pattern("Triple bottom", "chart", 1, D(p1.i), D(t),
                                   [(D(p.i), p.price, "") for p in (p1, p2, p3, p4, p5)],
                                   [((D(p2.i), nk), (D(t), nk), "neckline")],
                                   "confirmed" if price > nk else "forming", nk + (nk - min(ls)), min(ls)))
            elif p3.price < p1.price * (1 - tol) and p3.price < p5.price * (1 - tol) and \
                    abs(p1.price - p5.price) / p1.price < 2 * tol:
                nl = _line_at((p2.i, p2.price), (p4.i, p4.price), t)
                head_d = _line_at((p2.i, p2.price), (p4.i, p4.price), p3.i) - p3.price
                out.append(Pattern("Inverse head & shoulders", "chart", 1, D(p1.i), D(t),
                                   [(D(p1.i), p1.price, "L shoulder"), (D(p3.i), p3.price, "Head"),
                                    (D(p5.i), p5.price, "R shoulder")],
                                   [((D(p2.i), p2.price), (D(t), nl), "neckline")],
                                   "confirmed" if price > nl else "forming", nl + head_d, p5.price,
                                   "Classic bottom reversal; confirmed on a close above the neckline."))

    # --- triangles, wedges, rectangles, channels, broadening -----------------
    recent = [p for p in P if p.i >= t - 150][-6:]
    hi = [p for p in recent if p.kind == "H"]
    lw = [p for p in recent if p.kind == "L"]
    if len(hi) >= 2 and len(lw) >= 2 and len(hi) + len(lw) >= 4:
        fu = _fit([p.i for p in hi], [p.price for p in hi])
        fl = _fit([p.i for p in lw], [p.price for p in lw])
        x0 = min(hi[0].i, lw[0].i)
        span = t - x0
        if fu and fl and span >= 10:
            up_t, lo_t = fu[0] * t + fu[1], fl[0] * t + fl[1]
            su = fu[0] * span / price  # total % change of the line across the pattern
            sl = fl[0] * span / price
            flat = 0.025
            width0 = (fu[0] * x0 + fu[1]) - (fl[0] * x0 + fl[1])
            width_t = up_t - lo_t
            conv = width_t < 0.75 * width0
            div = width_t > 1.3 * width0
            name, bias = None, 0
            if abs(su) < flat and sl > flat and conv:
                name, bias = "Ascending triangle", 1
            elif abs(sl) < flat and su < -flat and conv:
                name, bias = "Descending triangle", -1
            elif su < -flat and sl > flat and conv:
                name, bias = "Symmetrical triangle", 0
            elif su > flat and sl > flat and conv and sl > su:
                name, bias = "Rising wedge", -1
            elif su < -flat and sl < -flat and conv and sl > su:
                name, bias = "Falling wedge", 1
            elif abs(su) < flat and abs(sl) < flat:
                name, bias = "Rectangle", 0
            elif su > flat and sl > flat and not conv and not div:
                name, bias = "Ascending channel", 1
            elif su < -flat and sl < -flat and not conv and not div:
                name, bias = "Descending channel", -1
            elif su > flat and sl < -flat and div:
                name, bias = "Broadening formation", 0
            if name and width_t > 0:
                status = "forming"
                tgt = None
                if price > up_t * (1 + 0.003):
                    status, tgt = "confirmed", up_t + width0
                    bias = 1 if name not in ("Ascending channel",) else bias
                elif price < lo_t * (1 - 0.003):
                    status, tgt = "confirmed", lo_t - width0
                    bias = -1 if name not in ("Descending channel",) else bias
                note = {"Ascending triangle": "Flat resistance with rising lows; usually breaks up.",
                        "Descending triangle": "Flat support with falling highs; usually breaks down.",
                        "Symmetrical triangle": "Coiling range; trade the breakout direction.",
                        "Rising wedge": "Converging uptrend; often breaks down.",
                        "Falling wedge": "Converging downtrend; often breaks up.",
                        "Rectangle": "Horizontal range between support and resistance.",
                        "Ascending channel": "Parallel uptrend; buy near support, trim near resistance.",
                        "Descending channel": "Parallel downtrend.",
                        "Broadening formation": "Expanding swings; volatile, often a topping sign."}[name]
                if status == "confirmed":
                    note += " Breakout confirmed by close."
                out.append(Pattern(
                    name, "chart", bias, D(x0), D(t), [(D(p.i), p.price, "") for p in recent],
                    [((D(x0), fu[0] * x0 + fu[1]), (D(t), up_t), "upper"),
                     ((D(x0), fl[0] * x0 + fl[1]), (D(t), lo_t), "lower")],
                    status, tgt, lo_t if bias >= 0 else up_t, note))

    # --- flags & pennants -----------------------------------------------------
    last = P[-1]
    if t - last.i >= 4 and t - last.i <= 25:
        prev = P[-2] if len(P) >= 2 else None
        if prev is not None:
            pole = (last.price - prev.price) / prev.price
            pole_len = last.i - prev.i
            seg = df.iloc[last.i + 1: t + 1]
            if len(seg) >= 4 and pole_len <= 20 and abs(pole) >= max(0.08, 4 * atrp):
                retr = (last.price - seg["Low"].min()) / (last.price - prev.price) if pole > 0 else \
                    (seg["High"].max() - last.price) / (prev.price - last.price)
                xs = np.arange(last.i + 1, t + 1)
                ku = np.polyfit(xs, seg["High"].values, 1)[0] * len(xs) / price
                kl = np.polyfit(xs, seg["Low"].values, 1)[0] * len(xs) / price
                w0 = seg["High"].iloc[:3].max() - seg["Low"].iloc[:3].min()
                w1 = seg["High"].iloc[-3:].max() - seg["Low"].iloc[-3:].min()
                if 0 < retr < 0.5:
                    bull_ = pole > 0
                    pennant = w1 < 0.6 * w0
                    against = (ku < 0 and kl < 0) if bull_ else (ku > 0 and kl > 0)
                    if pennant or against:
                        nm = ("Bull " if bull_ else "Bear ") + ("pennant" if pennant else "flag")
                        top = seg["High"].max()
                        bot = seg["Low"].min()
                        brk = price > top if bull_ else price < bot
                        out.append(Pattern(
                            nm, "chart", 1 if bull_ else -1, D(prev.i), D(t),
                            [(D(prev.i), prev.price, "Pole start"), (D(last.i), last.price, "Pole top" if bull_ else "Pole low")],
                            [((D(prev.i), prev.price), (D(last.i), last.price), "pole"),
                             ((D(last.i + 1), seg["High"].iloc[0]), (D(t), seg["High"].iloc[-1]), "upper"),
                             ((D(last.i + 1), seg["Low"].iloc[0]), (D(t), seg["Low"].iloc[-1]), "lower")],
                            "confirmed" if brk else "forming",
                            price + (last.price - prev.price), bot if bull_ else top,
                            f"{abs(pole):.0%} pole in {pole_len} bars, {retr:.0%} retracement. Measured move = pole length."))

    # --- cup & handle / rounding bottom & top ---------------------------------
    win = df.iloc[max(0, t - 180): t + 1]
    if len(win) >= 60:
        cl = win["Close"].values
        j = int(cl.argmin())
        left = cl[:j]
        if j > 15 and len(cl) - j > 15 and len(left):
            li = int(left.argmax())
            rim_l = left[li]
            right = cl[j:]
            ri = j + int(right.argmax())
            rim_r = cl[ri]
            depth = (rim_l - cl[j]) / rim_l
            xs = np.arange(li, ri + 1)
            if 0.12 <= depth <= 0.40 and abs(rim_r - rim_l) / rim_l < 0.06 and ri - li >= 30:
                coef = np.polyfit(xs - xs.mean(), cl[li:ri + 1], 2)
                fit = np.polyval(coef, xs - xs.mean())
                r2 = 1 - ((cl[li:ri + 1] - fit) ** 2).sum() / ((cl[li:ri + 1] - cl[li:ri + 1].mean()) ** 2).sum()
                if coef[0] > 0 and r2 > 0.55:
                    handle = cl[ri:]
                    h_dd = (rim_r - handle.min()) / (rim_r - cl[j]) if len(handle) > 1 else 0
                    base_i = win.index
                    if len(handle) >= 5 and 0.05 < h_dd < 0.5:
                        nm, note = "Cup & handle", "U-shaped base with a shallow handle; breakout above the rim."
                    elif len(handle) < 5:
                        nm, note = "Rounding bottom", "Gradual U-shaped bottom (saucer)."
                    else:
                        nm = None
                    if nm:
                        rim = max(rim_l, rim_r)
                        out.append(Pattern(
                            nm, "chart", 1, base_i[li], base_i[-1],
                            [(base_i[li], rim_l, "Left rim"), (base_i[j], cl[j], "Cup low"), (base_i[ri], rim_r, "Right rim")],
                            [((base_i[li], rim), (base_i[-1], rim), "rim")],
                            "confirmed" if cl[-1] > rim else "forming", rim + (rim - cl[j]), handle.min() if len(handle) > 1 else cl[j],
                            note + f" Depth {depth:.0%}, fit R² {r2:.2f}."))
        k = int(cl.argmax())
        if 15 < k < len(cl) - 15:
            xs = np.arange(len(cl))
            coef = np.polyfit(xs - xs.mean(), cl, 2)
            fit = np.polyval(coef, xs - xs.mean())
            r2 = 1 - ((cl - fit) ** 2).sum() / ((cl - cl.mean()) ** 2).sum()
            rise = (cl[k] - cl[0]) / cl[0]
            fall = (cl[k] - cl[-1]) / cl[k]
            if coef[0] < 0 and r2 > 0.7 and rise > 0.12 and fall > 0.10:
                out.append(Pattern("Rounding top", "chart", -1, win.index[0], win.index[-1],
                                   [(win.index[k], cl[k], "Top")], [], "forming", None, cl[k],
                                   f"Inverted-U distribution top, fit R² {r2:.2f}."))
    return out


# ============================================================== harmonics

HARMONICS = {
    # name: (AB/XA range, BC/AB range, CD/BC range, AD/XA range)
    "Gartley": ((0.58, 0.66), (0.382, 0.886), (1.13, 1.618), (0.75, 0.82)),
    "Bat": ((0.382, 0.5), (0.382, 0.886), (1.618, 2.618), (0.85, 0.92)),
    "Butterfly": ((0.75, 0.82), (0.382, 0.886), (1.618, 2.24), (1.24, 1.66)),
    "Crab": ((0.382, 0.618), (0.382, 0.886), (2.24, 3.618), (1.55, 1.68)),
    "Deep crab": ((0.85, 0.92), (0.382, 0.886), (2.0, 3.618), (1.55, 1.68)),
    "Shark": ((0.0, 9.9), (1.13, 1.618), (1.618, 2.24), (0.85, 1.16)),
}


def detect_harmonics(df: pd.DataFrame, pivs: list[Pivot], t: int | None = None) -> list[Pattern]:
    n = len(df)
    t = n - 1 if t is None else t
    P = _alt([p for p in pivs if p.confirmed <= t])
    out = []
    idx = df.index
    if len(P) >= 5:
        X, A, B, C_, D_ = P[-5:]
        xa, ab, bc, cd = abs(A.price - X.price), abs(B.price - A.price), abs(C_.price - B.price), abs(D_.price - C_.price)
        if min(xa, ab, bc) > 0:
            r = (ab / xa, bc / ab, cd / bc, abs(A.price - D_.price) / xa)
            bull = D_.kind == "L"
            for name, rng in HARMONICS.items():
                if all(lo_ * 0.95 <= v <= hi_ * 1.05 for v, (lo_, hi_) in zip(r, rng)):
                    pts = [(idx[p.i], p.price, lbl) for p, lbl in zip((X, A, B, C_, D_), "XABCD")]
                    lines = [((idx[a_.i], a_.price), (idx[b_.i], b_.price), "leg")
                             for a_, b_ in zip((X, A, B, C_), (A, B, C_, D_))]
                    tgt = D_.price + (0.618 * cd if bull else -0.618 * cd)
                    out.append(Pattern(f"{'Bullish' if bull else 'Bearish'} {name}", "harmonic", 1 if bull else -1,
                                       idx[X.i], idx[D_.i], pts, lines,
                                       "confirmed" if (df["Close"].iloc[t] > D_.price) == bull else "forming",
                                       tgt, D_.price * (0.97 if bull else 1.03),
                                       "Ratios AB/XA {:.2f}, BC/AB {:.2f}, CD/BC {:.2f}, AD/XA {:.2f}. "
                                       "Reversal zone at D; target = 61.8% of CD.".format(*r)))
                    break
    if len(P) >= 4:
        A, B, C_, D_ = P[-4:]
        ab, bc, cd = abs(B.price - A.price), abs(C_.price - B.price), abs(D_.price - C_.price)
        if ab > 0 and bc > 0 and 0.382 <= bc / ab <= 0.886 and 0.9 <= cd / ab <= 1.1 and \
                0.8 <= (D_.i - C_.i) / max(B.i - A.i, 1) <= 1.25:
            bull = D_.kind == "L"
            out.append(Pattern(f"{'Bullish' if bull else 'Bearish'} AB=CD", "harmonic", 1 if bull else -1,
                               idx[A.i], idx[D_.i], [(idx[p.i], p.price, lbl) for p, lbl in zip((A, B, C_, D_), "ABCD")],
                               [((idx[a_.i], a_.price), (idx[b_.i], b_.price), "leg") for a_, b_ in
                                zip((A, B, C_), (B, C_, D_))], "forming",
                               D_.price + (0.618 * cd if bull else -0.618 * cd), None,
                               f"CD = {cd / ab:.2f}× AB in {D_.i - C_.i} vs {B.i - A.i} bars."))
    return out


# ================================================================ Elliott


def detect_elliott(df: pd.DataFrame, pivs: list[Pivot]) -> list[Pattern]:
    P = _alt(pivs)
    out = []
    idx = df.index
    for k in range(len(P) - 6, max(len(P) - 10, -1), -1):
        if k < 0:
            break
        w = P[k:k + 6]
        if len(w) < 6:
            continue
        bull = w[0].kind == "L"
        s = 1 if bull else -1
        p = [x.price * s for x in w]
        w1, w2, w3, w4, w5 = p[1] - p[0], p[1] - p[2], p[3] - p[2], p[3] - p[4], p[5] - p[4]
        ok = (w1 > 0 and w3 > 0 and w5 > 0 and w2 > 0 and w4 > 0 and
              p[2] > p[0] and              # wave 2 never retraces past the start of 1
              w3 > min(w1, w5) and         # wave 3 is not the shortest
              p[4] > p[1])                 # wave 4 does not overlap wave 1
        if ok:
            pts = [(idx[x.i], x.price, str(lbl)) for x, lbl in zip(w, ["0", "1", "2", "3", "4", "5"])]
            lines = [((idx[a.i], a.price), (idx[b.i], b.price), "wave") for a, b in zip(w[:-1], w[1:])]
            total = abs(w[5].price - w[0].price)
            out.append(Pattern(
                f"Elliott 5-wave impulse ({'up' if bull else 'down'})", "wave", -s, idx[w[0].i], idx[w[5].i], pts, lines,
                "confirmed" if k + 6 == len(P) else "completed earlier",
                w[5].price - s * 0.382 * total, w[5].price * (1 + 0.02 * s),
                f"Rules pass: wave 3 = {w3 / w1:.2f}× wave 1, no 1–4 overlap. A completed impulse is usually followed "
                "by an A-B-C correction (target = 38.2% retracement). Wave counts are subjective; treat as context."))
            break
    return out


# ============================================================ divergences


def detect_divergences(df: pd.DataFrame, pivs: list[Pivot], lookback: int = 120) -> list[Pattern]:
    out = []
    idx = df.index
    n = len(df)
    oscs = {"RSI": ind.rsi(df["Close"]), "MACD": ind.macd(df["Close"])[2]}
    P = [p for p in _alt(pivs) if p.i >= n - lookback]
    for kind in ("L", "H"):
        pts = [p for p in P if p.kind == kind][-2:]
        if len(pts) < 2:
            continue
        a, b = pts
        for name, o in oscs.items():
            oa, ob = o.iloc[a.i], o.iloc[b.i]
            if not (np.isfinite(oa) and np.isfinite(ob)):
                continue
            if kind == "L":
                if b.price < a.price and ob > oa:
                    nm, bias, note = f"Bullish {name} divergence", 1, "Lower low in price, higher low in momentum."
                elif b.price > a.price and ob < oa:
                    nm, bias, note = f"Hidden bullish {name} divergence", 1, "Higher low in price, lower low in momentum: trend continuation."
                else:
                    continue
            else:
                if b.price > a.price and ob < oa:
                    nm, bias, note = f"Bearish {name} divergence", -1, "Higher high in price, lower high in momentum."
                elif b.price < a.price and ob > oa:
                    nm, bias, note = f"Hidden bearish {name} divergence", -1, "Lower high in price, higher high in momentum: downtrend continuation."
                else:
                    continue
            out.append(Pattern(nm, "divergence", bias, idx[a.i], idx[b.i],
                               [(idx[a.i], a.price, ""), (idx[b.i], b.price, "")],
                               [((idx[a.i], a.price), (idx[b.i], b.price), "price")], "confirmed", None, None,
                               note + f" ({name} {oa:.1f} → {ob:.1f})"))
    return out


# =================================================================== gaps


def detect_gaps(df: pd.DataFrame, lookback: int = 120, min_gap: float = 0.01) -> list[Pattern]:
    out = []
    n = len(df)
    h, lo, c, v = df["High"], df["Low"], df["Close"], df["Volume"]
    vavg = v.rolling(20).mean()
    for i in range(max(21, n - lookback), n):
        up = lo.iloc[i] > h.iloc[i - 1] * (1 + min_gap)
        dn = h.iloc[i] < lo.iloc[i - 1] * (1 - min_gap)
        if not (up or dn):
            continue
        g_lo, g_hi = (h.iloc[i - 1], lo.iloc[i]) if up else (h.iloc[i], lo.iloc[i - 1])
        after = df.iloc[i + 1:]
        filled = (after["Low"] <= g_lo).any() if up else (after["High"] >= g_hi).any()
        prior = c.iloc[i - 20:i]
        trend = c.iloc[i - 1] / c.iloc[i - 20] - 1
        vol_hi = v.iloc[i] > 1.5 * vavg.iloc[i - 1]
        if up and c.iloc[i - 1] <= prior.max() * 1.001 and c.iloc[i] > prior.max() and vol_hi and trend < 0.1:
            kind = "Breakaway"
        elif (up and trend > 0.15) or (dn and trend < -0.15):
            kind = "Exhaustion" if v.iloc[i] > 2 * vavg.iloc[i - 1] else "Runaway"
        elif dn and c.iloc[i] < prior.min() and vol_hi and trend > -0.1:
            kind = "Breakaway"
        elif (up and trend > 0.05) or (dn and trend < -0.05):
            kind = "Runaway"
        else:
            kind = "Common"
        bias = (1 if up else -1) * (-1 if kind == "Exhaustion" else 1)
        out.append(Pattern(f"{kind} gap {'up' if up else 'down'}", "gap", bias, df.index[i - 1], df.index[i],
                           [(df.index[i], g_hi if up else g_lo, "")], [((df.index[i - 1], g_lo), (df.index[-1], g_hi), "zone")],
                           "filled" if filled else "open", None, None,
                           f"{(g_hi / g_lo - 1):.1%} gap, {'filled' if filled else 'still open'}."))
    return out


# ============================================================ levels / lines


def trendlines(df: pd.DataFrame, pivs: list[Pivot], max_age: int = 250) -> list[Pattern]:
    """Support line through the two most recent rising lows (and resistance
    through falling highs) that price still respects; counts touches."""
    out = []
    n = len(df)
    idx = df.index
    P = [p for p in _alt(pivs) if p.i >= n - max_age]
    for kind, nm, bias in (("L", "Rising support trendline", 1), ("H", "Falling resistance trendline", -1)):
        pts = [p for p in P if p.kind == kind]
        best = None
        for a_ in range(len(pts) - 1):
            for b_ in range(a_ + 1, len(pts)):
                a, b = pts[a_], pts[b_]
                if (kind == "L" and b.price <= a.price) or (kind == "H" and b.price >= a.price) or b.i - a.i < 10:
                    continue
                line = lambda x: _line_at((a.i, a.price), (b.i, b.price), x)  # noqa: E731
                seg = df.iloc[a.i:]
                xs = np.arange(a.i, n)
                lv = np.array([line(x) for x in xs])
                if kind == "L":
                    breaks = (seg["Close"].values < lv * 0.99).sum()
                    touches = (np.abs(seg["Low"].values - lv) / lv < 0.01).sum()
                else:
                    breaks = (seg["Close"].values > lv * 1.01).sum()
                    touches = (np.abs(seg["High"].values - lv) / lv < 0.01).sum()
                if breaks <= 2 and touches >= 3 and (best is None or touches > best[0]):
                    best = (touches, a, b, line(n - 1))
        if best:
            touches, a, b, now = best
            out.append(Pattern(nm, "level", bias, idx[a.i], idx[-1], [(idx[a.i], a.price, ""), (idx[b.i], b.price, "")],
                               [((idx[a.i], a.price), (idx[-1], now), "trend")], "active", None, now,
                               f"{touches} touches; now at ₹{now:,.2f}."))
    return out


def fibonacci(df: pd.DataFrame, pivs: list[Pivot]) -> dict | None:
    P = _alt(pivs)
    if len(P) < 2:
        return None
    a, b = P[-2], P[-1]
    lo_, hi_ = min(a.price, b.price), max(a.price, b.price)
    up = b.kind == "H"
    rng = hi_ - lo_
    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
    levels = {f"{r:.1%}".replace(".0%", "%"): (hi_ - r * rng if up else lo_ + r * rng) for r in ratios}
    return {"from": df.index[a.i], "to": df.index[b.i], "up": up, "levels": levels}


def sr_zones(df: pd.DataFrame, pivs: list[Pivot], width: float = 0.015, min_touches: int = 2) -> pd.DataFrame:
    prices = sorted(p.price for p in pivs)
    zones = []
    for p in prices:
        if zones and p <= zones[-1]["hi"] * (1 + width):
            z = zones[-1]
            z["hi"] = max(z["hi"], p)
            z["prices"].append(p)
        else:
            zones.append({"lo": p, "hi": p, "prices": [p]})
    last = df["Close"].iloc[-1]
    rows = [{"Low": z["lo"], "High": z["hi"], "Touches": len(z["prices"]),
             "Type": "Resistance" if np.mean(z["prices"]) > last else "Support",
             "Distance %": (np.mean(z["prices"]) / last - 1) * 100}
            for z in zones if len(z["prices"]) >= min_touches]
    return pd.DataFrame(rows).sort_values("Distance %", key=abs) if rows else pd.DataFrame(
        columns=["Low", "High", "Touches", "Type", "Distance %"])


# =============================================================== all-in-one


def scan(df: pd.DataFrame, candle_window: int = 10) -> dict:
    pivs = zigzag(df)
    cf = candlestick_frame(df)
    recent = cf.iloc[-candle_window:]
    candles = []
    for d, row in recent.iterrows():
        for k, v in row.items():
            if v:
                candles.append(Pattern(k, "candle", CANDLE_BIAS.get(k, 0), d, d,
                                       [(d, float(df.loc[d, "High"] if CANDLE_BIAS.get(k, 0) < 0 else df.loc[d, "Low"]), k)]))
    return {
        "pivots": pivs,
        "candles": candles,
        "candle_frame": cf,
        "chart": detect_chart_patterns(df, pivs),
        "harmonic": detect_harmonics(df, pivs),
        "wave": detect_elliott(df, pivs),
        "divergence": detect_divergences(df, pivs),
        "gap": detect_gaps(df),
        "trend": trendlines(df, pivs),
        "fib": fibonacci(df, pivs),
        "zones": sr_zones(df, pivs),
    }


def pattern_history(df: pd.DataFrame, horizon: int = 20, step: int = 3) -> pd.DataFrame:
    """Replay the chart/harmonic detectors through history, using only pivots
    confirmed at each date, and measure what happened next. Look-ahead free."""
    pivs = zigzag(df)
    fwd = df["Close"].shift(-horizon) / df["Close"] - 1
    seen = {}
    for t in range(200, len(df) - horizon, step):
        for p in detect_chart_patterns(df, pivs, t) + detect_harmonics(df, pivs, t):
            key = (p.name, p.start)
            if key in seen:
                continue
            seen[key] = (p, t)
    rows = []
    for (name, _), (p, t) in seen.items():
        r = fwd.iloc[t]
        worked = float(r > 0) if p.bias > 0 else float(r < 0) if p.bias < 0 else np.nan
        rows.append({"Pattern": name, "bias": p.bias, "ret": float(r), "worked": worked})
    if not rows:
        return pd.DataFrame(columns=["Times", f"Avg {horizon}d", "Worked %"])
    x = pd.DataFrame(rows)
    return x.groupby("Pattern").agg(Times=("ret", "size"), **{f"Avg {horizon}d": ("ret", "mean")},
                                     **{"Worked %": ("worked", "mean")})
