"""Calendar effects and event studies: "does the market always move when X happens?"

* Calendar effects: weekday, month, week-of-month, turn-of-month, pre-holiday,
  F&O expiry day, each against all other days with a significance test.
* Event studies: give a rule ("2nd Friday of every month"), a built-in event
  set (Budget, RBI policy, US Fed decisions, election results, Diwali
  Muhurat), or paste your own dates (e.g. days a politician made a
  statement). JINNI measures the reaction (same day or next session, plus a
  window around it) and answers in plain words, including how often it held
  and whether it could just be luck.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------- built-in event calendars
# Dates are decision/announcement days. US Fed decisions are announced after
# Indian market hours, so their reaction is measured on the next NSE session.

UNION_BUDGET = ["2016-02-29", "2017-02-01", "2018-02-01", "2019-02-01", "2019-07-05", "2020-02-01", "2021-02-01",
                "2022-02-01", "2023-02-01", "2024-02-01", "2024-07-23", "2025-02-01", "2026-02-01"]
FOMC = [
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
]
RBI_MPC = [
    "2022-02-10", "2022-04-08", "2022-05-04", "2022-06-08", "2022-08-05", "2022-09-30", "2022-12-07",
    "2023-02-08", "2023-04-06", "2023-06-08", "2023-08-10", "2023-10-06", "2023-12-08",
    "2024-02-08", "2024-04-05", "2024-06-07", "2024-08-08", "2024-10-09", "2024-12-06",
    "2025-02-07", "2025-04-09", "2025-06-06", "2025-08-06", "2025-10-01", "2025-12-05",
]
ELECTION_RESULTS = ["2014-05-16", "2019-05-23", "2024-06-04"]
US_ELECTION = ["2016-11-09", "2020-11-04", "2024-11-06"]
MUHURAT = ["2019-10-27", "2020-11-14", "2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01", "2025-10-21"]

BUILTIN_EVENTS = {
    "Union Budget day": (UNION_BUDGET, 0),
    "RBI monetary policy decision": (RBI_MPC, 0),
    "US Fed (FOMC) decision → next NSE session": (FOMC, 1),
    "Indian general election results": (ELECTION_RESULTS, 0),
    "US presidential election result": (US_ELECTION, 0),
    "Diwali Muhurat trading": (MUHURAT, 0),
}
# reaction offset: 0 = same session, 1 = the next trading session


# ------------------------------------------------------------ date rules


def nth_weekday_dates(index: pd.DatetimeIndex, weekday: int, n: int) -> list[pd.Timestamp]:
    """n-th (1..5) or last (n=-1) given weekday (0=Mon) of each month, calendar-based;
    if that day is a holiday the next trading day is used."""
    out = []
    months = pd.period_range(index.min(), index.max(), freq="M")
    for m in months:
        days = pd.date_range(m.start_time, m.end_time, freq="D")
        wd = days[days.weekday == weekday]
        if n == -1:
            d = wd[-1]
        elif len(wd) >= n:
            d = wd[n - 1]
        else:
            continue
        pos = index.searchsorted(d)
        if pos < len(index) and (index[pos] - d).days <= 3:
            out.append(index[pos])
    return out


def monthly_expiry_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """NSE monthly F&O expiry: last Thursday until Aug 2025, last Tuesday from
    Sep 2025 (SEBI/NSE change); a holiday moves it to the previous session."""
    out = []
    for m in pd.period_range(index.min(), index.max(), freq="M"):
        wd = 3 if m.start_time < pd.Timestamp("2025-09-01") else 1
        days = pd.date_range(m.start_time, m.end_time, freq="D")
        d = days[days.weekday == wd][-1]
        pos = index.searchsorted(d, side="right") - 1
        if pos >= 0 and index[pos].month == d.month and (d - index[pos]).days <= 4:
            out.append(index[pos])
    return out


def pre_holiday_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Sessions followed by a non-weekend market holiday (derived from the data)."""
    out = []
    for a, b in zip(index[:-1], index[1:]):
        bdays = np.busday_count(a.date(), b.date())
        if bdays > 1:
            out.append(a)
    return out


def turn_of_month_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(index, index=index)
    g = s.groupby([index.year, index.month])
    out = set(g.tail(1).values) | set(g.head(3).values)
    return sorted(pd.Timestamp(x) for x in out)


# ------------------------------------------------------------ statistics


@dataclass
class EventResult:
    table: pd.DataFrame          # one row per event
    summary: dict
    path: pd.DataFrame           # average cumulative return around events
    verdict: str
    by_year: pd.DataFrame


def _snap(index: pd.DatetimeIndex, dates, offset: int) -> list[int]:
    pos = []
    for d in pd.to_datetime(list(dates)):
        p = index.searchsorted(d)
        if p >= len(index) or (index[p] - d).days > 4:
            continue
        p += offset
        if 0 < p < len(index):
            pos.append(p)
    return sorted(set(pos))


def event_study(close: pd.Series, dates, offset: int = 0, window: int = 5, label: str = "event") -> EventResult:
    """Reaction of ``close`` to events. Day-0 return = close(event)/close(previous)-1."""
    close = close.dropna()
    idx = close.index
    r = close.pct_change()
    pos = _snap(idx, dates, offset)
    rows = []
    for p in pos:
        pre = close.iloc[max(p - window - 1, 0)]
        rows.append({
            "Date": idx[p], "Day-0 %": r.iloc[p] * 100,
            f"{window}d before %": (close.iloc[p - 1] / pre - 1) * 100 if p - window - 1 >= 0 else np.nan,
            f"{window}d after %": (close.iloc[min(p + window, len(close) - 1)] / close.iloc[p] - 1) * 100
            if p + 1 < len(close) else np.nan,
            "Abs move %": abs(r.iloc[p]) * 100,
        })
    tbl = pd.DataFrame(rows).set_index("Date") if rows else pd.DataFrame()
    others = r.drop(index=[idx[p] for p in pos], errors="ignore").dropna()
    ev = r.iloc[pos].dropna() if pos else pd.Series(dtype=float)
    n = len(ev)
    summary = {"n": n, "label": label}
    if n:
        t_stat, p_mean = stats.ttest_ind(ev, others, equal_var=False) if n > 1 else (np.nan, np.nan)
        up = int((ev > 0).sum())
        binom = stats.binomtest(max(up, n - up), n, p=max((others > 0).mean(), 1 - (others > 0).mean())
                                if len(others) else 0.5, alternative="greater").pvalue if n >= 3 else np.nan
        _, p_vol = stats.mannwhitneyu(ev.abs(), others.abs(), alternative="greater") if n >= 3 else (np.nan, np.nan)
        summary.update({
            "mean": ev.mean() * 100, "median": ev.median() * 100, "others_mean": others.mean() * 100,
            "up": up, "down": n - up, "hit": up / n, "others_hit": float((others > 0).mean()),
            "avg_abs": ev.abs().mean() * 100, "others_abs": others.abs().mean() * 100,
            "p_mean": float(p_mean), "p_direction": float(binom), "p_bigger_moves": float(p_vol),
            "best": ev.max() * 100, "worst": ev.min() * 100,
        })
    # average path from -window to +window
    paths = []
    for p in pos:
        if p - window < 0 or p + window >= len(close):
            continue
        seg = close.iloc[p - window: p + window + 1].values
        paths.append(seg / close.iloc[p - 1] - 1)
    path = pd.DataFrame({"Avg %": np.mean(paths, axis=0) * 100 if paths else [],
                         "Median %": np.median(paths, axis=0) * 100 if paths else []},
                        index=range(-window, window + 1) if paths else [])
    by_year = tbl.groupby(tbl.index.year)["Day-0 %"].agg(["count", "mean", lambda x: (x > 0).mean()]) \
        .rename(columns={"count": "Events", "mean": "Avg day-0 %", "<lambda_0>": "Up %"}) if len(tbl) else pd.DataFrame()
    return EventResult(tbl, summary, path, verdict_text(summary), by_year)


def verdict_text(s: dict) -> str:
    n = s.get("n", 0)
    if n == 0:
        return "No events found in the price history."
    if n < 5:
        caution = f"Only {n} events: far too few to call it a pattern."
    elif n < 15:
        caution = f"{n} events: a small sample, so read this as a hint, not a rule."
    else:
        caution = f"{n} events."
    hit = s["hit"]
    direction = "up" if hit >= 0.5 else "down"
    share = max(hit, 1 - hit)
    always = ("Yes, every single time" if share == 1 else "Mostly" if share >= 0.75 else
              "No: it's mixed" if share < 0.6 else "Somewhat")
    sig_dir = s.get("p_direction", 1) < 0.05
    sig_mean = s.get("p_mean", 1) < 0.05
    sig_vol = s.get("p_bigger_moves", 1) < 0.05
    parts = [f"Does it always go {direction}? {always}: {direction} on {share:.0%} of these days "
             f"(normal days: {s['others_hit'] if direction == 'up' else 1 - s['others_hit']:.0%})."]
    parts.append(f"Average move {s['mean']:+.2f}% vs {s['others_mean']:+.2f}% on other days"
                 + (" (statistically significant)." if sig_mean else " (could easily be chance)."))
    if sig_vol:
        parts.append(f"Moves are bigger than usual ({s['avg_abs']:.2f}% vs {s['others_abs']:.2f}%): the event "
                     "matters for volatility even if direction is unpredictable.")
    elif not sig_dir and not sig_mean:
        parts.append("Overall: no reliable edge. Treat this day like any other.")
    if sig_dir and n >= 10:
        parts.append("The direction bias is unlikely to be pure luck (p < 0.05).")
    parts.append(caution)
    return " ".join(parts)


def calendar_effects(close: pd.Series) -> dict[str, pd.DataFrame]:
    """Average return / hit rate by weekday, month, week-of-month, and special days."""
    close = close.dropna()
    r = close.pct_change().dropna()
    idx = r.index
    base_mean, base_hit = r.mean(), (r > 0).mean()

    def table(groups, order=None):
        g = r.groupby(groups)
        t = pd.DataFrame({"Days": g.size(), "Avg %": g.mean() * 100, "Up %": g.apply(lambda x: (x > 0).mean()),
                          "p-value": g.apply(lambda x: stats.ttest_ind(x, r.drop(x.index), equal_var=False).pvalue
                                             if len(x) > 2 else np.nan)})
        if order is not None:
            t = t.reindex(order)
        t["Signif. (after testing all rows)"] = t["p-value"] * len(t) < 0.05  # Bonferroni
        return t

    out = {
        "Weekday": table(idx.day_name(), ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
        "Month": table(idx.month_name(), ["January", "February", "March", "April", "May", "June", "July",
                                          "August", "September", "October", "November", "December"]),
        "Week of month": table(((idx.day - 1) // 7 + 1).map(lambda w: f"Week {w}")),
    }
    special = {}
    for name, dates in (("Turn of month", turn_of_month_dates(close.index)),
                        ("Monthly F&O expiry", monthly_expiry_dates(close.index)),
                        ("Day before a holiday", pre_holiday_dates(close.index))):
        ev = r.reindex(pd.DatetimeIndex(dates)).dropna()
        oth = r.drop(ev.index, errors="ignore")
        special[name] = {"Days": len(ev), "Avg %": ev.mean() * 100, "Up %": (ev > 0).mean(),
                         "p-value": stats.ttest_ind(ev, oth, equal_var=False).pvalue if len(ev) > 2 else np.nan}
    sp = pd.DataFrame(special).T
    sp["Signif. (after testing all rows)"] = sp["p-value"] * len(sp) < 0.05
    out["Special days"] = sp
    out["_base"] = pd.DataFrame({"Avg %": [base_mean * 100], "Up %": [base_hit]}, index=["All days"])
    return out


def upcoming(close: pd.Series, days_ahead: int = 30) -> pd.DataFrame:
    """Recurring calendar events in the next ``days_ahead`` days with their past average reaction."""
    close = close.dropna()
    today = pd.Timestamp.today().normalize()
    fut = pd.bdate_range(today + pd.Timedelta(days=1), today + pd.Timedelta(days=days_ahead))
    ext = close.index.append(fut)
    rows = []
    hist_exp = monthly_expiry_dates(close.index)
    for d in monthly_expiry_dates(ext):
        if d in fut:
            er = event_study(close, hist_exp, 0, 3, "expiry").summary
            rows.append({"Date": d, "Event": "Monthly F&O expiry", "Past avg %": er.get("mean"),
                         "Past up %": er.get("hit"), "Past events": er.get("n")})
    for d in turn_of_month_dates(ext):
        if d in fut and d.day <= 3:
            er = event_study(close, turn_of_month_dates(close.index), 0, 3, "tom").summary
            rows.append({"Date": d, "Event": "Turn of month", "Past avg %": er.get("mean"), "Past up %": er.get("hit"),
                         "Past events": er.get("n")})
            break
    for name, (dates, off) in BUILTIN_EVENTS.items():
        for d in pd.to_datetime(dates):
            if today < d <= today + pd.Timedelta(days=days_ahead):
                er = event_study(close, dates, off, 3, name).summary
                rows.append({"Date": d, "Event": name, "Past avg %": er.get("mean"), "Past up %": er.get("hit"),
                             "Past events": er.get("n")})
    return pd.DataFrame(rows).sort_values("Date").set_index("Date") if rows else pd.DataFrame(
        columns=["Event", "Past avg %", "Past up %", "Past events"])
