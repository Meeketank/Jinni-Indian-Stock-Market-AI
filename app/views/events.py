"""Sentiment & Events: market mood, news tone, event studies, calendar effects."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import components as ui
from app import storage
from app.theme import BLUE, DOWN, GOLD, MUTED, UP
from jinni import events as jevents
from jinni import seasonality as se
from jinni import sentiment as sm
from jinni.universe import BENCHMARK, INDICES, NIFTY50, display_name, normalize_symbol

st.title("Sentiment & Events")
tab_mood, tab_news, tab_lab, tab_cal = st.tabs(["Market mood", "News tone", "Event lab", "Calendar"])

# ------------------------------------------------------------------- mood
with tab_mood:
    with st.spinner("Reading the market's mood…"):
        idx = ui.prices(("^NSEI", "^INDIAVIX", "GC=F"), "2y")
        stk = ui.prices(tuple(NIFTY50), "2y")
    ui.data_source_notice(idx)
    nifty = idx.close["^NSEI"].dropna()
    vix = idx.close.get("^INDIAVIX")
    gold = idx.close.get("GC=F")
    mood = sm.market_mood(nifty, vix, stk.close, gold)
    m1, m2 = st.columns([1, 1], gap="medium")
    with m1:
        sc = mood["score"]
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=sc, number=dict(valueformat=".0f"),
            title=dict(text=f"Market mood<br><span style='font-size:14px'>{mood['label']}</span>"),
            gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 20, 40, 60, 80, 100]),
                       bar=dict(color=GOLD), bgcolor="#111823", borderwidth=0,
                       steps=[dict(range=[0, 20], color="rgba(239,68,68,.35)"), dict(range=[20, 40], color="rgba(239,68,68,.15)"),
                              dict(range=[60, 80], color="rgba(34,197,94,.15)"), dict(range=[80, 100], color="rgba(34,197,94,.35)")])))
        ui.plot(fig, 280)
    with m2:
        comp = pd.Series(mood["components"]).dropna()
        fig = go.Figure(go.Bar(x=comp.values, y=comp.index, orientation="h",
                               marker_color=[UP if v >= 60 else DOWN if v < 40 else GOLD for v in comp.values],
                               text=[f"{v:.0f}" for v in comp.values], textposition="outside"))
        fig.add_vline(x=50, line=dict(color=MUTED, dash="dot"))
        fig.update_layout(title="Components (percentile vs last year)", xaxis=dict(range=[0, 108]))
        fig.update_yaxes(autorange="reversed")
        ui.plot(fig, 280)
    hist = sm.mood_history(nifty, vix, stk.close).dropna()
    if len(hist) > 20:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Mood", line=dict(color=GOLD, width=1.6)))
        fig.add_trace(go.Scatter(x=nifty.loc[hist.index[0]:].index, y=nifty.loc[hist.index[0]:], name="NIFTY 50",
                                 yaxis="y2", line=dict(color=MUTED, width=1)))
        fig.add_hrect(y0=0, y1=20, fillcolor="rgba(239,68,68,.08)", line_width=0)
        fig.add_hrect(y0=80, y1=100, fillcolor="rgba(34,197,94,.08)", line_width=0)
        fig.update_layout(title="Mood history vs NIFTY", yaxis=dict(range=[0, 100]),
                          yaxis2=dict(overlaying="y", side="right", showgrid=False))
        ui.plot(fig, 300)
        fwd = nifty.shift(-21) / nifty - 1
        z = pd.DataFrame({"mood": hist, "fwd": fwd.reindex(hist.index)}).dropna()
        if len(z) > 60:
            z["Zone"] = pd.cut(z["mood"], [0, 20, 40, 60, 80, 100], labels=["Extreme fear", "Fear", "Neutral", "Greed",
                                                                              "Extreme greed"])
            t = z.groupby("Zone", observed=True)["fwd"].agg(["size", "mean", lambda x: (x > 0).mean()])
            t.columns = ["Days", "NIFTY next month", "Up %"]
            st.markdown("**What happened next** (NIFTY's following month, by mood zone):")
            st.dataframe(t, width="stretch", column_config={
                "NIFTY next month": st.column_config.NumberColumn(format="percent"),
                "Up %": st.column_config.NumberColumn(format="percent")})
    st.caption("Contrarian reading: extreme fear has often preceded better returns than extreme greed, but not "
               "reliably enough to time the market on its own.")

# ------------------------------------------------------------------- news
with tab_news:
    default = ["^NSEI"] + storage.watchlist()[:8]
    picks = st.multiselect("Tickers", ["^NSEI", "^BSESN"] + list(NIFTY50), default=default[:9],
                           format_func=display_name)
    extra = st.text_input("Other symbols", "", placeholder="e.g. IRCTC, HAL")
    picks = picks + [normalize_symbol(x) for x in extra.replace(",", " ").split() if x.strip()]
    rows, all_news = [], []
    for t in picks:
        n = jevents.headlines(t, limit=12)
        s = sm.score_news(n)
        rows.append({"Ticker": display_name(t), "Tone": s["score"], "Mood": s["label"], "Headlines": s["n"],
                     "Positive": s["pos"], "Negative": s["neg"]})
        if not n.empty:
            all_news.append(n.assign(Ticker=display_name(t), Score=n["Headline"].map(sm.score_headline)))
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Ticker").sort_values("Tone"), width="stretch", column_config={
            "Tone": st.column_config.ProgressColumn(min_value=-1, max_value=1, format="%.2f")})
    if all_news:
        nn = pd.concat(all_news).sort_values("Time", ascending=False)
        st.dataframe(nn[["Time", "Ticker", "Score", "Headline", "Tags", "Link"]], width="stretch", hide_index=True,
                     column_config={"Link": st.column_config.LinkColumn(display_text="open"),
                                    "Score": st.column_config.NumberColumn(format="%+.2f")})
    else:
        st.info("No headlines available right now (feeds unreachable, or demo mode).")
    st.caption("Headline tone is a snapshot of the last few days: Yahoo's free feed has no long archive, so it "
               "can't be back-tested here. Use the Event lab with dated news to test reactions.")

# -------------------------------------------------------------------- lab
with tab_lab:
    st.markdown("Ask *\"when X happens, does the market always react?\"* Pick what to measure and define X, either "
                "as a recurring rule or by pasting dates. JINNI checks every past occurrence and answers honestly.")
    l1, l2 = st.columns(2)
    with l1:
        what = st.selectbox("Measure", list(INDICES) + list(NIFTY50) + ["Other…"], format_func=lambda t:
                            t if t == "Other…" else display_name(t))
        if what == "Other…":
            what = normalize_symbol(st.text_input("Symbol", "IRCTC"))
        years = st.select_slider("History", ["3y", "5y", "10y"], value="10y")
    with l2:
        mode = st.radio("Event", ["Recurring rule", "Built-in event", "Paste dates"], horizontal=True)
        offset, label = 0, ""
        if mode == "Recurring rule":
            wd_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            r1, r2 = st.columns(2)
            nth = r1.selectbox("Which", ["1st", "2nd", "3rd", "4th", "Last", "Every"], index=1)
            wd = r2.selectbox("Day", wd_names, index=4)
            label = f"{nth} {wd} of the month" if nth != "Every" else f"Every {wd}"
        elif mode == "Built-in event":
            label = st.selectbox("Event", list(se.BUILTIN_EVENTS))
            st.caption("Built-in dates cover recent years; please verify any you rely on.")
        else:
            txt = st.text_area("Dates (one per line or comma-separated, e.g. days a leader made a market statement)",
                               "2025-04-02\n2025-04-09\n2025-07-07\n2025-08-06", height=110)
            label = st.text_input("Name this event", "Tariff announcements")
            offset = st.radio("Reaction measured on", ["Same session", "Next session (event after 3:30 pm IST)"],
                              horizontal=True, index=1,
                              help="US announcements land after Indian market close, so the reaction is the next session."
                              ) != "Same session"
            offset = int(offset)
        window = st.slider("Window around the event (trading days)", 1, 15, 5)
    pd_ = ui.prices((what,), years)
    ui.data_source_notice(pd_)
    close = pd_.close[what].dropna() if what in pd_.close else pd.Series(dtype=float)
    if close.empty:
        st.error("No data for that symbol.")
    if close.empty:
        dates = []
    elif mode == "Recurring rule":
        wdi = wd_names.index(wd)
        if nth == "Every":
            dates = list(close.index[close.index.weekday == wdi])
        else:
            dates = se.nth_weekday_dates(close.index, wdi, {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "Last": -1}[nth])
    elif mode == "Built-in event":
        dates, offset = se.BUILTIN_EVENTS[label]
    else:
        raw = [x.strip() for x in txt.replace(",", "\n").splitlines() if x.strip()]
        dates = pd.to_datetime(raw, errors="coerce").dropna()
    res = se.event_study(close, dates, offset, window, label) if not close.empty else None
    s = res.summary if res else {"n": 0}
    if res is None:
        pass
    elif not s["n"]:
        st.warning("None of those dates fall inside the price history.")
    else:
        st.markdown(f"#### {label} → {display_name(what)}")
        st.info(res.verdict, icon=":material/psychology:")
        ui.kpis([
            ("Events", f"{s['n']}", f"reaction: {'next' if offset else 'same'} session"),
            ("Up on the day", ui.pct(s["hit"], 0), f"normal days {s['others_hit']:.0%}"),
            ("Average move", f"{s['mean']:+.2f}%", f"normal {s['others_mean']:+.2f}%", ui.tone(s["mean"])),
            ("Typical size", f"{s['avg_abs']:.2f}%", f"normal {s['others_abs']:.2f}%"),
            ("Best / worst", f"{s['best']:+.1f}% / {s['worst']:+.1f}%"),
            ("Luck test (p)", f"{s['p_mean']:.2f}", "< 0.05 = unlikely to be chance"),
        ])
        e1, e2 = st.columns(2, gap="medium")
        with e1:
            if not res.path.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res.path.index, y=res.path["Avg %"], name="Average", line=dict(color=GOLD, width=2)))
                fig.add_trace(go.Scatter(x=res.path.index, y=res.path["Median %"], name="Median",
                                         line=dict(color=BLUE, width=1.5, dash="dot")))
                fig.add_vline(x=0, line=dict(color=MUTED, dash="dot"))
                fig.update_layout(title="Average path around the event (% vs day before)", xaxis_title="Trading days")
                ui.plot(fig, 300)
        with e2:
            t = res.table
            fig = go.Figure(go.Bar(x=t.index, y=t["Day-0 %"], marker_color=[UP if v >= 0 else DOWN for v in t["Day-0 %"]]))
            fig.update_layout(title="Reaction on each occurrence (%)")
            ui.plot(fig, 300)
        if len(res.by_year) > 1:
            st.markdown("**Consistency by year**")
            st.dataframe(res.by_year, width="stretch", column_config={
                "Up %": st.column_config.NumberColumn(format="percent"),
                "Avg day-0 %": st.column_config.NumberColumn(format="%+.2f")})
        with st.expander("Every occurrence"):
            st.dataframe(res.table, width="stretch")
    st.caption("If you test many rules, some will look significant by chance. Prefer effects that hold across "
               "years and have an economic reason to exist.")

# --------------------------------------------------------------- calendar
with tab_cal:
    what_c = st.selectbox("Market / stock", list(INDICES) + list(NIFTY50), format_func=display_name, key="cal_what")
    cl = ui.prices((what_c,), "10y").close[what_c].dropna()
    cal = se.calendar_effects(cl)
    b = cal["_base"].iloc[0]
    st.markdown(f"Baseline: average day {b['Avg %']:+.3f}%, up {b['Up %']:.0%} of days.")
    for key in ("Month", "Weekday", "Week of month"):
        t = cal[key].dropna(subset=["Avg %"])
        fig = go.Figure(go.Bar(x=t.index, y=t["Avg %"], marker_color=[UP if x >= 0 else DOWN for x in t["Avg %"]],
                               text=[f"{u:.0%}" for u in t["Up %"]], textposition="outside",
                               hovertemplate="%{x}: %{y:.3f}% avg<extra></extra>"))
        fig.update_layout(title=f"By {key.lower()}: average daily return % (labels = % of days up)")
        ui.plot(fig, 280)
    st.markdown("**Special days**")
    st.dataframe(cal["Special days"], width="stretch", column_config={
        "Up %": st.column_config.NumberColumn(format="percent"), "p-value": st.column_config.NumberColumn(format="%.3f")})
    st.markdown("**Coming up in the next 30 days**")
    up = se.upcoming(cl)
    st.dataframe(up, width="stretch", column_config={"Past up %": st.column_config.NumberColumn(format="percent")})
    st.caption("F&O expiry follows NSE's schedule: monthly contracts expired on the last Thursday until Aug 2025 "
               "and on the last Tuesday since Sep 2025.")
ui.disclaimer()
