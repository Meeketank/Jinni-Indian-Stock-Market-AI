"""Prediction Journal: every tracked prediction scored against what really happened."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import components as ui
from app import storage
from app.theme import BLUE, DOWN, GOLD, MUTED, UP
from jinni import forecast as fc
from jinni.universe import display_name

st.title("Journal")
st.markdown('<span class="hint">JINNI\'s real report card. Every prediction you track is checked against the '
            "actual price path after it was made. This is the only accuracy number that counts.</span>",
            unsafe_allow_html=True)

preds = storage.predictions()
if preds.empty:
    st.info("Nothing tracked yet. In Return Finder, use **Track top 5** or **Track this trade plan**.")
    ui.disclaimer()
    st.stop()

with st.spinner("Scoring predictions against prices…"):
    pdata = ui.prices(tuple(preds["ticker"].unique()), "1y")
rows = []
for r in preds.itertuples():
    if r.ticker not in pdata.close:
        ev = {"status": "No data", "exit_date": None, "ret": np.nan, "days": 0}
    else:
        ev = fc.evaluate_tracked(pdata.ohlcv(r.ticker), r.entry_date, r.entry, r.target_price,
                                 r.stop_price if pd.notna(r.stop_price) else None, int(r.horizon))
    rows.append({"id": r.id, "Stock": display_name(r.ticker), "Status": ev["status"], "Return": ev["ret"],
                 "Predicted": r.p_target, "Goal": f"+{r.target_pct:.0%}", "Days": int(r.horizon), "Grade": r.grade,
                 "From": r.entry_date, "Days held": ev["days"], "Entry ₹": r.entry, "Target ₹": r.target_price,
                 "Stop ₹": r.stop_price, "Qty": r.qty, "Type": "Plan" if r.source == "plan" else "Pick"})
j = pd.DataFrame(rows)
closed = j[j["Status"].isin(["Target hit", "Stop hit", "Expired"])]
hit = (closed["Status"] == "Target hit")
pnl = (j["Return"] * j["Entry ₹"] * j["Qty"]).where(j["Type"] == "Plan").sum()

ui.kpis([
    ("Tracked", f"{len(j)}", f"{(j['Status'] == 'Open').sum()} still open"),
    ("Resolved", f"{len(closed)}", "target, stop or expired"),
    ("Target hit rate", ui.pct(hit.mean(), 0) if len(closed) else "—", "actual", "up"),
    ("Predicted (avg)", ui.pct(closed["Predicted"].mean(), 0) if len(closed) else "—", "what JINNI said"),
    ("Avg return", ui.pct(closed["Return"].mean(), 2, sign=True) if len(closed) else "—", "per resolved trade",
     ui.tone(closed["Return"].mean()) if len(closed) else ""),
    ("Plan P&L", ui.inr(pnl), "tracked plans, open + closed", ui.tone(pnl)),
])
if 0 < len(closed) < 30:
    st.caption(f"Only {len(closed)} resolved predictions so far. Accuracy needs about 30+ before it means much.")

status_filter = st.multiselect("Status", sorted(j["Status"].unique()), default=sorted(j["Status"].unique()))
show = j[j["Status"].isin(status_filter)]
st.dataframe(show.set_index("id"), width="stretch", column_config={
    "Predicted": st.column_config.ProgressColumn("P(target)", min_value=0, max_value=1, format="percent"),
    "Return": st.column_config.NumberColumn(format="percent"),
    "Entry ₹": st.column_config.NumberColumn(format="₹%.2f"),
    "Target ₹": st.column_config.NumberColumn(format="₹%.2f"),
    "Stop ₹": st.column_config.NumberColumn(format="₹%.2f"),
})

if len(closed) >= 5:
    ui.section("Calibration")
    b = closed.assign(bucket=pd.cut(closed["Predicted"], [0, 0.3, 0.45, 0.6, 0.75, 1.0]))
    cal = b.groupby("bucket", observed=True).agg(Predicted=("Predicted", "mean"),
                                                 Actual=("Status", lambda s: (s == "Target hit").mean()),
                                                 Count=("Status", "size"))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cal.index.astype(str), y=cal["Actual"], name="Actual", marker_color=BLUE,
                         text=cal["Count"].map(lambda c: f"n={c}"), textposition="outside"))
    fig.add_trace(go.Scatter(x=cal.index.astype(str), y=cal["Predicted"], name="Predicted", mode="lines+markers",
                             line=dict(color=GOLD)))
    fig.update_layout(title="When JINNI said X%, how often did it happen?", yaxis=dict(tickformat=".0%", range=[0, 1]))
    ui.plot(fig, 320)
    by_g = closed.groupby("Grade").agg(Trades=("Status", "size"), Hit=("Status", lambda s: (s == "Target hit").mean()),
                                       Avg_return=("Return", "mean"))
    st.dataframe(by_g, width="stretch", column_config={
        "Hit": st.column_config.NumberColumn("Hit rate", format="percent"),
        "Avg_return": st.column_config.NumberColumn("Avg return", format="percent")})

with st.expander("Manage", icon=":material/delete:"):
    rid = st.selectbox("Remove a prediction", j["id"],
                       format_func=lambda i: f"#{i} " + " · ".join(str(x) for x in j.loc[j['id'] == i, ['Stock', 'From', 'Goal']].values[0]))
    if st.button("Delete"):
        storage.delete_prediction(rid)
        st.rerun()
    st.download_button("Export journal CSV", j.to_csv(index=False).encode(), "jinni_journal.csv", "text/csv")
ui.disclaimer()
