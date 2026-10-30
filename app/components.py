"""Reusable Streamlit building blocks and cached data access."""

from __future__ import annotations

import html
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from jinni import data as jdata
from jinni.universe import DEFAULT_RISK_FREE, display_name

from .theme import DOWN, MUTED, UP

PLOT_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}


# ----------------------------------------------------------- cached data


@st.cache_data(ttl=900, show_spinner=False)
def prices(tickers: tuple[str, ...], period: str = "2y") -> jdata.PriceData:
    return jdata.get_prices(list(tickers), period)


@st.cache_data(ttl=3600, show_spinner=False)
def info(ticker: str) -> dict:
    return jdata.fetch_info(ticker)


def settings() -> dict:
    s = st.session_state
    return {
        "rf": s.get("rf", DEFAULT_RISK_FREE),
        "capital": s.get("capital", 1_000_000.0),
        "risk_pct": s.get("risk_pct", 0.01),
        "conf": s.get("conf", 0.95),
    }


def rules() -> dict:
    """Pre-trade risk rules: session > saved > defaults."""
    from jinni.riskcheck import DEFAULT_RULES

    from . import storage

    if "rules" not in st.session_state:
        saved = storage.load_setting("rules", {}) or {}
        st.session_state["rules"] = {**DEFAULT_RULES, **saved}
    return st.session_state["rules"]


# ----------------------------------------------------------- formatting


def inr(x: float, decimals: int = 0) -> str:
    """Indian-style grouping with lakh/crore suffix for big numbers."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:,.2f} Cr"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:,.2f} L"
    return f"{sign}₹{a:,.{decimals}f}"


def pct(x: float, decimals: int = 2, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x * 100:+.{decimals}f}%" if sign else f"{x * 100:.{decimals}f}%"


def num(x: float, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:,.{decimals}f}"


def tone(x: float) -> str:
    try:
        return "up" if x > 0 else "down" if x < 0 else ""
    except TypeError:
        return ""


# ----------------------------------------------------------- widgets


def kpis(items: list[tuple]) -> None:
    """Responsive KPI grid. Items: (label, value, sub=None, tone=None)."""
    cells = []
    for it in items:
        label, value = it[0], it[1]
        sub = it[2] if len(it) > 2 else None
        cls = it[3] if len(it) > 3 and it[3] else ""
        sub_html = f'<div class="sub {cls}">{html.escape(str(sub))}</div>' if sub else ""
        val_cls = cls if not sub else ""
        cells.append(
            f'<div class="kpi"><div class="lbl">{html.escape(label)}</div>'
            f'<div class="val {val_cls}">{html.escape(str(value))}</div>{sub_html}</div>'
        )
    st.markdown(f'<div class="kpi-grid">{"".join(cells)}</div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)


def rating_badge(rating: str) -> str:
    cls = {"STRONG BUY": "b-sbuy", "BUY": "b-buy", "HOLD": "b-hold", "SELL": "b-sell",
           "STRONG SELL": "b-ssell"}.get(rating, "b-hold")
    return f'<span class="badge {cls}">{html.escape(rating)}</span>'


def data_source_notice(pdata: jdata.PriceData) -> None:
    if pdata.source == "demo":
        st.markdown(
            '<span class="badge b-demo">DEMO DATA</span> '
            '<span class="hint">Live market data is unavailable right now, so a synthetic market is shown. '
            "Numbers are illustrative only.</span>",
            unsafe_allow_html=True,
        )
    if pdata.missing:
        st.caption("Skipped (no data): " + ", ".join(display_name(t) for t in pdata.missing))


def ticker_tape(rows: list[tuple[str, float, float]]) -> None:
    items = "".join(
        f'<span class="item"><b>{html.escape(n)}</b>{p:,.2f} '
        f'<span class="{"up" if c >= 0 else "down"}">{c * 100:+.2f}%</span></span>'
        for n, p, c in rows
    )
    st.markdown(f'<div class="tape"><div class="tape-inner">{items}{items}</div></div>', unsafe_allow_html=True)


def plot(fig: go.Figure, height: int = 380) -> None:
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


def colored_bar(values: pd.Series, title: str = "", height: int = 320, fmt: str = ".1%",
                horizontal: bool = True) -> go.Figure:
    colors = [UP if v >= 0 else DOWN for v in values]
    if horizontal:
        fig = go.Figure(go.Bar(x=values.values, y=values.index, orientation="h", marker_color=colors,
                               texttemplate=f"%{{x:{fmt}}}", textposition="outside", cliponaxis=False))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickformat=fmt)
    else:
        fig = go.Figure(go.Bar(x=values.index, y=values.values, marker_color=colors))
        fig.update_yaxes(tickformat=fmt)
    fig.update_layout(title=title, showlegend=False)
    return fig


def disclaimer() -> None:
    st.markdown(
        '<div class="disclaimer">JINNI is an analytics and education tool, not investment advice and not a '
        "SEBI-registered adviser. Market data from Yahoo Finance may be delayed. Model outputs are probabilistic "
        "and past performance does not guarantee future results.</div>",
        unsafe_allow_html=True,
    )


def muted(text: str) -> None:
    st.markdown(f'<span class="hint" style="color:{MUTED}">{html.escape(text)}</span>', unsafe_allow_html=True)
