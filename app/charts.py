"""Pro chart: candles (or Heikin-Ashi), any indicators, and auto-drawn patterns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jinni import patterns as pt
from jinni import ta

from .theme import BLUE, DOWN, GOLD, MUTED, SERIES, UP

LAYERS = ["Candlestick patterns", "Chart patterns", "Harmonics & Elliott", "Trendlines", "Fibonacci",
          "Support/resistance zones", "Divergences", "Gaps"]

_ROLE_STYLE = {"neckline": dict(color=GOLD, dash="dash", width=1.5), "upper": dict(color="#E86AF0", width=1.5),
               "lower": dict(color="#2EC4B6", width=1.5), "pole": dict(color=MUTED, dash="dot", width=1),
               "rim": dict(color=GOLD, dash="dash", width=1.5), "leg": dict(color="#A78BFA", width=1.5),
               "wave": dict(color="#FF8A4C", width=1.5), "trend": dict(color=BLUE, width=1.6),
               "price": dict(color="#FFD978", width=2)}


def build(df: pd.DataFrame, overlays: list[str], panels: list[str], layers: list[str], scan: dict | None,
          heikin: bool = False, view: int = 252) -> go.Figure:
    full = df
    d = df.iloc[-view:]
    start = d.index[0]
    rows = 1 + len(panels)
    heights = [0.62] + [0.38 / max(len(panels), 1)] * len(panels) if panels else [1.0]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.025,
                        subplot_titles=[""] + panels)
    cd = ta.heikin_ashi(full).iloc[-view:] if heikin else d
    fig.add_trace(go.Candlestick(x=d.index, open=cd["Open"], high=cd["High"], low=cd["Low"], close=cd["Close"],
                                 name="Heikin-Ashi" if heikin else "Price", increasing_line_color=UP,
                                 decreasing_line_color=DOWN, increasing_fillcolor=UP, decreasing_fillcolor=DOWN), 1, 1)

    # ---------------- overlays
    k = 0
    for name in overlays:
        out = ta.compute(full, name)
        frame = out.to_frame(name) if isinstance(out, pd.Series) else out
        frame = frame.iloc[-view:]
        if name == "Ichimoku":
            fig.add_trace(go.Scatter(x=frame.index, y=frame["Span A"], line=dict(width=0.8, color=UP), name="Span A"), 1, 1)
            fig.add_trace(go.Scatter(x=frame.index, y=frame["Span B"], line=dict(width=0.8, color=DOWN), name="Span B",
                                     fill="tonexty", fillcolor="rgba(138,148,166,0.10)"), 1, 1)
            for c, col in (("Tenkan", BLUE), ("Kijun", GOLD)):
                fig.add_trace(go.Scatter(x=frame.index, y=frame[c], line=dict(width=1.1, color=col), name=c), 1, 1)
            continue
        if name == "Parabolic SAR":
            fig.add_trace(go.Scatter(x=frame.index, y=frame.iloc[:, 0], mode="markers", name="PSAR",
                                     marker=dict(size=3, color=GOLD)), 1, 1)
            continue
        if name == "SuperTrend":
            st_ = ta.supertrend(full).iloc[-view:]
            for dirn, col in ((1, UP), (-1, DOWN)):
                y = st_["SuperTrend"].where(st_["ST_dir"] == dirn)
                fig.add_trace(go.Scatter(x=st_.index, y=y, line=dict(color=col, width=1.6),
                                         name=f"SuperTrend {'up' if dirn == 1 else 'down'}", connectgaps=False), 1, 1)
            continue
        for j, c in enumerate(frame.columns):
            col = SERIES[(k + j) % len(SERIES)]
            band = len(frame.columns) >= 2 and j == len(frame.columns) - 1 and \
                any(x in c for x in ("lower", "Lower", "Env lower"))
            fig.add_trace(go.Scatter(x=frame.index, y=frame[c], name=c, line=dict(width=1.2, color=col),
                                     fill="tonexty" if band else None,
                                     fillcolor="rgba(138,148,166,0.06)" if band else None), 1, 1)
        k += len(frame.columns)

    # ---------------- panels
    for r, name in enumerate(panels, start=2):
        out = ta.compute(full, name)
        frame = (out.to_frame(name) if isinstance(out, pd.Series) else out).iloc[-view:]
        for j, c in enumerate(frame.columns):
            if c.lower() in ("hist", "macd_hist") or c == "Hist":
                fig.add_trace(go.Bar(x=frame.index, y=frame[c], name=c, marker_color=np.where(frame[c] >= 0, UP, DOWN),
                                     opacity=0.7), r, 1)
            else:
                fig.add_trace(go.Scatter(x=frame.index, y=frame[c], name=c,
                                         line=dict(width=1.2, color=SERIES[(j + 1) % len(SERIES)])), r, 1)
        guides = {"RSI 14": (30, 70), "RSI 2": (10, 90), "Stochastic": (20, 80), "Stochastic RSI": (20, 80),
                  "MFI 14": (20, 80), "Williams %R": (-80, -20), "CCI 20": (-100, 100), "Ultimate oscillator": (30, 70),
                  "Connors RSI": (10, 90), "Schaff trend cycle": (25, 75), "Choppiness": (38.2, 61.8),
                  "ADX / DMI": (20, 25), "Bollinger %B": (0, 1), "Z-score 20": (-2, 2), "CMO": (-50, 50)}
        for y in guides.get(name, ()):
            fig.add_hline(y=y, line=dict(color=MUTED, dash="dot", width=1), row=r, col=1)

    lo, hi = d["Low"].min(), d["High"].max()
    # ---------------- patterns
    if scan:
        def in_view(ts):
            return ts is not None and ts >= start

        def draw(p: pt.Pattern, color):
            for (a, b, role) in p.lines:
                (d1, p1), (d2, p2) = a, b
                if d2 < start:
                    continue
                style = _ROLE_STYLE.get(role, dict(color=color, width=1.2))
                fig.add_trace(go.Scatter(x=[max(d1, start), d2], y=[p1 if d1 >= start else
                                                                   p1 + (p2 - p1) * ((start - d1) / (d2 - d1) if d2 != d1 else 0),
                                                                   p2],
                                         mode="lines", line=style, showlegend=False, hoverinfo="skip"), 1, 1)
            pts = [(x, y, lbl) for x, y, lbl in p.points if in_view(x)]
            if pts:
                fig.add_trace(go.Scatter(x=[x for x, _, _ in pts], y=[y for _, y, _ in pts], mode="markers+text",
                                         text=[lbl for _, _, lbl in pts], textposition="top center",
                                         textfont=dict(size=9, color=color), marker=dict(size=6, color=color),
                                         name=p.name, hovertext=p.note, showlegend=False), 1, 1)
            anchor = p.end if in_view(p.end) else d.index[-1]
            y = max([y for _, y, _ in p.points] or [d["High"].iloc[-1]])
            fig.add_annotation(x=anchor, y=y, text=f"{p.name}{' ✓' if p.status == 'confirmed' else ''}",
                               showarrow=True, arrowhead=0, ax=0, ay=-26, font=dict(size=10, color=color),
                               bgcolor="rgba(10,14,20,0.75)", bordercolor=color, borderwidth=1, row=1, col=1)
            if p.target and in_view(p.end):
                fig.add_hline(y=p.target, line=dict(color=color, dash="dot", width=1), row=1, col=1,
                              annotation_text=f"{p.name} target", annotation_font=dict(size=9, color=color))

        col_of = lambda p: UP if p.bias > 0 else DOWN if p.bias < 0 else GOLD  # noqa: E731
        if "Chart patterns" in layers:
            for p in scan["chart"]:
                draw(p, col_of(p))
        if "Harmonics & Elliott" in layers:
            for p in scan["harmonic"] + scan["wave"]:
                draw(p, col_of(p))
        if "Trendlines" in layers:
            for p in scan["trend"]:
                draw(p, BLUE)
        if "Divergences" in layers:
            for p in scan["divergence"]:
                draw(p, col_of(p))
        if "Gaps" in layers:
            for p in scan["gap"]:
                if p.status == "open" and in_view(p.end):
                    (d1, lo_), (_, hi_) = p.lines[0][0], p.lines[0][1]
                    fig.add_shape(type="rect", x0=p.end, x1=d.index[-1], y0=lo_, y1=hi_, line=dict(width=0),
                                  fillcolor="rgba(245,184,65,0.12)", row=1, col=1)
        if "Support/resistance zones" in layers and not scan["zones"].empty:
            zv = scan["zones"][(scan["zones"]["High"] >= lo * 0.96) & (scan["zones"]["Low"] <= hi * 1.04)]
            for _, z in zv.head(4).iterrows():
                c = "rgba(34,197,94,0.07)" if z["Type"] == "Support" else "rgba(239,68,68,0.07)"
                lo_, hi_ = z["Low"] * 0.998, z["High"] * 1.002
                fig.add_shape(type="rect", x0=start, x1=d.index[-1], y0=lo_, y1=hi_, line=dict(width=0), fillcolor=c,
                              row=1, col=1)
        if "Fibonacci" in layers and scan["fib"]:
            for lbl, lv in scan["fib"]["levels"].items():
                fig.add_hline(y=lv, line=dict(color="#A78BFA", dash="dot", width=0.8), row=1, col=1,
                              annotation_text=f"Fib {lbl}", annotation_position="left",
                              annotation_font=dict(size=9, color="#A78BFA"))
        if "Candlestick patterns" in layers:
            cf = scan["candle_frame"].iloc[-view:]
            for name in cf.columns:
                hits = cf.index[cf[name]]
                if not len(hits):
                    continue
                bias = pt.CANDLE_BIAS.get(name, 0)
                if bias == 0:
                    continue
                y = d.loc[hits, "Low"] * 0.985 if bias > 0 else d.loc[hits, "High"] * 1.015
                fig.add_trace(go.Scatter(x=hits, y=y, mode="markers", name=name, showlegend=False,
                                         marker=dict(symbol="triangle-up" if bias > 0 else "triangle-down", size=7,
                                                     color=UP if bias > 0 else DOWN, opacity=0.8),
                                         hovertemplate=f"{name}<extra></extra>"), 1, 1)

    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x unified", legend=dict(font=dict(size=10)))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    lo, hi = d["Low"].min(), d["High"].max()
    fig.update_yaxes(range=[lo * 0.96, hi * 1.04], row=1, col=1)
    for a in fig.layout.annotations:
        if a.text in panels:
            a.font = dict(size=11, color=MUTED)
            a.x = 0
            a.xanchor = "left"
    return fig
