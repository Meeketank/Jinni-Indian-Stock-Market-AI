"""Plotly figures as plain dicts (no plotly Python package needed in the serverless bundle).

The browser merges a dark theme into each layout (public/js/plot.js), so these
only describe data and chart-specific layout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jinni import patterns as pt
from jinni import ta

UP, DOWN, GOLD, BLUE, MUTED = "#22C55E", "#EF4444", "#F5B841", "#4C8DFF", "#8A94A6"
SERIES = ["#4C8DFF", "#F5B841", "#22C55E", "#E86AF0", "#2EC4B6", "#FF8A4C", "#A78BFA", "#94A3B8"]
PANEL = "#111823"


def fig(data: list, layout: dict | None = None) -> dict:
    return {"data": data, "layout": layout or {}}


def _v(x, nd: int = 4):
    """Series/array → list rounded to cut payload size; NaN stays NaN (cleaned to null later)."""
    if isinstance(x, pd.Series):
        x = x.values
    return np.round(np.asarray(x, dtype=float), nd).tolist()


def _x(idx):
    return [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(idx)] if isinstance(idx, (pd.DatetimeIndex, pd.Index)) \
        and len(idx) and isinstance(idx[0], (pd.Timestamp, np.datetime64)) else list(idx)


# ------------------------------------------------------------- basic charts


def bar_h(labels, values, title: str = "", fmt: str = ".0%", colors=None, texts=None, xrange=None,
          center: float | None = None) -> dict:
    values = [None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v) for v in values]
    colors = colors or [UP if (v or 0) >= (center or 0) else DOWN for v in values]
    tr = {"type": "bar", "orientation": "h", "x": values, "y": [str(l) for l in labels],
          "marker": {"color": colors}, "cliponaxis": False, "textposition": "outside",
          "hovertemplate": "%{y}: %{x:" + fmt + "}<extra></extra>"}
    tr["text"] = texts if texts else None
    if not texts:
        tr["texttemplate"] = "%{x:" + fmt + "}"
    lay = {"title": title, "showlegend": False, "yaxis": {"autorange": "reversed"}, "xaxis": {"tickformat": fmt}}
    if xrange:
        lay["xaxis"]["range"] = xrange
    return fig([tr], lay)


def bar_v(labels, values, title: str = "", fmt: str = ".1%", colors=None, texts=None) -> dict:
    values = [None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v) for v in values]
    colors = colors or [UP if (v or 0) >= 0 else DOWN for v in values]
    tr = {"type": "bar", "x": [str(l) for l in labels], "y": values, "marker": {"color": colors}, "text": texts,
          "textposition": "outside", "cliponaxis": False}
    return fig([tr], {"title": title, "showlegend": False, "yaxis": {"tickformat": fmt}})


def grouped_h(labels, series: dict, title: str = "", fmt: str = ".0%") -> dict:
    """series: name -> (values, color)"""
    data = [{"type": "bar", "orientation": "h", "name": n, "x": [None if pd.isna(v) else float(v) for v in vals],
             "y": [str(l) for l in labels], "marker": {"color": c}} for n, (vals, c) in series.items()]
    return fig(data, {"barmode": "group", "title": title, "xaxis": {"tickformat": fmt},
                      "yaxis": {"autorange": "reversed"}})


def lines(x, series: dict, title: str = "", yfmt: str | None = None, hover: str = "x unified", ylog: bool = False,
          yrange=None, extra_layout: dict | None = None) -> dict:
    """series: name -> dict(y=..., color=, width=, dash=, fill=, y2=bool, mode=)"""
    data = []
    for i, (n, s) in enumerate(series.items()):
        tr = {"type": "scatter", "mode": s.get("mode", "lines"), "name": n, "x": _x(x) if s.get("x") is None else s["x"],
              "y": _v(s["y"], s.get("nd", 4)), "line": {"color": s.get("color", SERIES[i % len(SERIES)]),
                                                         "width": s.get("width", 1.6), "dash": s.get("dash", "solid")}}
        if s.get("fill"):
            tr["fill"] = s["fill"]
            tr["fillcolor"] = s.get("fillcolor", "rgba(245,184,65,0.08)")
        if s.get("y2"):
            tr["yaxis"] = "y2"
        data.append(tr)
    lay = {"title": title, "hovermode": hover}
    if yfmt:
        lay["yaxis"] = {"tickformat": yfmt}
    if ylog:
        lay.setdefault("yaxis", {})["type"] = "log"
    if yrange:
        lay.setdefault("yaxis", {})["range"] = list(yrange)
    if any(s.get("y2") for s in series.values()):
        lay["yaxis2"] = {"overlaying": "y", "side": "right", "showgrid": False}
    if extra_layout:
        lay.update(extra_layout)
    return fig(data, lay)


def vline(x, color: str = GOLD, dash: str = "dash", text: str | None = None, width: float = 1.5) -> tuple[dict, dict | None]:
    shape = {"type": "line", "xref": "x", "yref": "paper", "x0": x, "x1": x, "y0": 0, "y1": 1,
             "line": {"color": color, "dash": dash, "width": width}}
    ann = {"x": x, "y": 1, "xref": "x", "yref": "paper", "text": text, "showarrow": False, "yanchor": "bottom",
           "font": {"size": 10, "color": color}} if text else None
    return shape, ann


def hline(y, color: str = MUTED, dash: str = "dot", text: str | None = None, width: float = 1, yref: str = "y") -> tuple:
    shape = {"type": "line", "xref": "paper", "yref": yref, "x0": 0, "x1": 1, "y0": y, "y1": y,
             "line": {"color": color, "dash": dash, "width": width}}
    ann = {"x": 1, "xref": "paper", "xanchor": "right", "y": y, "yref": yref, "text": text, "showarrow": False,
           "yanchor": "bottom", "font": {"size": 10, "color": color}} if text else None
    return shape, ann


def with_marks(f: dict, marks: list) -> dict:
    for shape, ann in marks:
        f["layout"].setdefault("shapes", []).append(shape)
        if ann:
            f["layout"].setdefault("annotations", []).append(ann)
    return f


def heatmap(z, x, y, title: str = "", zmin=None, zmax=None, scale=None, texts=None, fmt: str = ".0%") -> dict:
    z = [[None if (v is None or pd.isna(v)) else float(v) for v in r] for r in np.asarray(z, dtype=float).tolist()]
    tr = {"type": "heatmap", "z": z, "x": [str(a) for a in x], "y": [str(a) for a in y], "showscale": False,
          "colorscale": scale or [[0, "#111823"], [0.5, "#3B5B99"], [1, UP]],
          "hovertemplate": "%{y} / %{x}: %{z:" + fmt + "}<extra></extra>"}
    if zmin is not None:
        tr["zmin"], tr["zmax"] = zmin, zmax
    if texts is not None:
        tr["text"] = texts
        tr["texttemplate"] = "%{text}"
    return fig([tr], {"title": title, "yaxis": {"autorange": "reversed"}})


def gauge(value: float, rng, title: str, subtitle: str, color: str = GOLD, steps=None, ticks=None) -> dict:
    tr = {"type": "indicator", "mode": "gauge+number", "value": float(value), "number": {"valueformat": ".0f"},
          "title": {"text": f"{title}<br><span style='font-size:12px'>{subtitle}</span>"},
          "gauge": {"axis": {"range": list(rng), "tickvals": ticks}, "bar": {"color": color}, "bgcolor": PANEL,
                    "borderwidth": 0, "steps": steps or []}}
    return fig([tr], {"margin": {"l": 24, "r": 24, "t": 60, "b": 10}})


def histogram(values, nbins: int = 60, title: str = "", color: str = BLUE, marks=None) -> dict:
    v = np.asarray(values, dtype=float)
    cnt, edges = np.histogram(v[np.isfinite(v)], bins=nbins)
    mid = ((edges[:-1] + edges[1:]) / 2).round(2).tolist()
    f = fig([{"type": "bar", "x": mid, "y": cnt.tolist(), "marker": {"color": color}}],
            {"title": title, "showlegend": False, "bargap": 0.02})
    return with_marks(f, marks or [])


def donut(labels, values, title: str = "", center: str = "") -> dict:
    tr = {"type": "pie", "labels": list(labels), "values": [float(v) for v in values], "hole": 0.62, "sort": False,
          "marker": {"colors": SERIES, "line": {"color": "#0A0E14", "width": 2}}, "textinfo": "percent",
          "textposition": "inside"}
    lay = {"title": title}
    if center:
        lay["annotations"] = [{"text": center, "showarrow": False, "font": {"size": 15}, "x": 0.5, "y": 0.5}]
    return fig([tr], lay)


# -------------------------------------------------------------- price chart

_ROLE = {"neckline": dict(color=GOLD, dash="dash", width=1.5), "upper": dict(color="#E86AF0", width=1.5),
         "lower": dict(color="#2EC4B6", width=1.5), "pole": dict(color=MUTED, dash="dot", width=1),
         "rim": dict(color=GOLD, dash="dash", width=1.5), "leg": dict(color="#A78BFA", width=1.5),
         "wave": dict(color="#FF8A4C", width=1.5), "trend": dict(color=BLUE, width=1.6),
         "price": dict(color="#FFD978", width=2)}

PANEL_GUIDES = {"RSI 14": (30, 70), "RSI 2": (10, 90), "Stochastic": (20, 80), "Stochastic RSI": (20, 80),
                "MFI 14": (20, 80), "Williams %R": (-80, -20), "CCI 20": (-100, 100),
                "Ultimate oscillator": (30, 70), "Connors RSI": (10, 90), "Schaff trend cycle": (25, 75),
                "Choppiness": (38.2, 61.8), "ADX / DMI": (20, 25), "Bollinger %B": (0, 1), "Z-score 20": (-2, 2),
                "CMO": (-50, 50)}

LAYERS = ["Candlestick patterns", "Chart patterns", "Harmonics & Elliott", "Trendlines", "Fibonacci",
          "Support/resistance zones", "Divergences", "Gaps"]


class Stack:
    """Vertically stacked subplots sharing one x axis, built as raw Plotly dicts."""

    def __init__(self, n_panels: int):
        self.rows = 1 + n_panels
        main = 0.62 if n_panels else 1.0
        gap = 0.03
        heights = [main] + [(1 - main - gap * n_panels) / max(n_panels, 1)] * n_panels
        self.dom, top = [], 1.0
        for h in heights:
            self.dom.append((round(top - h, 4), round(top, 4)))
            top -= h + gap
        self.data, self.shapes, self.ann = [], [], []

    def ids(self, row: int):
        return ("x" if row == 0 else f"x{row + 1}", "y" if row == 0 else f"y{row + 1}")

    def add(self, tr: dict, row: int = 0):
        tr["xaxis"], tr["yaxis"] = self.ids(row)
        self.data.append(tr)

    def hline(self, y, row=0, color=MUTED, dash="dot", width=1, text=None):
        _, yid = self.ids(row)
        s, a = hline(y, color, dash, text, width, yref=yid)
        self.shapes.append(s)
        if a:
            self.ann.append(a)

    def layout(self, titles: list[str], y1_range=None) -> dict:
        lay = {"hovermode": "x unified", "shapes": self.shapes, "annotations": self.ann, "showlegend": True,
               "legend": {"font": {"size": 10}, "orientation": "h", "y": -0.04}}
        for r in range(self.rows):
            xid, yid = self.ids(r)
            xk, yk = ("xaxis" if r == 0 else f"xaxis{r + 1}"), ("yaxis" if r == 0 else f"yaxis{r + 1}")
            lay[xk] = {"anchor": yid, "domain": [0, 1], "rangeslider": {"visible": False},
                       "rangebreaks": [{"bounds": ["sat", "mon"]}], "showticklabels": r == self.rows - 1,
                       **({"matches": "x"} if r else {})}
            lay[yk] = {"anchor": xid, "domain": list(self.dom[r])}
        if y1_range:
            lay["yaxis"]["range"] = list(y1_range)
        for r, t in enumerate(titles, start=1):
            lay["annotations"].append({"text": t, "xref": "paper", "yref": "paper", "x": 0, "y": self.dom[r][1],
                                       "xanchor": "left", "yanchor": "bottom", "showarrow": False,
                                       "font": {"size": 11, "color": MUTED}})
        return lay


def price_chart(df: pd.DataFrame, overlays: list[str], panels: list[str], layers: list[str], scan: dict | None,
                heikin: bool = False, view: int = 126) -> dict:
    full = df
    d = df.iloc[-view:]
    start = d.index[0]
    S = Stack(len(panels))
    X = _x(d.index)
    cd = ta.heikin_ashi(full).iloc[-view:] if heikin else d
    S.add({"type": "candlestick", "x": X, "open": _v(cd["Open"], 3), "high": _v(cd["High"], 3),
           "low": _v(cd["Low"], 3), "close": _v(cd["Close"], 3), "name": "Heikin-Ashi" if heikin else "Price",
           "increasing": {"line": {"color": UP}, "fillcolor": UP}, "decreasing": {"line": {"color": DOWN}, "fillcolor": DOWN}})

    k = 0
    for name in overlays:
        out = ta.compute(full, name)
        frame = (out.to_frame(name) if isinstance(out, pd.Series) else out).iloc[-view:]
        if name == "Ichimoku":
            S.add({"type": "scatter", "mode": "lines", "x": X, "y": _v(frame["Span A"], 3), "name": "Span A",
                   "line": {"width": 0.8, "color": UP}})
            S.add({"type": "scatter", "mode": "lines", "x": X, "y": _v(frame["Span B"], 3), "name": "Span B",
                   "line": {"width": 0.8, "color": DOWN}, "fill": "tonexty", "fillcolor": "rgba(138,148,166,0.10)"})
            for c, colr in (("Tenkan", BLUE), ("Kijun", GOLD)):
                S.add({"type": "scatter", "mode": "lines", "x": X, "y": _v(frame[c], 3), "name": c,
                       "line": {"width": 1.1, "color": colr}})
            continue
        if name == "Parabolic SAR":
            S.add({"type": "scatter", "mode": "markers", "x": X, "y": _v(frame.iloc[:, 0], 3), "name": "PSAR",
                   "marker": {"size": 3, "color": GOLD}})
            continue
        if name == "SuperTrend":
            st_ = ta.supertrend(full).iloc[-view:]
            for dirn, colr in ((1, UP), (-1, DOWN)):
                y = st_["SuperTrend"].where(st_["ST_dir"] == dirn)
                S.add({"type": "scatter", "mode": "lines", "x": X, "y": _v(y, 3), "connectgaps": False,
                       "name": f"SuperTrend {'up' if dirn == 1 else 'down'}", "line": {"color": colr, "width": 1.6}})
            continue
        for j, c in enumerate(frame.columns):
            band = len(frame.columns) >= 2 and j == len(frame.columns) - 1 and "ower" in c
            tr = {"type": "scatter", "mode": "lines", "x": X, "y": _v(frame[c], 3), "name": c,
                  "line": {"width": 1.2, "color": SERIES[(k + j) % len(SERIES)]}}
            if band:
                tr["fill"], tr["fillcolor"] = "tonexty", "rgba(138,148,166,0.06)"
            S.add(tr)
        k += len(frame.columns)

    for r, name in enumerate(panels, start=1):
        out = ta.compute(full, name)
        frame = (out.to_frame(name) if isinstance(out, pd.Series) else out).iloc[-view:]
        for j, c in enumerate(frame.columns):
            if c in ("Hist", "macd_hist"):
                S.add({"type": "bar", "x": X, "y": _v(frame[c], 4), "name": c, "opacity": 0.7,
                       "marker": {"color": [UP if (v or 0) >= 0 else DOWN for v in np.nan_to_num(frame[c].values)]}}, r)
            else:
                S.add({"type": "scatter", "mode": "lines", "x": X, "y": _v(frame[c], 4), "name": c,
                       "line": {"width": 1.2, "color": SERIES[(j + 1) % len(SERIES)]}}, r)
        for y in PANEL_GUIDES.get(name, ()):
            S.hline(y, row=r)

    lo, hi = float(d["Low"].min()), float(d["High"].max())
    if scan:
        in_view = lambda ts: ts is not None and ts >= start  # noqa: E731
        col_of = lambda p: UP if p.bias > 0 else DOWN if p.bias < 0 else GOLD  # noqa: E731

        def draw(p, color):
            for (a, b, role) in p.lines:
                (d1, p1), (d2, p2) = a, b
                if d2 < start:
                    continue
                if d1 < start and d2 != d1:
                    p1 = p1 + (p2 - p1) * ((start - d1) / (d2 - d1))
                    d1 = start
                style = dict(_ROLE.get(role, dict(color=color, width=1.2)))
                S.add({"type": "scatter", "mode": "lines", "x": [d1.strftime("%Y-%m-%d"), d2.strftime("%Y-%m-%d")],
                       "y": [round(float(p1), 3), round(float(p2), 3)], "line": style, "showlegend": False,
                       "hoverinfo": "skip"})
            pts = [(x, y, l) for x, y, l in p.points if in_view(x)]
            if pts:
                S.add({"type": "scatter", "mode": "markers+text", "x": [x.strftime("%Y-%m-%d") for x, _, _ in pts],
                       "y": [round(float(y), 3) for _, y, _ in pts], "text": [l for _, _, l in pts],
                       "textposition": "top center", "textfont": {"size": 9, "color": color},
                       "marker": {"size": 6, "color": color}, "name": p.name, "hovertext": p.note, "showlegend": False})
            anchor = p.end if in_view(p.end) else d.index[-1]
            y = max([float(y) for _, y, _ in p.points] or [float(d["High"].iloc[-1])])
            S.ann.append({"x": anchor.strftime("%Y-%m-%d"), "y": y, "xref": "x", "yref": "y",
                          "text": f"{p.name}{' ✓' if p.status == 'confirmed' else ''}", "showarrow": True,
                          "arrowhead": 0, "ax": 0, "ay": -26, "font": {"size": 10, "color": color},
                          "bgcolor": "rgba(10,14,20,0.75)", "bordercolor": color, "borderwidth": 1})
            if p.target and in_view(p.end):
                S.hline(float(p.target), 0, color, "dot", 1, f"{p.name} target")

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
                    (_, glo), (_, ghi) = p.lines[0][0], p.lines[0][1]
                    S.shapes.append({"type": "rect", "xref": "x", "yref": "y", "x0": p.end.strftime("%Y-%m-%d"),
                                     "x1": d.index[-1].strftime("%Y-%m-%d"), "y0": glo, "y1": ghi,
                                     "line": {"width": 0}, "fillcolor": "rgba(245,184,65,0.12)", "layer": "below"})
        if "Support/resistance zones" in layers and not scan["zones"].empty:
            zv = scan["zones"][(scan["zones"]["High"] >= lo * 0.96) & (scan["zones"]["Low"] <= hi * 1.04)]
            for _, z in zv.head(4).iterrows():
                c = "rgba(34,197,94,0.07)" if z["Type"] == "Support" else "rgba(239,68,68,0.07)"
                S.shapes.append({"type": "rect", "xref": "x", "yref": "y", "x0": X[0], "x1": X[-1],
                                 "y0": float(z["Low"]) * 0.998, "y1": float(z["High"]) * 1.002,
                                 "line": {"width": 0}, "fillcolor": c, "layer": "below"})
        if "Fibonacci" in layers and scan["fib"]:
            for lbl, lv in scan["fib"]["levels"].items():
                S.hline(float(lv), 0, "#A78BFA", "dot", 0.8, f"Fib {lbl}")
        if "Candlestick patterns" in layers:
            cf = scan["candle_frame"].iloc[-view:]
            for name in cf.columns:
                bias = pt.CANDLE_BIAS.get(name, 0)
                hits = cf.index[cf[name]]
                if bias == 0 or not len(hits):
                    continue
                y = d.loc[hits, "Low"] * 0.985 if bias > 0 else d.loc[hits, "High"] * 1.015
                S.add({"type": "scatter", "mode": "markers", "x": _x(hits), "y": _v(y, 3), "name": name,
                       "showlegend": False, "hovertemplate": f"{name}<extra></extra>",
                       "marker": {"symbol": "triangle-up" if bias > 0 else "triangle-down", "size": 7,
                                  "color": UP if bias > 0 else DOWN, "opacity": 0.8}})
    lay = S.layout(panels, y1_range=[lo * 0.96, hi * 1.04])
    return {"data": S.data, "layout": lay}
