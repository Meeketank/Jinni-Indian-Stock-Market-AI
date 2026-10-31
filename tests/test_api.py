"""Web API + Vercel deployment tests (run in demo mode: no network needed)."""

import json
import os
import re
from pathlib import Path

import pytest

os.environ["JINNI_DEMO"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from index import app  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BLOCK_TYPES = {"kpis", "section", "text", "callout", "plot", "tabs", "cols", "tape", "news", "badge", "demo", "details",
               "table", "stockhead", "score", "gate"}
T = "RELIANCE.NS"
HOLD = [{"ticker": t, "quantity": q, "avg_cost": a} for t, q, a in
        [("RELIANCE.NS", 40, 1250), ("HDFCBANK.NS", 50, 800), ("INFY.NS", 45, 1500), ("ITC.NS", 150, 420)]]
NIFTY12 = "RELIANCE.NS,HDFCBANK.NS,ICICIBANK.NS,INFY.NS,TCS.NS,ITC.NS,LT.NS,SUNPHARMA.NS,MARUTI.NS,BHARTIARTL.NS,WIPRO.NS,SBIN.NS"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def strict_json(text: str):
    """Parse JSON refusing NaN/Infinity (browsers' JSON.parse would choke on them)."""
    def bad(c):
        raise ValueError(f"non-finite number in response: {c}")
    return json.loads(text, parse_constant=bad)


def walk_blocks(blocks):
    for b in blocks:
        assert b["type"] in BLOCK_TYPES, b["type"]
        yield b
        if b["type"] == "cols":
            for c in b["cols"]:
                yield from walk_blocks(c)
        elif b["type"] == "tabs":
            for t in b["tabs"]:
                yield from walk_blocks(t["blocks"])
        elif b["type"] == "details":
            yield from walk_blocks(b["blocks"])


GET_ENDPOINTS = [
    "/api/health", "/api/catalog", "/api/search?q=tata", "/api/home/pulse", "/api/home/sectors", "/api/home/movers", "/api/home/nifty50", "/api/mood", "/api/news?tickers=^NSEI,TCS",
    f"/api/stock/{T}/overview", f"/api/stock/{T}/chart", f"/api/stock/{T}/patterns", f"/api/stock/{T}/indicators",
    f"/api/stock/{T}/odds?target=0.07&hz=10", f"/api/stock/{T}/seasonality", f"/api/stock/{T}/plan?side=SHORT",
    f"/api/stock/{T}/fundamentals", f"/api/stock/{T}/news",
    f"/api/stock/{T}/chart?overlays=Ichimoku,SuperTrend,Parabolic SAR&panels=Stochastic,ADX / DMI,Klinger oscillator"
    "&layers=Candlestick patterns,Chart patterns,Harmonics %26 Elliott,Trendlines,Fibonacci,Gaps,Divergences&n=1&unit=year&heikin=1",
    "/api/events/study?symbol=^NSEI&mode=rule&nth=2nd&day=Friday",
    "/api/events/study?symbol=^NSEI&mode=builtin&event=Union Budget day",
    "/api/events/study?symbol=^NSEI&mode=dates&dates=2025-04-02,2025-07-07&offset=1", "/api/events/calendar?symbol=^NSEI",
    "/api/screen?tickers=nifty50", "/api/lab/backtest?ticker=TCS.NS&period=5y", "/api/lab/ml?tickers=RELIANCE.NS,TCS.NS,INFY.NS&hz=5&folds=3",
    "/api/market/universe",
]


@pytest.mark.parametrize("url", GET_ENDPOINTS)
def test_get_endpoint_ok(client, url):
    r = client.get(url)
    assert r.status_code == 200, r.text[:300]
    body = strict_json(r.text)
    if "blocks" in body:
        assert list(walk_blocks(body["blocks"]))
    if url not in ("/api/health",):
        assert "s-maxage" in r.headers["cache-control"]  # CDN-cacheable
    assert "Traceback" not in r.text


def test_table_blocks_are_well_formed(client):
    body = client.get(f"/api/stock/{T}/indicators").json()
    tables = [b for b in walk_blocks(body["blocks"]) if b["type"] == "table"]
    assert tables
    for t in tables:
        keys = {c["key"] for c in t["columns"]}
        assert t["rows"] and all(keys <= set(r) for r in t["rows"][:5])


def test_plot_blocks_have_traces(client):
    body = client.get(f"/api/stock/{T}/chart").json()
    plots = [b for b in walk_blocks(body["blocks"]) if b["type"] == "plot"]
    assert plots and plots[0]["fig"]["data"][0]["type"] == "candlestick"
    assert 120 <= len(plots[0]["fig"]["data"][0]["close"]) <= 135  # default span: 6 months


def test_market_scan_and_finder_flow(client):
    uni = client.get("/api/market/universe").json()
    assert uni["count"] == len(uni["rows"]) > 100
    tk = ",".join(r[0] for r in uni["rows"][:60])
    scan = client.get(f"/api/market/scan?tickers={tk}&target=0.07&hz=5&budget=1000").json()
    assert {"t", "price", "q", "why"} <= set(scan["rows"][0])
    assert any("budget" in r["why"] for r in scan["rows"])  # expensive stocks excluded for a ₹1,000 budget
    f = client.get(f"/api/finder?target=0.05&hz=21&stop=auto&tickers={NIFTY12}&budget=5000&max_names=3").json()
    rows = f["data"]["rows"]
    assert len(rows) >= 8 and {"ticker", "Grade", "p_target", "engines"} <= set(rows[0])
    assert set(f["data"]["plans"]) <= {"Diversified basket", "Single best pick"}
    table = next(b for b in f["blocks"] if b["type"] == "table")
    assert table["select"] == {"key": "ticker", "action": "detail"} and "ticker" in table["rows"][0]
    d = client.post("/api/finder/detail", json={"ticker": rows[0]["ticker"], "target": 0.05, "hz": 21, "stop": "auto",
                                                "holdings": HOLD, "row": rows[0]})
    assert d.status_code == 200 and d.headers["cache-control"] == "no-store"
    body = d.json()
    assert any(b["type"] == "tabs" for b in body["blocks"])
    assert body["data"]["track"] is None or {"ticker", "entry", "target_price", "qty"} <= set(body["data"]["track"])


@pytest.mark.parametrize("obj", ["max_sharpe", "min_vol", "risk_parity", "max_div", "equal"])
def test_optimizer_objectives(client, obj):
    r = client.post("/api/optimizer", json={"tickers": [h["ticker"] for h in HOLD], "holdings": HOLD, "objective": obj,
                                            "max_w": 0.5, "sector_cap": 0.6})
    assert r.status_code == 200, r.text[:300]
    strict_json(r.text)


def test_portfolio_risk_and_journal(client):
    r = client.post("/api/portfolio/risk", json={"holdings": HOLD, "lookback": "2y", "conf": 0.99, "horizon": 5})
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    tabs = next(b for b in strict_json(r.text)["blocks"] if b["type"] == "tabs")
    assert [t["label"] for t in tabs["tabs"]] == ["Exposure", "VaR & tails", "Risk decomposition", "Stress tests",
                                                  "Performance", "Liquidity"]
    j = client.post("/api/journal/score", json={"predictions": [
        {"id": 1, "ticker": "TCS.NS", "entry_date": "2026-09-01", "entry": 3000, "target_price": 3150, "stop_price": 2900,
         "target_pct": 0.05, "horizon": 21, "p_target": 0.4, "grade": "B", "qty": 3, "source": "plan"}]})
    assert j.status_code == 200


# ------------------------------------------------------------------ errors


@pytest.mark.parametrize("method,url,body,status", [
    ("get", "/api/finder?tickers=&target=0.05", None, 400),
    ("get", "/api/finder?target=5&tickers=TCS.NS", None, 422),
    ("get", "/api/events/study?mode=builtin&event=Nope", None, 400),
    ("get", "/api/events/study?mode=dates&dates=garbage", None, 400),
    ("post", "/api/optimizer", {"tickers": ["TCS.NS"]}, 400),
    ("post", "/api/portfolio/risk", {"holdings": []}, 400),
    ("post", "/api/optimizer", {"nonsense": 1}, 422),
])
def test_errors_are_json_without_stack_traces(client, method, url, body, status):
    r = getattr(client, method)(url, **({"json": body} if body is not None else {}))
    assert r.status_code == status
    assert set(r.json()) == {"error"} and "Traceback" not in r.text
    assert r.headers["cache-control"] == "no-store"


def test_unexpected_exception_is_hidden():
    from index import app as a

    @a.get("/api/_boom")
    def boom():
        raise RuntimeError("secret internal detail")

    r = TestClient(a, raise_server_exceptions=False).get("/api/_boom")
    assert r.status_code == 500 and "secret" not in r.text and "error" in r.json()


def test_live_failure_is_an_honest_503_not_fake_data(client, monkeypatch):
    """Demo off and Yahoo down: the API must say so rather than serve synthetic prices."""
    from jinni import data as jdata

    monkeypatch.delenv("JINNI_DEMO", raising=False)
    monkeypatch.setattr(jdata, "_download_yahoo", lambda *a, **k: None)
    r = client.get("/api/stock/FAKEONLY1.NS/chart")
    assert r.status_code == 503 and r.json()["code"] == "data_unavailable"
    assert r.headers["cache-control"] == "no-store"


def test_website_has_no_demo_switch(client, monkeypatch):
    from jinni import data as jdata

    monkeypatch.delenv("JINNI_DEMO", raising=False)
    monkeypatch.setattr(jdata, "_download_yahoo", lambda *a, **k: None)
    assert client.get("/api/stock/DEMOFLAG1.NS/odds").status_code == 503
    assert client.get("/api/stock/DEMOFLAG1.NS/odds?demo=1").status_code == 503  # the old query flag does nothing
    ui = "".join(p.read_text() for p in (ROOT / "public" / "js").rglob("*.js"))
    assert "demo" not in ui.lower().replace("'demo'", "")  # no demo toggle, banner or button left in the UI


def test_search(client):
    res = client.get("/api/search?q=reliance").json()["results"]
    assert any(x["t"] == "RELIANCE.NS" for x in res)
    assert client.get("/api/search?q=").json()["results"] == []


# ------------------------------------------------------ deployment wiring


def test_vercel_config_is_valid():
    cfg = json.loads((ROOT / "vercel.json").read_text())
    fn = cfg["functions"]["index.py"]
    assert (ROOT / "index.py").exists() and fn["maxDuration"] <= 300
    csp = next(h["value"] for h in cfg["headers"][0]["headers"] if h["key"] == "Content-Security-Policy")
    assert "script-src 'self'" in csp and "unsafe-eval" not in csp
    assert (ROOT / ".python-version").read_text().strip() in ("3.12", "3.13")


def test_runtime_requirements_are_lean():
    names = {re.split(r"[<>=~! ]", ln.strip())[0].lower() for ln in (ROOT / "requirements.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")}
    assert {"fastapi", "numpy", "pandas", "scipy", "yfinance"} <= names
    assert not names & {"streamlit", "plotly", "scikit-learn", "tensorflow", "torch", "xgboost"}  # blow the size limit


def test_pyproject_dependencies_match_requirements():
    """Vercel installs from pyproject.toml (+ uv.lock); pip users from requirements.txt. They must agree."""
    import tomllib

    py = tomllib.loads((ROOT / "pyproject.toml").read_text())
    req = [ln.strip() for ln in (ROOT / "requirements.txt").read_text().splitlines()
           if ln.strip() and not ln.startswith("#")]
    assert sorted(py["project"]["dependencies"]) == sorted(req)
    assert py["tool"]["vercel"]["entrypoint"] == "index:app"
    assert (ROOT / "uv.lock").exists()


def test_vercel_exclude_pattern_fits_the_256_char_limit():
    cfg = json.loads((ROOT / "vercel.json").read_text())
    assert len(cfg["functions"]["index.py"]["excludeFiles"]) <= 256  # Vercel rejects longer patterns
    assert cfg["framework"] == "fastapi"


def test_static_site_wiring():
    html = (ROOT / "public" / "index.html").read_text()
    for ref in re.findall(r'(?:src|href)="(/[^"#]+)"', html):
        assert (ROOT / "public" / ref.lstrip("/")).exists(), f"index.html references missing {ref}"
    manifest = json.loads((ROOT / "public" / "manifest.webmanifest").read_text())
    for i in manifest["icons"]:
        assert (ROOT / "public" / i["src"].lstrip("/")).exists()
    # every relative JS import resolves
    for js in (ROOT / "public" / "js").rglob("*.js"):
        for rel in re.findall(r"""(?:from|import\()\s*['"](\.[^'"]+)['"]""", js.read_text()):
            assert (js.parent / rel).resolve().exists(), f"{js.name} imports missing {rel}"
    main = (ROOT / "public" / "js" / "main.js").read_text()
    for page in re.findall(r"import\('\./pages/([a-z]+\.js)'\)", main):
        assert (ROOT / "public" / "js" / "pages" / page).exists()


def test_no_secrets_or_streamlit_in_runtime_path():
    for f in ("index.py", *[p.relative_to(ROOT).as_posix() for p in (ROOT / "server").glob("*.py")]):
        src = (ROOT / f).read_text()
        assert "import streamlit" not in src and "import plotly" not in src, f


def test_light_theme_is_default_and_csp_safe():
    html = (ROOT / "public" / "index.html").read_text()
    css = (ROOT / "public" / "css" / "app.css").read_text()
    assert '<script src="/js/theme-init.js"></script>' in html and "<script>" not in html  # no inline script (CSP)
    root = css[css.index(":root"):css.index("}", css.index(":root")) + 1]
    assert "--bg:#F4F6FA" in root.replace(" ", "")                    # light palette by default
    assert ':root[data-theme="dark"]' in css
    init = (ROOT / "public" / "js" / "theme-init.js").read_text()
    assert "'light'" in init                                          # default mode
    assert json.loads((ROOT / "public" / "manifest.webmanifest").read_text())["theme_color"] == "#F4F6FA"


def _candles(client, qs):
    body = client.get(f"/api/stock/{T}/chart?{qs}").json()
    return [b for b in walk_blocks(body["blocks"]) if b["type"] == "plot"][0]["fig"]


def test_chart_range_is_count_plus_unit(client):
    sizes = {}
    for n, unit in [(10, "day"), (3, "week"), (3, "month"), (1, "quarter"), (1, "year"), (1, "fy")]:
        f = _candles(client, f"n={n}&unit={unit}")
        sizes[(n, unit)] = len(f["data"][0]["close"])
    assert 6 <= sizes[(10, "day")] <= 9                       # 10 calendar days
    assert sizes[(3, "week")] < sizes[(3, "month")] <= sizes[(1, "year")]
    assert abs(sizes[(3, "month")] - sizes[(1, "quarter")]) <= 3
    assert client.get(f"/api/stock/{T}/chart?n=2&unit=fortnight").status_code == 422


def test_patterns_tab_draws_a_chart_and_follows_the_range(client):
    for qs in ("n=3&unit=month", "n=2&unit=year"):
        body = client.get(f"/api/stock/{T}/patterns?{qs}").json()
        kinds = [b["type"] for b in walk_blocks(body["blocks"])]
        assert "plot" in kinds and "table" in kinds


def test_pattern_detection_is_scoped_to_the_window():
    from jinni import data as jd
    from jinni import patterns as jp

    df = jd.get_prices(["TCS.NS"], "5y").ohlcv("TCS.NS")
    short, long_ = jp.scan_window(df, 63), jp.scan_window(df, 1000)
    assert short["bars"] == 63 and long_["bars"] == 1000
    assert min(p.i for p in short["pivots"]) >= 0 and len(short["pivots"]) != len(long_["pivots"])
    names = lambda s: sorted(p.name for p in s["chart"] + s["trend"])  # noqa: E731
    assert names(short) != names(long_)


def test_span_start_handles_financial_year_and_units():
    import pandas as pd

    from server.core import span_bars, span_start

    last = pd.Timestamp("2026-10-30")
    assert span_start(last, 1, "fy") == pd.Timestamp("2026-04-01")
    assert span_start(pd.Timestamp("2026-02-10"), 1, "fy") == pd.Timestamp("2025-04-01")
    assert span_start(last, 2, "fy") == pd.Timestamp("2025-04-01")
    assert span_start(last, 1, "quarter") == pd.Timestamp("2026-07-30")
    assert span_start(last, 2, "week") == pd.Timestamp("2026-10-16")
    df = pd.DataFrame({"Close": range(100)}, index=pd.bdate_range(end=last, periods=100))
    assert span_bars(df, 1, "day", minimum=5) == 5  # never fewer than the minimum
    assert span_bars(df, 50, "year") == 100


def test_nse_movers_come_from_the_live_screener(client, monkeypatch):
    from jinni import live
    from server import routes_market  # noqa: F401

    quotes = [{"symbol": "AAA.NS", "shortName": "Aaa Ltd", "regularMarketPrice": 120.5, "regularMarketChange": 5,
               "regularMarketChangePercent": 4.3, "regularMarketVolume": 1_000_000, "marketCap": 5e10,
               "fiftyTwoWeekHigh": 150, "trailingPE": 18.2}]

    class FakeYF:
        calls = []

        def screen(self, q, size=0, sortField=None, sortAsc=None):
            FakeYF.calls.append((sortField, sortAsc))
            return {"quotes": quotes}

    monkeypatch.setattr(live, "_yf", lambda: FakeYF())
    monkeypatch.setattr(live, "demo_forced", lambda: False)
    monkeypatch.setattr(routes_market.jdata, "demo_forced", lambda: False)
    df = live.board("gainers")
    assert list(df["t"]) == ["AAA.NS"] and abs(df.loc[0, "1D"] - 0.043) < 1e-9
    assert ("percentchange", False) in FakeYF.calls
    live.board("losers")
    assert ("percentchange", True) in FakeYF.calls

    class Boom:
        def screen(self, *a, **k):
            raise RuntimeError("429")

    monkeypatch.setattr(live, "_yf", lambda: Boom())
    assert live.board("gainers") is None  # never invents rows


def test_no_hardcoded_sample_data_in_the_front_end():
    store = (ROOT / "public" / "js" / "store.js").read_text()
    assert "SAMPLE_HOLDINGS" not in store and "RELIANCE" not in store
    assert "IRCTC, HAL" not in (ROOT / "public" / "js" / "pages" / "finder.js").read_text().replace("e.g. IRCTC, HAL, TRENT", "")


def test_phone_layout_guards_are_in_place():
    css = (ROOT / "public" / "css" / "app.css").read_text()
    ui = (ROOT / "public" / "js" / "ui.js").read_text()
    assert "overflow-x: hidden" in css and "touch-action: manipulation" in css
    assert re.search(r"max-width: 640px\) \{\s*input\[type=text\].*font-size: 16px", css, re.S)  # no iOS focus-zoom
    assert "tcards" in css and "as-cards" in ui                                                         # tables become cards


def test_search_matches_company_names_and_merges_yahoo(client, monkeypatch):
    from server import routes_market as rm

    local = rm._local_hits("TATA MOTORS")
    assert local and all("TATA" in r["name"].upper() and "MOTORS" in r["name"].upper() for r in local)
    assert rm._local_hits("RELIANCE")[0]["symbol"] == "RELIANCE"

    class FakeSearch:
        def __init__(self, q, **kw):
            self.quotes = [{"symbol": "JIOFIN.NS", "quoteType": "EQUITY", "longname": "Jio Financial Services Limited"},
                           {"symbol": "JIOFIN.BO", "quoteType": "EQUITY", "shortname": "JIOFIN"},
                           {"symbol": "AAPL", "quoteType": "EQUITY", "shortname": "Apple"}]

    class FakeYF:
        Search = FakeSearch

    monkeypatch.setattr(rm.jdata, "_yf", lambda: FakeYF())
    monkeypatch.setattr(rm.jdata, "demo_forced", lambda: False)
    got = rm._yahoo_hits("jio fin")
    assert [r["t"] for r in got] == ["JIOFIN.NS", "JIOFIN.BO"]          # foreign tickers dropped
    monkeypatch.setattr(rm.jdata, "_yf", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert rm._yahoo_hits("jio") == []                                  # a dead feed never breaks search


def test_new_listing_shows_what_exists_and_explains_the_rest(client, monkeypatch):
    from server import routes_stock as rs

    real = rs.need_series
    monkeypatch.setattr(rs, "need_series", lambda pdx, t, min_len=60: real(pdx, t, 1).tail(3))
    ov = client.get("/api/stock/NEWLIST1.NS/overview")
    assert ov.status_code == 200
    body = ov.json()
    assert body["data"]["rating"] == "NEW LISTING"
    assert "will not invent a score" in str(body["blocks"])
    for tab in ("chart", "patterns", "indicators", "odds"):
        r = client.get(f"/api/stock/NEWLIST1.NS/{tab}")
        assert r.status_code == 422 and r.json()["code"] == "new_listing" and "unlock" in r.json()["error"]


# ---------------------------------------------------------------- outlook & live sentiment

RSS = b"""<rss><channel>
<item><title>Sensex surges 600 points as FII inflows rebound - Mint</title><source>Mint</source><pubDate>{d}</pubDate><link>http://a</link></item>
<item><title>Exit polls ahead of results tomorrow, markets await outcome - ET</title><source>ET</source><pubDate>{d}</pubDate><link>http://b</link></item>
<item><title>RBI MPC meeting next week: rate cut hopes lift banks - Moneycontrol</title><source>Moneycontrol</source><pubDate>{d}</pubDate><link>http://c</link></item>
<item><title>Crude oil slumps as war fears ease - Reuters</title><source>Reuters</source><pubDate>{d}</pubDate><link>http://d</link></item>
<item><title>Sensex surges 600 points as FII inflows rebound - Mint</title><source>Mint</source><pubDate>{d}</pubDate><link>http://a</link></item>
</channel></rss>"""


def _fake_feeds(monkeypatch):
    from email.utils import format_datetime

    from jinni import newsfeeds as nf

    body = RSS.replace(b"{d}", format_datetime(pd.Timestamp.now(tz="UTC").to_pydatetime()).encode())
    from server import core

    core._MEMO.clear()
    monkeypatch.setattr(nf, "demo_forced", lambda: False)
    monkeypatch.setattr(nf, "_get", lambda url, timeout=7: body if "gdelt" not in url else json.dumps(
        {"timeline": [{"series": "Average Tone", "data": [
            {"date": (pd.Timestamp.now() - pd.Timedelta(days=60 - i)).strftime("%Y%m%dT000000Z"),
             "value": float(np.sin(i / 5))} for i in range(60)]}]}).encode())
    return nf


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def test_news_pipeline_scores_dedupes_and_flags_events(monkeypatch):
    nf = _fake_feeds(monkeypatch)
    df, status = nf.fetch_all()
    assert len(df) == 4                                         # the duplicate headline collapses
    assert set(df.columns) >= {"Headline", "Source", "Tone", "Topics"}
    assert df.loc[df["Headline"].str.startswith("Sensex surges"), "Tone"].iloc[0] > 0.2
    radar = nf.event_radar(pd.concat([df] * 3, ignore_index=True))
    assert "Election / results" in set(radar["Event"]) and radar.iloc[0]["Imminent"] >= 1
    assert any("Central banks" in t for t in df["Topics"].sum())
    assert all(isinstance(v, int) for v in status.values())     # every fake feed answered


def test_unreachable_feeds_mean_no_headlines_not_fake_ones(monkeypatch):
    from jinni import newsfeeds as nf

    monkeypatch.setattr(nf, "demo_forced", lambda: False)
    monkeypatch.setattr(nf, "_get", lambda *a, **k: None)
    df, status = nf.fetch_all()
    assert df.empty and set(status.values()) == {"unreachable"}


def test_gdelt_parser_and_tone_backtest(monkeypatch):
    nf = _fake_feeds(monkeypatch)
    from jinni import outlook as ol

    tone = nf.gdelt_tone()
    assert tone is not None and len(tone) == 60
    close = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 90)), index=pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=90))
    bt = ol.tone_vs_next_day(tone, close)
    assert bt and 25 <= bt["n"] <= 60 and -1 <= bt["rho"] <= 1 and 0 <= bt["p"] <= 1
    assert nf.parse_gdelt_timeline(b"not json") is None


def test_outlook_is_validated_capped_and_honest():
    from jinni import data as jd
    from jinni import outlook as ol

    px = jd.get_prices(ol.TICKERS, "10y")
    o = ol.fit_outlook(px.ohlcv("^NSEI"), px.close.drop(columns=["^NSEI"]), 1, True)
    assert 0.03 < o.p_cal < 0.97 and o.n_oos > 1000 and len(o.folds) >= 4
    assert 0.40 < o.auc < 0.62                          # synthetic random-walk data: must not claim a big edge
    assert o.edge in {"none", "weak"}
    assert len(o.drivers) == 8 and o.drivers["Push"].abs().iloc[0] >= o.drivers["Push"].abs().iloc[-1]
    # tilt: capped at +/-3 points, zero without enough headlines
    assert 0.028 <= ol.tilt(0.5, 1.0, 500)["shift"] <= ol.CAP
    assert -ol.CAP <= ol.tilt(0.5, -1.0, 500)["shift"] <= -0.028
    assert ol.tilt(0.5, 0.9, 3)["shift"] == 0.0


@pytest.mark.parametrize("url", ["/api/outlook/market?h=1", "/api/outlook/market?h=5", f"/api/stock/{T}/outlook?h=1",
                                  "/api/sentiment/live"])
def test_outlook_and_sentiment_endpoints(client, monkeypatch, url):
    _fake_feeds(monkeypatch)
    r = client.get(url)
    assert r.status_code == 200, r.text[:300]
    body = strict_json(r.text)
    text_ = json.dumps(body)
    if "outlook" in url:
        assert "not a prediction you can rely on" in text_ and "AUC" in text_
    else:
        assert "Event radar" in text_ and "Source status" in text_


def test_sentiment_says_so_when_every_feed_is_down(client, monkeypatch):
    from jinni import newsfeeds as nf

    monkeypatch.setattr(nf, "demo_forced", lambda: False)
    monkeypatch.setattr(nf, "_get", lambda *a, **k: None)
    from server import core

    core._MEMO.clear()
    r = client.get("/api/sentiment/live")
    assert r.status_code == 200 and "does not fill the gap" in r.text
