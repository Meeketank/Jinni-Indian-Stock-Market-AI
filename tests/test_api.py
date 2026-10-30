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
    "/api/health", "/api/catalog", "/api/search?q=tata", "/api/home", "/api/mood", "/api/news?tickers=^NSEI,TCS",
    f"/api/stock/{T}/overview", f"/api/stock/{T}/chart", f"/api/stock/{T}/patterns", f"/api/stock/{T}/indicators",
    f"/api/stock/{T}/odds?target=0.07&hz=10", f"/api/stock/{T}/seasonality", f"/api/stock/{T}/plan?side=SHORT",
    f"/api/stock/{T}/fundamentals", f"/api/stock/{T}/news",
    f"/api/stock/{T}/chart?overlays=Ichimoku,SuperTrend,Parabolic SAR&panels=Stochastic,ADX / DMI,Klinger oscillator"
    "&layers=Candlestick patterns,Chart patterns,Harmonics %26 Elliott,Trendlines,Fibonacci,Gaps,Divergences&zoom=1Y&heikin=1",
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
    assert len(plots[0]["fig"]["data"][0]["close"]) == 126


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


def test_demo_query_flag_switches_per_request(client, monkeypatch):
    from jinni import data as jdata

    monkeypatch.delenv("JINNI_DEMO", raising=False)
    monkeypatch.setattr(jdata, "_download_yahoo", lambda *a, **k: None)
    assert client.get("/api/stock/DEMOFLAG1.NS/odds").status_code == 503
    r = client.get("/api/stock/DEMOFLAG1.NS/odds?demo=1")
    assert r.status_code == 200 and "demo" in r.text
    assert client.get("/api/stock/DEMOFLAG1.NS/odds").status_code == 503  # the override did not leak


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
    assert {"fastapi", "numpy", "pandas", "scipy", "scikit-learn", "yfinance"} <= names
    assert not names & {"streamlit", "plotly", "tensorflow", "torch", "xgboost"}  # would blow Vercel's 500 MB limit


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
