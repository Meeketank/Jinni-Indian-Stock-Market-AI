"""Local persistence (SQLite) for holdings, watchlist and the analysis log.

An optional Firestore mirror is used when GOOGLE_APPLICATION_CREDENTIALS is set.
Note: on Streamlit Community Cloud the local disk is reset on redeploy, so use
the CSV export/import on the Portfolio page for anything you want to keep.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import pandas as pd

DB_PATH = os.getenv("JINNI_DB", "jinni.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS holdings (
    ticker TEXT PRIMARY KEY, quantity REAL NOT NULL, avg_cost REAL
);
CREATE TABLE IF NOT EXISTS watchlist (
    ticker TEXT PRIMARY KEY, added_at TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, entry_date TEXT, ticker TEXT, entry REAL,
    target_pct REAL, stop_pct REAL, target_price REAL, stop_price REAL, horizon INTEGER,
    p_target REAL, p_stop REAL, grade TEXT, verdict TEXT, qty INTEGER, source TEXT
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS analysis_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, ticker TEXT, price REAL, rating TEXT,
    composite REAL, trend REAL, momentum REAL, timing REAL, volume REAL, risk REAL,
    fundamental REAL, stop REAL, target1 REAL, target2 REAL, target3 REAL
);
"""

DEFAULT_HOLDINGS = pd.DataFrame({
    "ticker": ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "BHARTIARTL.NS",
               "LT.NS", "ITC.NS", "SUNPHARMA.NS", "MARUTI.NS"],
    "quantity": [40, 50, 60, 45, 18, 35, 20, 150, 40, 5],
    "avg_cost": [1250.0, 800.0, 1150.0, 1500.0, 3600.0, 1500.0, 3300.0, 420.0, 1600.0, 11500.0],
})


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    return con


def load_holdings() -> pd.DataFrame:
    try:
        with closing(_conn()) as con:
            df = pd.read_sql_query("SELECT ticker, quantity, avg_cost FROM holdings", con)
    except sqlite3.Error:
        df = pd.DataFrame()
    return df if not df.empty else DEFAULT_HOLDINGS.copy()


def save_holdings(df: pd.DataFrame) -> None:
    clean = df.dropna(subset=["ticker", "quantity"])
    clean = clean[clean["quantity"] > 0]
    with closing(_conn()) as con, con:
        con.execute("DELETE FROM holdings")
        con.executemany(
            "INSERT OR REPLACE INTO holdings (ticker, quantity, avg_cost) VALUES (?,?,?)",
            [(r.ticker, float(r.quantity), None if pd.isna(r.avg_cost) else float(r.avg_cost))
             for r in clean.itertuples()],
        )


def watchlist() -> list[str]:
    with closing(_conn()) as con:
        return [r[0] for r in con.execute("SELECT ticker FROM watchlist ORDER BY added_at")]


def add_watch(ticker: str) -> None:
    with closing(_conn()) as con, con:
        con.execute("INSERT OR IGNORE INTO watchlist VALUES (?,?)",
                    (ticker, datetime.now(timezone.utc).isoformat()))


def remove_watch(ticker: str) -> None:
    with closing(_conn()) as con, con:
        con.execute("DELETE FROM watchlist WHERE ticker=?", (ticker,))


def log_analysis(row: dict) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(), **row}
    cols = ",".join(row)
    with closing(_conn()) as con, con:
        con.execute(f"INSERT INTO analysis_log ({cols}) VALUES ({','.join('?' * len(row))})", list(row.values()))
    _firestore_mirror("analyses", row)


def recent_analyses(limit: int = 200) -> pd.DataFrame:
    with closing(_conn()) as con:
        return pd.read_sql_query(
            "SELECT ts, ticker, price, rating, composite, stop, target1 FROM analysis_log ORDER BY id DESC LIMIT ?",
            con, params=(limit,),
        )


def _firestore_mirror(collection: str, doc: dict) -> None:
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    try:
        from google.cloud import firestore

        firestore.Client().collection(collection).add(doc)
    except Exception:
        pass


# ------------------------------------------------------------ predictions


def track_prediction(row: dict) -> bool:
    """Store a prediction. Returns False if the same ticker/plan was already tracked today."""
    row = {"created": datetime.now(timezone.utc).isoformat(), **row}
    with closing(_conn()) as con, con:
        dup = con.execute(
            "SELECT 1 FROM predictions WHERE ticker=? AND entry_date=? AND target_pct=? AND horizon=?",
            (row["ticker"], row["entry_date"], row["target_pct"], row["horizon"]),
        ).fetchone()
        if dup:
            return False
        con.execute(f"INSERT INTO predictions ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                    list(row.values()))
    return True


def predictions() -> pd.DataFrame:
    with closing(_conn()) as con:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", con)


def delete_prediction(pid: int) -> None:
    with closing(_conn()) as con, con:
        con.execute("DELETE FROM predictions WHERE id=?", (int(pid),))


# ---------------------------------------------------------------- settings

import json as _json  # noqa: E402


def load_setting(key: str, default=None):
    with closing(_conn()) as con:
        r = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return _json.loads(r[0]) if r else default


def save_setting(key: str, value) -> None:
    with closing(_conn()) as con, con:
        con.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, _json.dumps(value)))
