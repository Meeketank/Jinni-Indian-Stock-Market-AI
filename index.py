"""JINNI web API: FastAPI entrypoint for Vercel (also runs locally with `python dev.py`).

Vercel serves the static app in public/ from its CDN and routes /api/* here.
Every endpoint returns JSON "blocks" that public/js renders (see server/core.py).
"""

import logging
import os
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from jinni import __version__
from jinni.data import DEMO_OVERRIDE
from server import routes_finder, routes_market, routes_port, routes_stock
from server.core import reply

log = logging.getLogger("jinni")
ROOT = Path(__file__).resolve().parent

app = FastAPI(title="JINNI API", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json",
              redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=1500)


class DemoSwitch:
    """Pure-ASGI middleware: ?demo=1 on any /api request switches that request to synthetic data.
    (A query parameter, not a header, so CDN caches keep demo and live responses apart.)"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            qs = parse_qs(scope.get("query_string", b"").decode())
            tok = DEMO_OVERRIDE.set(True if qs.get("demo", ["0"])[0] == "1" else None)
            try:
                await self.app(scope, receive, send)
            finally:
                DEMO_OVERRIDE.reset(tok)
        else:
            await self.app(scope, receive, send)


app.add_middleware(DemoSwitch)


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    d = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
    return JSONResponse(d, status_code=exc.status_code, headers={"Cache-Control": "no-store"})


@app.exception_handler(RequestValidationError)
async def bad_request(_: Request, exc: RequestValidationError):
    msg = "; ".join(f"{'.'.join(str(x) for x in e['loc'][1:])}: {e['msg']}" for e in exc.errors()[:3])
    return JSONResponse({"error": f"Invalid request ({msg})"}, status_code=422, headers={"Cache-Control": "no-store"})


@app.exception_handler(ValueError)
async def value_error(_: Request, exc: ValueError):
    # Engine errors such as "Need at least ~6 months of daily bars." are written for users.
    return JSONResponse({"error": str(exc)[:300]}, status_code=422, headers={"Cache-Control": "no-store"})


@app.exception_handler(Exception)
async def unexpected(_: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse({"error": "Something went wrong on our side. Please try again."}, status_code=500,
                        headers={"Cache-Control": "no-store"})


for r in (routes_market.router, routes_stock.router, routes_finder.router, routes_port.router):
    app.include_router(r, prefix="/api")


@app.get("/api")
def api_root():
    return reply({"name": "JINNI API", "version": __version__, "docs": "/api/docs"})


@app.get("/", include_in_schema=False)
def home_fallback():
    """Normally Vercel's CDN serves public/index.html before this runs; this covers other hosts."""
    p = ROOT / "public" / "index.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>JINNI API</h1><p>See <a href='/api/docs'>/api/docs</a>.</p>")
