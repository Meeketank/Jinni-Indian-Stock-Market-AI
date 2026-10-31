"""Local development server: the API plus the static front end on one port.

    python dev.py            # http://localhost:8000
    JINNI_DEMO=1 python dev.py   # offline, synthetic data

(On Vercel the CDN serves public/ itself, so index.py does not mount it.)
"""

import json
import os

import uvicorn
from fastapi.staticfiles import StaticFiles

from index import ROOT, app

# Apply the same response headers vercel.json sets (CSP etc.) so local testing matches production.
_cfg = json.loads((ROOT / "vercel.json").read_text())
_SECURITY = {h["key"]: h["value"] for h in _cfg["headers"][0]["headers"]}


@app.middleware("http")
async def security_headers(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/api"):
        for k, v in _SECURITY.items():
            resp.headers.setdefault(k, v)
    return resp


app.mount("/", StaticFiles(directory=ROOT / "public", html=True), name="public")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
