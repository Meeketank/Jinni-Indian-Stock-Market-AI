"""Writable cache location that works on laptops and on read-only serverless filesystems."""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def cache_dir() -> Path:
    """First writable of: $JINNI_CACHE, ~/.cache/jinni, <tmp>/jinni (Vercel only allows /tmp)."""
    candidates = []
    if os.getenv("JINNI_CACHE"):
        candidates.append(Path(os.environ["JINNI_CACHE"]))
    if not os.getenv("VERCEL"):
        candidates.append(Path.home() / ".cache" / "jinni")
    candidates.append(Path(tempfile.gettempdir()) / "jinni")
    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".w"
            probe.write_text("1")
            probe.unlink()
            return p
        except OSError:
            continue
    return Path(tempfile.gettempdir())
