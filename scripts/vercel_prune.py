"""Vercel build step: delete files the API never imports from the installed packages.

Vercel's Python functions have a ~225 MB size limit. numpy, pandas, SciPy and scikit-learn together ship
~115 MB of their own unit tests (plus type stubs), which are never imported at runtime. Removing them keeps
the bundle comfortably under the limit without changing any behaviour.

Runs automatically via [tool.vercel.scripts] build in pyproject.toml. Safe to run locally: it only touches
site-packages directories of the active environment.
"""

from __future__ import annotations

import glob
import os
import shutil
import sys

# Only these packages are pruned (never touch anything we have not checked).
PACKAGES = ("numpy", "scipy", "pandas")
TEST_DIRS = {"tests", "test", "_tests"}
# SciPy sub-packages that nothing in JINNI (or scipy.stats / scipy.optimize) imports.
SCIPY_UNUSED = ("io", "signal", "odr", "cluster", "misc", "datasets")


def site_packages() -> list[str]:
    found = {p for p in sys.path if p.endswith("site-packages") and os.path.isdir(p)}
    for base in filter(None, [os.environ.get("VIRTUAL_ENV"), os.path.join(".vercel", "python", ".venv"), ".venv"]):
        found.update(glob.glob(os.path.join(base, "lib", "python*", "site-packages")))
    return sorted(found)


def size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e6


def self_check() -> None:
    """Fail the build loudly if pruning broke an import the API needs."""
    import subprocess

    code = "import numpy, pandas, scipy.stats, scipy.optimize, scipy.special, yfinance; print('imports ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(f"[prune] self-check: {(r.stdout or r.stderr).strip()[-300:]}")
    if r.returncode != 0:
        raise SystemExit("[prune] pruning broke an import; aborting the build")


def main() -> None:
    saved = 0.0
    for sp in site_packages():
        for name in SCIPY_UNUSED:
            p = os.path.join(sp, "scipy", name)
            if os.path.isdir(p):
                saved += size_mb(p)
                shutil.rmtree(p, ignore_errors=True)
        for pkg in PACKAGES:
            top = os.path.join(sp, pkg)
            if not os.path.isdir(top):
                continue
            for root, dirs, files in os.walk(top):
                for d in [d for d in dirs if d in TEST_DIRS]:
                    p = os.path.join(root, d)
                    saved += size_mb(p)
                    shutil.rmtree(p, ignore_errors=True)
                    dirs.remove(d)
                for f in files:
                    if f.endswith(".pyi"):
                        p = os.path.join(root, f)
                        saved += os.path.getsize(p) / 1e6
                        os.remove(p)
        print(f"[prune] {sp}")
    print(f"[prune] removed {saved:.1f} MB of tests, type stubs and unused SciPy modules")
    self_check()


if __name__ == "__main__":
    main()
