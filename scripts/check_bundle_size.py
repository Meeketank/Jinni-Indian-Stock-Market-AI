"""CI guard: estimate the Vercel function bundle size and fail before it can break a deployment.

Vercel's Python functions are limited to 225 MiB uncompressed. This sums the installed runtime dependencies
(after scripts/vercel_prune.py) plus the project code that ships, using the same accounting as Vercel's builder.
Run it inside a virtualenv that has only requirements.txt installed:

    uv venv .bundle-venv && uv pip install --python .bundle-venv/bin/python -r requirements.txt
    .bundle-venv/bin/python scripts/vercel_prune.py && .bundle-venv/bin/python scripts/check_bundle_size.py
"""

from __future__ import annotations

import glob
import os
import sys

LIMIT_MIB = 225.0      # Vercel's hard limit
FAIL_AT_MIB = 220.0    # fail early: leave room for dependency drift and bytecode
SKIP = {"pip", "setuptools", "wheel", "pkg_resources", "_distutils_hack"}
SHIPPED = ("index.py", "pyproject.toml", "requirements.txt", "uv.lock", "vercel.json", "jinni", "server")


def tree_bytes(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files if not f.endswith(".pyc"))
    return total


def main() -> int:
    sps = glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages"))
    if not sps:
        print("could not find site-packages; run inside the bundle virtualenv")
        return 2
    deps = {}
    for entry in os.listdir(sps[0]):
        if entry.split("-")[0].split(".")[0] in SKIP:
            continue
        deps[entry] = tree_bytes(os.path.join(sps[0], entry))
    code = sum(tree_bytes(p) for p in SHIPPED if os.path.exists(p))
    total = (sum(deps.values()) + code) / 2**20
    print("largest dependencies (MiB):")
    for name, size in sorted(deps.items(), key=lambda x: -x[1])[:8]:
        print(f"  {size / 2**20:7.1f}  {name}")
    print(f"project code: {code / 2**20:.2f} MiB")
    print(f"estimated function bundle: {total:.1f} MiB  (Vercel limit {LIMIT_MIB:.0f}, CI fails at {FAIL_AT_MIB:.0f})")
    if total > FAIL_AT_MIB:
        print("FAIL: bundle is too close to Vercel's limit. Remove a dependency or prune more in scripts/vercel_prune.py.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
