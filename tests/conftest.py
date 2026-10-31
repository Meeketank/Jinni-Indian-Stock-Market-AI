"""Offline test setup: synthetic prices and fixture stock lists. The website itself never uses either."""

import os
from pathlib import Path

os.environ.setdefault("JINNI_DEMO", "1")
os.environ.setdefault("JINNI_FIXTURES", str(Path(__file__).parent / "fixtures"))
