#!/usr/bin/env python3
"""CLI: honesty-weighted recheck → artifacts/diagnostics/honesty_weighted_recheck.md."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.diag.honesty_weighted_recheck import main  # noqa: E402

if __name__ == "__main__":
    main()
