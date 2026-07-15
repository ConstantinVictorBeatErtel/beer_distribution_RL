#!/usr/bin/env python3
"""CLI: shortage-gaming analysis → artifacts/diagnostics/shortage_gaming.md."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.diag.shortage_gaming import main  # noqa: E402

if __name__ == "__main__":
    main()
