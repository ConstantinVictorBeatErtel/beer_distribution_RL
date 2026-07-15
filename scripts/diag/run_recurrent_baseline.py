#!/usr/bin/env python3
"""CLI: recurrent baseline report → artifacts/diagnostics/recurrent_baseline.md."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.diag.recurrent_baseline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
