#!/usr/bin/env python3
"""CLI: Regime-B AR(1) signal-content analysis → artifacts/diagnostics/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.diag.v11_signal_content import main  # noqa: E402

if __name__ == "__main__":
    main()
