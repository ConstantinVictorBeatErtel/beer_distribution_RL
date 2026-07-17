#!/usr/bin/env python3
"""Serve the live Beer Game spectator UI.

Usage:
  pip install -e ".[web]"
  python scripts/serve_spectator.py
  # open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Beer Game spectator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed-ms", type=int, default=500)
    parser.add_argument("--reload", action="store_true", help="Dev auto-reload")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print(
            "Missing web deps. Install with: pip install -e \".[web]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    from beer_distribution_rl.web.runner import EpisodeRunner
    from beer_distribution_rl.web.server import create_app

    runner = EpisodeRunner(speed_ms=args.speed_ms, seed=args.seed)
    app = create_app(runner)

    print(f"Beer Game Spectator → http://{args.host}:{args.port}")
    # reload requires an import string; keep object path for the common case
    if args.reload:
        uvicorn.run(
            "beer_distribution_rl.web.server:app",
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
