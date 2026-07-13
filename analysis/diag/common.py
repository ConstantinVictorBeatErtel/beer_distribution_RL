"""Shared helpers for diagnostics (checkpoint load, CIs, paths)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M3_DIR = ROOT / "artifacts" / "runs" / "ippo" / "m3"
M2_A = ROOT / "artifacts" / "runs" / "ippo" / "regimeA_seed0"
M2_C = ROOT / "artifacts" / "runs" / "ippo" / "regimeC_seed0"
FIG_DIR = ROOT / "analysis" / "figs" / "diag"
CACHE_DIR = ROOT / "analysis" / "diag" / "cache"
REPORT_PATH = ROOT / "analysis" / "DIAGNOSTICS.md"

CAP_ORDER = ["inf", "1p5mu", "1p2mu", "1p0mu", "0p8mu"]
CAP_LABEL = {
    "inf": "∞",
    "1p5mu": "1.5μ",
    "1p2mu": "1.2μ",
    "1p0mu": "1.0μ",
    "0p8mu": "0.8μ",
}
ROLES_ORDER = ["retailer", "wholesaler", "distributor", "factory"]
TIGHT_CAPS = {"1p0mu", "0p8mu"}

# Fixed eval seed offset (distinct from training eval offset +10_000).
DIAG_EVAL_SEED_OFFSET = 70_000


def ci95(xs: list[float] | np.ndarray) -> tuple[float, float, float]:
    """Return (mean, half-width CI95, n)."""
    arr = np.asarray(list(xs), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = float(arr.mean())
    if n < 2:
        return m, 0.0, n
    se = float(arr.std(ddof=1)) / math.sqrt(n)
    return m, 1.96 * se, n


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_index(path: Path | None = None) -> list[dict[str, Any]]:
    return json.loads((path or (M3_DIR / "index.json")).read_text())


def list_m3_runs(
    *,
    rationing: str | None = "proportional",
    capacity_tags: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    rows = load_index()
    out = []
    for r in rows:
        if r.get("regime") != "B":
            continue
        if rationing is not None and r.get("rationing") != rationing:
            continue
        tag = r.get("capacity_tag", "inf")
        if capacity_tags is not None and tag not in capacity_tags:
            continue
        if seeds is not None and int(r["seed"]) not in seeds:
            continue
        run_dir = M3_DIR / r["run"]
        ckpt = run_dir / "checkpoints"
        if not (ckpt / "policy_retailer.pt").exists():
            continue
        out.append({**r, "run_dir": str(run_dir)})
    return out


def load_trainer(run_dir: Path | str):
    """Load frozen IPPOTrainer from a completed run directory."""
    from beer_distribution_rl.agents.ippo import IPPOConfig, IPPOTrainer

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    fields = IPPOConfig.__dataclass_fields__
    cfg = IPPOConfig(**{k: v for k, v in meta["config"].items() if k in fields})
    trainer = IPPOTrainer(cfg)
    trainer.load(run_dir / "checkpoints")
    return trainer, meta


def signal_feature_slice(obs_dim: int, ship_delay: int = 2, order_delay: int = 1) -> slice:
    """Indices of delayed signal-board features in IPPO obs."""
    base = 6 + ship_delay + order_delay + 3
    return slice(base, obs_dim)


def inventory_index() -> int:
    return 0


@dataclass
class AblationCondition:
    name: str
    mode: str  # intact | zero | shuffle | random


ABLATIONS = [
    AblationCondition("intact", "intact"),
    AblationCondition("zeroed", "zero"),
    AblationCondition("shuffled", "shuffle"),
    AblationCondition("random", "random"),
]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
