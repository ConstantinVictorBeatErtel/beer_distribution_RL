#!/usr/bin/env python3
"""D6 — Information value of the training demand process."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from beer_distribution_rl.env.demand import AR1Demand, UniformDemand  # noqa: E402

from analysis.diag.common import CACHE_DIR, FIG_DIR, ensure_dirs, write_json  # noqa: E402


def _series(process, horizon: int, n_traj: int, seed: int) -> np.ndarray:
    """Return array [n_traj, horizon] of integer demand."""
    out = np.zeros((n_traj, horizon), dtype=float)
    for i in range(n_traj):
        rng = __import__("random").Random(seed + i)
        process.reset(rng)
        # Fresh process copy-ish for AR1 state
        if isinstance(process, AR1Demand):
            proc = AR1Demand(
                mu=process.mu,
                phi=process.phi,
                sigma=process.sigma,
                regime_shift_week=process.regime_shift_week,
                mu_after=process.mu_after,
            )
        else:
            proc = UniformDemand(process.low, process.high)
        proc.reset(rng)
        for t in range(1, horizon + 1):
            out[i, t - 1] = proc(t, rng)
    return out


def _lag1_r2(series: np.ndarray) -> float:
    """R² of predicting d_{t+1} from d_t (pooled)."""
    x = series[:, :-1].ravel()
    y = series[:, 1:].ravel()
    if x.size < 2:
        return float("nan")
    # Linear regression y = a + b x
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def _lag1_corr(series: np.ndarray) -> float:
    x = series[:, :-1].ravel()
    y = series[:, 1:].ravel()
    if x.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _discrete_mi_bits(series: np.ndarray, max_val: int = 40) -> float:
    """Simple plug-in mutual information I(d_t; d_{t+1}) in bits."""
    x = series[:, :-1].ravel().astype(int)
    y = series[:, 1:].ravel().astype(int)
    x = np.clip(x, 0, max_val)
    y = np.clip(y, 0, max_val)
    n = x.size
    # joint
    joint = np.zeros((max_val + 1, max_val + 1), dtype=float)
    for a, b in zip(x, y):
        joint[a, b] += 1.0
    joint /= n
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in range(max_val + 1):
        for j in range(max_val + 1):
            pxy = joint[i, j]
            if pxy <= 0 or px[i] <= 0 or py[j] <= 0:
                continue
            mi += pxy * np.log2(pxy / (px[i] * py[j]))
    return float(mi)


def run() -> dict:
    ensure_dirs()
    horizon = 52
    n_traj = 2000
    seed = 0

    uni = UniformDemand(0, 15)
    ar1 = AR1Demand(mu=7.5, phi=0.7, sigma=2.0)

    s_uni = _series(uni, horizon, n_traj, seed)
    s_ar1 = _series(ar1, horizon, n_traj, seed + 10_000)

    result = {
        "uniform": {
            "name": "U[0,15]",
            "lag1_r2": _lag1_r2(s_uni),
            "lag1_corr": _lag1_corr(s_uni),
            "mi_bits": _discrete_mi_bits(s_uni, max_val=15),
            "mean": float(s_uni.mean()),
            "var": float(s_uni.var()),
        },
        "ar1": {
            "name": "AR(1) φ=0.7, μ=7.5, σ=2",
            "lag1_r2": _lag1_r2(s_ar1),
            "lag1_corr": _lag1_corr(s_ar1),
            "mi_bits": _discrete_mi_bits(s_ar1, max_val=40),
            "mean": float(s_ar1.mean()),
            "var": float(s_ar1.var()),
            "theoretical_lag1_corr": 0.7,
            "theoretical_r2": 0.7**2,
        },
        "paragraph": "",
    }

    u = result["uniform"]
    a = result["ar1"]
    result["paragraph"] = (
        f"Under training demand U[0,15], next-week demand is essentially unpredictable from "
        f"this week: lag-1 R²≈{u['lag1_r2']:.3f}, corr≈{u['lag1_corr']:.3f}, "
        f"I(d_t;d_{{t+1}})≈{u['mi_bits']:.3f} bits. A truthful demand broadcast therefore "
        f"cannot reduce an upstream agent's one-step forecast error in principle — there is "
        f"almost nothing to communicate about future demand. By contrast, AR(1) with φ=0.7 "
        f"(proposed v1.1) has lag-1 R²≈{a['lag1_r2']:.3f} (theory φ²={a['theoretical_r2']:.2f}) "
        f"and MI≈{a['mi_bits']:.2f} bits, so a truthful current-demand signal is informative "
        f"about next week and creates a real incentive for listening when capacity/rationing "
        f"also binds."
    )

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    metrics = ["lag1_r2", "mi_bits"]
    titles = ["Lag-1 R² (predict d_{t+1} from d_t)", "I(d_t; d_{t+1}) bits"]
    for ax, key, title in zip(axes, metrics, titles):
        ax.bar(
            ["U[0,15]", "AR(1) φ=0.7"],
            [result["uniform"][key], result["ar1"][key]],
            color=["#6b7c8a", "#2f6f4e"],
        )
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("D6 demand information value", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "d6_demand_info.png", dpi=160)
    plt.close(fig)

    # Scatter lag plots
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, series, name in zip(axes, [s_uni, s_ar1], ["U[0,15]", "AR(1) φ=0.7"]):
        x = series[:50, :-1].ravel()
        y = series[:50, 1:].ravel()
        ax.scatter(x, y, s=6, alpha=0.25, color="#4a6fa5")
        ax.set_xlabel("d_t")
        ax.set_ylabel("d_{t+1}")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "d6_lag_scatter.png", dpi=160)
    plt.close(fig)

    interpretation = (
        "demand uninformative (STRUCTURAL)"
        if abs(u["lag1_r2"]) < 0.05
        else "demand informative"
    )
    result["interpretation_key"] = interpretation
    write_json(CACHE_DIR / "d6_result.json", result)
    return result


if __name__ == "__main__":
    out = run()
    print(out["paragraph"])
    print("→", out["interpretation_key"])
