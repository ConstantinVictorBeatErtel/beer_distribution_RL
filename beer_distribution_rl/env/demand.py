"""Demand processes for the beer distribution game.

Pluggable processes (env v1.1):
  - ``uniform`` / U[0,15] — retained for backward comparison only
  - ``ar1`` — training default (φ=0.7, μ=7.5); lag-1 R²≈φ² so demand carries info
  - ``regime_switch`` — two-state Markov (easy case for communication value)
  - ``classic_step`` — published baseline (4→8)

Only the retailer observes true consumer demand (as ``last_demand_or_order``).
Upstream roles see their own incoming orders only — never ``customer_demand``.
"""

from __future__ import annotations

import copy
import math
import random
import warnings
from dataclasses import dataclass, fields
from typing import Any, Literal, Mapping, Protocol

from beer_distribution_rl.env.core_types import Role

DemandName = Literal["uniform", "ar1", "regime_switch", "classic_step", "correlated_y"]

# B1 / v1.1 calibration: raise hard clamp so AR(1)+relative-Δ ratchets rarely bind.
DEFAULT_ORDER_CAP = 128
BOUNDARY_ACTION_WARN_FRAC = 0.05


class DemandProcess(Protocol):
    def reset(self, rng: random.Random) -> None: ...

    def __call__(self, t: int, rng: random.Random) -> int:
        """Return non-negative integer demand for week ``t`` (1-indexed)."""
        ...


@dataclass
class ClassicStepDemand:
    """Classic MIT step: demand ``pre`` for weeks < switch_week, else ``post``."""

    pre: int = 4
    post: int = 8
    switch_week: int = 5

    def reset(self, rng: random.Random) -> None:
        return None

    def __call__(self, t: int, rng: random.Random) -> int:
        return self.pre if t < self.switch_week else self.post

    @property
    def name(self) -> str:
        return "classic_step"


@dataclass
class UniformDemand:
    """Stationary uniform integer demand on [low, high] inclusive.

    Retained for backward comparison with M2/M3. Lag-1 R²≈0 — little to communicate.
    """

    low: int = 0
    high: int = 15

    def reset(self, rng: random.Random) -> None:
        return None

    def __call__(self, t: int, rng: random.Random) -> int:
        return rng.randint(self.low, self.high)

    @property
    def name(self) -> str:
        return "uniform"

    def theoretical_mean(self) -> float:
        return 0.5 * (self.low + self.high)

    def theoretical_var(self) -> float:
        n = self.high - self.low + 1
        return (n * n - 1) / 12.0

    def theoretical_lag1_corr(self) -> float:
        return 0.0


@dataclass
class AR1Demand:
    """Gaussian AR(1) demand rounded to non-negative ints.

    Training default: φ=0.7, μ=7.5, σ=2.0 (D6: lag-1 R²≈0.47 vs ≈0 for uniform).
    Optional one-shot mean shift via ``regime_shift_week`` / ``mu_after``.
    """

    mu: float = 7.5
    phi: float = 0.7
    sigma: float = 2.0
    regime_shift_week: int | None = None
    mu_after: float | None = None
    x0: float | None = None

    def __post_init__(self) -> None:
        self._x: float = self.mu if self.x0 is None else self.x0

    def reset(self, rng: random.Random) -> None:
        self._x = self.mu if self.x0 is None else self.x0

    def __call__(self, t: int, rng: random.Random) -> int:
        mu = self.mu
        if (
            self.regime_shift_week is not None
            and self.mu_after is not None
            and t >= self.regime_shift_week
        ):
            mu = self.mu_after
        eps = rng.gauss(0.0, self.sigma)
        self._x = mu + self.phi * (self._x - mu) + eps
        return max(0, int(round(self._x)))

    @property
    def name(self) -> str:
        return "ar1"

    def stationary_sigma(self) -> float:
        denom = 1.0 - self.phi * self.phi
        if denom <= 1e-12:
            return float("inf")
        return self.sigma / math.sqrt(denom)

    def theoretical_mean(self) -> float:
        return float(self.mu)

    def theoretical_var(self) -> float:
        s = self.stationary_sigma()
        return float(s * s)

    def theoretical_lag1_corr(self) -> float:
        return float(self.phi)

    def conditional_mean(self, d_t: float, t: int | None = None) -> float:
        """E[latent_{t+1} | latent_t ≈ d_t] under the AR(1) law (pre-rounding)."""
        mu = self.mu
        if (
            t is not None
            and self.regime_shift_week is not None
            and self.mu_after is not None
            and (t + 1) >= self.regime_shift_week
        ):
            mu = self.mu_after
        return mu + self.phi * (d_t - mu)


@dataclass
class RegimeSwitchDemand:
    """Two-state Markov demand (low / high mean) — high information value.

    Persistent regimes make a truthful current-demand broadcast highly informative
    about next-week demand (the "easy" communication case).
    """

    mu_low: float = 4.0
    mu_high: float = 12.0
    sigma: float = 1.5
    p_stay_low: float = 0.90
    p_stay_high: float = 0.90
    start_high: bool | None = None

    def __post_init__(self) -> None:
        self._high: bool = bool(self.start_high) if self.start_high is not None else False

    def reset(self, rng: random.Random) -> None:
        if self.start_high is None:
            # Stationary occupancy of the high state.
            p_lh = 1.0 - self.p_stay_low
            p_hl = 1.0 - self.p_stay_high
            pi_high = p_lh / (p_lh + p_hl) if (p_lh + p_hl) > 0 else 0.5
            self._high = rng.random() < pi_high
        else:
            self._high = bool(self.start_high)

    def __call__(self, t: int, rng: random.Random) -> int:
        if self._high:
            if rng.random() > self.p_stay_high:
                self._high = False
        else:
            if rng.random() > self.p_stay_low:
                self._high = True
        mu = self.mu_high if self._high else self.mu_low
        return max(0, int(round(rng.gauss(mu, self.sigma))))

    @property
    def name(self) -> str:
        return "regime_switch"

    def stationary_pi_high(self) -> float:
        p_lh = 1.0 - self.p_stay_low
        p_hl = 1.0 - self.p_stay_high
        if p_lh + p_hl <= 0:
            return 0.5
        return p_lh / (p_lh + p_hl)

    def theoretical_mean(self) -> float:
        pi_h = self.stationary_pi_high()
        return (1.0 - pi_h) * self.mu_low + pi_h * self.mu_high

    def theoretical_var(self) -> float:
        """Law of total variance: E[Var|s] + Var(E[·|s])."""
        pi_h = self.stationary_pi_high()
        pi_l = 1.0 - pi_h
        mean = self.theoretical_mean()
        within = self.sigma * self.sigma
        between = pi_l * (self.mu_low - mean) ** 2 + pi_h * (self.mu_high - mean) ** 2
        return within + between

    def theoretical_lag1_corr(self) -> float:
        """Approximate lag-1 corr of the latent (pre-rounding) mixture process."""
        pi_h = self.stationary_pi_high()
        pi_l = 1.0 - pi_h
        mean = self.theoretical_mean()
        var = self.theoretical_var()
        if var <= 1e-12:
            return 0.0
        # P(H_{t+1}|H_t)=p_stay_high, etc. Cov via regime persistence.
        # E[X_t X_{t+1}] = sum_{i,j} P(s_t=i)P(s_{t+1}=j|i) μ_i μ_j + σ² terms ≈
        # (innovation independent across t for different weeks' eps).
        e_prod = (
            pi_l * self.p_stay_low * self.mu_low * self.mu_low
            + pi_l * (1.0 - self.p_stay_low) * self.mu_low * self.mu_high
            + pi_h * self.p_stay_high * self.mu_high * self.mu_high
            + pi_h * (1.0 - self.p_stay_high) * self.mu_high * self.mu_low
        )
        # Cross-week innovations uncorrelated ⇒ Cov = E[μ_s μ_s'] - μ²
        cov = e_prod - mean * mean
        return float(cov / var)

    def conditional_mean(self, d_t: float) -> float:
        """Soft regime posterior given observed demand, then one-step predictive mean."""
        # Likelihood under each Gaussian regime (unnormalized).
        def _ll(mu: float) -> float:
            z = (d_t - mu) / max(self.sigma, 1e-6)
            return math.exp(-0.5 * z * z)

        pi_h = self.stationary_pi_high()
        w_h = pi_h * _ll(self.mu_high)
        w_l = (1.0 - pi_h) * _ll(self.mu_low)
        post_h = w_h / (w_h + w_l) if (w_h + w_l) > 0 else pi_h
        # Next-step regime probs
        p_h_next = post_h * self.p_stay_high + (1.0 - post_h) * (1.0 - self.p_stay_low)
        return (1.0 - p_h_next) * self.mu_low + p_h_next * self.mu_high


@dataclass
class CorrelatedYDemand:
    """Two retailers share a common AR(1) factor plus idiosyncratic noise.

    ``d_{i,t} = max(0, round(mu + common_t + eps_{i,t}))`` with
    ``common_t = phi * common_{t-1} + eta_t``.

    The shared factor makes a rival's cheap-talk signal informative (and
    therefore strategically worth lying about under shortage gaming / P3).
    """

    mu: float = 7.5
    phi: float = 0.7
    sigma_common: float = 2.0
    sigma_idio: float = 1.5
    roles: tuple[Role, Role] = (Role.RETAILER, Role.RETAILER_B)
    common0: float | None = None

    def __post_init__(self) -> None:
        self._common: float = 0.0 if self.common0 is None else float(self.common0)

    def reset(self, rng: random.Random) -> None:
        self._common = 0.0 if self.common0 is None else float(self.common0)

    def demands(self, t: int, rng: random.Random) -> dict[Role, int]:
        eta = rng.gauss(0.0, self.sigma_common)
        self._common = self.phi * self._common + eta
        out: dict[Role, int] = {}
        for role in self.roles:
            eps = rng.gauss(0.0, self.sigma_idio)
            out[role] = max(0, int(round(self.mu + self._common + eps)))
        return out

    def __call__(self, t: int, rng: random.Random) -> int:
        """Sum of both retailer streams (capacity μ calibration / single-int API)."""
        return int(sum(self.demands(t, rng).values()))

    @property
    def name(self) -> str:
        return "correlated_y"

    def theoretical_mean(self) -> float:
        return float(2.0 * self.mu)


@dataclass
class TwinCustomerDemand:
    """Independent single-stream processes for each Y-topology retailer.

    Used when the matrix asks for ``regime_switch`` (or ``ar1``) on Y without
    the shared-factor structure of ``CorrelatedYDemand``. Each customer gets its
    own process clone — signals about *rival* demand are less informative than
    under correlated_y, which is a useful ablation for P3.
    """

    base: DemandProcess
    roles: tuple[Role, ...] = (Role.RETAILER, Role.RETAILER_B)

    def __post_init__(self) -> None:
        self._streams: dict[Role, DemandProcess] = {
            r: copy.deepcopy(self.base) for r in self.roles
        }

    def reset(self, rng: random.Random) -> None:
        for i, r in enumerate(self.roles):
            # Offset child RNGs so streams diverge.
            child = random.Random(rng.randint(0, 2**31 - 1) + i * 9973)
            self._streams[r].reset(child)

    def demands(self, t: int, rng: random.Random) -> dict[Role, int]:
        out: dict[Role, int] = {}
        for i, r in enumerate(self.roles):
            child = random.Random(rng.randint(0, 2**31 - 1) + i)
            out[r] = int(self._streams[r](t, child))
        return out

    def __call__(self, t: int, rng: random.Random) -> int:
        return int(sum(self.demands(t, rng).values()))

    @property
    def name(self) -> str:
        base_name = getattr(self.base, "name", type(self.base).__name__)
        return f"twin_{base_name}"

    def theoretical_mean(self) -> float:
        if hasattr(self.base, "theoretical_mean"):
            return float(len(self.roles) * self.base.theoretical_mean())  # type: ignore[operator]
        return float("nan")


class MultiCustomerDemand(Protocol):
    """Demand streams keyed by customer-facing roles (Y-topology)."""

    def reset(self, rng: random.Random) -> None: ...

    def demands(self, t: int, rng: random.Random) -> Mapping[Role, int]: ...


def sample_customer_demands(
    process: DemandProcess | MultiCustomerDemand,
    t: int,
    rng: random.Random,
    customers: tuple[Role, ...],
) -> dict[Role, int]:
    """Normalize single- and multi-stream demand into a per-customer map."""
    if hasattr(process, "demands"):
        raw = process.demands(t, rng)  # type: ignore[attr-defined]
        return {c: int(raw[c]) for c in customers}
    if len(customers) != 1:
        raise ValueError(
            f"Single-stream demand cannot serve {len(customers)} customers; "
            "use CorrelatedYDemand (or another MultiCustomerDemand)."
        )
    return {customers[0]: int(process(t, rng))}  # type: ignore[operator]


def make_demand(name: str | DemandProcess, **kwargs: Any) -> DemandProcess:
    """Factory for named demand processes (pluggable env v1.1 API)."""
    if not isinstance(name, str):
        return name
    key = name.lower().replace("-", "_").replace("(", "").replace(")", "")
    if key in ("uniform", "u", "u015"):
        low = int(kwargs.pop("low", 0))
        high = int(kwargs.pop("high", 15))
        if kwargs:
            raise TypeError(f"unexpected kwargs for uniform: {kwargs}")
        return UniformDemand(low=low, high=high)
    if key in ("ar1", "ar"):
        allowed = {f.name for f in fields(AR1Demand)}
        return AR1Demand(**{k: v for k, v in kwargs.items() if k in allowed})
    if key in ("regime_switch", "regime", "markov", "two_state"):
        allowed = {f.name for f in fields(RegimeSwitchDemand)}
        return RegimeSwitchDemand(**{k: v for k, v in kwargs.items() if k in allowed})
    if key in ("classic_step", "classic", "step"):
        allowed = {f.name for f in fields(ClassicStepDemand)}
        return ClassicStepDemand(**{k: v for k, v in kwargs.items() if k in allowed})
    if key in ("correlated_y", "correlated", "y_demand", "y"):
        allowed = {f.name for f in fields(CorrelatedYDemand)}
        return CorrelatedYDemand(**{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(
        f"unknown demand process {name!r}; "
        "expected one of: uniform, ar1, regime_switch, classic_step, correlated_y"
    )


def resolve_matrix_demand(demand_name: str, topology: str) -> DemandProcess:
    """Map matrix demand × topology to a concrete process.

    - serial: ar1 / regime_switch as named
    - y + ar1: CorrelatedYDemand (shared factor — rival signals informative)
    - y + regime_switch: TwinCustomerDemand of independent RegimeSwitch streams
    """
    topo = topology.lower().strip()
    key = demand_name.lower().strip()
    if topo in ("y", "y_topology", "ytopology"):
        if key in ("ar1", "correlated_y", "correlated"):
            return make_demand("correlated_y")
        if key in ("regime_switch", "regime", "markov"):
            return TwinCustomerDemand(base=make_demand("regime_switch"))
        return make_demand(key)
    return make_demand(key)


def clone_demand(process: DemandProcess) -> DemandProcess:
    """Deep-copy a demand process (including latent state)."""
    return copy.deepcopy(process)


def mean_demand(process: DemandProcess, horizon: int, seed: int = 0) -> float:
    """Empirical mean demand over ``horizon`` weeks (for capacity sweeps)."""
    rng = random.Random(seed)
    proc = clone_demand(process)
    proc.reset(rng)
    total = 0
    for t in range(1, horizon + 1):
        total += int(proc(t, rng))
    return total / horizon


def sample_demand_series(
    process: DemandProcess,
    horizon: int,
    n_traj: int = 1,
    seed: int = 0,
) -> list[list[int]]:
    """Sample ``n_traj`` demand trajectories of length ``horizon``."""
    out: list[list[int]] = []
    for i in range(n_traj):
        rng = random.Random(seed + i)
        proc = clone_demand(process)
        proc.reset(rng)
        out.append([int(proc(t, rng)) for t in range(1, horizon + 1)])
    return out


def _mse(y_true: list[float], y_hat: list[float]) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    return sum((a - b) ** 2 for a, b in zip(y_true, y_hat)) / n


def _forecaster_next(process: DemandProcess, d_t: float, t: int) -> float:
    """One-step predictive mean under a fixed process-aware base-stock forecaster.

    Conditions on truthful current demand ``d_t`` only (the broadcast payload),
    not on privileged calendar knowledge beyond what the process law encodes.
    """
    if isinstance(process, AR1Demand):
        return process.conditional_mean(d_t, t=t)
    if isinstance(process, RegimeSwitchDemand):
        return process.conditional_mean(d_t)
    if isinstance(process, ClassicStepDemand):
        # After the step, demand is constant — persistence is the optimal forecast.
        # Before the step, persistence is also optimal week-to-week except at the jump.
        return float(d_t)
    if isinstance(process, UniformDemand):
        return process.theoretical_mean()
    return float(d_t)


def truthful_broadcast_info_value(
    process: DemandProcess,
    *,
    horizon: int = 52,
    n_traj: int = 2000,
    seed: int = 0,
) -> dict[str, float | str]:
    """Information value of a truthful retailer demand broadcast.

    Compares one-step forecast MSE for an upstream agent using a fixed base-stock
    forecaster:

    - **blind**: unconditional mean of the process (no channel)
    - **informed**: process-conditional E[d_{t+1} | d_t] given truthful current demand

    Returns absolute and relative forecast-error reduction (paper justification metric).
    """
    series = sample_demand_series(process, horizon=horizon, n_traj=n_traj, seed=seed)
    # Unconditional mean estimated on a held-out burn-in of the same process family.
    all_vals = [v for traj in series for v in traj]
    mu_hat = sum(all_vals) / len(all_vals)

    y_true: list[float] = []
    y_blind: list[float] = []
    y_informed: list[float] = []
    for traj in series:
        for t_idx in range(len(traj) - 1):
            d_t = float(traj[t_idx])
            d_next = float(traj[t_idx + 1])
            week = t_idx + 1  # 1-indexed week of d_t
            y_true.append(d_next)
            y_blind.append(mu_hat)
            y_informed.append(_forecaster_next(process, d_t, week))

    mse_blind = _mse(y_true, y_blind)
    mse_informed = _mse(y_true, y_informed)
    reduction = mse_blind - mse_informed
    rel = reduction / mse_blind if mse_blind > 1e-12 else 0.0

    # Lag-1 R² of the series (same definition as D6).
    xs = [float(traj[i]) for traj in series for i in range(len(traj) - 1)]
    ys = [float(traj[i + 1]) for traj in series for i in range(len(traj) - 1)]
    if len(xs) >= 2:
        x_bar = sum(xs) / len(xs)
        y_bar = sum(ys) / len(ys)
        var_x = sum((x - x_bar) ** 2 for x in xs)
        cov = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
        b = cov / var_x if var_x > 1e-12 else 0.0
        a = y_bar - b * x_bar
        ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - y_bar) ** 2 for y in ys)
        lag1_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    else:
        lag1_r2 = float("nan")

    name = getattr(process, "name", type(process).__name__)
    return {
        "process": str(name),
        "mse_blind": float(mse_blind),
        "mse_informed": float(mse_informed),
        "mse_reduction": float(reduction),
        "relative_mse_reduction": float(rel),
        "lag1_r2": float(lag1_r2),
        "mean": float(mu_hat),
        "var": float(sum((v - mu_hat) ** 2 for v in all_vals) / len(all_vals)),
        "n_pairs": float(len(y_true)),
    }


def info_value_table(
    *,
    horizon: int = 52,
    n_traj: int = 2000,
    seed: int = 0,
) -> dict[str, dict[str, float | str]]:
    """Recompute info value under each named process (paper table)."""
    processes = {
        "uniform": make_demand("uniform"),
        "ar1": make_demand("ar1"),
        "regime_switch": make_demand("regime_switch"),
        "classic_step": make_demand("classic_step"),
    }
    return {
        k: truthful_broadcast_info_value(p, horizon=horizon, n_traj=n_traj, seed=seed)
        for k, p in processes.items()
    }


def recommend_order_cap(
    process: DemandProcess | None = None,
    *,
    delta_max: int = 8,
    horizon: int = 52,
    z_hi: float = 3.0,
) -> dict[str, float | int | str]:
    """Re-derive env ``order_cap`` from the AR(1) demand distribution (B1 hand-off).

    Under relative actions ``order = clip(last + Δ, 0, cap)`` with Δ∈[-δ,δ], a hard
    clamp must sit above plausible bullwhip ratchets. B1 recommended **128**.
    """
    proc = process if process is not None else AR1Demand()
    if isinstance(proc, AR1Demand):
        d_hi = proc.mu + z_hi * proc.stationary_sigma()
        rationale = (
            f"AR(1) μ={proc.mu}, φ={proc.phi}, σ={proc.sigma}: "
            f"high demand ≈ μ+{z_hi}σ_stat = {d_hi:.1f}; "
            f"relative +{delta_max}/week over T={horizon} can ratchet far above 64; "
            f"cap={DEFAULT_ORDER_CAP} keeps the hard clamp rarely binding."
        )
    elif isinstance(proc, UniformDemand):
        d_hi = float(proc.high)
        rationale = f"Uniform high={proc.high}; cap={DEFAULT_ORDER_CAP} for parity with AR(1) default."
    elif isinstance(proc, RegimeSwitchDemand):
        d_hi = proc.mu_high + z_hi * proc.sigma
        rationale = (
            f"Regime-switch high≈{d_hi:.1f}; cap={DEFAULT_ORDER_CAP} matches AR(1) headroom."
        )
    else:
        d_hi = 16.0
        rationale = f"Fallback d_hi={d_hi}; cap={DEFAULT_ORDER_CAP}."

    # Absolute catch-up need under backlog recovery (B1 probe): L≈3 + buffer weeks.
    max_plausible_abs = d_hi * (3 + 8)
    return {
        "suggested_hard_cap": int(DEFAULT_ORDER_CAP),
        "d_hi_approx": float(d_hi),
        "max_plausible_need_abs": float(max_plausible_abs),
        "relative_ratchet_horizon": float(d_hi + horizon * delta_max),
        "rationale": rationale,
    }


def frac_actions_at_boundary(
    orders: list[int] | tuple[int, ...],
    order_cap: int,
) -> float:
    """Fraction of placed orders exactly equal to the hard action cap."""
    if not orders:
        return 0.0
    return sum(1 for o in orders if int(o) == int(order_cap)) / len(orders)


def warn_if_boundary_saturated(
    frac: float,
    *,
    threshold: float = BOUNDARY_ACTION_WARN_FRAC,
    context: str = "",
) -> None:
    """Log a warning when boundary hit-rate exceeds the healthy-run threshold (~5%)."""
    if frac > threshold:
        msg = (
            f"action-cap boundary fraction {frac:.3f} > {threshold:.2f} "
            f"(healthy runs ≈ 0). Consider raising order_cap."
        )
        if context:
            msg = f"{context}: {msg}"
        warnings.warn(msg, stacklevel=2)
