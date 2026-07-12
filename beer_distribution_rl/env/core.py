"""Beer Distribution Game state transition, costs, and delays."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Mapping

from beer_distribution_rl.env.core_types import ROLE_NAMES, ROLES, Role
from beer_distribution_rl.env.demand import ClassicStepDemand, DemandProcess
from beer_distribution_rl.env.rationing import (
    ProportionalRationing,
    RationContext,
    RationingPolicy,
)
from beer_distribution_rl.env.signals import HonestyMetrics, Signal, SignalChannel

# Re-export for callers expecting types from core.
__all__ = [
    "BeerGameCore",
    "EnvConfig",
    "ROLES",
    "Role",
    "RoleCosts",
    "RoleState",
    "Signal",
    "StepInfo",
    "classic_env_config",
    "dqn_paper_env_config",
]


@dataclass(frozen=True)
class RoleCosts:
    holding: float
    backlog: float


def _classic_costs() -> tuple[RoleCosts, ...]:
    return tuple(RoleCosts(holding=0.5, backlog=1.0) for _ in ROLES)


@dataclass
class RoleState:
    inventory: int
    backlog: int
    ship_pipeline: list[int]
    order_pipeline: list[int]
    on_order: int = 0  # outstanding replenishment (ordered, not yet received)
    last_order_placed: int = 0
    last_shipment_received: int = 0
    last_demand_or_order: int = 0

    def inventory_position(self) -> int:
        """On-hand − backlog + on-order (classic base-stock inventory position)."""
        return self.inventory - self.backlog + self.on_order

    def copy(self) -> RoleState:
        return RoleState(
            inventory=self.inventory,
            backlog=self.backlog,
            ship_pipeline=list(self.ship_pipeline),
            order_pipeline=list(self.order_pipeline),
            on_order=self.on_order,
            last_order_placed=self.last_order_placed,
            last_shipment_received=self.last_shipment_received,
            last_demand_or_order=self.last_demand_or_order,
        )


@dataclass(frozen=True)
class EnvConfig:
    horizon: int = 52
    order_cap: int = 64
    ship_delay: int = 2
    order_delay: int = 1
    costs: tuple[RoleCosts, ...] = field(default_factory=_classic_costs)
    demand: DemandProcess = field(default_factory=ClassicStepDemand)
    capacity: float | None = None
    rationing: RationingPolicy = field(default_factory=ProportionalRationing)
    signaling_enabled: bool = False
    regime: Literal["A", "B", "C"] = "A"
    init_inventory: tuple[int, ...] = (12, 12, 12, 12)
    init_pipeline_ship: int = 4
    init_pipeline_order: int = 4
    seed: int | None = None
    honesty_ema_alpha: float = 0.2


@dataclass
class StepInfo:
    shipments: dict[Role, int]
    orders_placed: dict[Role, int]
    factory_production: int
    rationed: bool
    signals_sent: dict[Role, Signal | None]
    signals_received: dict[Role, dict[Role, Signal | None]]
    honesty: dict[Role, dict[str, float]]
    local_costs: dict[Role, float]
    system_cost: float
    orders_clamped: dict[Role, bool] = field(default_factory=dict)
    incoming_orders: dict[Role, int] = field(default_factory=dict)
    shipments_received: dict[Role, int] = field(default_factory=dict)


def classic_env_config(**overrides) -> EnvConfig:
    """Research classic config (PROJECT_SPEC): step 4→8, h=0.5, b=1.0, L_s=2, L_o=1."""
    base = dict(
        horizon=36,
        demand=ClassicStepDemand(),
        costs=_classic_costs(),
        ship_delay=2,
        order_delay=1,
        regime="A",
        signaling_enabled=False,
        capacity=None,
    )
    base.update(overrides)
    return EnvConfig(**base)


def dqn_paper_env_config(**overrides) -> EnvConfig:
    """Oroojlooy et al. arXiv:1708.05924 §4 validation config.

    Demand U{0,1,2}, L_s=L_o=2, ch=[2,2,2,2], cp=[2,0,0,0].
    Published all-base-stock mean cost/period ≈ 2.008 with S=[9,5,3,1].
    """
    from beer_distribution_rl.env.demand import UniformDemand

    base = dict(
        horizon=36,
        demand=UniformDemand(0, 2),
        costs=(
            RoleCosts(holding=2.0, backlog=2.0),
            RoleCosts(holding=2.0, backlog=0.0),
            RoleCosts(holding=2.0, backlog=0.0),
            RoleCosts(holding=2.0, backlog=0.0),
        ),
        ship_delay=2,
        order_delay=2,
        regime="A",
        signaling_enabled=False,
        capacity=None,
        init_inventory=(0, 0, 0, 0),
        init_pipeline_ship=0,
        init_pipeline_order=0,
    )
    base.update(overrides)
    return EnvConfig(**base)


class BeerGameCore:
    """Faithful serial beer-game simulator with optional capacity and cheap talk."""

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        if len(self.config.costs) != 4:
            raise ValueError("costs must have length 4 (one per role)")
        if len(self.config.init_inventory) != 4:
            raise ValueError("init_inventory must have length 4")
        self._rng = random.Random(self.config.seed)
        self._t = 0
        self._states: dict[Role, RoleState] = {}
        self._channel = SignalChannel(delay=1)
        self._honesty_ema: dict[Role, float] = {r: 0.0 for r in ROLES}
        self._terminated = False
        self._last_signal_board: dict[Role, Signal | None] = {r: None for r in ROLES}

    @property
    def t(self) -> int:
        return self._t

    @property
    def states(self) -> dict[Role, RoleState]:
        return {r: s.copy() for r, s in self._states.items()}

    def reset(self, seed: int | None = None) -> dict[Role, RoleState]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self.config.seed is not None:
            self._rng = random.Random(self.config.seed)
        self.config.demand.reset(self._rng)
        self._t = 0
        self._terminated = False
        self._honesty_ema = {r: 0.0 for r in ROLES}
        self._channel.reset()
        self._last_signal_board = {r: None for r in ROLES}
        cfg = self.config
        self._states = {}
        for role in ROLES:
            ship_pipe = [cfg.init_pipeline_ship] * cfg.ship_delay
            self._states[role] = RoleState(
                inventory=int(cfg.init_inventory[int(role)]),
                backlog=0,
                ship_pipeline=ship_pipe,
                order_pipeline=[cfg.init_pipeline_order] * cfg.order_delay,
                # Goods already in transit count as on-order at reset.
                on_order=sum(ship_pipe),
            )
        return self.states

    def observe(self, role: Role) -> dict:
        """Local observation — no privileged cross-role inventory."""
        s = self._states[role]
        costs = self.config.costs[int(role)]
        obs: dict = {
            "role": int(role),
            "role_name": ROLE_NAMES[role],
            "t": self._t,
            "inventory": s.inventory,
            "backlog": s.backlog,
            "ship_pipeline": list(s.ship_pipeline),
            "order_pipeline": list(s.order_pipeline),
            "last_order_placed": s.last_order_placed,
            "last_shipment_received": s.last_shipment_received,
            "last_demand_or_order": s.last_demand_or_order,
            "inventory_position": s.inventory_position(),
            "on_order": s.on_order,
            "holding_cost": costs.holding,
            "backlog_cost": costs.backlog,
            "order_cap": self.config.order_cap,
            "regime": self.config.regime,
        }
        if self.config.signaling_enabled:
            # Delayed board as last received; empty until first step fills it.
            obs["signals"] = {ROLE_NAMES[r]: None for r in ROLES}
        return obs

    def step(
        self,
        orders: Mapping[Role, int],
        signals: Mapping[Role, Signal | None] | None = None,
    ) -> tuple[dict[Role, RoleState], dict[Role, float], bool, StepInfo]:
        if not self._states:
            self.reset()
        if self._terminated:
            raise RuntimeError("Episode terminated; call reset()")

        cfg = self.config
        # Week index for demand is 1-based after increment; we are about to play week t+1.
        week = self._t + 1

        # --- 1. Advance ship pipelines; receive shipments ---
        received: dict[Role, int] = {}
        for role in ROLES:
            st = self._states[role]
            incoming_ship = st.ship_pipeline.pop(0) if st.ship_pipeline else 0
            received[role] = incoming_ship
            st.last_shipment_received = incoming_ship
            st.on_order = max(0, st.on_order - incoming_ship)

        # --- 2. Advance order pipelines / customer demand ---
        incoming: dict[Role, int] = {}
        customer_demand = int(cfg.demand(week, self._rng))
        for role in ROLES:
            st = self._states[role]
            if role == Role.RETAILER:
                inc = customer_demand
            else:
                inc = st.order_pipeline.pop(0) if st.order_pipeline else 0
            incoming[role] = inc
            st.last_demand_or_order = inc

        # --- 3. Resolve shipments (downstream → upstream fill) ---
        # Each role ships to its single downstream customer (serial chain).
        # Available goods = on-hand + just-received. Need = backlog + incoming order.
        shipments: dict[Role, int] = {}
        rationed = False
        for role in ROLES:
            st = self._states[role]
            available = st.inventory + received[role]
            need = st.backlog + incoming[role]

            if role == Role.FACTORY and cfg.capacity is not None:
                # Capacity limits how much the factory can liberate this week from
                # inventory+receipts toward filling demand (production is separate).
                # Spec: factory production capped at C; for fill we still use inventory.
                pass

            ship = min(available, need)
            if ship < need:
                rationed = True

            # Serial chain: single claimant — rationing policies agree; still call
            # for API uniformity / future multi-downstream topologies.
            alloc = cfg.rationing.allocate(
                {role: need},
                available,
                RationContext(honesty_ema=dict(self._honesty_ema)),
            )
            ship = int(alloc.get(role, ship))
            ship = max(0, min(ship, available, need))
            shipments[role] = ship

            leftover = available - ship
            new_backlog = need - ship
            st.inventory = leftover
            st.backlog = new_backlog
            assert st.inventory >= 0 and st.backlog >= 0

        # --- 4. Push shipments into downstream ship pipelines ---
        # Role i ships to role i-1; retailer ships to customers (leaves system).
        for role in ROLES:
            qty = shipments[role]
            if role == Role.RETAILER:
                continue  # sold to end customers
            downstream = Role(int(role) - 1)
            self._states[downstream].ship_pipeline.append(qty)

        # Ensure ship pipeline lengths: after pop, append exactly once per role that
        # receives from upstream. Factory receives via production in step 5.
        # Non-factory roles that didn't get an append yet (shouldn't happen): pad.
        for role in ROLES:
            st = self._states[role]
            # Downstream roles get append from upstream shipment above.
            # Factory pipeline is filled in step 5.
            while len(st.ship_pipeline) < cfg.ship_delay - 1:
                # After pop, length is ship_delay-1 before append; upstream append
                # restores ship_delay. If missing, pad with 0 (should not occur).
                break

        # --- 5. Accept orders; push into upstream order pipelines / production ---
        orders_clamped: dict[Role, bool] = {}
        orders_placed: dict[Role, int] = {}
        factory_production = 0

        for role in ROLES:
            raw = int(orders.get(role, 0))
            clamped = raw < 0 or raw > cfg.order_cap
            qty = max(0, min(cfg.order_cap, raw))
            orders_clamped[role] = clamped or (raw != qty)
            orders_placed[role] = qty
            self._states[role].last_order_placed = qty

            if role == Role.FACTORY:
                # Production enters factory ship pipeline (lead time = ship_delay).
                prod = qty
                if cfg.capacity is not None:
                    cap = int(cfg.capacity)
                    if prod > cap:
                        rationed = True
                        prod = cap
                factory_production = prod
                self._states[Role.FACTORY].on_order += prod
                self._states[Role.FACTORY].ship_pipeline.append(prod)
            else:
                self._states[role].on_order += qty
                upstream = Role(int(role) + 1)
                self._states[upstream].order_pipeline.append(qty)

        # Pad order pipelines to order_delay after pop+optional append.
        for role in ROLES:
            st = self._states[role]
            # Retailer never receives via order_pipeline; keep length stable with 0s.
            if role == Role.RETAILER:
                while len(st.order_pipeline) < cfg.order_delay:
                    st.order_pipeline.append(0)
            else:
                while len(st.order_pipeline) < cfg.order_delay:
                    st.order_pipeline.append(0)
            while len(st.ship_pipeline) < cfg.ship_delay:
                st.ship_pipeline.append(0)
            # Truncate if somehow over-length.
            if len(st.ship_pipeline) > cfg.ship_delay:
                st.ship_pipeline = st.ship_pipeline[: cfg.ship_delay]
            if len(st.order_pipeline) > cfg.order_delay:
                st.order_pipeline = st.order_pipeline[: cfg.order_delay]

        # --- 6. Signals (optional, unverified, delayed) ---
        signals_sent: dict[Role, Signal | None] = {r: None for r in ROLES}
        signals_received: dict[Role, dict[Role, Signal | None]] = {
            r: {o: None for o in ROLES} for r in ROLES
        }
        honesty_raw: dict[Role, HonestyMetrics] = {}
        honesty_out: dict[Role, dict[str, float]] = {}

        if cfg.signaling_enabled:
            if signals is None:
                signals = {r: None for r in ROLES}
            signals_sent = {r: signals.get(r) for r in ROLES}
            truths = {
                r: {
                    "demand": incoming[r],
                    "inventory": self._states[r].inventory,
                }
                for r in ROLES
            }
            honesty_raw = self._channel.measure_honesty(signals_sent, truths)
            # Update EMA of −mean_abs_error (higher ⇒ more honest). None ⇒ no update.
            alpha = cfg.honesty_ema_alpha
            for r, hm in honesty_raw.items():
                if hm.mean_abs_error is not None:
                    score = -float(hm.mean_abs_error)
                    self._honesty_ema[r] = (1 - alpha) * self._honesty_ema[r] + alpha * score
                honesty_out[r] = {
                    "abs_demand_error": float(hm.abs_demand_error)
                    if hm.abs_demand_error is not None
                    else float("nan"),
                    "abs_inventory_error": float(hm.abs_inventory_error)
                    if hm.abs_inventory_error is not None
                    else float("nan"),
                    "mean_abs_error": float(hm.mean_abs_error)
                    if hm.mean_abs_error is not None
                    else float("nan"),
                    "honesty_ema": self._honesty_ema[r],
                }
            signals_received = self._channel.receive()
            self._last_signal_board = dict(signals_received[Role.RETAILER])
            self._channel.send(signals_sent)
        elif signals is not None and any(v is not None for v in signals.values()):
            # Signals provided but channel disabled — ignore (do not validate truth).
            signals_sent = {r: signals.get(r) for r in ROLES}

        # --- 7. Accrue costs / rewards ---
        local_costs: dict[Role, float] = {}
        for role in ROLES:
            st = self._states[role]
            c = cfg.costs[int(role)]
            local_costs[role] = c.holding * st.inventory + c.backlog * st.backlog
        system_cost = float(sum(local_costs.values()))

        if cfg.regime == "C":
            rewards = {r: -system_cost for r in ROLES}
        else:
            # Regimes A and B: strictly local costs. Honesty never enters reward.
            rewards = {r: -local_costs[r] for r in ROLES}

        self._t = week
        terminated = self._t >= cfg.horizon
        self._terminated = terminated

        info = StepInfo(
            shipments=shipments,
            orders_placed=orders_placed,
            factory_production=factory_production,
            rationed=rationed,
            signals_sent=signals_sent,
            signals_received=signals_received,
            honesty=honesty_out,
            local_costs=local_costs,
            system_cost=system_cost,
            orders_clamped=orders_clamped,
            incoming_orders=incoming,
            shipments_received=received,
        )
        return self.states, rewards, terminated, info
