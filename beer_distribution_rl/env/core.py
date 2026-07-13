"""Beer Distribution Game state transition, costs, and delays.

Generalized from a hardcoded serial chain to a DAG of nodes (``env/topology.py``).
The classic 4-node serial config remains the default so prior results stay
reproducible (golden-trajectory regression). Y-topology adds two competing
retailers under one wholesaler — the multi-claimant structure required for P3.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Mapping

from beer_distribution_rl.env.core_types import ROLE_NAMES, ROLES, Role, Y_ROLES
from beer_distribution_rl.env.demand import (
    ClassicStepDemand,
    CorrelatedYDemand,
    DEFAULT_ORDER_CAP,
    DemandProcess,
    sample_customer_demands,
)
from beer_distribution_rl.env.rationing import (
    ProportionalRationing,
    RationContext,
    RationingPolicy,
)
from beer_distribution_rl.env.signals import HonestyMetrics, Signal, SignalChannel
from beer_distribution_rl.env.topology import Topology, get_topology, serial_topology

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
    "y_topology_env_config",
]


@dataclass(frozen=True)
class RoleCosts:
    holding: float
    backlog: float


def _classic_costs(roles: tuple[Role, ...] = ROLES) -> tuple[RoleCosts, ...]:
    n = max(int(r) for r in roles) + 1
    out = [RoleCosts(holding=0.0, backlog=0.0) for _ in range(n)]
    for r in roles:
        out[int(r)] = RoleCosts(holding=0.5, backlog=1.0)
    return tuple(out)


def _default_init_inventory(roles: tuple[Role, ...] = ROLES, fill: int = 12) -> tuple[int, ...]:
    n = max(int(r) for r in roles) + 1
    out = [0] * n
    for r in roles:
        out[int(r)] = fill
    return tuple(out)


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
    # Multi-claimant bookkeeping (populated for nodes with downstream roles).
    # On a serial chain each dict has a single key — physically equivalent to scalars.
    claimant_backlog: dict[Role, int] = field(default_factory=dict)
    order_pipelines: dict[Role, list[int]] = field(default_factory=dict)

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
            claimant_backlog=dict(self.claimant_backlog),
            order_pipelines={k: list(v) for k, v in self.order_pipelines.items()},
        )


@dataclass(frozen=True)
class EnvConfig:
    horizon: int = 52
    # v1.1 (B1): raise hard clamp so AR(1)+relative-Δ ratchets rarely bind.
    order_cap: int = DEFAULT_ORDER_CAP
    ship_delay: int = 2
    order_delay: int = 1
    costs: tuple[RoleCosts, ...] = field(default_factory=_classic_costs)
    demand: DemandProcess = field(default_factory=ClassicStepDemand)
    capacity: float | None = None
    rationing: RationingPolicy = field(default_factory=ProportionalRationing)
    signaling_enabled: bool = False
    regime: Literal["A", "B", "C"] = "A"
    init_inventory: tuple[int, ...] = field(default_factory=_default_init_inventory)
    init_pipeline_ship: int = 4
    init_pipeline_order: int = 4
    seed: int | None = None
    honesty_ema_alpha: float = 0.2
    # "serial" (default) or "y" / Topology — serial keeps all prior results reproducible.
    topology: str | Topology = "serial"


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
    customer_demand: int | None = None
    frac_actions_at_cap: float = 0.0
    # Y-topology / multi-claimant diagnostics
    customer_demands: dict[Role, int] = field(default_factory=dict)
    allocations: dict[Role, dict[Role, int]] = field(default_factory=dict)
    # sender → roles that heard the delayed board this week (includes rivals)
    signal_listeners: dict[Role, tuple[Role, ...]] = field(default_factory=dict)
    # D5 / Tier-1 logs: first-class bind events (no recompute from checkpoints)
    capacity_binds: bool = False
    allocation_triggers: bool = False


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
        topology="serial",
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
        topology="serial",
    )
    base.update(overrides)
    return EnvConfig(**base)


def y_topology_env_config(**overrides) -> EnvConfig:
    """Two competing retailers → Wholesaler → Distributor → Factory.

    Demand: correlated AR(1) factor + idiosyncratic noise (rival signals informative).
    """
    roles = Y_ROLES
    base = dict(
        horizon=52,
        demand=CorrelatedYDemand(),
        costs=_classic_costs(roles),
        init_inventory=_default_init_inventory(roles, 12),
        ship_delay=2,
        order_delay=1,
        regime="A",
        signaling_enabled=False,
        capacity=None,
        topology="y",
    )
    base.update(overrides)
    return EnvConfig(**base)


def _resolve_topology(spec: str | Topology) -> Topology:
    if isinstance(spec, Topology):
        return spec
    return get_topology(str(spec))


def _aggregate_pipelines(pipes: dict[Role, list[int]], delay: int) -> list[int]:
    if not pipes:
        return [0] * delay
    out = [0] * delay
    for pipe in pipes.values():
        for i, v in enumerate(pipe):
            if i < delay:
                out[i] += int(v)
    return out


class BeerGameCore:
    """Beer-game simulator on a role DAG with optional capacity and cheap talk."""

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.topology = _resolve_topology(self.config.topology)
        self.roles = self.topology.roles
        self.role_names = self.topology.role_names
        n_cost = max(int(r) for r in self.roles) + 1
        if len(self.config.costs) < n_cost:
            raise ValueError(
                f"costs length {len(self.config.costs)} < required {n_cost} for topology "
                f"{self.topology.name!r}"
            )
        if len(self.config.init_inventory) < n_cost:
            raise ValueError(
                f"init_inventory length {len(self.config.init_inventory)} < required {n_cost}"
            )
        self._rng = random.Random(self.config.seed)
        self._t = 0
        self._states: dict[Role, RoleState] = {}
        self._channel = SignalChannel(delay=1, roles=self.roles)
        self._honesty_ema: dict[Role, float] = {r: 0.0 for r in self.roles}
        self._terminated = False
        self._last_signal_board: dict[Role, Signal | None] = {r: None for r in self.roles}
        # Private: true consumer demand(s). Never copied into upstream observations.
        self._last_customer_demand: int | None = None
        self._last_customer_demands: dict[Role, int] = {}
        self._boundary_hits: int = 0
        self._boundary_orders: int = 0

    @property
    def t(self) -> int:
        return self._t

    @property
    def states(self) -> dict[Role, RoleState]:
        return {r: s.copy() for r, s in self._states.items()}

    def _cost(self, role: Role) -> RoleCosts:
        return self.config.costs[int(role)]

    def _init_inv(self, role: Role) -> int:
        return int(self.config.init_inventory[int(role)])

    def _outstanding_from_upstream(self, role: Role) -> int:
        """Orders in transit to upstream + unfilled backlog owed to ``role``."""
        up_role = self.topology.upstream[role]
        if up_role is None:
            return 0
        up = self._states[up_role]
        if role in up.order_pipelines:
            in_transit = sum(up.order_pipelines[role])
        else:
            # Serial-compat fallback when pipelines not yet split.
            in_transit = sum(up.order_pipeline)
        owed = int(up.claimant_backlog.get(role, 0))
        return in_transit + owed

    def _sync_aggregate_order_pipeline(self, role: Role) -> None:
        st = self._states[role]
        if st.order_pipelines:
            st.order_pipeline = _aggregate_pipelines(
                st.order_pipelines, self.config.order_delay
            )
        elif self.topology.is_customer(role):
            while len(st.order_pipeline) < self.config.order_delay:
                st.order_pipeline.append(0)
            if len(st.order_pipeline) > self.config.order_delay:
                st.order_pipeline = st.order_pipeline[: self.config.order_delay]

    def reset(self, seed: int | None = None) -> dict[Role, RoleState]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self.config.seed is not None:
            self._rng = random.Random(self.config.seed)
        self.config.demand.reset(self._rng)
        self._t = 0
        self._terminated = False
        self._honesty_ema = {r: 0.0 for r in self.roles}
        self._channel = SignalChannel(delay=1, roles=self.roles)
        self._channel.reset()
        self._last_signal_board = {r: None for r in self.roles}
        self._last_customer_demand = None
        self._last_customer_demands = {}
        self._boundary_hits = 0
        self._boundary_orders = 0
        cfg = self.config
        topo = self.topology
        self._states = {}
        for role in self.roles:
            ship_pipe = [cfg.init_pipeline_ship] * cfg.ship_delay
            claimants = topo.downstream[role]
            if topo.is_customer(role):
                # Customer-facing: exogenous demand, no inbound order pipelines.
                order_pipes: dict[Role, list[int]] = {}
                order_pipe = [0] * cfg.order_delay
                claimant_bl: dict[Role, int] = {}
            else:
                order_pipes = {
                    c: [cfg.init_pipeline_order] * cfg.order_delay for c in claimants
                }
                order_pipe = _aggregate_pipelines(order_pipes, cfg.order_delay)
                claimant_bl = {c: 0 for c in claimants}
            self._states[role] = RoleState(
                inventory=self._init_inv(role),
                backlog=0,
                ship_pipeline=ship_pipe,
                order_pipeline=order_pipe,
                on_order=0,
                claimant_backlog=claimant_bl,
                order_pipelines=order_pipes,
            )
        # on_order = own ship pipeline + outstanding at upstream (B1: include order delay).
        for role in self.roles:
            ship_sum = sum(self._states[role].ship_pipeline)
            if topo.is_factory(role):
                self._states[role].on_order = ship_sum
            else:
                self._states[role].on_order = ship_sum + self._outstanding_from_upstream(role)
        return self.states

    def observe(self, role: Role) -> dict:
        """Local observation — no privileged cross-role inventory / demand.

        Information asymmetry (non-negotiable for the cheap-talk premise):
        only customer-facing roles see true consumer demand, via
        ``last_demand_or_order``. Upstream roles see their own incoming *orders*
        under that same key. True consumer demand is never placed in upstream
        observation vectors.
        """
        s = self._states[role]
        costs = self._cost(role)
        obs: dict = {
            "role": int(role),
            "role_name": self.role_names.get(role, ROLE_NAMES.get(role, str(role))),
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
            "topology": self.topology.name,
        }
        assert "customer_demand" not in obs
        assert "true_demand" not in obs
        if self.config.signaling_enabled:
            # Delayed board as last received; empty until first step fills it.
            # Values are claimed (unverified) signals only — never ground-truth demand.
            # Includes rival retailers on Y-topology.
            obs["signals"] = {
                self.role_names.get(r, ROLE_NAMES.get(r, str(r))): None for r in self.roles
            }
        return obs

    @property
    def last_customer_demand(self) -> int | None:
        """True consumer demand (sum) from the most recent step (diagnostics only)."""
        return self._last_customer_demand

    def boundary_action_fraction(self) -> float:
        """Fraction of placed orders at the hard ``order_cap`` since last reset."""
        if self._boundary_orders == 0:
            return 0.0
        return self._boundary_hits / self._boundary_orders

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
        topo = self.topology
        week = self._t + 1

        # --- 1. Advance ship pipelines; receive shipments ---
        received: dict[Role, int] = {}
        for role in self.roles:
            st = self._states[role]
            incoming_ship = st.ship_pipeline.pop(0) if st.ship_pipeline else 0
            received[role] = incoming_ship
            st.last_shipment_received = incoming_ship
            st.on_order = max(0, st.on_order - incoming_ship)

        # --- 2. Advance order pipelines / customer demand ---
        customer_demands = sample_customer_demands(
            cfg.demand, week, self._rng, topo.customers
        )
        customer_demand_sum = int(sum(customer_demands.values()))
        self._last_customer_demand = customer_demand_sum
        self._last_customer_demands = dict(customer_demands)

        incoming: dict[Role, int] = {}
        incoming_by_claimant: dict[Role, dict[Role, int]] = {}
        for role in self.roles:
            st = self._states[role]
            if topo.is_customer(role):
                inc = int(customer_demands[role])
                incoming[role] = inc
                incoming_by_claimant[role] = {role: inc}
            else:
                by_c: dict[Role, int] = {}
                for c in topo.downstream[role]:
                    pipe = st.order_pipelines.get(c)
                    if pipe is None:
                        by_c[c] = 0
                    else:
                        by_c[c] = pipe.pop(0) if pipe else 0
                incoming_by_claimant[role] = by_c
                incoming[role] = int(sum(by_c.values()))
            st.last_demand_or_order = incoming[role]

        # --- 3. Resolve shipments (multi-claimant rationing where applicable) ---
        # On a serial chain each node has one claimant, so all three rationing
        # policies are identical (identity fill). Y-topology wholesaler is where
        # they diverge — see env/rationing.py module docstring.
        shipments: dict[Role, int] = {}
        allocations: dict[Role, dict[Role, int]] = {}
        rationed = False
        honesty_ctx = RationContext(honesty_ema=dict(self._honesty_ema))

        for role in self.roles:
            st = self._states[role]
            available = st.inventory + received[role]
            by_c = incoming_by_claimant[role]

            if topo.is_customer(role):
                # Exogenous customer: single virtual claimant keyed by self.
                need = st.backlog + by_c[role]
                requested = {role: need}
            else:
                requested = {
                    c: int(st.claimant_backlog.get(c, 0)) + int(by_c.get(c, 0))
                    for c in topo.downstream[role]
                }
                need = int(sum(requested.values()))

            if need > available:
                rationed = True

            alloc = cfg.rationing.allocate(requested, available, honesty_ctx)
            # Belt-and-suspenders: respect per-claimant caps; policies already
            # guarantee sum(alloc) ≤ available.
            alloc = {
                c: max(0, min(int(alloc.get(c, 0)), int(requested[c])))
                for c in requested
            }
            if sum(alloc.values()) > available:
                alloc = ProportionalRationing().allocate(requested, available, honesty_ctx)

            ship_total = int(sum(alloc.values()))
            shipments[role] = ship_total
            allocations[role] = dict(alloc)

            st.inventory = available - ship_total
            if topo.is_customer(role):
                st.backlog = need - ship_total
                st.claimant_backlog = {}
            else:
                st.claimant_backlog = {
                    c: int(requested[c]) - int(alloc.get(c, 0)) for c in requested
                }
                st.backlog = int(sum(st.claimant_backlog.values()))
            assert st.inventory >= 0 and st.backlog >= 0
            assert not (st.inventory > 0 and st.backlog > 0)

        # --- 4. Push shipments into downstream ship pipelines ---
        for role in self.roles:
            if topo.is_customer(role):
                continue  # sold to end customers — leaves system
            for claimant, qty in allocations[role].items():
                self._states[claimant].ship_pipeline.append(int(qty))

        # --- 5. Accept orders; push into upstream order pipelines / production ---
        orders_clamped: dict[Role, bool] = {}
        orders_placed: dict[Role, int] = {}
        factory_production = 0

        for role in self.roles:
            raw = int(orders.get(role, 0))
            clamped = raw < 0 or raw > cfg.order_cap
            qty = max(0, min(cfg.order_cap, raw))
            orders_clamped[role] = clamped or (raw != qty)
            orders_placed[role] = qty
            self._states[role].last_order_placed = qty
            self._boundary_orders += 1
            if qty == cfg.order_cap:
                self._boundary_hits += 1

            if topo.is_factory(role):
                prod = qty
                if cfg.capacity is not None:
                    cap = int(cfg.capacity)
                    if prod > cap:
                        rationed = True
                        prod = cap
                factory_production = prod
                self._states[role].on_order += prod
                self._states[role].ship_pipeline.append(prod)
            else:
                self._states[role].on_order += qty
                upstream = topo.upstream[role]
                assert upstream is not None
                up = self._states[upstream]
                if role not in up.order_pipelines:
                    up.order_pipelines[role] = [0] * cfg.order_delay
                up.order_pipelines[role].append(qty)

        # Pad / truncate pipelines.
        for role in self.roles:
            st = self._states[role]
            for c, pipe in list(st.order_pipelines.items()):
                while len(pipe) < cfg.order_delay:
                    pipe.append(0)
                if len(pipe) > cfg.order_delay:
                    st.order_pipelines[c] = pipe[: cfg.order_delay]
            self._sync_aggregate_order_pipeline(role)
            while len(st.ship_pipeline) < cfg.ship_delay:
                st.ship_pipeline.append(0)
            if len(st.ship_pipeline) > cfg.ship_delay:
                st.ship_pipeline = st.ship_pipeline[: cfg.ship_delay]

        # --- 6. Signals (optional, unverified, delayed, broadcast to all incl. rivals) ---
        signals_sent: dict[Role, Signal | None] = {r: None for r in self.roles}
        signals_received: dict[Role, dict[Role, Signal | None]] = {
            r: {o: None for o in self.roles} for r in self.roles
        }
        signal_listeners: dict[Role, tuple[Role, ...]] = {
            r: self._channel.listeners_of(r) for r in self.roles
        }
        honesty_out: dict[Role, dict[str, float]] = {}

        if cfg.signaling_enabled:
            if signals is None:
                signals = {r: None for r in self.roles}
            signals_sent = {r: signals.get(r) for r in self.roles}
            truths = {
                r: {
                    "demand": incoming[r],
                    "inventory": self._states[r].inventory,
                }
                for r in self.roles
            }
            honesty_raw = self._channel.measure_honesty(signals_sent, truths, self.roles)
            # Update EMA of −mean_abs_error (higher ⇒ more honest). None ⇒ no update.
            # EMA feeds honesty-weighted *allocation* only — never the reward (see step 7).
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
            # Log who hears what: full board to every observer (incl. rival retailer).
            listener = next(iter(self.roles))
            self._last_signal_board = dict(signals_received[listener])
            self._channel.send(signals_sent)
        elif signals is not None and any(v is not None for v in signals.values()):
            # Signals provided but channel disabled — ignore (do not validate truth).
            signals_sent = {r: signals.get(r) for r in self.roles}

        # --- 7. Accrue costs / rewards ---
        # Regimes A/B: strictly LOCAL costs. Honesty / rationing weights never enter.
        local_costs: dict[Role, float] = {}
        for role in self.roles:
            st = self._states[role]
            c = self._cost(role)
            local_costs[role] = c.holding * st.inventory + c.backlog * st.backlog
        system_cost = float(sum(local_costs.values()))

        if cfg.regime == "C":
            rewards = {r: -system_cost for r in self.roles}
        else:
            rewards = {r: -local_costs[r] for r in self.roles}

        self._t = week
        terminated = self._t >= cfg.horizon
        self._terminated = terminated

        # Capacity bind: factory wanted more than the production cap (D5 definition).
        capacity_binds = False
        if cfg.capacity is not None:
            for frole in topo.factories:
                if int(orders_placed.get(frole, 0)) > int(cfg.capacity):
                    capacity_binds = True
                    break
        # Allocation shortfall: any node still has backlog after fill.
        allocation_triggers = any(int(self._states[r].backlog) > 0 for r in self.roles)

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
            customer_demand=customer_demand_sum,
            frac_actions_at_cap=self.boundary_action_fraction(),
            customer_demands=dict(customer_demands),
            allocations=allocations,
            signal_listeners=signal_listeners,
            capacity_binds=capacity_binds,
            allocation_triggers=allocation_triggers,
        )
        return self.states, rewards, terminated, info
