"""Immutable scenario definitions and portable seed derivation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal

Role = Literal[
    "retailer",
    "retailer_a",
    "retailer_b",
    "wholesaler",
    "distributor",
    "factory",
]
Split = Literal["development", "validation", "test"]
Variant = Literal["headline", "t5_control_base_rival", "t5_control_uniform"]

SERIAL_ROLES: tuple[Role, ...] = (
    "retailer",
    "wholesaler",
    "distributor",
    "factory",
)
Y_ROLES: tuple[Role, ...] = (
    "retailer_a",
    "retailer_b",
    "wholesaler",
    "distributor",
    "factory",
)
SPLIT_SIZES: dict[Split, int] = {
    "development": 3,
    "validation": 5,
    "test": 10,
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def master_seed_hex(split: Split, index: int) -> str:
    if index < 0 or index >= SPLIT_SIZES[split]:
        raise ValueError(f"seed index {index} is outside {split!r} split")
    label = f"beer-agent-v1|{split}|{index:05d}".encode()
    return hashlib.sha256(label).hexdigest()[:16]


def derive_seed(seed_hex: str, namespace: str) -> int:
    payload = f"beer-agent-v1|{seed_hex}|{namespace}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _shock(seed_hex: str) -> tuple[int, float]:
    payload = f"beer-agent-v1|{seed_hex}|mechanism/shock".encode()
    digest = hashlib.sha256(payload).digest()
    return (15, 19, 23)[digest[0] % 3], (4.0, 12.0)[digest[1] % 2]


@dataclass(frozen=True)
class ScenarioSpec:
    schema_version: str
    environment_version: str
    scenario_id: str
    tier: int
    variant: Variant
    split: Split
    seed_index: int
    master_seed_hex: str
    topology: Literal["serial", "y"]
    roles: tuple[Role, ...]
    horizon: int
    order_delay: int
    shipment_delay: int
    order_cap: int
    holding_cost: float
    backlog_cost: float
    initial_inventory: int
    initial_shipment_pipeline: int
    initial_order_pipeline: int
    observation_mode: Literal["shipment_notices", "aggregate_supply_line"]
    history_window: int
    demand_process: str
    demand_parameters: dict[str, int | float | None]
    capacity: int | None
    rationing: Literal["proportional", "uniform"]
    counterparty_policy: str
    aggressive_retailers: bool

    @property
    def settlement_weeks(self) -> int:
        return self.order_delay + self.shipment_delay

    def to_dict(self) -> dict:
        return asdict(self)

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def episode_id(self, controlled_role: Role) -> str:
        payload = f"{self.canonical_json()}|{controlled_role}".encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def scenario_from_dict(data: dict) -> ScenarioSpec:
    values = dict(data)
    values["roles"] = tuple(values["roles"])
    return ScenarioSpec(**values)


def scenario_for(
    tier: int,
    split: Split,
    seed_index: int,
    variant: Variant = "headline",
) -> ScenarioSpec:
    if tier not in range(1, 6):
        raise ValueError("tier must be in 1..5")
    if tier != 5 and variant != "headline":
        raise ValueError("Tier 5 controls are only valid for tier 5")

    seed_hex = master_seed_hex(split, seed_index)
    common = dict(
        schema_version="1.0.0",
        environment_version="0.2.0",
        tier=tier,
        variant=variant,
        split=split,
        seed_index=seed_index,
        master_seed_hex=seed_hex,
        horizon=36,
        order_delay=1,
        shipment_delay=2,
        order_cap=128,
        holding_cost=0.5,
        backlog_cost=1.0,
        initial_inventory=12,
        initial_shipment_pipeline=4,
        initial_order_pipeline=4,
        history_window=8,
    )

    if tier == 1:
        return ScenarioSpec(
            **common,
            scenario_id="t1-steady-serial-v2",
            topology="serial",
            roles=SERIAL_ROLES,
            observation_mode="shipment_notices",
            demand_process="constant_v1",
            demand_parameters={"value": 8},
            capacity=None,
            rationing="proportional",
            counterparty_policy="adaptive_base_stock_v2",
            aggressive_retailers=False,
        )

    if tier == 2:
        return ScenarioSpec(
            **common,
            scenario_id="t2-ar1-serial-v2",
            topology="serial",
            roles=SERIAL_ROLES,
            observation_mode="shipment_notices",
            demand_process="ar1_v1",
            demand_parameters={"mu": 7.5, "phi": 0.7, "sigma": 2.0, "x0": 7.5},
            capacity=None,
            rationing="proportional",
            counterparty_policy="adaptive_base_stock_v2",
            aggressive_retailers=False,
        )

    shift_week, mu_after = _shock(seed_hex)
    if tier in (3, 4):
        return ScenarioSpec(
            **common,
            scenario_id=(
                "t3-shift-serial-v2" if tier == 3 else "t4-partial-shift-serial-v2"
            ),
            topology="serial",
            roles=SERIAL_ROLES,
            observation_mode=(
                "shipment_notices" if tier == 3 else "aggregate_supply_line"
            ),
            demand_process="shifted_ar1_v1",
            demand_parameters={
                "mu_before": 7.5,
                "phi": 0.7,
                "sigma": 2.0,
                "x0": 7.5,
                "shift_week": shift_week,
                "mu_after": mu_after,
            },
            capacity=None,
            rationing="proportional",
            counterparty_policy="adaptive_base_stock_v2",
            aggressive_retailers=False,
        )

    aggressive = variant != "t5_control_base_rival"
    rationing = "uniform" if variant == "t5_control_uniform" else "proportional"
    suffix = {
        "headline": "strategic",
        "t5_control_base_rival": "control-base-rival",
        "t5_control_uniform": "control-uniform",
    }[variant]
    return ScenarioSpec(
        **common,
        scenario_id=f"t5-{suffix}-y-v2",
        topology="y",
        roles=Y_ROLES,
        observation_mode="aggregate_supply_line",
        demand_process="correlated_y_ar1_v1",
        demand_parameters={
            "mu": 7.5,
            "phi": 0.7,
            "sigma_common": 2.0,
            "sigma_idiosyncratic": 1.5,
            "common0": 0.0,
        },
        capacity=22,
        rationing=rationing,
        counterparty_policy="adaptive_base_stock_v2",
        aggressive_retailers=aggressive,
    )


def roles_for(spec: ScenarioSpec, role_mode: Literal["core", "all"]) -> tuple[Role, ...]:
    if role_mode == "all":
        return spec.roles
    return ("retailer_a",) if spec.topology == "y" else ("retailer",)
