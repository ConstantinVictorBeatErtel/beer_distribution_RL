"""Supply-chain topologies as DAGs of roles.

Serial chain (4 nodes) remains the default so prior results stay reproducible.
Y-topology adds a second retailer under the wholesaler — the multi-claimant
structure that makes proportional vs honesty-weighted rationing distinguishable
(and therefore makes P3 testable).
"""

from __future__ import annotations

from dataclasses import dataclass

from beer_distribution_rl.env.core_types import (
    ROLE_NAMES,
    ROLES,
    Role,
    Y_ROLE_NAMES,
    Y_ROLES,
)


@dataclass(frozen=True)
class Topology:
    """Directed acyclic supply network.

    Edges run upstream for orders and downstream for shipments:
    ``upstream[r]`` is who ``r`` orders from (None for factories);
    ``downstream[r]`` are the claimants ``r`` ships to (empty for customer-facing
    leaves — those ship to exogenous demand).
    """

    name: str
    roles: tuple[Role, ...]
    upstream: dict[Role, Role | None]
    downstream: dict[Role, tuple[Role, ...]]
    customers: tuple[Role, ...]
    factories: tuple[Role, ...]
    role_names: dict[Role, str]

    def suppliers(self) -> tuple[Role, ...]:
        """Nodes that fill orders from one or more claimants (incl. customer leaves)."""
        return self.roles

    def claimants_of(self, role: Role) -> tuple[Role, ...]:
        """Downstream roles this node allocates to. Empty ⇒ exogenous customer demand."""
        return self.downstream[role]

    def is_customer(self, role: Role) -> bool:
        return role in self.customers

    def is_factory(self, role: Role) -> bool:
        return role in self.factories


def serial_topology() -> Topology:
    """Classic 4-node chain: Retailer → Wholesaler → Distributor → Factory.

    Each node has a single claimant, so proportional / uniform / honesty-weighted
    rationing are mathematically identical (see ``env/rationing.py``).
    """
    up = {
        Role.RETAILER: Role.WHOLESALER,
        Role.WHOLESALER: Role.DISTRIBUTOR,
        Role.DISTRIBUTOR: Role.FACTORY,
        Role.FACTORY: None,
    }
    down = {
        Role.FACTORY: (Role.DISTRIBUTOR,),
        Role.DISTRIBUTOR: (Role.WHOLESALER,),
        Role.WHOLESALER: (Role.RETAILER,),
        Role.RETAILER: (),
    }
    return Topology(
        name="serial",
        roles=ROLES,
        upstream=up,
        downstream=down,
        customers=(Role.RETAILER,),
        factories=(Role.FACTORY,),
        role_names=dict(ROLE_NAMES),
    )


def y_topology() -> Topology:
    """Two competing retailers → Wholesaler → Distributor → Factory.

    Wholesaler is the multi-claimant node where rationing mechanisms diverge.
    Both retailers hear each other's broadcasts (cheap-talk board is global).
    """
    up = {
        Role.RETAILER: Role.WHOLESALER,
        Role.RETAILER_B: Role.WHOLESALER,
        Role.WHOLESALER: Role.DISTRIBUTOR,
        Role.DISTRIBUTOR: Role.FACTORY,
        Role.FACTORY: None,
    }
    down = {
        Role.FACTORY: (Role.DISTRIBUTOR,),
        Role.DISTRIBUTOR: (Role.WHOLESALER,),
        Role.WHOLESALER: (Role.RETAILER, Role.RETAILER_B),
        Role.RETAILER: (),
        Role.RETAILER_B: (),
    }
    return Topology(
        name="y",
        roles=Y_ROLES,
        upstream=up,
        downstream=down,
        customers=(Role.RETAILER, Role.RETAILER_B),
        factories=(Role.FACTORY,),
        role_names=dict(Y_ROLE_NAMES),
    )


def get_topology(name: str) -> Topology:
    key = name.strip().lower().replace("-", "_")
    if key in ("serial", "serial_chain", "chain"):
        return serial_topology()
    if key in ("y", "y_topology", "ytopology"):
        return y_topology()
    raise ValueError(f"Unknown topology {name!r}; expected 'serial' or 'y'")
