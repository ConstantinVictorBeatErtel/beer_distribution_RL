"""Shared lightweight types to avoid circular imports between env modules."""

from __future__ import annotations

from enum import IntEnum


class Role(IntEnum):
    RETAILER = 0
    WHOLESALER = 1
    DISTRIBUTOR = 2
    FACTORY = 3
    # Second retailer for Y-topology only (enum value 4 keeps serial indices stable).
    RETAILER_B = 4


ROLES: tuple[Role, ...] = (
    Role.RETAILER,
    Role.WHOLESALER,
    Role.DISTRIBUTOR,
    Role.FACTORY,
)

Y_ROLES: tuple[Role, ...] = (
    Role.RETAILER,
    Role.RETAILER_B,
    Role.WHOLESALER,
    Role.DISTRIBUTOR,
    Role.FACTORY,
)

ROLE_NAMES: dict[Role, str] = {
    Role.RETAILER: "retailer",
    Role.WHOLESALER: "wholesaler",
    Role.DISTRIBUTOR: "distributor",
    Role.FACTORY: "factory",
}

Y_ROLE_NAMES: dict[Role, str] = {
    Role.RETAILER: "retailer_a",
    Role.RETAILER_B: "retailer_b",
    Role.WHOLESALER: "wholesaler",
    Role.DISTRIBUTOR: "distributor",
    Role.FACTORY: "factory",
}
