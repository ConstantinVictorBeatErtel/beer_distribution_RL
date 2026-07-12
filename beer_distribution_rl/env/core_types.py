"""Shared lightweight types to avoid circular imports between env modules."""

from __future__ import annotations

from enum import IntEnum


class Role(IntEnum):
    RETAILER = 0
    WHOLESALER = 1
    DISTRIBUTOR = 2
    FACTORY = 3


ROLES: tuple[Role, ...] = (
    Role.RETAILER,
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
