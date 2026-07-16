"""Grammar / schema for constrained order decoding (relative Δ ∈ [-δ, δ])."""

from __future__ import annotations

from typing import Any

# Matches IPPO ``action_mode=relative`` / ``action_delta_max=8`` (DECISIONS).
DEFAULT_DELTA_MAX = 8
DEFAULT_ORDER_CAP = 128


def delta_json_schema(delta_max: int = DEFAULT_DELTA_MAX) -> dict[str, Any]:
    """JSON Schema for Ollama / vLLM structured generation.

    The model may only emit ``{"delta": <int>}`` with delta in ``[-delta_max, delta_max]``.
    Absolute order is derived as ``clip(demand + delta, 0, order_cap)``.
    """
    return {
        "type": "object",
        "properties": {
            "delta": {
                "type": "integer",
                "minimum": -int(delta_max),
                "maximum": int(delta_max),
            }
        },
        "required": ["delta"],
        "additionalProperties": False,
    }


# Default schema object (mutable copies via ``delta_json_schema()`` preferred).
DELTA_JSON_SCHEMA: dict[str, Any] = delta_json_schema(DEFAULT_DELTA_MAX)


def order_delta_gbnf(delta_max: int = DEFAULT_DELTA_MAX) -> str:
    """GBNF grammar for vLLM / llama.cpp backends (documentation + portable path).

    Emits exactly ``{"delta": N}`` with N ∈ [-delta_max, delta_max].
    """
    # Integer alternatives: -8|-7|...|-1|0|1|...|8
    negs = "|".join(f"-{i}" for i in range(delta_max, 0, -1))
    poss = "|".join(str(i) for i in range(0, delta_max + 1))
    return (
        "root ::= \"{\" ws \"\\\"delta\\\"\" ws \":\" ws delta ws \"}\"\n"
        f"delta ::= {negs}|{poss}\n"
        "ws ::= [ \\t\\n]*\n"
    )


ORDER_DELTA_GBNF: str = order_delta_gbnf(DEFAULT_DELTA_MAX)


def map_delta_to_order(
    delta: int,
    last_demand_or_order: int,
    *,
    order_cap: int = DEFAULT_ORDER_CAP,
    delta_max: int = DEFAULT_DELTA_MAX,
) -> int:
    """Map relative Δ to absolute order quantity (IPPO decode parity)."""
    d = int(delta)
    if d < -delta_max or d > delta_max:
        raise ValueError(f"delta {d} outside [{-delta_max}, {delta_max}]")
    raw = int(last_demand_or_order) + d
    return max(0, min(int(order_cap), raw))
