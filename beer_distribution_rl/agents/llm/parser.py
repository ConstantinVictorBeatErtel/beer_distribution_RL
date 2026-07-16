"""Parse LLM text → relative Δ / absolute order.

Primary path: JSON ``{"delta": N}`` produced under grammar/schema constraints.
Legacy path: Check-5 ``ORDER: <int>`` regex (post-hoc) for before/after baselines.
"""

from __future__ import annotations

import json
import re
from typing import Any

from beer_distribution_rl.agents.llm.grammar import (
    DEFAULT_DELTA_MAX,
    DEFAULT_ORDER_CAP,
    map_delta_to_order,
)

ORDER_RE = re.compile(r"ORDER:\s*(\d+)", re.IGNORECASE)
DELTA_JSON_RE = re.compile(
    r'\{\s*"delta"\s*:\s*(-?\d+)\s*\}',
    re.IGNORECASE,
)


def parse_delta_json(
    text: str,
    *,
    delta_max: int = DEFAULT_DELTA_MAX,
) -> int | None:
    """Extract Δ from constrained JSON (or a single JSON object in the string).

    Returns ``None`` on parse failure (triggers resample). Valid only if
    ``delta ∈ [-delta_max, delta_max]``.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    # Prefer strict JSON load of the whole string.
    try:
        obj: Any = json.loads(raw)
        if isinstance(obj, dict) and "delta" in obj:
            d = int(obj["delta"])
            if -delta_max <= d <= delta_max:
                return d
            return None
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # Fallback: find first {"delta": N} substring (still validates range).
    m = DELTA_JSON_RE.search(raw)
    if not m:
        return None
    d = int(m.group(1))
    if -delta_max <= d <= delta_max:
        return d
    return None


def parse_order_from_delta_text(
    text: str,
    last_demand_or_order: int,
    *,
    order_cap: int = DEFAULT_ORDER_CAP,
    delta_max: int = DEFAULT_DELTA_MAX,
) -> int | None:
    """Parse constrained JSON Δ and map to absolute order, or ``None``."""
    delta = parse_delta_json(text, delta_max=delta_max)
    if delta is None:
        return None
    return map_delta_to_order(
        delta, last_demand_or_order, order_cap=order_cap, delta_max=delta_max
    )


def parse_order_legacy(text: str, order_cap: int = DEFAULT_ORDER_CAP) -> int | None:
    """Check-5 post-hoc regex parser: ``ORDER: <int>`` in ``[0, order_cap]``."""
    m = ORDER_RE.search((text or "").strip())
    if not m:
        return None
    qty = int(m.group(1))
    if qty < 0 or qty > order_cap:
        return None
    return qty
