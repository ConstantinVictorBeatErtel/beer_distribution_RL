"""LLM text I/O: obs→prompt serializer + grammar-constrained order parser.

Product layer clearing readiness Check 2 (text I/O). Order-only agents;
broadcast / cheap-talk channel is out of scope. Action space matches IPPO
relative Δ ∈ [-8, 8] → absolute order clipped to ``order_cap`` (default 128).
"""

from __future__ import annotations

from beer_distribution_rl.agents.llm.decode import (
    ConstrainedOrderDecoder,
    DecodeResult,
    ParseFailStats,
)
from beer_distribution_rl.agents.llm.grammar import (
    DELTA_JSON_SCHEMA,
    ORDER_DELTA_GBNF,
    delta_json_schema,
    map_delta_to_order,
)
from beer_distribution_rl.agents.llm.memory import AgentMemory, WeekRecord
from beer_distribution_rl.agents.llm.parser import parse_delta_json, parse_order_legacy
from beer_distribution_rl.agents.llm.serializer import (
    FORBIDDEN_SUBSTRINGS,
    OWN_HISTORY_FIELDS,
    observe_local,
    prompt_leak_report,
    serialize_prompt,
)

__all__ = [
    "AgentMemory",
    "ConstrainedOrderDecoder",
    "DELTA_JSON_SCHEMA",
    "DecodeResult",
    "FORBIDDEN_SUBSTRINGS",
    "ORDER_DELTA_GBNF",
    "OWN_HISTORY_FIELDS",
    "ParseFailStats",
    "WeekRecord",
    "delta_json_schema",
    "map_delta_to_order",
    "observe_local",
    "parse_delta_json",
    "parse_order_legacy",
    "prompt_leak_report",
    "serialize_prompt",
]
