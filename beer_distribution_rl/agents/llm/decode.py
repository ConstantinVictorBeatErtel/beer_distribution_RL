"""Constrained decoding client (Ollama JSON-schema) + parse-fail logging."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from beer_distribution_rl.agents.llm.grammar import (
    DEFAULT_DELTA_MAX,
    DEFAULT_ORDER_CAP,
    delta_json_schema,
    map_delta_to_order,
)
from beer_distribution_rl.agents.llm.parser import parse_delta_json, parse_order_legacy


@dataclass
class DecodeResult:
    order: int
    delta: int | None
    raw: str
    parse_ok: bool
    n_attempts: int
    used_fallback: bool
    constrained: bool


@dataclass
class ParseFailStats:
    """Accumulate parse attempts/failures, optionally stratified by key."""

    attempts: int = 0
    failures: int = 0
    by_key: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, *, key: str, failed: bool) -> None:
        self.attempts += 1
        if failed:
            self.failures += 1
        bucket = self.by_key.setdefault(key, {"attempts": 0, "failures": 0})
        bucket["attempts"] += 1
        if failed:
            bucket["failures"] += 1

    @property
    def rate(self) -> float:
        return self.failures / max(self.attempts, 1)

    def rate_by_key(self) -> dict[str, float]:
        return {
            k: v["failures"] / max(v["attempts"], 1) for k, v in self.by_key.items()
        }

    def resampling_multiplier(self) -> float:
        """Implied cost multiplier ``1/(1-p)`` from geometric resampling."""
        p = self.rate
        if p >= 1.0:
            return float("inf")
        return 1.0 / (1.0 - p)

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "failures": self.failures,
            "rate": self.rate,
            "resampling_multiplier": self.resampling_multiplier(),
            "by_key": {
                k: {
                    **v,
                    "rate": v["failures"] / max(v["attempts"], 1),
                }
                for k, v in self.by_key.items()
            },
        }


class ConstrainedOrderDecoder:
    """Grammar-constrained order sampling via Ollama JSON Schema ``format``.

    Failures should be rare by construction; resample + log rate as fallback.
    Unconstrained / legacy ``ORDER:`` mode retained for before/after baselines.
    """

    def __init__(
        self,
        *,
        model: str = "qwen2.5:3b",
        host: str = "http://127.0.0.1:11434",
        delta_max: int = DEFAULT_DELTA_MAX,
        order_cap: int = DEFAULT_ORDER_CAP,
        max_parse_retries: int = 3,
        temperature: float = 0.0,
        constrained: bool = True,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.delta_max = int(delta_max)
        self.order_cap = int(order_cap)
        self.max_parse_retries = int(max_parse_retries)
        self.temperature = float(temperature)
        self.constrained = bool(constrained)
        self.stats = ParseFailStats()
        self.schema = delta_json_schema(self.delta_max)

    def _generate(self, prompt: str) -> str:
        if self.constrained:
            body: dict[str, Any] = {
                "model": self.model,
                "stream": False,
                "format": self.schema,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 24,
                },
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Emit only JSON matching the schema: "
                            f'{{"delta": integer in [{-self.delta_max}, {self.delta_max}]}}.'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            }
        else:
            body = {
                "model": self.model,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 12,
                },
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You output only: ORDER: <integer>\n"
                            "Example: ORDER: 7\n"
                            "Never omit the ORDER: prefix. No other words."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode())
        msg = data.get("message") or {}
        return str(msg.get("content", data.get("response", "")))

    def sample_order(
        self,
        prompt: str,
        last_demand_or_order: int,
        *,
        stats_key: str = "default",
    ) -> DecodeResult:
        """Sample an absolute order; resample on parse failure; log fail rate."""
        raw = ""
        delta: int | None = None
        order: int | None = None
        attempts = 0
        for _ in range(self.max_parse_retries):
            attempts += 1
            raw = self._generate(prompt)
            failed = False
            if self.constrained:
                delta = parse_delta_json(raw, delta_max=self.delta_max)
                if delta is None:
                    failed = True
                    self.stats.record(key=stats_key, failed=True)
                else:
                    order = map_delta_to_order(
                        delta,
                        last_demand_or_order,
                        order_cap=self.order_cap,
                        delta_max=self.delta_max,
                    )
                    self.stats.record(key=stats_key, failed=False)
                    return DecodeResult(
                        order=order,
                        delta=delta,
                        raw=raw,
                        parse_ok=True,
                        n_attempts=attempts,
                        used_fallback=False,
                        constrained=True,
                    )
            else:
                order = parse_order_legacy(raw, self.order_cap)
                if order is None:
                    failed = True
                    self.stats.record(key=stats_key, failed=True)
                else:
                    self.stats.record(key=stats_key, failed=False)
                    return DecodeResult(
                        order=order,
                        delta=None,
                        raw=raw,
                        parse_ok=True,
                        n_attempts=attempts,
                        used_fallback=False,
                        constrained=False,
                    )
            _ = failed
        # Exhausted retries — safe demand-matching fallback.
        fallback = max(0, min(self.order_cap, int(last_demand_or_order)))
        return DecodeResult(
            order=fallback,
            delta=0 if self.constrained else None,
            raw=raw,
            parse_ok=False,
            n_attempts=attempts,
            used_fallback=True,
            constrained=self.constrained,
        )
