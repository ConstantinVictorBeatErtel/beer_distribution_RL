#!/usr/bin/env python3
"""Reduce Verifiers traces to compact, replayable evaluation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_file", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    raw = args.trace_file.read_bytes()
    traces = [json.loads(line) for line in raw.splitlines() if line.strip()]
    episodes = []
    for trace in traces:
        outcome = trace["info"]["beer_game"]
        scenario = outcome["scenario"]
        grade = outcome["grade"]
        role = outcome["controlled_role"]
        usage = [node.get("usage", {}) for node in trace["nodes"]]
        episodes.append(
            {
                "seed_index": scenario["seed_index"],
                "master_seed_hex": scenario["master_seed_hex"],
                "episode_id": outcome["episode_id"],
                "scenario_id": scenario["scenario_id"],
                "environment_version": scenario["environment_version"],
                "controlled_role": role,
                "actions": [
                    row["orders"][role] for row in outcome["operational_transitions"]
                ],
                "reward": grade["episode_reward"],
                "local_total_cost": grade["primary"]["local_total_cost"],
                "paired_base_stock_local_total_cost": grade["primary"][
                    "paired_base_stock_local_total_cost"
                ],
                "system_total_cost": grade["costs"]["system_total_cost"],
                "immediate_fill_rate": grade["service"]["immediate_fill_rate"],
                "bullwhip_ratio": grade["stability"]["bullwhip_ratio"],
                "protocol_clean": grade["protocol_clean"],
                "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in usage),
                "completion_tokens": sum(
                    item.get("completion_tokens", 0) for item in usage
                ),
            }
        )

    result = {
        "schema_version": "1.0.0",
        "source_trace_sha256": hashlib.sha256(raw).hexdigest(),
        "episode_count": len(episodes),
        "episodes": sorted(episodes, key=lambda row: row["seed_index"]),
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
