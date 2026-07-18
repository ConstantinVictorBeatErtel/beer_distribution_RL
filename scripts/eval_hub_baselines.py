#!/usr/bin/env python3
"""Evaluate deterministic Hub baselines and write raw plus aggregate results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.policies import adaptive_policy, random_policy
from beer_distribution_game.scenario import SPLIT_SIZES, Split, scenario_for


POLICIES = ("adaptive_base_stock", "uniform_random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("development", "validation", "test"),
        default=("development", "validation"),
        help="Test is intentionally opt-in until the benchmark version is frozen.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/hub_baselines/development_validation"),
    )
    return parser.parse_args()


def run_policy(spec, role, policy_name: str) -> dict[str, Any]:
    episode = BeerEpisode(spec, role, include_reference=False)
    if policy_name == "adaptive_base_stock":
        policy = adaptive_policy(spec, role)
    elif policy_name == "uniform_random":
        policy = random_policy(spec, role)
    else:
        raise ValueError(policy_name)

    observation = episode.start()
    while not episode.done:
        result = episode.place_order(policy.act(observation))
        if not result["done"]:
            observation = result["next_observation"]
    assert episode.outcome is not None
    return episode.outcome


def episode_rows(splits: list[Split]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        for seed_index in range(SPLIT_SIZES[split]):
            for tier in range(1, 6):
                variants = (
                    ("headline",)
                    if tier < 5
                    else (
                        "headline",
                        "t5_control_base_rival",
                        "t5_control_uniform",
                    )
                )
                for variant in variants:
                    spec = scenario_for(tier, split, seed_index, variant)
                    for role in spec.roles:
                        outcomes = {
                            policy: run_policy(spec, role, policy)
                            for policy in POLICIES
                        }
                        base_cost = outcomes["adaptive_base_stock"]["grade"][
                            "primary"
                        ]["local_total_cost"]
                        for policy, outcome in outcomes.items():
                            grade = outcome["grade"]
                            local_cost = grade["primary"]["local_total_cost"]
                            rows.append(
                                {
                                    "environment_version": spec.environment_version,
                                    "scenario_id": spec.scenario_id,
                                    "episode_id": outcome["episode_id"],
                                    "split": split,
                                    "seed_index": seed_index,
                                    "master_seed_hex": spec.master_seed_hex,
                                    "tier": tier,
                                    "variant": variant,
                                    "role": role,
                                    "policy": policy,
                                    "local_total_cost": local_cost,
                                    "system_total_cost": grade["costs"][
                                        "system_total_cost"
                                    ],
                                    "cost_score": base_cost / (base_cost + local_cost),
                                    "immediate_fill_rate": grade["service"][
                                        "immediate_fill_rate"
                                    ],
                                    "horizon_fulfillment": grade["service"][
                                        "horizon_fulfillment"
                                    ],
                                    "bullwhip_ratio": grade["stability"][
                                        "bullwhip_ratio"
                                    ],
                                    "normalized_order_volatility": grade["stability"][
                                        "normalized_order_volatility"
                                    ],
                                    "order_cap_hit_rate": grade["stability"][
                                        "order_cap_hit_rate"
                                    ],
                                }
                            )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["split"],
            row["tier"],
            row["variant"],
            row["role"],
            row["policy"],
        )
        groups.setdefault(key, []).append(row)

    metrics = (
        "local_total_cost",
        "system_total_cost",
        "cost_score",
        "immediate_fill_rate",
        "horizon_fulfillment",
        "bullwhip_ratio",
        "normalized_order_volatility",
        "order_cap_hit_rate",
    )
    output = []
    for key, group in sorted(groups.items()):
        summary: dict[str, Any] = {
            "split": key[0],
            "tier": key[1],
            "variant": key[2],
            "role": key[3],
            "policy": key[4],
            "n": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            summary[f"{metric}_mean"] = mean(values) if values else None
            summary[f"{metric}_sd"] = stdev(values) if len(values) >= 2 else None
            summary[f"{metric}_median"] = median(values) if values else None
        output.append(summary)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = episode_rows(list(args.splits))
    summaries = aggregate(rows)
    write_csv(args.output_dir / "episodes.csv", rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    (args.output_dir / "results.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "splits": list(args.splits),
                "episode_count": len(rows),
                "episodes": rows,
                "summary": summaries,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} episodes to {args.output_dir}")


if __name__ == "__main__":
    main()
