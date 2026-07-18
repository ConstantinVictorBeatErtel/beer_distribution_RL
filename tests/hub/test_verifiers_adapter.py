import asyncio
import importlib

import pytest

vf = pytest.importorskip("verifiers.v1")

from beer_distribution_game import BeerHarness, BeerTaskset
from beer_distribution_game.harness import PROGRAM_SOURCE
from beer_distribution_game.taskset import (
    BeerTasksetConfig,
    BeerToolset,
)


def test_package_exports_exactly_one_taskset_and_one_harness():
    module = importlib.import_module("beer_distribution_game")
    exported = [getattr(module, name) for name in module.__all__]
    assert sum(issubclass(value, vf.Taskset) for value in exported) == 1
    assert sum(issubclass(value, vf.Harness) for value in exported) == 1
    assert BeerTaskset in exported
    assert BeerHarness in exported


def test_bundled_program_requires_one_tool_call_and_retries_transport_errors():
    assert 'tool_choice="required"' in PROGRAM_SOURCE
    assert "parallel_tool_calls=False" in PROGRAM_SOURCE
    assert "max_retries=2" in PROGRAM_SOURCE
    assert "timeout=120.0" in PROGRAM_SOURCE


def test_taskset_loads_typed_reproducible_rows():
    config = BeerTasksetConfig(
        id="beer-distribution-game",
        split="development",
        tiers=[1, 2, 3, 4, 5],
        role_mode="core",
        seed_limit=1,
    )
    first = BeerTaskset(config).load()
    second = BeerTaskset(config).load()
    assert len(first) == 5
    assert [task.data.model_dump() for task in first] == [
        task.data.model_dump() for task in second
    ]


def test_taskset_selects_only_y_wholesaler():
    config = BeerTasksetConfig(
        id="beer-distribution-game",
        split="development",
        tiers=[5],
        controlled_roles=["wholesaler"],
        seed_limit=1,
    )
    tasks = BeerTaskset(config).load()
    assert len(tasks) == 1
    assert tasks[0].data.controlled_role == "wholesaler"
    assert tasks[0].data.scenario["topology"] == "y"
    assert ":wholesaler:development:0" in tasks[0].data.name


def test_taskset_rejects_role_missing_from_selected_topology():
    config = BeerTasksetConfig(
        id="beer-distribution-game",
        split="development",
        tiers=[5],
        controlled_roles=["retailer"],
        seed_limit=1,
    )
    with pytest.raises(ValueError, match="unavailable in tier 5"):
        BeerTaskset(config).load()


def test_taskset_rejects_duplicate_controlled_roles():
    config = BeerTasksetConfig(
        id="beer-distribution-game",
        split="development",
        tiers=[5],
        controlled_roles=["wholesaler", "wholesaler"],
        seed_limit=1,
    )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        BeerTaskset(config).load()


def test_toolset_completes_an_episode_and_persists_grade():
    config = BeerTasksetConfig(
        id="beer-distribution-game",
        split="development",
        tiers=[1],
        role_mode="core",
        seed_limit=1,
    )
    task = BeerTaskset(config).load()[0]
    toolset = BeerToolset(vf.ToolsetConfig())

    async def run():
        await toolset.setup_task(task.data)
        for _ in range(36):
            result = await toolset.place_order(8)
        return result

    final = asyncio.run(run())
    assert '"done":true' in final
    assert toolset.state.done is True
    assert toolset.state.outcome["grade"]["status"] == "scored"


def test_protocol_error_does_not_advance_week_and_zeros_finished_reward():
    config = BeerTasksetConfig(
        id="beer-distribution-game",
        split="development",
        tiers=[1],
        role_mode="core",
        seed_limit=1,
    )
    task = BeerTaskset(config).load()[0]
    toolset = BeerToolset(vf.ToolsetConfig())

    async def run():
        await toolset.setup_task(task.data)
        first = await toolset.record_protocol_error("missing_tool_call")
        assert toolset.episode.core.week == 0
        second = await toolset.record_protocol_error("missing_tool_call")
        return first, second

    first, second = asyncio.run(run())
    assert '"retry_allowed":true' in first
    assert '"done":true' in second
    assert toolset.state.outcome["grade"]["episode_reward"] == 0.0
