import json

import pytest

from beer_distribution_game.core import BeerGameCore
from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.policies import adaptive_policy
from beer_distribution_game.scenario import canonical_json, scenario_for


def _fixed_transitions(tier: int, quantity: int = 8):
    spec = scenario_for(tier, "development", 0)
    core = BeerGameCore(spec)
    rows = []
    observations = []
    history = {role: [] for role in spec.roles}
    costs = {role: 0.0 for role in spec.roles}
    for _ in range(spec.horizon):
        core.prepare_week()
        observations.append(
            core.observation(
                spec.roles[0],
                episode_id=spec.episode_id(spec.roles[0]),
                recent_history=history[spec.roles[0]],
                cumulative_local_cost=costs[spec.roles[0]],
            )
        )
        rows.append(core.commit_orders({role: quantity for role in spec.roles}))
    return rows, observations


def test_same_scenario_and_actions_are_byte_identical():
    first, _ = _fixed_transitions(2)
    second, _ = _fixed_transitions(2)
    assert canonical_json(first) == canonical_json(second)


def test_distinct_cores_own_distinct_demand_objects():
    spec = scenario_for(2, "development", 0)
    a = BeerGameCore(spec)
    b = BeerGameCore(spec)
    assert a.demand is not b.demand
    a.prepare_week()
    b.prepare_week()
    assert a.prepared.incoming_by_claimant == b.prepared.incoming_by_claimant


def test_invalid_or_missing_action_does_not_mutate_prepared_state():
    spec = scenario_for(5, "development", 0)
    core = BeerGameCore(spec)
    core.prepare_week()
    before = canonical_json(core.snapshot())
    with pytest.raises(ValueError, match="missing"):
        core.commit_orders({role: 8 for role in spec.roles if role != "retailer_b"})
    assert canonical_json(core.snapshot()) == before
    assert core.week == 0


def test_future_downstream_order_pipeline_is_not_observed():
    spec = scenario_for(3, "development", 0)
    core = BeerGameCore(spec)
    core.prepare_week()
    obs = core.observation(
        "wholesaler",
        episode_id=spec.episode_id("wholesaler"),
        recent_history=[],
        cumulative_local_cost=0.0,
    )
    assert "order_pipeline" not in json.dumps(obs)


def test_tier3_and_tier4_hidden_transitions_match_but_observations_differ():
    rows3, obs3 = _fixed_transitions(3)
    rows4, obs4 = _fixed_transitions(4)
    assert canonical_json(rows3) == canonical_json(rows4)
    assert "inbound_shipment_pipeline" in obs3[0]["state"]
    assert "inbound_shipment_pipeline" not in obs4[0]["state"]
    stripped = dict(obs3[0])
    stripped["state"] = dict(stripped["state"])
    stripped["state"].pop("inbound_shipment_pipeline")
    stripped["scenario_id"] = obs4[0]["scenario_id"]
    stripped["observation_mode"] = obs4[0]["observation_mode"]
    stripped["episode_id"] = obs4[0]["episode_id"]
    assert stripped == obs4[0]


def test_base_stock_reference_scores_exactly_half_in_every_tier():
    for tier in range(1, 6):
        spec = scenario_for(tier, "development", 0)
        role = "retailer_a" if tier == 5 else "retailer"
        episode = BeerEpisode(spec, role)
        observation = episode.start()
        policy = adaptive_policy(spec, role)
        while not episode.done:
            result = episode.place_order(policy.act(observation))
            if not result["done"]:
                observation = result["next_observation"]
        assert episode.outcome["grade"]["episode_reward"] == 0.5


def test_settlement_adds_no_new_customer_demand():
    spec = scenario_for(2, "development", 0)
    episode = BeerEpisode(spec, "retailer")
    observation = episode.start()
    policy = adaptive_policy(spec, "retailer")
    while not episode.done:
        result = episode.place_order(policy.act(observation))
        if not result["done"]:
            observation = result["next_observation"]
    assert len(episode.settlement_transitions) == 3
    for row in episode.settlement_transitions:
        assert row["operational"] is False
        assert sum(row["incoming_by_claimant"]["retailer"].values()) == 0


def test_constant_demand_bullwhip_is_null():
    spec = scenario_for(1, "development", 0)
    episode = BeerEpisode(spec, "retailer")
    observation = episode.start()
    policy = adaptive_policy(spec, "retailer")
    while not episode.done:
        result = episode.place_order(policy.act(observation))
        if not result["done"]:
            observation = result["next_observation"]
    assert episode.outcome["grade"]["stability"]["bullwhip_ratio"] is None
