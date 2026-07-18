import json

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.grading import _service_metrics, grade_episode
from beer_distribution_game.policies import adaptive_policy
from beer_distribution_game.scenario import scenario_for


def test_service_is_calculated_per_claimant_not_from_aggregate_backlog():
    transition = {
        "incoming_by_claimant": {
            "wholesaler": {"retailer_a": 0, "retailer_b": 10}
        },
        "backlog_before_by_claimant": {
            "wholesaler": {"retailer_a": 10, "retailer_b": 0}
        },
        "allocations": {"wholesaler": {"retailer_a": 0, "retailer_b": 10}},
        "states_after_fulfillment": {"wholesaler": {"backlog": 10}},
    }
    metrics = _service_metrics([transition], "wholesaler")
    assert metrics["immediate_fill_rate"] == 1.0
    assert metrics["cycle_service_level"] == 1.0


def test_lower_cost_produces_higher_score_for_fixed_reference():
    spec = scenario_for(1, "development", 0)

    def row(cost):
        return {
            "local_costs": {role: cost for role in spec.roles},
            "incoming_by_claimant": {
                role: {f"c:{role}": 1} for role in spec.roles
            },
            "backlog_before_by_claimant": {
                role: {f"c:{role}": 0} for role in spec.roles
            },
            "allocations": {role: {f"c:{role}": 1} for role in spec.roles},
            "states_after_fulfillment": {
                role: {"backlog": 0} for role in spec.roles
            },
            "orders": {role: 8 for role in spec.roles},
        }

    common = dict(
        spec=spec,
        controlled_role="retailer",
        settlement=[],
        terminal_inventory_positions={role: 0 for role in spec.roles},
        protocol_clean=True,
        base_reference={"local_total_cost": 100.0, "system_total_cost": 400.0},
    )
    low = grade_episode(operational=[row(1.0), row(1.0)], **common)
    high = grade_episode(operational=[row(5.0), row(5.0)], **common)
    assert low["episode_reward"] > high["episode_reward"]


def test_protocol_gate_zeros_an_otherwise_valid_score():
    spec = scenario_for(1, "development", 0)
    episode = BeerEpisode(spec, "retailer")
    observation = episode.start()
    policy = adaptive_policy(spec, "retailer")
    episode.mark_protocol_error()
    while not episode.done:
        result = episode.place_order(policy.act(observation))
        if not result["done"]:
            observation = result["next_observation"]
    assert episode.outcome["grade"]["primary"]["cost_score"] == 0.5
    assert episode.outcome["grade"]["episode_reward"] == 0.0


def test_complete_outcome_is_strict_json():
    spec = scenario_for(2, "development", 0)
    episode = BeerEpisode(spec, "retailer")
    observation = episode.start()
    policy = adaptive_policy(spec, "retailer")
    while not episode.done:
        result = episode.place_order(policy.act(observation))
        if not result["done"]:
            observation = result["next_observation"]
    json.dumps(episode.outcome, allow_nan=False, sort_keys=True)
