from statistics import median

from beer_distribution_game.core import BeerGameCore
from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.policies import (
    adaptive_policy,
    counterparty_policies,
    random_policy,
)
from beer_distribution_game.scenario import SPLIT_SIZES, scenario_for


def _run_episode(spec, role, policy_kind="base"):
    episode = BeerEpisode(spec, role)
    policy = (
        adaptive_policy(spec, role)
        if policy_kind == "base"
        else random_policy(spec, role)
    )
    observation = episode.start()
    while not episode.done:
        result = episode.place_order(policy.act(observation))
        if not result["done"]:
            observation = result["next_observation"]
    return episode


def test_base_stock_beats_random_in_every_development_validation_cell():
    for tier in range(1, 6):
        specs = [
            scenario_for(tier, split, index)
            for split in ("development", "validation")
            for index in range(SPLIT_SIZES[split])
        ]
        for role in specs[0].roles:
            base_costs = [
                _run_episode(spec, role).outcome["grade"]["primary"][
                    "local_total_cost"
                ]
                for spec in specs
            ]
            random_costs = [
                _run_episode(spec, role, "random").outcome["grade"]["primary"][
                    "local_total_cost"
                ]
                for spec in specs
            ]
            assert median(base_costs) < median(random_costs), (tier, role)


def test_tier5_base_rival_capacity_binding_is_in_calibration_band():
    bound = []
    for index in range(SPLIT_SIZES["validation"]):
        spec = scenario_for(5, "validation", index, "t5_control_base_rival")
        episode = _run_episode(spec, "retailer_a")
        bound.extend(row["capacity_bound"] for row in episode.operational_transitions)
    rate = sum(bound) / len(bound)
    assert 0.10 <= rate <= 0.70


def test_tier5_aggressive_policy_does_not_collapse_into_order_cap():
    cap_hits = []
    for index in range(SPLIT_SIZES["validation"]):
        spec = scenario_for(5, "validation", index)
        # Controlling the wholesaler makes both scripted retailers aggressive.
        episode = _run_episode(spec, "wholesaler")
        for row in episode.operational_transitions:
            cap_hits.extend(
                row["orders"][role] == spec.order_cap
                for role in ("retailer_a", "retailer_b")
            )
    assert sum(cap_hits) / len(cap_hits) <= 0.25


def _scripted_action_trace(spec):
    core = BeerGameCore(spec)
    policies = counterparty_policies(spec, "factory")
    policies["factory"] = adaptive_policy(spec, "factory")
    actions = []
    for _ in range(spec.horizon):
        core.prepare_week()
        observations = {
            role: core.observation(
                role,
                episode_id="calibration",
                recent_history=[],
                cumulative_local_cost=0.0,
            )
            for role in spec.roles
        }
        orders = {role: policies[role].act(observations[role]) for role in spec.roles}
        actions.append(orders)
        core.commit_orders(orders)
    return actions


def _replay(spec, actions):
    core = BeerGameCore(spec)
    transitions = []
    for orders in actions:
        core.prepare_week()
        transitions.append(core.commit_orders(orders))
    return transitions


def test_tier5_rationing_controls_change_shortage_allocations():
    shortage_weeks = 0
    differing_weeks = 0
    for index in range(SPLIT_SIZES["validation"]):
        proportional = scenario_for(5, "validation", index, "headline")
        uniform = scenario_for(5, "validation", index, "t5_control_uniform")
        actions = _scripted_action_trace(proportional)
        proportional_rows = _replay(proportional, actions)
        uniform_rows = _replay(uniform, actions)
        for prop_row, uniform_row in zip(proportional_rows, uniform_rows):
            role = "wholesaler"
            requested = {
                claimant: (
                    prop_row["backlog_before_by_claimant"][role].get(claimant, 0)
                    + incoming
                )
                for claimant, incoming in prop_row["incoming_by_claimant"][role].items()
            }
            allocation = prop_row["allocations"][role]
            if sum(allocation.values()) < sum(requested.values()):
                shortage_weeks += 1
                differing_weeks += int(
                    allocation != uniform_row["allocations"][role]
                )
    assert shortage_weeks > 0
    assert differing_weeks / shortage_weeks >= 0.10
