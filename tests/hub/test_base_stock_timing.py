from beer_distribution_game.core import BeerGameCore
from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.policies import adaptive_policy
from beer_distribution_game.scenario import scenario_for


def test_order_placed_after_week_one_demand_arrives_at_start_of_week_four():
    """A one-unit impulse proves the decision-boundary lead time is three weeks."""
    spec = scenario_for(1, "development", 0)
    control = BeerGameCore(spec)
    impulse = BeerGameCore(spec)
    receipt_differences = []

    for week in range(1, 6):
        control_prepared = control.prepare_week()
        impulse_prepared = impulse.prepare_week()
        receipt_differences.append(
            impulse_prepared.received["retailer"]
            - control_prepared.received["retailer"]
        )

        zero_orders = {role: 0 for role in spec.roles}
        impulse_orders = dict(zero_orders)
        if week == 1:
            impulse_orders["retailer"] = 1
        control.commit_orders(zero_orders)
        impulse.commit_orders(impulse_orders)

    assert receipt_differences == [0, 0, 0, 1, 0]


def test_corrected_base_stock_matches_steady_flow_from_first_decision():
    spec = scenario_for(1, "development", 0)
    episode = BeerEpisode(spec, "retailer")
    policy = adaptive_policy(spec, "retailer")
    observation = episode.start()
    orders = []

    while not episode.done:
        order = policy.act(observation)
        orders.append(order)
        result = episode.place_order(order)
        if not result["done"]:
            observation = result["next_observation"]

    assert orders == [8] * spec.horizon
    # Initial pipelines contain four units rather than the steady eight, so the
    # exact score includes the documented startup transient and settlement.
    assert episode.outcome["grade"]["primary"]["local_total_cost"] == 69.0
