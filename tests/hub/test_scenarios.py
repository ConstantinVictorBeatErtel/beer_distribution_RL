from beer_distribution_game.scenario import (
    SPLIT_SIZES,
    master_seed_hex,
    scenario_for,
    scenario_from_dict,
)


def test_seed_ids_are_portable_stable_hex_strings():
    assert master_seed_hex("development", 0) == "e2056d6d52741a08"
    for split, count in SPLIT_SIZES.items():
        seeds = [master_seed_hex(split, index) for index in range(count)]
        assert len(seeds) == len(set(seeds))
        assert all(len(seed) == 16 and int(seed, 16) >= 0 for seed in seeds)


def test_every_split_has_upward_and_downward_shocks():
    for split, count in SPLIT_SIZES.items():
        post_means = {
            scenario_for(3, split, index).demand_parameters["mu_after"]
            for index in range(count)
        }
        assert post_means == {4.0, 12.0}


def test_capacity_is_fixed_across_tier5_seeds_and_controls():
    for index in range(SPLIT_SIZES["test"]):
        for variant in (
            "headline",
            "t5_control_base_rival",
            "t5_control_uniform",
        ):
            assert scenario_for(5, "test", index, variant).capacity == 22


def test_scenario_round_trip_preserves_canonical_json():
    spec = scenario_for(5, "validation", 2)
    assert scenario_from_dict(spec.to_dict()).canonical_json() == spec.canonical_json()
