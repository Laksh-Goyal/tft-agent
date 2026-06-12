import numpy as np

from tft_sim.env.state import GameState


def test_same_rng_seed_produces_identical_shops():
    g1 = GameState(n_players=2, rng=np.random.default_rng(42))
    g2 = GameState(n_players=2, rng=np.random.default_rng(42))

    g1.start_round()
    g2.start_round()

    assert g1.players[0].current_shop == g2.players[0].current_shop
    assert g1.players[1].current_shop == g2.players[1].current_shop


def test_different_seeds_produce_different_shops():
    g1 = GameState(n_players=2, rng=np.random.default_rng(1))
    g2 = GameState(n_players=2, rng=np.random.default_rng(2))

    g1.start_round()
    g2.start_round()

    assert g1.players[0].current_shop != g2.players[0].current_shop or (
        g1.players[1].current_shop != g2.players[1].current_shop
    )
