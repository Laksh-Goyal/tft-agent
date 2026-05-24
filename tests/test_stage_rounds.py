from tft_sim.env.state import GameState, max_rounds_in_stage


def test_max_rounds_in_stage():
    assert max_rounds_in_stage(1) == 4
    assert max_rounds_in_stage(2) == 6
    assert max_rounds_in_stage(4) == 6
    assert max_rounds_in_stage(5) == 5
    assert max_rounds_in_stage(6) == 6


def test_advances_from_stage_4_to_5_after_six_rounds():
    import numpy as np

    game = GameState(n_players=2, rng=np.random.default_rng(0))
    game.stage = 4
    game.round_in_stage = 6
    game.start_round()
    assert game.stage == 5
    assert game.round_in_stage == 1


def test_stage_5_has_five_rounds():
    import numpy as np

    game = GameState(n_players=2, rng=np.random.default_rng(0))
    game.stage = 5
    game.round_in_stage = 5
    game.start_round()
    assert game.stage == 6
    assert game.round_in_stage == 1
