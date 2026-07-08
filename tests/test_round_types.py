"""Tests for carousel / PvE / PvP round types."""
import numpy as np

from tft_sim.env.state import GameState
from tft_sim.env.tft_env import TFTEnv
from tft_sim.env.action_space import ACTION_PASS
from tft_sim.game.rounds import determine_round_type


def test_round_type_mapping():
    assert determine_round_type(1, 1) == "carousel"
    assert determine_round_type(1, 3) == "pve_creep"
    assert determine_round_type(3, 2) == "pvp"


def test_carousel_free_pick():
    env = TFTEnv(n_players=4)
    env.reset(seed=0)
    assert env.game.round_type == "carousel"
    gold_before = env.game.agent.gold
    mask = env.game.action_mask()
    buy_actions = [i for i in range(3, 8) if mask[i]]
    if buy_actions:
        env.step(buy_actions[0])
        assert env.game.agent.gold == gold_before


def test_pve_no_hp_loss_on_loss():
    game = GameState(n_players=2, rng=np.random.default_rng(1))
    game.start_round()
    while game.round_type != "pve_creep":
        game.start_round()
    hp = game.agent.health
    reward = game.resolve_round()
    assert game.agent.health == hp
    assert isinstance(reward, float)


def test_stage1_curriculum_truncates():
    env = TFTEnv(n_players=4, curriculum_mode="stage1_economy")
    env.reset(seed=2)
    steps = 0
    truncated = False
    while steps < 500 and not truncated:
        mask = env.game.action_mask()
        legal = np.where(mask)[0]
        action = int(legal[0]) if len(legal) else ACTION_PASS
        _, _, term, trunc, _ = env.step(action)
        truncated = trunc or term
        steps += 1
    assert truncated or steps < 500
