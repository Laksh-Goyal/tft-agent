"""Tests for terminal placement reward and reward helpers."""
import numpy as np

from tft_sim.env.metrics import placement_reward, agent_placement
from tft_sim.env.state import GameState
from tft_sim.env.tft_env import TFTEnv
from tft_sim.env.action_space import ACTION_PASS


def test_placement_reward_scale():
    assert placement_reward(1) == 4.0
    assert placement_reward(8) == 0.5


def test_episode_end_includes_placement_reward():
    env = TFTEnv(n_players=2)
    env.reset(seed=0)

    # Force agent win: eliminate opponent via direct state edit after one pass
    env.game.players[1].health = 0
    env.game.players[1].is_eliminated = True

    _, reward, terminated, truncated, _ = env.step(ACTION_PASS)
    assert terminated or truncated
    assert reward >= placement_reward(1)


def test_eliminated_agent_gets_low_placement_reward():
    game = GameState(n_players=4, rng=np.random.default_rng(0))
    game.agent.health = 0
    game.agent.is_eliminated = True
    for p in game.players:
        if p is not game.agent and not p.is_eliminated:
            p.health = 50
    placement = agent_placement(game)
    assert placement == 4
    assert placement_reward(placement) == 2.5
