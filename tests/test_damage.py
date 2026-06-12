import copy
from unittest.mock import patch

import numpy as np
import pytest

from tft_sim.env.state import GameState
from tft_sim.game.player import Player
from tft_sim.game.units import UnitDatabase


def _game_with_agent_match(agent_is_p1: bool):
    """Two-player game so the agent is in exactly one PvP match."""
    game = GameState(n_players=2, rng=np.random.default_rng(0))
    game.stage = 2
    game.round_in_stage = 1
    game.players = [Player(health=100), Player(health=100)]
    for pl in game.players:
        pl.is_agent = False

    db = game.unit_db
    strong = [db.create_unit(0)] + [None] * 9
    weak = [None] * 10

    if agent_is_p1:
        game.players[0].board = copy.deepcopy(strong)
        game.players[1].board = copy.deepcopy(weak)
        game.agent = game.players[0]
    else:
        game.players[0].board = copy.deepcopy(weak)
        game.players[1].board = copy.deepcopy(strong)
        game.agent = game.players[1]
    game.agent.is_agent = True

    return game


def test_agent_takes_damage_when_losing_as_team_b():
    game = _game_with_agent_match(agent_is_p1=False)
    hp_before = game.agent.health
    with patch("tft_sim.env.state.resolve_combat", return_value=("A", 1)):
        game.resolve_round()
    assert game.agent.health < hp_before


def test_agent_takes_damage_when_losing_as_team_a():
    game = _game_with_agent_match(agent_is_p1=True)
    hp_before = game.agent.health
    with patch("tft_sim.env.state.resolve_combat", return_value=("B", 1)):
        game.resolve_round()
    assert game.agent.health < hp_before


def test_illegal_action_raises():
    game = GameState(n_players=2, rng=np.random.default_rng(0))
    game.start_round()
    with pytest.raises(ValueError, match="Illegal action"):
        game.apply_action(999, game.agent, count_agent_action=True)
