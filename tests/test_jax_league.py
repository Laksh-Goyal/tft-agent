"""Tests for JAX LeagueManager and policy-slot assignment."""
import jax
import jax.numpy as jnp
import pytest

from tft_sim.jax_port.league import LeagueManager
from tft_sim.jax_port.static_data import (
    load_static_data, WIN_RATE_WINDOW, GRADUATION_THRESHOLD, MAX_POLICY_BOTS,
    OPPONENT_POLICY,
)
from tft_sim.jax_port.step import apply_policy_slots_jax, reset_jax
from tft_sim.jax_port.game_state import make_game_state


@pytest.fixture
def static():
    s, _ = load_static_data()
    return s


class TestLeagueManager:
    def test_win_rate_and_threshold(self, tmp_path):
        league = LeagueManager(save_dir=tmp_path)
        for _ in range(WIN_RATE_WINDOW):
            league.record_episode(1)
        assert league.win_rate() == 1.0
        assert league.maybe_graduate()
        assert league.n_policy_bots == 1
        assert (tmp_path / "league.json").exists()

    def test_does_not_graduate_below_threshold(self):
        league = LeagueManager()
        for _ in range(WIN_RATE_WINDOW):
            league.record_episode(2)
        assert league.win_rate() == 0.0
        assert not league.maybe_graduate()

    def test_caps_at_max_policy_bots(self, tmp_path):
        league = LeagueManager(save_dir=tmp_path)
        for _ in range(MAX_POLICY_BOTS):
            for _ in range(WIN_RATE_WINDOW):
                league.record_episode(1)
            assert league.maybe_graduate()
        for _ in range(WIN_RATE_WINDOW):
            league.record_episode(1)
        assert not league.maybe_graduate()
        assert league.n_policy_bots == MAX_POLICY_BOTS


class TestPolicySlots:
    def test_first_n_opponents_marked(self, static):
        state = make_game_state(8, static, jax.random.PRNGKey(0))
        state = apply_policy_slots_jax(state, jnp.int32(2))
        assert int(state.players.opponent_type[0]) == 0
        assert int(state.players.opponent_type[1]) == OPPONENT_POLICY
        assert int(state.players.opponent_type[2]) == OPPONENT_POLICY
        assert int(state.players.opponent_type[3]) == 0
        assert int(state.players.policy_bot_index[1]) == 0
        assert int(state.players.policy_bot_index[2]) == 1
