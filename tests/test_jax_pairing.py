"""Tests for JAX 8-player PvP pairing, PvE-for-all, and placement."""
import jax
import jax.numpy as jnp
import pytest

from tft_sim.jax_port.static_data import load_static_data
from tft_sim.jax_port.game_state import (
    make_game_state, BOARD_SIZE, ROUND_PVP, ROUND_PVE,
)
from tft_sim.jax_port.step import (
    pair_living_players, apply_match_jax, resolve_round_jax,
    agent_placement_jax, is_last_player_standing_jax,
)
from tft_sim.jax_port.combat import BASE_DAMAGE, STAGE_DAMAGE_SCALE


@pytest.fixture(scope="module")
def static():
    s, _ = load_static_data()
    return s


def _empty_board():
    return jnp.full(BOARD_SIZE, -1, dtype=jnp.int32), jnp.ones(BOARD_SIZE, dtype=jnp.int32)


def _board_with(units):
    ids = jnp.full(BOARD_SIZE, -1, dtype=jnp.int32)
    stars = jnp.ones(BOARD_SIZE, dtype=jnp.int32)
    for i, (uid, star) in enumerate(units):
        ids = ids.at[i].set(uid)
        stars = stars.at[i].set(star)
    return ids, stars


def _set_board(state, pid, units):
    ids, stars = _board_with(units)
    p = state.players
    return state.replace(players=p.replace(
        board_ids=p.board_ids.at[pid].set(ids),
        board_stars=p.board_stars.at[pid].set(stars),
    ))


def _set_empty(state, pid):
    ids, stars = _empty_board()
    p = state.players
    return state.replace(players=p.replace(
        board_ids=p.board_ids.at[pid].set(ids),
        board_stars=p.board_stars.at[pid].set(stars),
    ))


def _eliminate(state, pids):
    p = state.players
    elim = p.is_eliminated
    for pid in pids:
        elim = elim.at[pid].set(True)
    return state.replace(players=p.replace(is_eliminated=elim))


def _pvp_state(static, key=None, stage=2):
    if key is None:
        key = jax.random.PRNGKey(0)
    state = make_game_state(8, static, key)
    return state.replace(round_type=jnp.int32(ROUND_PVP), stage=jnp.int32(stage))


class TestPairLivingPlayers:
    def test_eight_alive_four_matches(self):
        eliminated = jnp.zeros(8, dtype=jnp.bool_)
        match_a, match_b, valid = pair_living_players(eliminated, jax.random.PRNGKey(1))
        assert int(valid.sum()) == 4
        assert not bool(valid[-1])
        paired = [int(x) for x in list(match_a[:4]) + list(match_b[:4])]
        assert sorted(paired) == list(range(8))

    def test_three_alive_pair_plus_ghost(self):
        eliminated = jnp.array([False, False, False, True, True, True, True, True])
        match_a, match_b, valid = pair_living_players(eliminated, jax.random.PRNGKey(2))
        assert int(valid.sum()) == 2
        assert bool(valid[0])
        assert bool(valid[-1])
        leftover = int(match_a[-1])
        ghost = int(match_b[-1])
        living = {0, 1, 2}
        assert leftover in living
        assert ghost in living
        assert leftover != ghost
        first = {int(match_a[0]), int(match_b[0])}
        assert ghost in first
        assert leftover not in first

    def test_one_alive_no_matches(self):
        eliminated = jnp.array([False, True, True, True, True, True, True, True])
        _, _, valid = pair_living_players(eliminated, jax.random.PRNGKey(3))
        assert int(valid.sum()) == 0


class TestApplyMatch:
    def test_winner_unhurt_loser_takes_damage(self, static):
        state = _pvp_state(static)
        state = _set_board(state, 0, [(0, 3)])
        state = _set_empty(state, 1)
        before = state.players.health.copy()
        players, agent_won, agent_lost = apply_match_jax(
            state.players, jnp.int32(0), jnp.int32(1), jnp.bool_(True),
            state.stage, static,
        )
        assert bool(agent_won)
        assert not bool(agent_lost)
        assert int(players.health[0]) == int(before[0])
        assert int(players.health[1]) < int(before[1])
        assert int(players.win_streak[0]) == 1
        assert int(players.loss_streak[1]) == 1

    def test_tie_no_damage_or_streak(self, static):
        state = _pvp_state(static)
        players, won, lost = apply_match_jax(
            state.players, jnp.int32(0), jnp.int32(1), jnp.bool_(True),
            state.stage, static,
        )
        assert not bool(won)
        assert not bool(lost)
        assert jnp.array_equal(players.health, state.players.health)
        assert jnp.array_equal(players.win_streak, state.players.win_streak)
        assert jnp.array_equal(players.loss_streak, state.players.loss_streak)

    def test_ghost_can_take_damage_twice(self, static):
        state = _pvp_state(static, stage=2)
        eliminated = jnp.array([False, False, False, True, True, True, True, True])
        state = state.replace(players=state.players.replace(is_eliminated=eliminated))
        match_a, match_b, valid = pair_living_players(eliminated, jax.random.PRNGKey(4))
        leftover = int(match_a[-1])
        ghost = int(match_b[-1])
        first_a, first_b = int(match_a[0]), int(match_b[0])
        if first_a == ghost:
            state = _set_empty(state, first_a)
            state = _set_board(state, first_b, [(0, 3)])
        else:
            state = _set_board(state, first_a, [(0, 3)])
            state = _set_empty(state, first_b)
        state = _set_board(state, leftover, [(0, 3)])
        before = int(state.players.health[ghost])
        p = state.players
        p, _, _ = apply_match_jax(
            p, match_a[0], match_b[0], valid[0], state.stage, static
        )
        mid = int(p.health[ghost])
        p, _, _ = apply_match_jax(
            p, match_a[-1], match_b[-1], valid[-1], state.stage, static
        )
        after = int(p.health[ghost])
        min_hit = int(BASE_DAMAGE + 2 * STAGE_DAMAGE_SCALE)
        assert mid < before
        assert after < mid
        assert before - after >= 2 * min_hit


class TestResolveRoundPvp:
    def test_eight_alive_losers_take_damage(self, static):
        state = _pvp_state(static, jax.random.PRNGKey(10))
        for pid in range(8):
            state = _set_empty(state, pid)
        state = _set_board(state, 5, [(0, 3), (0, 3), (0, 3)])
        new_state, _, _ = resolve_round_jax(state, static)
        damaged = [
            i for i in range(8)
            if int(new_state.players.health[i]) < int(state.players.health[i])
        ]
        assert len(damaged) == 1
        assert int(new_state.players.health[5]) == int(state.players.health[5])

    def test_one_alive_no_combat(self, static):
        state = _pvp_state(static)
        state = _eliminate(state, list(range(1, 8)))
        state = _set_board(state, 0, [(0, 1)])
        new_state, reward, terminated = resolve_round_jax(state, static)
        assert jnp.array_equal(new_state.players.health, state.players.health)
        assert float(reward) == 0.0
        assert not bool(terminated)

    def test_elimination_returns_units_to_pool(self, static):
        state = _pvp_state(static, stage=2)
        state = _eliminate(state, list(range(2, 8)))
        state = _set_board(state, 0, [(0, 3), (0, 3), (0, 3)])
        state = _set_board(state, 1, [(0, 1)])
        p = state.players
        state = state.replace(players=p.replace(health=p.health.at[1].set(1)))
        cost = int(static.unit_costs[0])
        before = int(state.pool.pool_counts[cost])
        new_state, _, _ = resolve_round_jax(state, static)
        assert bool(new_state.players.is_eliminated[1])
        assert int(new_state.pool.pool_counts[cost]) > before


class TestPlacement:
    def test_agent_last_standing_is_first(self, static):
        state = make_game_state(8, static, jax.random.PRNGKey(0))
        state = _eliminate(state, list(range(1, 8)))
        assert int(agent_placement_jax(state)) == 1
        assert bool(is_last_player_standing_jax(state))

    def test_agent_dead_with_five_alive_is_sixth(self, static):
        state = make_game_state(8, static, jax.random.PRNGKey(0))
        # Alive: 1–5. Dead: 0 (agent), 6, 7. Agent is first among dead → place 6.
        p = state.players
        health = p.health
        for i, h in enumerate([0, 90, 80, 70, 60, 50, 0, 0]):
            health = health.at[i].set(h)
        elim = jnp.array([True, False, False, False, False, False, True, True])
        state = state.replace(players=p.replace(health=health, is_eliminated=elim))
        assert int(agent_placement_jax(state)) == 6
        assert not bool(is_last_player_standing_jax(state))

    def test_living_sorted_by_health(self, static):
        state = make_game_state(8, static, jax.random.PRNGKey(0))
        p = state.players
        health = p.health
        health = health.at[0].set(10)
        health = health.at[1].set(50)
        state = state.replace(players=p.replace(health=health))
        # All alive; agent has lowest health → place 8.
        assert int(agent_placement_jax(state)) == 8


class TestPveAllLiving:
    def test_living_opponents_gain_gold_eliminated_do_not(self, static):
        key = jax.random.PRNGKey(0)
        state = make_game_state(8, static, key)
        state = state.replace(round_type=jnp.int32(ROUND_PVE), stage=jnp.int32(1))
        strong = [(0, 3)] * 7
        for pid in range(8):
            state = _set_board(state, pid, strong)
        state = _eliminate(state, [7])
        p = state.players
        state = state.replace(players=p.replace(gold=jnp.zeros(8, dtype=jnp.int32)))
        new_state, _, terminated = resolve_round_jax(state, static)
        assert not bool(terminated)
        for pid in range(7):
            assert int(new_state.players.gold[pid]) > 0
        assert int(new_state.players.gold[7]) == 0
        assert jnp.array_equal(new_state.players.health, state.players.health)
