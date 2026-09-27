"""Tests for JAX unit combining (3 copies -> next star)."""
import jax
import jax.numpy as jnp
import pytest

from tft_sim.jax_port.static_data import load_static_data, ACTION_BUY_UNIT_START
from tft_sim.jax_port.game_state import make_game_state, BOARD_SIZE, BENCH_SIZE
from tft_sim.jax_port.step import try_combine_jax, apply_action_jax, count_copies_jax


@pytest.fixture(scope="module")
def static():
    s, _ = load_static_data()
    return s


def _empty():
    return (
        jnp.full(BOARD_SIZE, -1, dtype=jnp.int32),
        jnp.ones(BOARD_SIZE, dtype=jnp.int32),
        jnp.full(BENCH_SIZE, -1, dtype=jnp.int32),
        jnp.ones(BENCH_SIZE, dtype=jnp.int32),
    )


class TestCombine:
    def test_three_one_stars_become_two_star(self, static):
        board_ids, board_stars, bench_ids, bench_stars = _empty()
        uid = jnp.int32(0)
        bench_ids = bench_ids.at[0].set(uid).at[1].set(uid).at[2].set(uid)
        bench_stars = bench_stars.at[0].set(1).at[1].set(1).at[2].set(1)
        board_ids, board_stars, bench_ids, bench_stars = try_combine_jax(
            board_ids, board_stars, bench_ids, bench_stars, uid, jnp.int32(1)
        )
        n1 = int(count_copies_jax(board_ids, board_stars, bench_ids, bench_stars, uid, jnp.int32(1)))
        n2 = int(count_copies_jax(board_ids, board_stars, bench_ids, bench_stars, uid, jnp.int32(2)))
        assert n1 == 0
        assert n2 == 1
        # Upgrade lands on first empty bench (all three were cleared)
        assert int(bench_ids[0]) == 0
        assert int(bench_stars[0]) == 2

    def test_bench_consumed_before_board(self, static):
        board_ids, board_stars, bench_ids, bench_stars = _empty()
        uid = jnp.int32(1)
        bench_ids = bench_ids.at[0].set(uid).at[1].set(uid)
        board_ids = board_ids.at[0].set(uid)
        board_ids, board_stars, bench_ids, bench_stars = try_combine_jax(
            board_ids, board_stars, bench_ids, bench_stars, uid, jnp.int32(1)
        )
        assert int(bench_ids[0]) == -1 or int(bench_stars[0]) == 2
        assert int(board_ids[0]) == -1
        n2 = int(count_copies_jax(board_ids, board_stars, bench_ids, bench_stars, uid, jnp.int32(2)))
        assert n2 == 1

    def test_cascade_to_three_star(self, static):
        board_ids, board_stars, bench_ids, bench_stars = _empty()
        uid = jnp.int32(0)
        # 9 one-stars: 3->2 three times, then 3 two-stars -> one 3-star
        for i in range(9):
            bench_ids = bench_ids.at[i].set(uid)
            bench_stars = bench_stars.at[i].set(1)
        board_ids, board_stars, bench_ids, bench_stars = try_combine_jax(
            board_ids, board_stars, bench_ids, bench_stars, uid, jnp.int32(1)
        )
        n1 = int(count_copies_jax(board_ids, board_stars, bench_ids, bench_stars, uid, jnp.int32(1)))
        n2 = int(count_copies_jax(board_ids, board_stars, bench_ids, bench_stars, uid, jnp.int32(2)))
        n3 = int(count_copies_jax(board_ids, board_stars, bench_ids, bench_stars, uid, jnp.int32(3)))
        assert n1 == 0
        assert n2 == 0
        assert n3 == 1

    def test_two_copies_do_not_combine(self, static):
        board_ids, board_stars, bench_ids, bench_stars = _empty()
        uid = jnp.int32(0)
        bench_ids = bench_ids.at[0].set(uid).at[1].set(uid)
        board_ids, board_stars, bench_ids, bench_stars = try_combine_jax(
            board_ids, board_stars, bench_ids, bench_stars, uid, jnp.int32(1)
        )
        n1 = int(count_copies_jax(board_ids, board_stars, bench_ids, bench_stars, uid, jnp.int32(1)))
        assert n1 == 2

    def test_combine_on_buy(self, static):
        key = jax.random.PRNGKey(0)
        state = make_game_state(8, static, key)
        uid = jnp.int32(0)
        shop = jnp.array([0, -1, -1, -1, -1], dtype=jnp.int32)
        bench = jnp.full(BENCH_SIZE, -1, dtype=jnp.int32).at[0].set(0).at[1].set(0)
        state = state.replace(
            players=state.players.replace(
                shop=state.players.shop.at[0].set(shop),
                bench_ids=state.players.bench_ids.at[0].set(bench),
                gold=state.players.gold.at[0].set(20),
            ),
        )
        state = apply_action_jax(state, jnp.int32(0), jnp.int32(ACTION_BUY_UNIT_START), static)
        n2 = int(count_copies_jax(
            state.players.board_ids[0], state.players.board_stars[0],
            state.players.bench_ids[0], state.players.bench_stars[0],
            uid, jnp.int32(2),
        ))
        assert n2 == 1

    def test_combine_jit(self, static):
        board_ids, board_stars, bench_ids, bench_stars = _empty()
        uid = jnp.int32(0)
        bench_ids = bench_ids.at[0].set(uid).at[1].set(uid).at[2].set(uid)
        fn = jax.jit(try_combine_jax)
        _, _, new_bench, new_stars = fn(
            board_ids, board_stars, bench_ids, bench_stars, uid, jnp.int32(1)
        )
        assert int(jnp.max(new_stars)) == 2
