"""Tests for JAX pool tracking, shop rolls, and start_round economy."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tft_sim.jax_port.static_data import load_static_data, ACTION_REROLL, ACTION_BUY_UNIT_START, ACTION_SELL_BENCH_START
from tft_sim.jax_port.game_state import (
    make_game_state, make_pool, return_to_pool, reserve_from_pool,
    roll_shop_jax, calculate_income_jax, return_ids_to_pool,
)
from tft_sim.jax_port.step import (
    apply_action_jax, start_round_jax, reset_jax, step_jax,
)


@pytest.fixture(scope="module")
def static():
    s, _ = load_static_data()
    return s


class TestPoolOps:
    def test_reserve_decrements_count(self, static):
        pool = make_pool(static)
        uid = jnp.int32(0)
        cost = static.unit_costs[0]
        before = int(pool.pool_counts[cost])
        new_pool, ok = reserve_from_pool(pool, uid, cost)
        assert bool(ok)
        assert int(new_pool.pool_counts[cost]) == before - 1

    def test_return_restores_count(self, static):
        pool = make_pool(static)
        uid = jnp.int32(0)
        cost = static.unit_costs[0]
        pool, _ = reserve_from_pool(pool, uid, cost)
        mid = int(pool.pool_counts[cost])
        pool = return_to_pool(pool, uid, cost)
        assert int(pool.pool_counts[cost]) == mid + 1

    def test_return_empty_is_noop(self, static):
        pool = make_pool(static)
        before = np.array(pool.pool_counts)
        pool = return_to_pool(pool, jnp.int32(-1), jnp.int32(1))
        np.testing.assert_array_equal(np.array(pool.pool_counts), before)

    def test_roll_shop_depletes_pool(self, static):
        pool = make_pool(static)
        total_before = int(jnp.sum(pool.pool_counts))
        key = jax.random.PRNGKey(0)
        new_pool, shop, _ = roll_shop_jax(pool, jnp.int32(1), key, static)
        n_filled = int(jnp.sum(shop >= 0))
        total_after = int(jnp.sum(new_pool.pool_counts))
        assert n_filled == 5
        assert total_after == total_before - n_filled
        # Level 1 shop is all cost-1 units
        for uid in np.array(shop):
            assert int(static.unit_costs[int(uid)]) == 1

    def test_roll_shop_jit(self, static):
        pool = make_pool(static)
        fn = jax.jit(lambda p, k: roll_shop_jax(p, jnp.int32(3), k, static)[1])
        shop = fn(pool, jax.random.PRNGKey(1))
        assert shop.shape == (5,)


class TestBuySellPool:
    def test_buy_does_not_double_reserve(self, static):
        key = jax.random.PRNGKey(2)
        state = make_game_state(8, static, key)
        pool, shop, key = roll_shop_jax(state.pool, jnp.int32(1), key, static)
        state = state.replace(
            pool=pool,
            players=state.players.replace(
                shop=state.players.shop.at[0].set(shop),
                gold=state.players.gold.at[0].set(20),
            ),
        )
        before = int(jnp.sum(state.pool.pool_counts))
        state = apply_action_jax(state, jnp.int32(0), jnp.int32(ACTION_BUY_UNIT_START), static)
        # Shop slot was already reserved at roll; buy must not reserve again
        assert int(jnp.sum(state.pool.pool_counts)) == before
        assert int(state.players.shop[0, 0]) == -1
        assert int(state.players.bench_ids[0, 0]) >= 0

    def test_sell_returns_one_copy(self, static):
        key = jax.random.PRNGKey(3)
        state = make_game_state(8, static, key)
        uid = jnp.int32(0)
        cost = static.unit_costs[0]
        state = state.replace(
            players=state.players.replace(
                bench_ids=state.players.bench_ids.at[0, 0].set(uid),
                bench_stars=state.players.bench_stars.at[0, 0].set(1),
                gold=state.players.gold.at[0].set(10),
            ),
        )
        before = int(state.pool.pool_counts[cost])
        state = apply_action_jax(
            state, jnp.int32(0), jnp.int32(ACTION_SELL_BENCH_START), static
        )
        assert int(state.pool.pool_counts[cost]) == before + 1
        assert int(state.players.bench_ids[0, 0]) == -1
        assert int(state.players.bench_stars[0, 0]) == 1
        assert int(state.players.gold[0]) == 10 + int(cost)

    def test_reroll_returns_old_shop(self, static):
        key = jax.random.PRNGKey(4)
        state = make_game_state(8, static, key)
        pool, shop, key = roll_shop_jax(state.pool, jnp.int32(1), key, static)
        total_after_roll = int(jnp.sum(pool.pool_counts))
        state = state.replace(
            pool=pool,
            rng_key=key,
            players=state.players.replace(
                shop=state.players.shop.at[0].set(shop),
                gold=state.players.gold.at[0].set(20),
            ),
        )
        state = apply_action_jax(state, jnp.int32(0), jnp.int32(ACTION_REROLL), static)
        total_after_reroll = int(jnp.sum(state.pool.pool_counts))
        n_new = int(jnp.sum(state.players.shop[0] >= 0))
        # Returned 5, then reserved n_new — net zero if both shops full
        assert total_after_reroll == total_after_roll + 5 - n_new
        assert int(state.players.gold[0]) == 18


class TestStartRound:
    def test_reset_starts_carousel_with_shops(self, static):
        state = reset_jax(static, jax.random.PRNGKey(5))
        assert int(state.stage) == 1
        assert int(state.round_in_stage) == 1
        assert int(state.round_type) == 0  # carousel
        # Carousel shops filled, no income on 1-1
        assert int(state.players.gold[0]) == 0
        assert int(jnp.sum(state.players.shop[0, :4] >= 0)) == 4

    def test_pve_round_grants_early_income(self, static):
        state = reset_jax(static, jax.random.PRNGKey(6))
        # PASS carousel -> start 1-2 PvE
        result = step_jax(state, jnp.int32(0), static)
        assert int(result.state.round_in_stage) == 2
        assert int(result.state.round_type) == 1  # pve
        assert int(result.state.players.gold[0]) == 2  # early income

    def test_income_formula(self):
        # Stage 2, 30 gold, no streak: 5 + 3 interest = 8
        inc = calculate_income_jax(
            jnp.int32(30), jnp.int32(0), jnp.int32(0), jnp.int32(2), jnp.int32(2)
        )
        assert int(inc) == 8
        # Stage 1 round 2: early 2
        inc = calculate_income_jax(
            jnp.int32(50), jnp.int32(0), jnp.int32(0), jnp.int32(1), jnp.int32(2)
        )
        assert int(inc) == 2
        # 5-win streak: +3 streak bonus
        inc = calculate_income_jax(
            jnp.int32(10), jnp.int32(5), jnp.int32(0), jnp.int32(2), jnp.int32(2)
        )
        assert int(inc) == 5 + 1 + 3

    def test_start_round_jit(self, static):
        state = make_game_state(8, static, jax.random.PRNGKey(7))
        fn = jax.jit(lambda s: start_round_jax(s, static))
        out = fn(state)
        assert int(out.round_in_stage) == 1
