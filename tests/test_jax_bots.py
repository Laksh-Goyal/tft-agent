"""Tests for JAX scripted bot planning."""
import jax
import jax.numpy as jnp
import pytest

from tft_sim.jax_port.static_data import (
    load_static_data, ACTION_PASS, ACTION_BUY_XP, ACTION_REROLL, ACTION_BUY_UNIT_START,
    ARCHETYPE_HYPER_BUYER, ARCHETYPE_INTEREST_SAVER, ARCHETYPE_BALANCED,
    ARCHETYPE_LEVEL_RUSHER, ARCHETYPE_ROLLER,
)
from tft_sim.jax_port.game_state import make_game_state, make_player, BOARD_SIZE, BENCH_SIZE, SHOP_SIZE
from tft_sim.jax_port.step import (
    compute_action_mask_jax, run_bot_planning_phase_jax, reset_jax, apply_action_jax,
)
from tft_sim.jax_port.bots import choose_bot_action, pick_priciest_buy, pick_cheapest_buy


@pytest.fixture(scope="module")
def static():
    s, _ = load_static_data()
    return s


def _player_with(**kwargs):
    p = make_player(is_agent=False)
    return p.replace(**kwargs)


class TestArchetypeFirstAction:
    def _mask(self, player, static):
        return compute_action_mask_jax(player, static)

    def test_hyper_buyer_buys_priciest(self, static):
        shop = jnp.array([0, 5, -1, -1, -1], dtype=jnp.int32)  # cost 1 and whatever 5 is
        player = _player_with(
            gold=jnp.int32(50),
            shop=shop,
        )
        mask = self._mask(player, static)
        action = int(choose_bot_action(player, mask, static, jnp.int32(ARCHETYPE_HYPER_BUYER)))
        expected = int(pick_priciest_buy(player, mask, static))
        assert action == expected
        assert action != ACTION_PASS

    def test_interest_saver_below_cap_does_not_spend(self, static):
        player = _player_with(gold=jnp.int32(40), shop=jnp.array([0, 1, 2, -1, -1], dtype=jnp.int32))
        mask = self._mask(player, static)
        action = int(choose_bot_action(player, mask, static, jnp.int32(ARCHETYPE_INTEREST_SAVER)))
        assert action != ACTION_BUY_UNIT_START
        assert action != ACTION_REROLL
        assert action != ACTION_BUY_XP

    def test_balanced_buys_when_rich(self, static):
        player = _player_with(gold=jnp.int32(20), shop=jnp.array([0, 1, 2, -1, -1], dtype=jnp.int32))
        mask = self._mask(player, static)
        action = int(choose_bot_action(player, mask, static, jnp.int32(ARCHETYPE_BALANCED)))
        expected = int(pick_cheapest_buy(player, mask, static))
        assert action == expected

    def test_level_rusher_buys_xp(self, static):
        player = _player_with(gold=jnp.int32(20), level=jnp.int32(1), shop=jnp.full(SHOP_SIZE, -1, dtype=jnp.int32))
        mask = self._mask(player, static)
        action = int(choose_bot_action(player, mask, static, jnp.int32(ARCHETYPE_LEVEL_RUSHER)))
        assert action == ACTION_BUY_XP

    def test_roller_rerolls(self, static):
        player = _player_with(gold=jnp.int32(20), shop=jnp.full(SHOP_SIZE, -1, dtype=jnp.int32))
        mask = self._mask(player, static)
        action = int(choose_bot_action(player, mask, static, jnp.int32(ARCHETYPE_ROLLER)))
        assert action == ACTION_REROLL


class TestBotPhase:
    def test_does_not_increment_agent_actions(self, static):
        state = reset_jax(static, jax.random.PRNGKey(0))
        # Move off carousel so bots spend a budget
        from tft_sim.jax_port.step import step_jax
        state = step_jax(state, jnp.int32(ACTION_PASS), static).state
        before = int(state.actions_this_round)
        state = run_bot_planning_phase_jax(state, static)
        assert int(state.actions_this_round) == before

    def test_opponents_acquire_units(self, static):
        state = reset_jax(static, jax.random.PRNGKey(1))
        state = state.replace(
            players=state.players.replace(
                gold=state.players.gold.at[:].set(30),
                bot_strategy_id=state.players.bot_strategy_id.at[1].set(ARCHETYPE_HYPER_BUYER),
            ),
            round_type=jnp.int32(2),  # pvp so full planning loop
            stage=jnp.int32(1),
        )
        state = run_bot_planning_phase_jax(state, static)
        opp_units = int(jnp.sum(state.players.bench_ids[1] >= 0) + jnp.sum(state.players.board_ids[1] >= 0))
        assert opp_units >= 1

    def test_agent_not_acted_on(self, static):
        state = reset_jax(static, jax.random.PRNGKey(2))
        agent_gold = int(state.players.gold[0])
        state = run_bot_planning_phase_jax(state, static)
        # Carousel: agent is skipped
        assert int(state.players.gold[0]) == agent_gold
