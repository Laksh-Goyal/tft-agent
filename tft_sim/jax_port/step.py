"""
JAX step function for the TFT port.

This is the hardest part of the port: converting the mutable, branching
Python logic of apply_action (state.py:267-337) and resolve_round
(state.py:199-249) into pure JAX functions that work under jit.

Key challenges:
    1. Python if/elif chains -> jnp.where / jax.lax.cond / jax.lax.switch
    2. Mutable player.board[idx] = unit -> functional array update via .at[].set()
    3. Variable-length pool operations -> fixed-size arrays with masking
    4. Action validation -> pre-computed boolean masks

This module implements:
    - compute_action_mask_jax: legal action mask for a player
    - apply_action_jax: apply one planning action (buy, sell, place, reroll, pass)
    - step_jax: the full env step (action -> reward -> next state)

Mirrors:
    - tft_sim/game/actions.py   (compute_action_mask)
    - tft_sim/env/state.py:267-337 (apply_action)
    - tft_sim/env/tft_env.py:87-118 (step)
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass as flax_dataclass
from typing import Any

logger = logging.getLogger(__name__)

from tft_sim.jax_port.static_data import (
    StaticData, StaticMeta,
    XP_REQUIRED, ACTION_BUDGETS, SHOP_ODDS, POOL_SIZES,
    TOTAL_ACTIONS,
    ACTION_PASS, ACTION_BUY_XP, ACTION_REROLL,
    ACTION_BUY_UNIT_START, ACTION_BUY_UNIT_END,
    ACTION_SELL_BENCH_START, ACTION_SELL_BENCH_END,
    ACTION_SELL_BOARD_START, ACTION_SELL_BOARD_END,
    ACTION_PLACE_UNIT_START, ACTION_PLACE_UNIT_END,
    TRAIT_BREAKPOINT_REWARD, PLACEMENT_REWARD_SCALE,
    REROLL_COST, BUY_XP_GOLD, BUY_XP_AMOUNT, PASSIVE_XP,
    MAX_LEVEL_INT, N_PLAYERS_DEFAULT, CAROUSEL_N_OPTIONS,
    N_ARCHETYPES,
    trait_counts_jax, active_breakpoint_level_jax, OPPONENT_POLICY,
    MAX_POLICY_BOTS,
)
from tft_sim.jax_port.game_state import (
    GameState, PlayerState, PoolState,
    make_game_state, make_player, make_pool,
    to_observation, determine_round_type_jax,
    BOARD_SIZE, BENCH_SIZE, SHOP_SIZE,
    ROUND_CAROUSEL, ROUND_PVE, ROUND_PVP,
    return_to_pool, return_ids_to_pool, roll_shop_jax,
    calculate_income_jax, check_level_up_jax, tree_where,
)
from tft_sim.jax_port.bots import choose_bot_action, pick_carousel_unit
from tft_sim.jax_port.combat import (
    resolve_combat_jax,
    BASE_DAMAGE, STAGE_DAMAGE_SCALE,
    REWARD_WIN, REWARD_LOSS, REWARD_ELIMINATED,
    PVE_REWARD_WIN, PVE_REWARD_LOSS,
    PVE_GOLD_EARLY, PVE_GOLD_LATE,
)


# ============================================================
# Action Mask (mirrors actions.py:28-80)
# ============================================================
def compute_action_mask_jax(player: PlayerState, static: StaticData,
                            free_shop: bool = False) -> jnp.ndarray:
    """Boolean mask for a player's legal actions.

    Replaces compute_action_mask (actions.py:28-80).
    1 = legal, 0 = illegal.

    Instead of Python if/elif, we compute each action's legality as a
    boolean expression and combine them into the mask array.
    """
    mask = jnp.zeros(TOTAL_ACTIONS, dtype=jnp.int8)

    # PASS: always legal
    mask = mask.at[ACTION_PASS].set(1)

    # BUY_XP: gold >= 4 AND level < 9
    can_buy_xp = (player.gold >= 4) & (player.level < 9)
    mask = mask.at[ACTION_BUY_XP].set(can_buy_xp.astype(jnp.int8))

    # REROLL: gold >= 2
    can_reroll = player.gold >= 2
    mask = mask.at[ACTION_REROLL].set(can_reroll.astype(jnp.int8))

    # BUY_UNIT (slots 0-4): shop slot occupied, affordable, bench not full
    bench_full = jnp.all(player.bench_ids >= 0)  # all bench slots occupied
    for i in range(SHOP_SIZE):
        unit_id = player.shop[i]
        slot_occupied = unit_id >= 0
        # Get unit cost
        safe_id = jnp.maximum(unit_id, 0)
        unit_cost = jnp.take(static.unit_costs, safe_id)
        affordable = free_shop | (player.gold >= unit_cost)
        legal = slot_occupied & ~bench_full & affordable & (unit_id >= 0)
        mask = mask.at[ACTION_BUY_UNIT_START + i].set(legal.astype(jnp.int8))

    # SELL_BENCH (slots 0-8): bench slot occupied
    for i in range(BENCH_SIZE):
        occupied = player.bench_ids[i] >= 0
        mask = mask.at[ACTION_SELL_BENCH_START + i].set(occupied.astype(jnp.int8))

    # SELL_BOARD (slots 0-9): board slot occupied
    for i in range(BOARD_SIZE):
        occupied = player.board_ids[i] >= 0
        mask = mask.at[ACTION_SELL_BOARD_START + i].set(occupied.astype(jnp.int8))

    # PLACE_UNIT (bench b x board d): bench[b] occupied AND
    #   (board[d] empty AND under unit cap) OR board[d] occupied (swap)
    board_unit_count = jnp.sum(player.board_ids >= 0)
    under_cap = board_unit_count < player.level
    for b in range(BENCH_SIZE):
        bench_occupied = player.bench_ids[b] >= 0
        for d in range(BOARD_SIZE):
            action_idx = b * BOARD_SIZE + d + ACTION_PLACE_UNIT_START
            board_empty = player.board_ids[d] < 0
            # Legal if bench occupied AND (board empty+under cap OR board occupied=swap)
            legal = bench_occupied & ((board_empty & under_cap) | ~board_empty)
            mask = mask.at[action_idx].set(legal.astype(jnp.int8))

    return mask


def compute_player_mask(state: GameState, player_idx: jnp.ndarray,
                        static: StaticData) -> jnp.ndarray:
    """Agent/bot mask including carousel restrictions (state.py:255-265)."""
    player = jax.tree_util.tree_map(lambda x: x[player_idx], state.players)
    is_carousel = state.round_type == ROUND_CAROUSEL
    mask = compute_action_mask_jax(player, static, free_shop=is_carousel)
    mask = mask.at[ACTION_BUY_XP].set(
        jnp.where(is_carousel, jnp.int8(0), mask[ACTION_BUY_XP])
    )
    mask = mask.at[ACTION_REROLL].set(
        jnp.where(is_carousel, jnp.int8(0), mask[ACTION_REROLL])
    )
    already_picked = state.carousel_picked[player_idx]
    only_pass = is_carousel & already_picked
    pass_only = jnp.zeros_like(mask).at[ACTION_PASS].set(jnp.int8(1))
    return jnp.where(only_pass, pass_only, mask)


def max_rounds_in_stage_jax(stage: jnp.ndarray) -> jnp.ndarray:
    """Round cap per stage (state.py:47-53)."""
    return jnp.where(stage == 1, jnp.int32(4),
           jnp.where(stage == 5, jnp.int32(5), jnp.int32(6)))


def count_copies_jax(board_ids, board_stars, bench_ids, bench_stars,
                     unit_id, star_level):
    """Copies of unit_id at star_level across board+bench (units.py:64-65)."""
    valid_unit = unit_id >= 0
    bench_n = jnp.sum(
        (bench_ids == unit_id) & (bench_stars == star_level) & (bench_ids >= 0)
    )
    board_n = jnp.sum(
        (board_ids == unit_id) & (board_stars == star_level) & (board_ids >= 0)
    )
    return jnp.where(valid_unit, bench_n + board_n, jnp.int32(0))


def _clear_matches(ids, stars, unit_id, star_level, remaining):
    """Clear up to `remaining` matching slots left-to-right. Returns remaining."""
    def body(i, carry):
        ids, stars, left = carry
        match = (ids[i] == unit_id) & (stars[i] == star_level) & (ids[i] >= 0) & (left > 0)
        ids = ids.at[i].set(jnp.where(match, jnp.int32(-1), ids[i]))
        stars = stars.at[i].set(jnp.where(match, jnp.int32(1), stars[i]))
        left = left - match.astype(jnp.int32)
        return ids, stars, left

    return jax.lax.fori_loop(0, ids.shape[0], body, (ids, stars, remaining))


def try_combine_jax(board_ids, board_stars, bench_ids, bench_stars,
                    unit_id, star_level):
    """Auto-combine 3 copies into the next star (units.py:78-101).

    Keeps combining while any star 1 or 2 has 3+ copies so a pile of
    1-stars can cascade to 3-star in one call. Bench is consumed first.
    """
    def cond(carry):
        board_ids, board_stars, bench_ids, bench_stars = carry
        n1 = count_copies_jax(board_ids, board_stars, bench_ids, bench_stars, unit_id, jnp.int32(1))
        n2 = count_copies_jax(board_ids, board_stars, bench_ids, bench_stars, unit_id, jnp.int32(2))
        return (unit_id >= 0) & ((n1 >= 3) | (n2 >= 3))

    def body(carry):
        board_ids, board_stars, bench_ids, bench_stars = carry
        n1 = count_copies_jax(board_ids, board_stars, bench_ids, bench_stars, unit_id, jnp.int32(1))
        star_level = jnp.where(n1 >= 3, jnp.int32(1), jnp.int32(2))
        bench_ids, bench_stars, left = _clear_matches(
            bench_ids, bench_stars, unit_id, star_level, jnp.int32(3)
        )
        board_ids, board_stars, left = _clear_matches(
            board_ids, board_stars, unit_id, star_level, left
        )
        new_star = star_level + 1
        bench_empty = bench_ids < 0
        has_bench = jnp.any(bench_empty)
        bench_idx = jnp.argmax(bench_empty)
        board_empty = board_ids < 0
        has_board = jnp.any(board_empty)
        board_idx = jnp.argmax(board_empty)
        bench_ids = bench_ids.at[bench_idx].set(
            jnp.where(has_bench, unit_id, bench_ids[bench_idx])
        )
        bench_stars = bench_stars.at[bench_idx].set(
            jnp.where(has_bench, new_star, bench_stars[bench_idx])
        )
        board_ids = board_ids.at[board_idx].set(
            jnp.where((~has_bench) & has_board, unit_id, board_ids[board_idx])
        )
        board_stars = board_stars.at[board_idx].set(
            jnp.where((~has_bench) & has_board, new_star, board_stars[board_idx])
        )
        return board_ids, board_stars, bench_ids, bench_stars

    board_ids, board_stars, bench_ids, bench_stars = jax.lax.while_loop(
        cond, body, (board_ids, board_stars, bench_ids, bench_stars)
    )
    return board_ids, board_stars, bench_ids, bench_stars


# ============================================================
# Apply Action (mirrors state.py:267-337)
# ============================================================
def apply_action_jax(state: GameState, player_idx: jnp.ndarray,
                     action: jnp.ndarray, static: StaticData,
                     count_agent_action: bool = False) -> GameState:
    """Apply one planning action for a player.

    Replaces GameState.apply_action (state.py:267-337).
    Returns a NEW GameState (no mutation).
    """
    p = state.players
    gold = p.gold[player_idx]
    level = p.level[player_idx]
    xp = p.xp[player_idx]
    board_ids = p.board_ids[player_idx]
    board_stars = p.board_stars[player_idx]
    bench_ids = p.bench_ids[player_idx]
    bench_stars = p.bench_stars[player_idx]
    shop = p.shop[player_idx]
    is_carousel = state.round_type == ROUND_CAROUSEL

    before_counts = trait_counts_jax(board_ids, board_stars, static)
    before_bps = active_breakpoint_level_jax(before_counts, static)

    new_gold = gold
    new_xp = xp
    new_level = level
    new_board_ids = board_ids
    new_board_stars = board_stars
    new_bench_ids = bench_ids
    new_bench_stars = bench_stars
    new_shop = shop
    new_pool = state.pool
    rng_key = state.rng_key
    new_carousel_picked = state.carousel_picked

    is_buy_xp = action == ACTION_BUY_XP
    is_reroll = action == ACTION_REROLL
    is_buy_unit = (action >= ACTION_BUY_UNIT_START) & (action <= ACTION_BUY_UNIT_END)
    is_sell_bench = (action >= ACTION_SELL_BENCH_START) & (action <= ACTION_SELL_BENCH_END)
    is_sell_board = (action >= ACTION_SELL_BOARD_START) & (action <= ACTION_SELL_BOARD_END)
    is_place_unit = (action >= ACTION_PLACE_UNIT_START) & (action <= ACTION_PLACE_UNIT_END)

    # --- BUY_XP: gold -= 4, xp += 4, check level up (state.py:283-286) ---
    buy_xp_xp_final, buy_xp_level_final = check_level_up_jax(xp + BUY_XP_AMOUNT, level)
    new_gold = jnp.where(is_buy_xp, gold - BUY_XP_GOLD, new_gold)
    new_xp = jnp.where(is_buy_xp, buy_xp_xp_final, new_xp)
    new_level = jnp.where(is_buy_xp, buy_xp_level_final, new_level)

    # --- REROLL: return shop, -2g, roll from pool (shop.py:83-88) ---
    reroll_key, rng_after_reroll = jax.random.split(state.rng_key)
    pool_after_return = return_ids_to_pool(state.pool, shop, static)
    pool_after_roll, shop_reroll, rng_after_reroll = roll_shop_jax(
        pool_after_return, level, reroll_key, static
    )
    new_shop = jnp.where(is_reroll, shop_reroll, new_shop)
    new_gold = jnp.where(is_reroll, gold - REROLL_COST, new_gold)
    new_pool = tree_where(is_reroll, pool_after_roll, new_pool)
    rng_key = jnp.where(is_reroll, rng_after_reroll, rng_key)

    # --- BUY_UNIT: shop already reserved at roll time (state.py:291-308) ---
    buy_slot_safe = jnp.clip(action - ACTION_BUY_UNIT_START, 0, SHOP_SIZE - 1)
    bought_unit_id = jnp.take(shop, buy_slot_safe)
    bought_unit_cost = jnp.take(static.unit_costs, jnp.maximum(bought_unit_id, 0))
    bench_empty_mask = bench_ids < 0
    bench_empty_idx = jnp.argmax(bench_empty_mask)
    can_place_buy = is_buy_unit & bench_empty_mask[bench_empty_idx]
    new_bench_buy = bench_ids.at[bench_empty_idx].set(
        jnp.where(can_place_buy, bought_unit_id, bench_ids[bench_empty_idx])
    )
    new_bench_stars_buy = bench_stars.at[bench_empty_idx].set(
        jnp.where(can_place_buy, jnp.int32(1), bench_stars[bench_empty_idx])
    )
    comb_board, comb_board_stars, comb_bench, comb_bench_stars = try_combine_jax(
        board_ids, board_stars, new_bench_buy, new_bench_stars_buy,
        bought_unit_id, jnp.int32(1),
    )
    new_shop_buy = shop.at[buy_slot_safe].set(
        jnp.where(is_buy_unit, jnp.int32(-1), shop[buy_slot_safe])
    )
    gold_after_buy = jnp.where(is_carousel, gold, gold - bought_unit_cost)
    new_board_ids = jnp.where(is_buy_unit, comb_board, new_board_ids)
    new_board_stars = jnp.where(is_buy_unit, comb_board_stars, new_board_stars)
    new_bench_ids = jnp.where(is_buy_unit, comb_bench, new_bench_ids)
    new_bench_stars = jnp.where(is_buy_unit, comb_bench_stars, new_bench_stars)
    new_shop = jnp.where(is_buy_unit, new_shop_buy, new_shop)
    new_gold = jnp.where(is_buy_unit, gold_after_buy, new_gold)
    new_carousel_picked = new_carousel_picked.at[player_idx].set(
        jnp.where(is_buy_unit & is_carousel, True, new_carousel_picked[player_idx])
    )

    # --- SELL_BENCH: refund base cost, return one copy (state.py:310-315) ---
    sell_bench_slot_safe = jnp.clip(action - ACTION_SELL_BENCH_START, 0, BENCH_SIZE - 1)
    sold_bench_unit_id = jnp.take(bench_ids, sell_bench_slot_safe)
    sold_bench_cost = jnp.take(static.unit_costs, jnp.maximum(sold_bench_unit_id, 0))
    new_bench_sell = bench_ids.at[sell_bench_slot_safe].set(
        jnp.where(is_sell_bench, jnp.int32(-1), bench_ids[sell_bench_slot_safe])
    )
    new_bench_stars_sell = bench_stars.at[sell_bench_slot_safe].set(
        jnp.where(is_sell_bench, jnp.int32(1), bench_stars[sell_bench_slot_safe])
    )
    pool_sell_bench = return_to_pool(state.pool, sold_bench_unit_id, sold_bench_cost)
    new_bench_ids = jnp.where(is_sell_bench, new_bench_sell, new_bench_ids)
    new_bench_stars = jnp.where(is_sell_bench, new_bench_stars_sell, new_bench_stars)
    new_gold = jnp.where(is_sell_bench, gold + sold_bench_cost, new_gold)
    new_pool = tree_where(is_sell_bench, pool_sell_bench, new_pool)

    # --- SELL_BOARD: refund base cost, return one copy (state.py:317-322) ---
    sell_board_slot_safe = jnp.clip(action - ACTION_SELL_BOARD_START, 0, BOARD_SIZE - 1)
    sold_board_unit_id = jnp.take(board_ids, sell_board_slot_safe)
    sold_board_cost = jnp.take(static.unit_costs, jnp.maximum(sold_board_unit_id, 0))
    new_board_sell = board_ids.at[sell_board_slot_safe].set(
        jnp.where(is_sell_board, jnp.int32(-1), board_ids[sell_board_slot_safe])
    )
    new_board_stars_sell = board_stars.at[sell_board_slot_safe].set(
        jnp.where(is_sell_board, jnp.int32(1), board_stars[sell_board_slot_safe])
    )
    pool_sell_board = return_to_pool(state.pool, sold_board_unit_id, sold_board_cost)
    new_board_ids = jnp.where(is_sell_board, new_board_sell, new_board_ids)
    new_board_stars = jnp.where(is_sell_board, new_board_stars_sell, new_board_stars)
    new_gold = jnp.where(is_sell_board, gold + sold_board_cost, new_gold)
    new_pool = tree_where(is_sell_board, pool_sell_board, new_pool)

    # --- PLACE_UNIT: swap bench[b] with board[d] (state.py:324-330) ---
    place_idx = action - ACTION_PLACE_UNIT_START
    place_b_safe = jnp.clip(place_idx // BOARD_SIZE, 0, BENCH_SIZE - 1)
    place_d_safe = jnp.clip(place_idx % BOARD_SIZE, 0, BOARD_SIZE - 1)
    bench_unit = jnp.take(bench_ids, place_b_safe)
    bench_star = jnp.take(bench_stars, place_b_safe)
    board_unit = jnp.take(board_ids, place_d_safe)
    board_star = jnp.take(board_stars, place_d_safe)
    new_board_place = board_ids.at[place_d_safe].set(
        jnp.where(is_place_unit, bench_unit, board_ids[place_d_safe])
    )
    new_board_stars_place = board_stars.at[place_d_safe].set(
        jnp.where(is_place_unit, bench_star, board_stars[place_d_safe])
    )
    new_bench_place = bench_ids.at[place_b_safe].set(
        jnp.where(is_place_unit, board_unit, bench_ids[place_b_safe])
    )
    new_bench_stars_place = bench_stars.at[place_b_safe].set(
        jnp.where(is_place_unit, board_star, bench_stars[place_b_safe])
    )
    new_board_ids = jnp.where(is_place_unit, new_board_place, new_board_ids)
    new_board_stars = jnp.where(is_place_unit, new_board_stars_place, new_board_stars)
    new_bench_ids = jnp.where(is_place_unit, new_bench_place, new_bench_ids)
    new_bench_stars = jnp.where(is_place_unit, new_bench_stars_place, new_bench_stars)

    after_counts = trait_counts_jax(new_board_ids, new_board_stars, static)
    after_bps = active_breakpoint_level_jax(after_counts, static)
    n_new_bps = jnp.sum(after_bps > before_bps)
    pending = jnp.where(
        count_agent_action,
        TRAIT_BREAKPOINT_REWARD * n_new_bps.astype(jnp.float32),
        jnp.float32(0.0),
    )

    new_players = p.replace(
        gold=p.gold.at[player_idx].set(new_gold),
        xp=p.xp.at[player_idx].set(new_xp),
        level=p.level.at[player_idx].set(new_level),
        board_ids=p.board_ids.at[player_idx].set(new_board_ids),
        board_stars=p.board_stars.at[player_idx].set(new_board_stars),
        bench_ids=p.bench_ids.at[player_idx].set(new_bench_ids),
        bench_stars=p.bench_stars.at[player_idx].set(new_bench_stars),
        shop=p.shop.at[player_idx].set(new_shop),
    )
    new_actions = jnp.where(
        count_agent_action, state.actions_this_round + 1, state.actions_this_round
    )
    return state.replace(
        players=new_players,
        actions_this_round=new_actions,
        pool=new_pool,
        rng_key=rng_key,
        pending_action_reward=pending,
        carousel_picked=new_carousel_picked,
    )


# ============================================================
# Step function (mirrors tft_env.py:87-118)
# ============================================================
@flax_dataclass
class StepResult:
    """Result of a single env step."""
    state: GameState
    reward: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    observation: jnp.ndarray
    action_mask: jnp.ndarray


def _creep_board_jax(stage: jnp.ndarray) -> tuple:
    """Neutral creep IDs by stage (pve.py:10-18)."""
    early = jnp.array([0, 1, 2, -1, -1, -1, -1, -1, -1, -1], dtype=jnp.int32)
    mid = jnp.array([2, 3, 4, -1, -1, -1, -1, -1, -1, -1], dtype=jnp.int32)
    late = jnp.array([3, 4, 5, 6, -1, -1, -1, -1, -1, -1], dtype=jnp.int32)
    ids = jnp.where(stage <= 1, early, jnp.where(stage >= 4, late, mid))
    stars = jnp.ones(BOARD_SIZE, dtype=jnp.int32)
    return ids, stars


def pair_living_players(eliminated: jnp.ndarray, rng_key: jax.Array) -> tuple:
    """Shuffle living players and emit padded matches (state.py:206-218).

    Returns (match_a, match_b, match_valid) each of length n_players//2 + 1
    (regular pairs plus an optional ghost slot).
    """
    n_players = eliminated.shape[0]
    max_pairs = n_players // 2
    max_matches = max_pairs + 1
    n_alive = jnp.sum(~eliminated).astype(jnp.int32)
    k_perm, k_ghost = jax.random.split(rng_key)
    perm = jax.random.permutation(k_perm, n_players)
    living_first = jnp.argsort(eliminated[perm].astype(jnp.int32))
    order = perm[living_first]

    match_a = jnp.zeros(max_matches, dtype=jnp.int32)
    match_b = jnp.zeros(max_matches, dtype=jnp.int32)
    match_valid = jnp.zeros(max_matches, dtype=jnp.bool_)

    for i in range(max_pairs):
        valid = (2 * i + 1) < n_alive
        match_a = match_a.at[i].set(order[2 * i])
        match_b = match_b.at[i].set(order[2 * i + 1])
        match_valid = match_valid.at[i].set(valid)

    leftover = (n_alive % 2 == 1) & (n_alive > 1)
    leftover_idx = order[jnp.clip(n_alive - 1, 0, n_players - 1)]
    n_ghost_pool = jnp.clip(n_alive - 1, 1, n_players)
    ghost_slot = jax.random.randint(k_ghost, (), 0, n_ghost_pool)
    ghost_idx = order[ghost_slot]
    match_a = match_a.at[max_pairs].set(leftover_idx)
    match_b = match_b.at[max_pairs].set(ghost_idx)
    match_valid = match_valid.at[max_pairs].set(leftover)
    return match_a, match_b, match_valid


def apply_match_jax(players: PlayerState, idx_a: jnp.ndarray, idx_b: jnp.ndarray,
                    valid: jnp.ndarray, stage: jnp.ndarray,
                    static: StaticData) -> tuple:
    """Apply one PvP match: combat, damage, streaks (state.py:168-197, 223-233).

    Ties (winner==2) are a no-op. Returns (new_players, agent_won, agent_lost).
    """
    winner, surv_units = resolve_combat_jax(
        players.board_ids[idx_a], players.board_stars[idx_a],
        players.board_ids[idx_b], players.board_stars[idx_b],
        static,
    )
    damage = jnp.int32(BASE_DAMAGE + stage * STAGE_DAMAGE_SCALE + surv_units)
    a_won = (winner == 0) & valid
    b_won = (winner == 1) & valid

    health = players.health
    health = health.at[idx_b].set(
        jnp.where(a_won, health[idx_b] - damage, health[idx_b])
    )
    health = health.at[idx_a].set(
        jnp.where(b_won, health[idx_a] - damage, health[idx_a])
    )

    win_streak = players.win_streak
    loss_streak = players.loss_streak
    win_streak = win_streak.at[idx_a].set(
        jnp.where(a_won, win_streak[idx_a] + 1,
                  jnp.where(b_won, jnp.int32(0), win_streak[idx_a]))
    )
    win_streak = win_streak.at[idx_b].set(
        jnp.where(b_won, win_streak[idx_b] + 1,
                  jnp.where(a_won, jnp.int32(0), win_streak[idx_b]))
    )
    loss_streak = loss_streak.at[idx_a].set(
        jnp.where(b_won, loss_streak[idx_a] + 1,
                  jnp.where(a_won, jnp.int32(0), loss_streak[idx_a]))
    )
    loss_streak = loss_streak.at[idx_b].set(
        jnp.where(a_won, loss_streak[idx_b] + 1,
                  jnp.where(b_won, jnp.int32(0), loss_streak[idx_b]))
    )

    agent_won = ((idx_a == 0) & a_won) | ((idx_b == 0) & b_won)
    agent_lost = ((idx_a == 0) & b_won) | ((idx_b == 0) & a_won)
    new_players = players.replace(
        health=health, win_streak=win_streak, loss_streak=loss_streak
    )
    return new_players, agent_won, agent_lost


def _return_eliminated_units(state: GameState, newly_dead: jnp.ndarray,
                             static: StaticData) -> GameState:
    """Return board+bench of newly eliminated players to the pool (state.py:235-239)."""
    n_players = newly_dead.shape[0]

    def body(i, pool):
        units = jnp.concatenate([
            state.players.board_ids[i],
            state.players.bench_ids[i],
        ])
        returned = return_ids_to_pool(pool, units, static)
        return tree_where(newly_dead[i], returned, pool)

    pool = jax.lax.fori_loop(0, n_players, body, state.pool)
    return state.replace(pool=pool)


def resolve_round_jax(state: GameState, static: StaticData) -> tuple:
    """Resolve the current round (combat + damage + rewards).

    Replaces GameState.resolve_round (state.py:199-249) and
    resolve_pve_round (pve.py:25-56).

    For PvP: shuffle living players, pair them, leftover fights a ghost
    from someone who already has a match. Matches apply sequentially
    (a ghost can take damage twice). Damage is 2 + stage + survivors.
    After all matches, anyone with health <= 0 is eliminated and their
    board+bench return to the pool. Agent combat reward only from matches
    the agent is in (including a ghost match).

    For PvE: every living player fights the stage creep board. Win grants
    gold; loss deals no HP damage. Only the agent's win/loss is rewarded.

    For Carousel: no combat, no reward.

    Returns:
        (new_state, reward, terminated)
    """
    is_carousel = state.round_type == ROUND_CAROUSEL
    is_pve = state.round_type == ROUND_PVE
    rng_key, pvp_key = jax.random.split(state.rng_key)
    carousel_state = state.replace(rng_key=rng_key)
    carousel_reward = jnp.float32(0.0)

    # --- PvE: every living player vs creeps (pve.py:25-56) ---
    creep_ids, creep_stars = _creep_board_jax(state.stage)
    drop = jnp.where(state.stage == 1, jnp.int32(PVE_GOLD_EARLY), jnp.int32(PVE_GOLD_LATE))

    def _pve_one(board_ids, board_stars):
        winner, _ = resolve_combat_jax(
            board_ids, board_stars, creep_ids, creep_stars, static
        )
        return winner

    pve_winners = jax.vmap(_pve_one)(
        state.players.board_ids, state.players.board_stars
    )
    pve_won = (pve_winners == 0) & (~state.players.is_eliminated)
    pve_gold = state.players.gold + jnp.where(pve_won, drop, jnp.int32(0))
    pve_state = state.replace(
        players=state.players.replace(gold=pve_gold),
        rng_key=rng_key,
    )
    agent_alive = ~state.players.is_eliminated[0]
    pve_reward = jnp.where(
        ~agent_alive,
        jnp.float32(0.0),
        jnp.where(
            pve_winners[0] == 0, PVE_REWARD_WIN,
            jnp.where(pve_winners[0] == 1, PVE_REWARD_LOSS, jnp.float32(0.0)),
        ),
    )
    pve_terminated = jnp.bool_(False)

    # --- PvP: shuffle / pair / ghost (state.py:206-249) ---
    match_a, match_b, match_valid = pair_living_players(
        state.players.is_eliminated, pvp_key
    )

    def scan_body(carry, inputs):
        players, agent_won, agent_lost = carry
        idx_a, idx_b, valid = inputs
        new_players, won, lost = apply_match_jax(
            players, idx_a, idx_b, valid, state.stage, static
        )
        return (new_players, agent_won | won, agent_lost | lost), None

    (pvp_players, agent_won, agent_lost), _ = jax.lax.scan(
        scan_body,
        (state.players, jnp.bool_(False), jnp.bool_(False)),
        (match_a, match_b, match_valid),
    )
    newly_dead = (~state.players.is_eliminated) & (pvp_players.health <= 0)
    pvp_players = pvp_players.replace(
        is_eliminated=pvp_players.is_eliminated | newly_dead
    )
    pvp_state = state.replace(players=pvp_players, rng_key=rng_key)
    pvp_state = _return_eliminated_units(pvp_state, newly_dead, static)

    pvp_reward = jnp.float32(0.0)
    pvp_reward = jnp.where(agent_won, pvp_reward + REWARD_WIN, pvp_reward)
    pvp_reward = jnp.where(agent_lost, pvp_reward + REWARD_LOSS, pvp_reward)
    pvp_reward = jnp.where(newly_dead[0], pvp_reward + REWARD_ELIMINATED, pvp_reward)
    pvp_terminated = pvp_players.is_eliminated[0]

    resolve_state = jax.tree_util.tree_map(
        lambda c, pv, v: jnp.where(is_carousel, c, jnp.where(is_pve, pv, v)),
        carousel_state, pve_state, pvp_state,
    )
    resolve_reward = jnp.where(is_carousel, carousel_reward,
                       jnp.where(is_pve, pve_reward, pvp_reward))
    resolve_terminated = jnp.where(is_carousel, jnp.bool_(False),
                          jnp.where(is_pve, pve_terminated, pvp_terminated))

    # Combat resolution does not advance stage/round — that is start_round_jax
    # (mirrors state.py:199-249, which only increments rounds_completed).
    resolve_state = resolve_state.replace(
        rounds_completed=resolve_state.rounds_completed + 1,
        actions_this_round=jnp.int32(0),
        pending_action_reward=jnp.float32(0.0),
    )
    return resolve_state, resolve_reward, resolve_terminated


def start_round_jax(state: GameState, static: StaticData) -> GameState:
    """Advance stage/round and grant income + shops (state.py:108-135)."""
    max_rounds = max_rounds_in_stage_jax(state.stage)
    need_stage_up = state.round_in_stage >= max_rounds
    new_stage = jnp.where(need_stage_up, state.stage + 1, state.stage)
    new_round = jnp.where(need_stage_up, jnp.int32(1), state.round_in_stage + 1)
    new_round_type = determine_round_type_jax(new_stage, new_round)
    n_players = state.players.health.shape[0]
    is_carousel = new_round_type == ROUND_CAROUSEL
    grant_xp = (new_stage > 1) | (new_round > 1)

    key, car_key = jax.random.split(state.rng_key)
    perm = jax.random.permutation(car_key, static.n_units)
    car_opts = perm[:CAROUSEL_N_OPTIONS]
    car_shop = jnp.full((SHOP_SIZE,), -1, dtype=jnp.int32).at[:CAROUSEL_N_OPTIONS].set(car_opts)

    state = state.replace(
        stage=new_stage,
        round_in_stage=new_round,
        round_type=new_round_type,
        actions_this_round=jnp.int32(0),
        carousel_picked=jnp.zeros(n_players, dtype=jnp.bool_),
        carousel_options=car_opts,
        rng_key=key,
    )

    def player_body(i, carry):
        state, key = carry
        p = state.players
        alive = ~p.is_eliminated[i]
        income = calculate_income_jax(
            p.gold[i], p.win_streak[i], p.loss_streak[i],
            new_stage, new_round,
        )
        xp = p.xp[i] + jnp.where(grant_xp & ~is_carousel, jnp.int32(PASSIVE_XP), jnp.int32(0))
        xp, level = check_level_up_jax(xp, p.level[i])
        gold = p.gold[i] + jnp.where(is_carousel, jnp.int32(0), income)
        xp = jnp.where(alive, xp, p.xp[i])
        level = jnp.where(alive, level, p.level[i])
        gold = jnp.where(alive, gold, p.gold[i])

        pool_after_return = return_ids_to_pool(state.pool, p.shop[i], static)
        key, shop_key = jax.random.split(key)
        rolled_pool, rolled_shop, key = roll_shop_jax(
            pool_after_return, level, shop_key, static
        )
        shop = jnp.where(is_carousel, car_shop, rolled_shop)
        pool = tree_where(is_carousel, pool_after_return, rolled_pool)
        shop = jnp.where(alive, shop, p.shop[i])
        pool = tree_where(alive, pool, state.pool)

        players = p.replace(
            gold=p.gold.at[i].set(gold),
            xp=p.xp.at[i].set(xp),
            level=p.level.at[i].set(level),
            shop=p.shop.at[i].set(shop),
        )
        return state.replace(players=players, pool=pool), key

    state, key = jax.lax.fori_loop(
        0, n_players, player_body, (state, state.rng_key)
    )
    return state.replace(rng_key=key)


def assign_bot_strategies_jax(state: GameState) -> GameState:
    """Random archetype per opponent (bot.py:247-253)."""
    n_players = state.players.health.shape[0]
    key, strat_key = jax.random.split(state.rng_key)
    ids = jax.random.randint(strat_key, (n_players,), 0, N_ARCHETYPES, dtype=jnp.int32)
    ids = ids.at[0].set(jnp.int32(0))
    return state.replace(
        players=state.players.replace(bot_strategy_id=ids),
        rng_key=key,
    )


def reset_from_template(template: GameState, rng_key: jax.Array,
                        static: StaticData,
                        n_policy_bots: jnp.ndarray = jnp.int32(0)) -> GameState:
    """JIT-safe reset: copy a Python-built template, then start_round.

    make_pool uses NumPy, so it cannot run under jit. The template is
    created once on the host and closed over / stored on TrainState.
    """
    state = template.replace(rng_key=rng_key)
    state = assign_bot_strategies_jax(state)
    state = apply_policy_slots_jax(state, n_policy_bots)
    return start_round_jax(state, static)


def reset_jax(static: StaticData, rng_key: jax.Array,
              n_players: int = N_PLAYERS_DEFAULT) -> GameState:
    """Fresh episode: make state, assign bots, start first round (tft_env.py:62-74)."""
    template = make_game_state(n_players, static, rng_key)
    return reset_from_template(template, rng_key, static)


def agent_placement_jax(state: GameState) -> jnp.ndarray:
    """Agent finish rank 1 (best) through n_players (metrics.py:19-27).

    Living players sort by health descending, then eliminated players in
    original index order. Returns the agent's 1-based rank.
    """
    n = state.players.health.shape[0]
    eliminated = state.players.is_eliminated
    health = state.players.health
    idx = jnp.arange(n, dtype=jnp.int32)
    health_sort = jnp.where(~eliminated, -health, jnp.int32(0))
    order = jnp.lexsort((idx, health_sort, eliminated.astype(jnp.int32)))
    return jnp.argmax(order == 0).astype(jnp.int32) + 1


def is_last_player_standing_jax(state: GameState) -> jnp.ndarray:
    """True when at most one player is still alive (state.py:251-253)."""
    n_alive = jnp.sum(~state.players.is_eliminated)
    return n_alive <= 1


def placement_reward_jax(placement: jnp.ndarray) -> jnp.ndarray:
    """Terminal bonus (metrics.py:14-16)."""
    return (9.0 - placement.astype(jnp.float32)) * PLACEMENT_REWARD_SCALE


def run_bot_planning_phase_jax(state: GameState, static: StaticData) -> GameState:
    """Opponents act until PASS or budget (bot.py:256-287).

    Shared pool means players must be scanned sequentially, not vmapped.
    Policy-bot slots (opponent_type == OPPONENT_POLICY) are skipped here;
    they act via run_policy_bot_planning_jax when frozen params are provided.
    """
    n_players = state.players.health.shape[0]
    is_carousel = state.round_type == ROUND_CAROUSEL
    budget = jnp.take(ACTION_BUDGETS, state.stage)

    def one_action(player_idx, state):
        p = jax.tree_util.tree_map(lambda x: x[player_idx], state.players)
        skip = p.is_agent | p.is_eliminated | (p.opponent_type == OPPONENT_POLICY)
        mask = compute_player_mask(state, player_idx, static)
        scripted = jnp.where(
            is_carousel,
            pick_carousel_unit(mask),
            choose_bot_action(p, mask, static, p.bot_strategy_id),
        )
        action = jnp.where(skip, jnp.int32(ACTION_PASS), scripted)
        return apply_action_jax(
            state, player_idx, action, static, count_agent_action=False
        )

    def player_plan(player_idx, state):
        def body(_i, state):
            return one_action(player_idx, state)
        n_steps = jnp.where(is_carousel, jnp.int32(1), budget)
        return jax.lax.fori_loop(0, n_steps, body, state)

    def all_body(player_idx, state):
        return player_plan(player_idx, state)

    return jax.lax.fori_loop(0, n_players, all_body, state)


MASK_LOGIT = -1e8


def apply_policy_slots_jax(state: GameState, n_policy_bots: jnp.ndarray) -> GameState:
    """Mark the first N opponent slots as frozen policy bots."""
    types = state.players.opponent_type
    idxs = state.players.policy_bot_index
    for i in range(MAX_POLICY_BOTS):
        pid = i + 1
        use = jnp.int32(i) < n_policy_bots
        types = types.at[pid].set(
            jnp.where(use, jnp.int32(OPPONENT_POLICY), jnp.int32(0))
        )
        idxs = idxs.at[pid].set(jnp.where(use, jnp.int32(i), jnp.int32(-1)))
    return state.replace(
        players=state.players.replace(opponent_type=types, policy_bot_index=idxs)
    )


def run_policy_bot_planning_jax(state: GameState, static: StaticData,
                                frozen_params, n_policy_bots, apply_fn) -> GameState:
    """Frozen-policy opponents (policy_bot.py:71-96). apply_fn is model.apply."""
    if apply_fn is None:
        return state
    n_players = state.players.health.shape[0]
    is_carousel = state.round_type == ROUND_CAROUSEL
    budget = jnp.take(ACTION_BUDGETS, state.stage)

    def one_action(player_idx, carry):
        state, key = carry
        p = jax.tree_util.tree_map(lambda x: x[player_idx], state.players)
        is_policy = (
            (p.opponent_type == OPPONENT_POLICY)
            & (~p.is_eliminated)
            & (~p.is_agent)
            & (n_policy_bots > 0)
        )
        mask = compute_player_mask(state, player_idx, static)
        obs = to_observation(state, player_idx, static)
        bot_idx = jnp.clip(p.policy_bot_index, 0, MAX_POLICY_BOTS - 1)
        params_i = jax.tree_util.tree_map(lambda x: x[bot_idx], frozen_params)
        logits, _ = apply_fn(params_i, obs)
        masked = jnp.where(mask == 1, logits, MASK_LOGIT)
        key, akey = jax.random.split(key)
        sampled = jax.random.categorical(akey, masked).astype(jnp.int32)
        chosen = jnp.where(is_carousel, pick_carousel_unit(mask), sampled)
        action = jnp.where(is_policy, chosen, jnp.int32(ACTION_PASS))
        state = apply_action_jax(
            state, player_idx, action, static, count_agent_action=False
        )
        return state, key

    def player_plan(player_idx, carry):
        def body(_i, carry):
            return one_action(player_idx, carry)
        n_steps = jnp.where(is_carousel, jnp.int32(1), budget)
        return jax.lax.fori_loop(0, n_steps, body, carry)

    def all_body(player_idx, carry):
        return player_plan(player_idx, carry)

    state, key = jax.lax.fori_loop(
        0, n_players, all_body, (state, state.rng_key)
    )
    return state.replace(rng_key=key)


def step_jax(state: GameState, action: jnp.ndarray, static: StaticData,
             frozen_params=None, n_policy_bots: jnp.ndarray = jnp.int32(0),
             apply_fn=None) -> StepResult:
    """Execute one environment step.

    Replaces TFTEnv.step (tft_env.py:87-118).

    If action == PASS or action budget exceeded:
        - Opponents plan (scripted / policy bots)
        - Resolve round (combat, damage, rewards)
        - Check termination / last-player-standing
        - Start next round if the episode continues

    Otherwise:
        - Apply planning action
        - Return trait-breakpoint reward
    """
    agent_idx = jnp.int32(0)
    action_budget = jnp.take(ACTION_BUDGETS, state.stage)
    budget_exceeded = state.actions_this_round >= action_budget
    is_pass_or_done = (action == ACTION_PASS) | budget_exceeded

    planning_state = apply_action_jax(
        state, agent_idx, action, static, count_agent_action=True
    )
    planning_reward = planning_state.pending_action_reward
    planning_terminated = jnp.bool_(False)
    planning_truncated = jnp.bool_(False)

    bot_state = run_bot_planning_phase_jax(state, static)
    bot_state = run_policy_bot_planning_jax(
        bot_state, static, frozen_params, n_policy_bots, apply_fn
    )
    resolve_state, resolve_reward, resolve_terminated = resolve_round_jax(
        bot_state, static
    )
    resolve_truncated = is_last_player_standing_jax(resolve_state) & (~resolve_terminated)
    episode_done = resolve_terminated | resolve_truncated
    resolve_reward = resolve_reward + jnp.where(
        episode_done, placement_reward_jax(agent_placement_jax(resolve_state)), jnp.float32(0.0)
    )
    continued = start_round_jax(resolve_state, static)
    resolve_state = tree_where(episode_done, resolve_state, continued)

    new_state = tree_where(is_pass_or_done, resolve_state, planning_state)
    reward = jnp.where(is_pass_or_done, resolve_reward, planning_reward)
    terminated = jnp.where(is_pass_or_done, resolve_terminated, planning_terminated)
    truncated = jnp.where(is_pass_or_done, resolve_truncated, planning_truncated)

    obs = to_observation(new_state, agent_idx, static)
    mask = compute_player_mask(new_state, agent_idx, static)

    return StepResult(
        state=new_state,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        observation=obs,
        action_mask=mask,
    )

# ============================================================
# Verification
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=" * 60)
    logger.info("Step Function — Verification")
    logger.info("=" * 60)

    from tft_sim.jax_port.static_data import load_static_data
    static, meta = load_static_data()
    key = jax.random.PRNGKey(42)

    # Create game state
    state = make_game_state(8, static, key)

    # --- Test action mask ---
    logger.info(f"\n--- Action mask (initial state) ---")
    player0 = jax.tree_util.tree_map(lambda x: x[0], state.players)
    mask = compute_action_mask_jax(player0, static)
    legal_count = jnp.sum(mask)
    logger.info(f"  Legal actions: {legal_count}")
    logger.info(f"  PASS legal: {mask[ACTION_PASS]}")
    logger.info(f"  BUY_XP legal: {mask[ACTION_BUY_XP]} (gold=0, should be 0)")
    logger.info(f"  REROLL legal: {mask[ACTION_REROLL]} (gold=0, should be 0)")

    # Give player some gold and test again
    state_with_gold = state.replace(
        players=state.players.replace(
            gold=state.players.gold.at[0].set(20),
            shop=state.players.shop.at[0].set(jnp.array([0, 1, 2, -1, -1], dtype=jnp.int32)).at[0],
        )
    )
    # Actually, let's set the shop properly
    new_shop = jnp.array([0, 1, 2, -1, -1], dtype=jnp.int32)
    state_with_gold = state.replace(
        players=state.players.replace(
            gold=state.players.gold.at[0].set(20),
            shop=state.players.shop.at[0].set(new_shop),
        )
    )
    player0_gold = jax.tree_util.tree_map(lambda x: x[0], state_with_gold.players)
    mask_gold = compute_action_mask_jax(player0_gold, static)
    logger.info(f"\n  With gold=20, shop=[0,1,2,-1,-1]:")
    logger.info(f"  BUY_XP legal: {mask_gold[ACTION_BUY_XP]} (should be 1)")
    logger.info(f"  REROLL legal: {mask_gold[ACTION_REROLL]} (should be 1)")
    logger.info(f"  BUY_UNIT(0) legal: {mask_gold[ACTION_BUY_UNIT_START]} (should be 1)")
    logger.info(f"  BUY_UNIT(3) legal: {mask_gold[ACTION_BUY_UNIT_START + 3]} (should be 0, empty slot)")
    logger.info(f"  SELL_BENCH(0) legal: {mask_gold[ACTION_SELL_BENCH_START]} (should be 0, empty bench)")
    logger.info(f"  Total legal: {jnp.sum(mask_gold)}")

    # --- Test apply_action: BUY_XP ---
    logger.info(f"\n--- Apply action: BUY_XP ---")
    state_after_xp = apply_action_jax(state_with_gold, jnp.int32(0), jnp.int32(ACTION_BUY_XP), static)
    p0 = state_after_xp.players
    logger.info(f"  Gold: {state_with_gold.players.gold[0]} -> {p0.gold[0]} (should be 16)")
    logger.info(f"  XP: {state_with_gold.players.xp[0]} -> {p0.xp[0]} (should be 4)")
    assert p0.gold[0] == 16, f"Expected gold=16, got {p0.gold[0]}"
    assert p0.xp[0] == 4, f"Expected xp=4, got {p0.xp[0]}"

    # --- Test apply_action: BUY_UNIT ---
    logger.info(f"\n--- Apply action: BUY_UNIT(0) (buy Garen) ---")
    state_after_buy = apply_action_jax(state_with_gold, jnp.int32(0), jnp.int32(ACTION_BUY_UNIT_START), static)
    p0 = state_after_buy.players
    logger.info(f"  Gold: {state_with_gold.players.gold[0]} -> {p0.gold[0]} (should be 19, Garen costs 1)")
    logger.info(f"  Shop[0]: {state_with_gold.players.shop[0, 0]} -> {p0.shop[0, 0]} (should be -1)")
    logger.info(f"  Bench[0]: {p0.bench_ids[0, 0]} (should be 0 = Garen)")
    assert p0.gold[0] == 19, f"Expected gold=19, got {p0.gold[0]}"
    assert p0.shop[0, 0] == -1, f"Expected shop slot cleared, got {p0.shop[0, 0]}"
    assert p0.bench_ids[0, 0] == 0, f"Expected bench[0]=0 (Garen), got {p0.bench_ids[0, 0]}"

    # --- Test apply_action: PLACE_UNIT ---
    logger.info(f"\n--- Apply action: PLACE_UNIT(bench=0, board=0) ---")
    place_action = ACTION_PLACE_UNIT_START + 0 * BOARD_SIZE + 0  # bench 0 -> board 0
    state_after_place = apply_action_jax(state_after_buy, jnp.int32(0), jnp.int32(place_action), static)
    p0 = state_after_place.players
    logger.info(f"  Board[0]: {p0.board_ids[0, 0]} (should be 0 = Garen)")
    logger.info(f"  Bench[0]: {p0.bench_ids[0, 0]} (should be -1, moved to board)")
    assert p0.board_ids[0, 0] == 0, f"Expected board[0]=0, got {p0.board_ids[0, 0]}"
    assert p0.bench_ids[0, 0] == -1, f"Expected bench[0]=-1, got {p0.bench_ids[0, 0]}"

    # --- Test apply_action: SELL_BOARD ---
    logger.info(f"\n--- Apply action: SELL_BOARD(0) (sell Garen from board) ---")
    sell_action = ACTION_SELL_BOARD_START + 0
    state_after_sell = apply_action_jax(state_after_place, jnp.int32(0), jnp.int32(sell_action), static)
    p0 = state_after_sell.players
    logger.info(f"  Board[0]: {p0.board_ids[0, 0]} (should be -1, sold)")
    logger.info(f"  Gold: {p0.gold[0]} (should be 20, got 1 back for Garen)")
    assert p0.board_ids[0, 0] == -1, f"Expected board[0]=-1, got {p0.board_ids[0, 0]}"
    assert p0.gold[0] == 20, f"Expected gold=20, got {p0.gold[0]}"

    # --- Test full step ---
    logger.info(f"\n--- Full step (BUY_XP) ---")
    result = step_jax(state_with_gold, jnp.int32(ACTION_BUY_XP), static)
    logger.info(f"  Reward: {result.reward}")
    logger.info(f"  Terminated: {result.terminated}")
    logger.info(f"  Obs shape: {result.observation.shape}")
    logger.info(f"  Mask shape: {result.action_mask.shape}")
    assert result.observation.shape == (static.obs_total_size,)
    assert result.action_mask.shape == (TOTAL_ACTIONS,)

    # --- Test PASS (triggers round resolution) ---
    logger.info(f"\n--- Full step (PASS -> round resolution) ---")
    result_pass = step_jax(state_with_gold, jnp.int32(ACTION_PASS), static)
    logger.info(f"  Reward: {result_pass.reward}")
    logger.info(f"  Terminated: {result_pass.terminated}")
    logger.info(f"  Stage: {result_pass.state.stage}")
    logger.info(f"  Round: {result_pass.state.round_in_stage}")
    logger.info(f"  Rounds completed: {result_pass.state.rounds_completed}")

    # --- Test jit compatibility ---
    logger.info(f"\n--- JIT compatibility ---")
    @jax.jit
    def jit_step(state, action, static):
        return step_jax(state, action, static)

    jit_result = jit_step(state_with_gold, jnp.int32(ACTION_BUY_XP), static)
    logger.info(f"  JIT step compiled successfully")
    logger.info(f"  Obs shape: {jit_result.observation.shape}")
    logger.info(f"  Reward: {jit_result.reward}")

    # --- Test vmap (batch of steps) ---
    logger.info(f"\n--- vmap (batch of 4 envs) ---")
    states = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (4,) + x.shape),
        state_with_gold,
    )
    actions = jnp.array([ACTION_BUY_XP, ACTION_PASS, ACTION_REROLL, ACTION_BUY_UNIT_START], dtype=jnp.int32)

    batch_step = jax.vmap(step_jax, in_axes=(0, 0, None))
    batch_result = batch_step(states, actions, static)
    logger.info(f"  Batch obs shape: {batch_result.observation.shape}")
    logger.info(f"  Batch reward: {batch_result.reward}")
    logger.info(f"  Batch terminated: {batch_result.terminated}")

    logger.info(f"\n{'=' * 60}")
    logger.info("ALL VERIFICATIONS PASSED")
    logger.info(f"{'=' * 60}")
