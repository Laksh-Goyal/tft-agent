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
)
from tft_sim.jax_port.game_state import (
    GameState, PlayerState, PoolState,
    make_game_state, make_player, make_pool,
    to_observation, determine_round_type_jax,
    BOARD_SIZE, BENCH_SIZE, SHOP_SIZE,
    ROUND_CAROUSEL, ROUND_PVE, ROUND_PVP,
)
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


# ============================================================
# Apply Action (mirrors state.py:267-337)
# ============================================================
def apply_action_jax(state: GameState, player_idx: jnp.ndarray,
                     action: jnp.ndarray, static: StaticData,
                     count_agent_action: bool = False) -> GameState:
    """Apply one planning action for a player.

    Replaces GameState.apply_action (state.py:267-337).
    Returns a NEW GameState (no mutation).

    The action is applied using functional updates: we compute the new
    player state and return state.replace(players=new_players).
    """
    p = state.players

    # Extract current player fields
    health = p.health[player_idx]
    gold = p.gold[player_idx]
    level = p.level[player_idx]
    xp = p.xp[player_idx]
    board_ids = p.board_ids[player_idx]
    board_stars = p.board_stars[player_idx]
    bench_ids = p.bench_ids[player_idx]
    bench_stars = p.bench_stars[player_idx]
    shop = p.shop[player_idx]

    # --- Determine action type and apply ---
    # We use jnp.where to branch without Python if/elif.
    # Each action type computes its new state, and we select the right one.

    # Default: no change
    new_gold = gold
    new_xp = xp
    new_level = level
    new_board_ids = board_ids
    new_board_stars = board_stars
    new_bench_ids = bench_ids
    new_bench_stars = bench_stars
    new_shop = shop

    is_pass = action == ACTION_PASS
    is_buy_xp = action == ACTION_BUY_XP
    is_reroll = action == ACTION_REROLL
    is_buy_unit = (action >= ACTION_BUY_UNIT_START) & (action <= ACTION_BUY_UNIT_END)
    is_sell_bench = (action >= ACTION_SELL_BENCH_START) & (action <= ACTION_SELL_BENCH_END)
    is_sell_board = (action >= ACTION_SELL_BOARD_START) & (action <= ACTION_SELL_BOARD_END)
    is_place_unit = (action >= ACTION_PLACE_UNIT_START) & (action <= ACTION_PLACE_UNIT_END)

    # --- BUY_XP: gold -= 4, xp += 4, check level up ---
    # (state.py:283-286)
    buy_xp_gold = gold - 4
    buy_xp_xp = xp + 4
    # Level up loop (simplified — check once, not in a while loop)
    # In JAX, we use lax.while_loop or just check iteratively
    def check_level_up(xp_val, level_val):
        def body(val):
            xp, lvl = val
            new_lvl = lvl + 1
            return xp, new_lvl
        cond = lambda val: (val[1] < 9) & (val[0] >= jnp.take(XP_REQUIRED, val[1] + 1))
        return jax.lax.while_loop(cond, body, (xp_val, level_val))

    buy_xp_xp_final, buy_xp_level_final = check_level_up(buy_xp_xp, level)
    new_gold = jnp.where(is_buy_xp, buy_xp_gold, new_gold)
    new_xp = jnp.where(is_buy_xp, buy_xp_xp_final, new_xp)
    new_level = jnp.where(is_buy_xp, buy_xp_level_final, new_level)

    # --- REROLL: return shop to pool, roll new shop ---
    # (state.py:288-289, shop.py:83-88)
    # Simplified: just clear and re-roll shop (pool management omitted for now)
    reroll_gold = gold - 2
    # Generate new shop (simplified — just random unit IDs)
    reroll_key = jax.random.fold_in(state.rng_key, action)
    new_shop_reroll = jax.random.randint(
        reroll_key, (SHOP_SIZE,), 0, static.n_units, dtype=jnp.int32
    )
    new_shop = jnp.where(is_reroll, new_shop_reroll, new_shop)
    new_gold = jnp.where(is_reroll, reroll_gold, new_gold)

    # --- BUY_UNIT: buy from shop slot ---
    # (state.py:291-308)
    buy_slot = action - ACTION_BUY_UNIT_START
    buy_slot_safe = jnp.clip(buy_slot, 0, SHOP_SIZE - 1)
    bought_unit_id = jnp.take(shop, buy_slot_safe)
    bought_unit_cost = jnp.take(static.unit_costs, jnp.maximum(bought_unit_id, 0))

    # Find first empty bench slot
    bench_empty_mask = bench_ids < 0
    bench_empty_idx = jnp.argmax(bench_empty_mask)  # first True index

    # Place unit in bench
    new_bench_buy = bench_ids.at[bench_empty_idx].set(
        jnp.where(is_buy_unit & bench_empty_mask[bench_empty_idx], bought_unit_id, bench_ids[bench_empty_idx])
    )
    new_bench_stars_buy = bench_stars.at[bench_empty_idx].set(
        jnp.where(is_buy_unit & bench_empty_mask[bench_empty_idx], 1, bench_stars[bench_empty_idx])
    )

    # Clear shop slot and deduct gold
    new_shop_buy = shop.at[buy_slot_safe].set(
        jnp.where(is_buy_unit, -1, shop[buy_slot_safe])
    )
    new_gold_buy = gold - bought_unit_cost

    new_bench_ids = jnp.where(is_buy_unit, new_bench_buy, new_bench_ids)
    new_bench_stars = jnp.where(is_buy_unit, new_bench_stars_buy, new_bench_stars)
    new_shop = jnp.where(is_buy_unit, new_shop_buy, new_shop)
    new_gold = jnp.where(is_buy_unit, new_gold_buy, new_gold)

    # --- SELL_BENCH: sell bench unit ---
    # (state.py:310-315)
    sell_bench_slot = action - ACTION_SELL_BENCH_START
    sell_bench_slot_safe = jnp.clip(sell_bench_slot, 0, BENCH_SIZE - 1)
    sold_bench_unit_id = jnp.take(bench_ids, sell_bench_slot_safe)
    sold_bench_cost = jnp.take(static.unit_costs, jnp.maximum(sold_bench_unit_id, 0))

    new_bench_sell = bench_ids.at[sell_bench_slot_safe].set(
        jnp.where(is_sell_bench, -1, bench_ids[sell_bench_slot_safe])
    )
    new_gold_sell = gold + sold_bench_cost

    new_bench_ids = jnp.where(is_sell_bench, new_bench_sell, new_bench_ids)
    new_gold = jnp.where(is_sell_bench, new_gold_sell, new_gold)

    # --- SELL_BOARD: sell board unit ---
    # (state.py:317-322)
    sell_board_slot = action - ACTION_SELL_BOARD_START
    sell_board_slot_safe = jnp.clip(sell_board_slot, 0, BOARD_SIZE - 1)
    sold_board_unit_id = jnp.take(board_ids, sell_board_slot_safe)
    sold_board_cost = jnp.take(static.unit_costs, jnp.maximum(sold_board_unit_id, 0))

    new_board_sell = board_ids.at[sell_board_slot_safe].set(
        jnp.where(is_sell_board, -1, board_ids[sell_board_slot_safe])
    )
    new_gold_sell_board = gold + sold_board_cost

    new_board_ids = jnp.where(is_sell_board, new_board_sell, new_board_ids)
    new_gold = jnp.where(is_sell_board, new_gold_sell_board, new_gold)

    # --- PLACE_UNIT: swap bench[b] with board[d] ---
    # (state.py:324-330)
    place_idx = action - ACTION_PLACE_UNIT_START
    place_b = place_idx // BOARD_SIZE
    place_d = place_idx % BOARD_SIZE
    place_b_safe = jnp.clip(place_b, 0, BENCH_SIZE - 1)
    place_d_safe = jnp.clip(place_d, 0, BOARD_SIZE - 1)

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

    # --- Write back to player ---
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

    new_actions = jnp.where(count_agent_action,
                            state.actions_this_round + 1,
                            state.actions_this_round)

    return state.replace(players=new_players, actions_this_round=new_actions)


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


def resolve_round_jax(state: GameState, static: StaticData) -> tuple:
    """Resolve the current round (combat + damage + rewards).

    Replaces GameState.resolve_round (state.py:199-249).

    For PvP: agent (player 0) fights opponent (player 1). Loser takes
    damage = BASE_DAMAGE + stage + surviving_units. Winner gets a small
    positive reward; loser gets a small negative reward. Elimination
    gives a large negative reward.

    For PvE: agent fights creeps. Win grants gold; loss deals no HP damage.

    For Carousel: no combat, no reward.

    Returns:
        (new_state, reward, terminated)
    """
    agent_idx = jnp.int32(0)
    opp_idx = jnp.int32(1)

    # --- Carousel: no combat ---
    is_carousel = state.round_type == ROUND_CAROUSEL
    carousel_reward = jnp.float32(0.0)

    # --- PvE: agent vs creeps ---
    is_pve = state.round_type == ROUND_PVE
    # Creep board scales with stage (from pve.py:10-18)
    creep_ids = jnp.where(state.stage <= 1,
                         jnp.array([0, 1, 2, -1, -1, -1, -1, -1, -1, -1], dtype=jnp.int32),
                         jnp.array([2, 3, 4, -1, -1, -1, -1, -1, -1, -1], dtype=jnp.int32))
    creep_stars = jnp.ones(BOARD_SIZE, dtype=jnp.int32)

    pve_winner, _ = resolve_combat_jax(
        state.players.board_ids[agent_idx], state.players.board_stars[agent_idx],
        creep_ids, creep_stars,
        static,
    )
    pve_won = pve_winner == 0
    pve_gold = jnp.where(state.stage == 1, PVE_GOLD_EARLY, PVE_GOLD_LATE)
    pve_reward = jnp.where(pve_won, PVE_REWARD_WIN, PVE_REWARD_LOSS)
    # Add gold to agent on win
    pve_gold_update = jnp.where(pve_won, pve_gold, jnp.int32(0))
    pve_players = state.players.replace(
        gold=state.players.gold.at[agent_idx].set(
            state.players.gold[agent_idx] + pve_gold_update
        ),
    )
    pve_state = state.replace(players=pve_players)
    pve_terminated = jnp.bool_(False)  # PvE never eliminates

    # --- PvP: agent vs opponent ---
    is_pvp = state.round_type == ROUND_PVP
    pvp_winner, surv_units = resolve_combat_jax(
        state.players.board_ids[agent_idx], state.players.board_stars[agent_idx],
        state.players.board_ids[opp_idx], state.players.board_stars[opp_idx],
        static,
    )
    damage = jnp.int32(BASE_DAMAGE + state.stage * STAGE_DAMAGE_SCALE + surv_units)

    agent_won = pvp_winner == 0
    agent_lost = pvp_winner == 1

    # Apply damage to loser
    agent_health = state.players.health[agent_idx]
    opp_health = state.players.health[opp_idx]
    new_agent_health = jnp.where(agent_lost, agent_health - damage, agent_health)
    new_opp_health = jnp.where(agent_won, opp_health - damage, opp_health)

    # Update streaks
    new_agent_win_streak = jnp.where(agent_won, state.players.win_streak[agent_idx] + 1, 0)
    new_agent_loss_streak = jnp.where(agent_lost, state.players.loss_streak[agent_idx] + 1, 0)
    new_opp_win_streak = jnp.where(agent_lost, state.players.win_streak[opp_idx] + 1, 0)
    new_opp_loss_streak = jnp.where(agent_won, state.players.loss_streak[opp_idx] + 1, 0)

    # Check elimination
    agent_eliminated = new_agent_health <= 0
    opp_eliminated = new_opp_health <= 0

    pvp_players = state.players.replace(
        health=state.players.health.at[agent_idx].set(new_agent_health).at[opp_idx].set(new_opp_health),
        win_streak=state.players.win_streak.at[agent_idx].set(new_agent_win_streak).at[opp_idx].set(new_opp_win_streak),
        loss_streak=state.players.loss_streak.at[agent_idx].set(new_agent_loss_streak).at[opp_idx].set(new_opp_loss_streak),
        is_eliminated=state.players.is_eliminated.at[agent_idx].set(
            state.players.is_eliminated[agent_idx] | agent_eliminated
        ).at[opp_idx].set(
            state.players.is_eliminated[opp_idx] | opp_eliminated
        ),
    )
    pvp_state = state.replace(players=pvp_players)

    # PvP reward: win bonus, loss penalty, elimination penalty
    pvp_reward = jnp.where(agent_won, REWARD_WIN, jnp.where(agent_lost, REWARD_LOSS, jnp.float32(0.0)))
    pvp_reward = jnp.where(agent_eliminated, pvp_reward + REWARD_ELIMINATED, pvp_reward)
    pvp_terminated = agent_eliminated

    # --- Select round type branch ---
    # Default to carousel (no-op)
    resolve_state = jax.tree_util.tree_map(
        lambda c, p, v: jnp.where(is_carousel, c, jnp.where(is_pve, p, v)),
        state, pve_state, pvp_state,
    )
    resolve_reward = jnp.where(is_carousel, carousel_reward,
                       jnp.where(is_pve, pve_reward, pvp_reward))
    resolve_terminated = jnp.where(is_carousel, jnp.bool_(False),
                          jnp.where(is_pve, pve_terminated, pvp_terminated))

    # --- Advance round/stage ---
    resolve_state = resolve_state.replace(
        rounds_completed=resolve_state.rounds_completed + 1,
        actions_this_round=jnp.int32(0),
        pending_action_reward=jnp.float32(0.0),
    )

    max_rounds = jnp.where(resolve_state.stage == 1, 4,
                  jnp.where(resolve_state.stage == 5, 5, 6))
    next_round = resolve_state.round_in_stage + 1
    need_stage_up = next_round > max_rounds
    new_stage = jnp.where(need_stage_up, resolve_state.stage + 1, resolve_state.stage)
    new_round = jnp.where(need_stage_up, 1, next_round)
    new_round_type = determine_round_type_jax(new_stage, new_round)

    resolve_state = resolve_state.replace(
        stage=new_stage,
        round_in_stage=new_round,
        round_type=new_round_type,
    )

    return resolve_state, resolve_reward, resolve_terminated


def step_jax(state: GameState, action: jnp.ndarray, static: StaticData) -> StepResult:
    """Execute one environment step.

    Replaces TFTEnv.step (tft_env.py:87-118).

    If action == PASS or action budget exceeded:
        - Resolve round (combat, damage, rewards)
        - Check termination
        - Start next round

    Otherwise:
        - Apply planning action
        - Return small reward (trait breakpoint)
    """
    agent_idx = jnp.int32(0)  # player 0 is always the agent
    action_budget = jnp.take(ACTION_BUDGETS, state.stage)
    budget_exceeded = state.actions_this_round >= action_budget
    is_pass_or_done = (action == ACTION_PASS) | budget_exceeded

    # --- Planning branch: apply action ---
    planning_state = apply_action_jax(state, agent_idx, action, static,
                                       count_agent_action=True)
    planning_reward = state.pending_action_reward  # trait breakpoint reward
    planning_terminated = jnp.bool_(False)
    planning_truncated = jnp.bool_(False)

    # --- Round resolution branch: resolve combat, advance round ---
    resolve_state, resolve_reward, resolve_terminated = resolve_round_jax(state, static)
    resolve_truncated = jnp.bool_(False)

    # Select branch using tree_map (jnp.where doesn't work on PyTrees)
    new_state = jax.tree_util.tree_map(
        lambda a, b: jnp.where(is_pass_or_done, b, a),
        planning_state, resolve_state,
    )

    reward = jnp.where(is_pass_or_done, resolve_reward, planning_reward)
    terminated = jnp.where(is_pass_or_done, resolve_terminated, planning_terminated)
    truncated = jnp.where(is_pass_or_done, resolve_truncated, planning_truncated)

    # Build observation and action mask for the new state
    obs = to_observation(new_state, agent_idx, static)
    agent_player = jax.tree_util.tree_map(lambda x: x[agent_idx], new_state.players)
    mask = compute_action_mask_jax(agent_player, static)

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
