"""
JIT-compatible scripted opponent planning for the JAX TFT port.

Ports the five archetypes in agents/bot.py to integer-ID dispatch with
vectorized action selection. No Python strategy objects inside jit.

Mirrors:
    - tft_sim/agents/bot.py
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp

from tft_sim.jax_port.static_data import (
    StaticData,
    ACTION_PASS, ACTION_BUY_XP, ACTION_REROLL,
    ACTION_BUY_UNIT_START, ACTION_BUY_UNIT_END,
    ACTION_PLACE_UNIT_START, ACTION_PLACE_UNIT_END,
    STAR_MULT_2, STAR_MULT_3,
    UNIT_STRENGTH_COST_WEIGHT,
    ARCHETYPE_HYPER_BUYER, ARCHETYPE_INTEREST_SAVER, ARCHETYPE_BALANCED,
    ARCHETYPE_LEVEL_RUSHER, ARCHETYPE_ROLLER,
    INTEREST_SAVER_CAP, INTEREST_SAVER_REROLL_GOLD,
    BALANCED_BUY_GOLD, BALANCED_XP_GOLD, BALANCED_XP_LEVEL, BALANCED_REROLL_GOLD,
    LEVEL_RUSHER_XP_LEVEL, LEVEL_RUSHER_XP_GOLD, LEVEL_RUSHER_BUY_GOLD,
    LEVEL_RUSHER_REROLL_GOLD,
    ROLLER_REROLL_GOLD, ROLLER_XP_LEVEL, ROLLER_XP_GOLD,
    OPPONENT_POLICY,
)
from tft_sim.jax_port.game_state import PlayerState, BOARD_SIZE, BENCH_SIZE, SHOP_SIZE

logger = logging.getLogger(__name__)


def _star_mult(star: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(star == 2, STAR_MULT_2, jnp.where(star == 3, STAR_MULT_3, 1.0))


def unit_strength_jax(unit_id: jnp.ndarray, star: jnp.ndarray,
                      static: StaticData) -> jnp.ndarray:
    """Rough combat value (bot.py:28-30). Empty slots score -1."""
    valid = unit_id >= 0
    safe_id = jnp.maximum(unit_id, 0)
    cost = jnp.take(static.unit_costs, safe_id).astype(jnp.float32)
    stats = jnp.take(static.unit_stats, safe_id, axis=0)
    mult = _star_mult(star)
    hp = stats[0] * mult
    ad = stats[3] * mult
    strength = cost * UNIT_STRENGTH_COST_WEIGHT + hp + ad
    return jnp.where(valid, strength, jnp.float32(-1.0))


def _illegal():
    return jnp.int32(-1)


def pick_priciest_buy(player: PlayerState, mask: jnp.ndarray,
                      static: StaticData) -> jnp.ndarray:
    """Highest-cost legal shop buy; -1 if none (bot.py:61-64)."""
    costs = jnp.zeros(SHOP_SIZE, dtype=jnp.int32)
    legal = jnp.zeros(SHOP_SIZE, dtype=jnp.bool_)
    for i in range(SHOP_SIZE):
        action = ACTION_BUY_UNIT_START + i
        uid = player.shop[i]
        cost = jnp.take(static.unit_costs, jnp.maximum(uid, 0))
        ok = (mask[action] == 1) & (uid >= 0)
        costs = costs.at[i].set(jnp.where(ok, cost, jnp.int32(-1)))
        legal = legal.at[i].set(ok)
    has = jnp.any(legal)
    # max cost, then lowest index
    best_cost = jnp.max(costs)
    best_idx = jnp.argmax((costs == best_cost) & legal)
    return jnp.where(has, jnp.int32(ACTION_BUY_UNIT_START + best_idx), _illegal())


def pick_cheapest_buy(player: PlayerState, mask: jnp.ndarray,
                      static: StaticData) -> jnp.ndarray:
    """Lowest-cost legal shop buy; -1 if none (bot.py:55-58)."""
    costs = jnp.full(SHOP_SIZE, 999, dtype=jnp.int32)
    legal = jnp.zeros(SHOP_SIZE, dtype=jnp.bool_)
    for i in range(SHOP_SIZE):
        action = ACTION_BUY_UNIT_START + i
        uid = player.shop[i]
        cost = jnp.take(static.unit_costs, jnp.maximum(uid, 0))
        ok = (mask[action] == 1) & (uid >= 0)
        costs = costs.at[i].set(jnp.where(ok, cost, jnp.int32(999)))
        legal = legal.at[i].set(ok)
    has = jnp.any(legal)
    best_cost = jnp.min(costs)
    best_idx = jnp.argmax((costs == best_cost) & legal)
    return jnp.where(has, jnp.int32(ACTION_BUY_UNIT_START + best_idx), _illegal())


def first_place_on_empty_board(player: PlayerState, mask: jnp.ndarray) -> jnp.ndarray:
    """First legal bench->empty-board place (bot.py:67-75)."""
    found = jnp.bool_(False)
    chosen = jnp.int32(ACTION_PASS)
    for b in range(BENCH_SIZE):
        for d in range(BOARD_SIZE):
            action = ACTION_PLACE_UNIT_START + b * BOARD_SIZE + d
            ok = (mask[action] == 1) & (player.board_ids[d] < 0) & (player.bench_ids[b] >= 0)
            chosen = jnp.where(ok & ~found, jnp.int32(action), chosen)
            found = found | ok
    return jnp.where(found, chosen, _illegal())


def best_place_strongest_on_empty(player: PlayerState, mask: jnp.ndarray,
                                  static: StaticData) -> jnp.ndarray:
    """Place the strongest bench unit onto an empty board slot (bot.py:78-91)."""
    best_strength = jnp.float32(-1.0)
    chosen = jnp.int32(ACTION_PASS)
    found = jnp.bool_(False)
    for b in range(BENCH_SIZE):
        strength = unit_strength_jax(player.bench_ids[b], player.bench_stars[b], static)
        for d in range(BOARD_SIZE):
            action = ACTION_PLACE_UNIT_START + b * BOARD_SIZE + d
            ok = (mask[action] == 1) & (player.board_ids[d] < 0) & (player.bench_ids[b] >= 0)
            better = ok & (strength > best_strength)
            chosen = jnp.where(better, jnp.int32(action), chosen)
            best_strength = jnp.where(better, strength, best_strength)
            found = found | better
    return jnp.where(found, chosen, _illegal())


def first_legal_place(mask: jnp.ndarray) -> jnp.ndarray:
    """First legal place/swap action (bot.py:94-98)."""
    place_mask = mask[ACTION_PLACE_UNIT_START: ACTION_PLACE_UNIT_END + 1]
    has = jnp.any(place_mask == 1)
    idx = jnp.argmax(place_mask)
    return jnp.where(has, jnp.int32(ACTION_PLACE_UNIT_START + idx), _illegal())


def pick_carousel_unit(mask: jnp.ndarray) -> jnp.ndarray:
    """First legal shop buy, else PASS (bot.py:240-244)."""
    found = jnp.bool_(False)
    chosen = jnp.int32(ACTION_PASS)
    for i in range(SHOP_SIZE):
        action = ACTION_BUY_UNIT_START + i
        ok = mask[action] == 1
        chosen = jnp.where(ok & ~found, jnp.int32(action), chosen)
        found = found | ok
    return chosen


def _or_action(preferred: jnp.ndarray, fallback: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(preferred >= 0, preferred, fallback)


def choose_hyper_buyer(player, mask, static):
    buy = pick_priciest_buy(player, mask, static)
    bench_space = jnp.any(player.bench_ids < 0)
    first = jnp.where(bench_space, buy, _illegal())
    place = best_place_strongest_on_empty(player, mask, static)
    reroll = jnp.where(mask[ACTION_REROLL] == 1, jnp.int32(ACTION_REROLL), _illegal())
    return _or_action(first, _or_action(place, _or_action(reroll, jnp.int32(ACTION_PASS))))


def choose_interest_saver(player, mask, static):
    below_cap = player.gold <= INTEREST_SAVER_CAP
    place_any = first_legal_place(mask)
    below = jnp.where(place_any >= 0, place_any, jnp.int32(ACTION_PASS))
    board_n = jnp.sum(player.board_ids >= 0)
    cheap = pick_cheapest_buy(player, mask, static)
    fill = jnp.where(board_n < player.level, cheap, _illegal())
    reroll = jnp.where(
        (player.gold > INTEREST_SAVER_REROLL_GOLD) & (mask[ACTION_REROLL] == 1),
        jnp.int32(ACTION_REROLL), _illegal(),
    )
    place_empty = first_place_on_empty_board(player, mask)
    above = _or_action(fill, _or_action(reroll, _or_action(place_empty, jnp.int32(ACTION_PASS))))
    return jnp.where(below_cap, below, above)


def choose_balanced(player, mask, static):
    bench_space = jnp.any(player.bench_ids < 0)
    cheap = pick_cheapest_buy(player, mask, static)
    buy = jnp.where((player.gold > BALANCED_BUY_GOLD) & bench_space, cheap, _illegal())
    xp = jnp.where(
        (player.level < BALANCED_XP_LEVEL) & (player.gold > BALANCED_XP_GOLD) & (mask[ACTION_BUY_XP] == 1),
        jnp.int32(ACTION_BUY_XP), _illegal(),
    )
    reroll = jnp.where(
        (player.gold > BALANCED_REROLL_GOLD) & (mask[ACTION_REROLL] == 1),
        jnp.int32(ACTION_REROLL), _illegal(),
    )
    place = first_place_on_empty_board(player, mask)
    return _or_action(buy, _or_action(xp, _or_action(reroll, _or_action(place, jnp.int32(ACTION_PASS)))))


def choose_level_rusher(player, mask, static):
    xp = jnp.where(
        (player.level < LEVEL_RUSHER_XP_LEVEL) & (player.gold > LEVEL_RUSHER_XP_GOLD) & (mask[ACTION_BUY_XP] == 1),
        jnp.int32(ACTION_BUY_XP), _illegal(),
    )
    board_n = jnp.sum(player.board_ids >= 0)
    cheap = pick_cheapest_buy(player, mask, static)
    buy = jnp.where(
        (player.gold > LEVEL_RUSHER_BUY_GOLD) & (board_n < player.level), cheap, _illegal()
    )
    place = first_place_on_empty_board(player, mask)
    reroll = jnp.where(
        (player.gold > LEVEL_RUSHER_REROLL_GOLD) & (board_n < player.level) & (mask[ACTION_REROLL] == 1),
        jnp.int32(ACTION_REROLL), _illegal(),
    )
    return _or_action(xp, _or_action(buy, _or_action(place, _or_action(reroll, jnp.int32(ACTION_PASS)))))


def choose_roller(player, mask, static):
    reroll = jnp.where(
        (player.gold > ROLLER_REROLL_GOLD) & (mask[ACTION_REROLL] == 1),
        jnp.int32(ACTION_REROLL), _illegal(),
    )
    bench_space = jnp.any(player.bench_ids < 0)
    pricey = pick_priciest_buy(player, mask, static)
    buy = jnp.where(bench_space, pricey, _illegal())
    xp = jnp.where(
        (player.level < ROLLER_XP_LEVEL) & (player.gold > ROLLER_XP_GOLD) & (mask[ACTION_BUY_XP] == 1),
        jnp.int32(ACTION_BUY_XP), _illegal(),
    )
    place = best_place_strongest_on_empty(player, mask, static)
    return _or_action(reroll, _or_action(buy, _or_action(xp, _or_action(place, jnp.int32(ACTION_PASS)))))


def choose_bot_action(player: PlayerState, mask: jnp.ndarray,
                      static: StaticData, strategy_id: jnp.ndarray) -> jnp.ndarray:
    """Dispatch one archetype (bot.py choose_action methods)."""
    branches = (
        lambda: choose_hyper_buyer(player, mask, static),
        lambda: choose_interest_saver(player, mask, static),
        lambda: choose_balanced(player, mask, static),
        lambda: choose_level_rusher(player, mask, static),
        lambda: choose_roller(player, mask, static),
    )
    sid = jnp.clip(strategy_id, 0, 4)
    return jax.lax.switch(sid, branches)
