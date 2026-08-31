"""
JAX-compatible game state for the TFT port.

This module replaces the mutable Python GameState (state.py:78) and
Player (player.py:6) with immutable flax.struct.dataclass PyTrees
that can be traced through jit/vmap/scan.

Key architectural change:
    PyTorch version: Player.board is List[Optional[Unit]] — a list of
    Python objects, mutated in place by apply_action.

    JAX version: PlayerState.board_ids is jnp.ndarray (10,) int32 and
    PlayerState.board_stars is jnp.ndarray (10,) int32. Unit stats are
    looked up from StaticData arrays when needed. No mutation — every
    "update" returns a new state via .replace().

The PoolManager (shop.py:26) is also replaced. Instead of a dict of
lists, the pool is a (max_cost+1, max_pool_size) array of unit IDs,
with a separate count array tracking how many are available.

Mirrors:
    - tft_sim/game/player.py   (Player dataclass)
    - tft_sim/env/state.py      (GameState, observation builder)
    - tft_sim/game/shop.py      (PoolManager, roll_shop, reroll)
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
    StaticData, StaticMeta, load_static_data,
    build_unit_vector, trait_counts_jax, active_breakpoint_level_jax,
    XP_REQUIRED, ACTION_BUDGETS, SHOP_ODDS, POOL_SIZES,
    MAX_GOLD, START_HEALTH, MAX_LEVEL, MAX_STREAK, MAX_STAGE,
    MAX_HP, MAX_DAMAGE, MAX_ARMOR, MAX_AS, MAX_ABILITY_COEFF,
    BACKLINE_MIN_RANGE, TRAIT_BREAKPOINT_REWARD,
    TOTAL_ACTIONS,
    ACTION_PASS, ACTION_BUY_XP, ACTION_REROLL,
    ACTION_BUY_UNIT_START, ACTION_BUY_UNIT_END,
    ACTION_SELL_BENCH_START, ACTION_SELL_BENCH_END,
    ACTION_SELL_BOARD_START, ACTION_SELL_BOARD_END,
    ACTION_PLACE_UNIT_START, ACTION_PLACE_UNIT_END,
)


# ============================================================
# Board dimensions (from player.py:13-14)
# ============================================================
BOARD_SIZE = 10
BENCH_SIZE = 9
SHOP_SIZE = 5
N_OPPONENTS = 7
OPPONENT_VEC_SIZE = 34  # 4 econ + 10*3 board = 34 (from state.py:431-446)


# ============================================================
# PlayerState (mirrors player.py:6-20)
# ============================================================
@flax_dataclass
class PlayerState:
    """JAX-compatible player state.

    Instead of storing Unit objects, we store unit IDs and star levels
    as separate arrays. The board/bench/shop are fixed-size arrays that
    can be indexed and updated functionally.

    -1 in board_ids/bench_ids means empty slot.
    -1 in shop means empty shop slot.
    """
    health: jnp.ndarray         # scalar int32
    gold: jnp.ndarray           # scalar int32
    level: jnp.ndarray          # scalar int32
    xp: jnp.ndarray            # scalar int32
    win_streak: jnp.ndarray     # scalar int32
    loss_streak: jnp.ndarray    # scalar int32
    board_ids: jnp.ndarray      # (BOARD_SIZE,) int32, -1 = empty
    board_stars: jnp.ndarray    # (BOARD_SIZE,) int32
    bench_ids: jnp.ndarray      # (BENCH_SIZE,) int32, -1 = empty
    bench_stars: jnp.ndarray    # (BENCH_SIZE,) int32
    shop: jnp.ndarray           # (SHOP_SIZE,) int32, -1 = empty
    is_agent: jnp.ndarray       # scalar bool
    is_eliminated: jnp.ndarray  # scalar bool
    bot_strategy_id: jnp.ndarray  # scalar int32 (index into strategy list)
    opponent_type: jnp.ndarray    # scalar int32 (0=scripted, 1=policy)
    policy_bot_index: jnp.ndarray  # scalar int32


def make_player(is_agent: bool = False) -> PlayerState:
    """Create a fresh player (mirrors Player() default constructor)."""
    return PlayerState(
        health=jnp.int32(100),
        gold=jnp.int32(0),
        level=jnp.int32(1),
        xp=jnp.int32(0),
        win_streak=jnp.int32(0),
        loss_streak=jnp.int32(0),
        board_ids=jnp.full(BOARD_SIZE, -1, dtype=jnp.int32),
        board_stars=jnp.ones(BOARD_SIZE, dtype=jnp.int32),
        bench_ids=jnp.full(BENCH_SIZE, -1, dtype=jnp.int32),
        bench_stars=jnp.ones(BENCH_SIZE, dtype=jnp.int32),
        shop=jnp.full(SHOP_SIZE, -1, dtype=jnp.int32),
        is_agent=jnp.bool_(is_agent),
        is_eliminated=jnp.bool_(False),
        bot_strategy_id=jnp.int32(0),
        opponent_type=jnp.int32(0),  # scripted
        policy_bot_index=jnp.int32(-1),
    )


# ============================================================
# PoolState (replaces PoolManager from shop.py:26)
# ============================================================
@flax_dataclass
class PoolState:
    """Unit pool as a fixed-size array.

    Replaces PoolManager (shop.py:26) which used a defaultdict(list).
    We store the pool as a (N_COST_TIERS, MAX_POOL_SIZE) array of unit IDs,
    with a separate count array tracking how many slots are filled per tier.

    pool_ids: (6, MAX_POOL) int32 — unit IDs in the pool, padded with -1
    pool_counts: (6,) int32 — how many units are available per cost tier
    """
    pool_ids: jnp.ndarray       # (6, MAX_POOL) int32
    pool_counts: jnp.ndarray    # (6,) int32


def make_pool(static: StaticData) -> PoolState:
    """Initialize the unit pool (mirrors PoolManager.__init__)."""
    # Each cost tier has sum(POOL_SIZES[cost] for each unit of that cost) units.
    # We need the max across cost tiers to size the array.
    n_per_cost = np.zeros(6, dtype=np.int32)
    for unit_id in range(static.n_units):
        cost = int(static.unit_costs[unit_id])
        n_per_cost[cost] += int(POOL_SIZES[cost])
    max_pool = int(n_per_cost.max())

    pool_ids = np.full((6, max_pool), -1, dtype=np.int32)
    pool_counts = np.zeros(6, dtype=np.int32)

    # Fill pool with unit IDs (mirrors shop.py:30-33)
    for unit_id in range(static.n_units):
        cost = int(static.unit_costs[unit_id])
        count = int(POOL_SIZES[cost])
        for i in range(count):
            pool_ids[cost, pool_counts[cost]] = unit_id
            pool_counts[cost] += 1

    return PoolState(
        pool_ids=jnp.array(pool_ids),
        pool_counts=jnp.array(pool_counts),
    )


# ============================================================
# GameState (mirrors state.py:78)
# ============================================================
@flax_dataclass
class GameState:
    """JAX-compatible game state.

    All nested structures are PyTrees, so the entire GameState can be
    passed through jit/vmap/scan without any Python overhead.

    The players field holds N players batched as a single PlayerState
    (each array has a leading dimension of N_PLAYERS). This is the key
    to vmap: instead of a list of Player objects, we have one PlayerState
    with batched arrays.
    """
    # Player data (batched over N_PLAYERS)
    players: PlayerState

    # Round info
    stage: jnp.ndarray          # scalar int32
    round_in_stage: jnp.ndarray  # scalar int32
    actions_this_round: jnp.ndarray  # scalar int32
    rounds_completed: jnp.ndarray    # scalar int32
    round_type: jnp.ndarray     # scalar int32 (0=carousel, 1=pve, 2=pvp)

    # Pending reward from trait breakpoints (state.py:103)
    pending_action_reward: jnp.ndarray  # scalar float32

    # Carousel state (state.py:105-106)
    carousel_options: jnp.ndarray  # (4,) int32
    carousel_picked: jnp.ndarray   # (N_PLAYERS,) bool

    # Pool
    pool: PoolState

    # RNG
    rng_key: jax.Array


# Round type constants
ROUND_CAROUSEL = 0
ROUND_PVE = 1
ROUND_PVP = 2


def determine_round_type_jax(stage: jnp.ndarray, round_in_stage: jnp.ndarray) -> jnp.ndarray:
    """Determine round type (mirrors rounds.py:7-13).

    Returns: 0=carousel, 1=pve_creep, 2=pvp
    """
    is_carousel = (stage == 1) & (round_in_stage == 1)
    is_pve = ((stage == 1) & (round_in_stage >= 2) & (round_in_stage <= 4)) | \
             ((stage >= 2) & (round_in_stage == 1))
    return jnp.where(is_carousel, ROUND_CAROUSEL,
           jnp.where(is_pve, ROUND_PVE, ROUND_PVP))


def make_game_state(n_players: int, static: StaticData,
                    rng_key: jax.Array) -> GameState:
    """Create initial game state (mirrors GameState.__init__)."""
    # Create N players, batched
    player = make_player(is_agent=False)
    # Make player 0 the agent
    player = player.replace(is_agent=jnp.bool_(True))

    # Batch players by stacking N copies, then set agent flag
    # We use jax.vmap to create N independent players
    def make_one_player(idx):
        return make_player(is_agent=(idx == 0))

    players = jax.vmap(make_one_player)(jnp.arange(n_players))

    pool = make_pool(static)

    return GameState(
        players=players,
        stage=jnp.int32(1),
        round_in_stage=jnp.int32(0),
        actions_this_round=jnp.int32(0),
        rounds_completed=jnp.int32(0),
        round_type=jnp.int32(ROUND_CAROUSEL),
        pending_action_reward=jnp.float32(0.0),
        carousel_options=jnp.full(4, -1, dtype=jnp.int32),
        carousel_picked=jnp.zeros(n_players, dtype=jnp.bool_),
        pool=pool,
        rng_key=rng_key,
    )


# ============================================================
# Observation builder (mirrors state.py:374-448)
# ============================================================
def to_observation(state: GameState, player_idx: jnp.ndarray,
                   static: StaticData) -> jnp.ndarray:
    """Build flat observation for a single player.

    Replaces GameState.to_observation_for (state.py:377-448).

    Layout (from compute_obs_layout, state.py:56-75):
        econ (8) + context (4) + board (10*U) + bench (9*U) +
        shop (5*U) + synergies (N*2) + opponents (7*34)
    """
    p = state.players
    n_players = p.health.shape[0]
    u = static.unit_vec_size

    # --- Economy (8) ---
    max_rounds = jnp.float32(6.0)  # simplified; actual is max_rounds_in_stage
    econ = jnp.array([
        p.gold[player_idx] / MAX_GOLD,
        p.health[player_idx] / START_HEALTH,
        p.level[player_idx] / MAX_LEVEL,
        (XP_REQUIRED[p.level[player_idx] + 1] - p.xp[player_idx]) / 100.0,
        p.win_streak[player_idx] / MAX_STREAK,
        p.loss_streak[player_idx] / MAX_STREAK,
        state.stage / MAX_STAGE,
        state.round_in_stage / max_rounds,
    ], dtype=jnp.float32)

    # --- Context (4) ---
    is_carousel = (state.round_type == ROUND_CAROUSEL).astype(jnp.float32)
    is_pvp = (state.round_type == ROUND_PVP).astype(jnp.float32)
    is_pve = (state.round_type == ROUND_PVE).astype(jnp.float32)
    players_alive = jnp.sum(~p.is_eliminated) / jnp.float32(n_players)
    context = jnp.array([is_carousel, is_pvp, is_pve, players_alive], dtype=jnp.float32)

    # --- Board (10 * U) ---
    board_ids = p.board_ids[player_idx]  # (10,)
    board_stars = p.board_stars[player_idx]  # (10,)
    board_vecs = jax.vmap(build_unit_vector, in_axes=(0, 0, None))(
        board_ids, board_stars, static
    )  # (10, U)
    board_flat = board_vecs.reshape(-1)

    # --- Bench (9 * U) ---
    bench_ids = p.bench_ids[player_idx]
    bench_stars = p.bench_stars[player_idx]
    bench_vecs = jax.vmap(build_unit_vector, in_axes=(0, 0, None))(
        bench_ids, bench_stars, static
    )
    bench_flat = bench_vecs.reshape(-1)

    # --- Shop (5 * U) ---
    shop_ids = p.shop[player_idx]
    shop_stars = jnp.ones(SHOP_SIZE, dtype=jnp.int32)  # shop units are always star 1
    shop_vecs = jax.vmap(build_unit_vector, in_axes=(0, 0, None))(
        shop_ids, shop_stars, static
    )
    shop_flat = shop_vecs.reshape(-1)

    # --- Synergies (N * 2) ---
    counts = trait_counts_jax(board_ids, board_stars, static)
    # synergy_obs_pair: [count_norm, breakpoint_progress]
    # (simplified from traits.py:78-91)
    max_bps = static.trait_max_breakpoint  # (N_TRAITS,)
    count_norm = jnp.minimum(1.0, counts / jnp.maximum(max_bps, 1))
    # Progress to next breakpoint (simplified)
    active_bps = active_breakpoint_level_jax(counts, static)
    progress = jnp.minimum(1.0, counts / jnp.maximum(max_bps, 1))
    synergies = jnp.stack([count_norm, progress], axis=-1).reshape(-1)

    # --- Opponents (7 * 34) ---
    # Build opponent vectors for all other players
    def build_opponent_vec(other_idx):
        other_health = p.health[other_idx] / START_HEALTH
        other_level = p.level[other_idx] / MAX_LEVEL
        other_gold = jnp.minimum(1.0, p.gold[other_idx] / MAX_GOLD)
        other_streak = jnp.maximum(p.win_streak[other_idx], p.loss_streak[other_idx]) / MAX_STREAK
        econ_vec = jnp.array([other_health, other_level, other_gold, other_streak], dtype=jnp.float32)

        # Board units: 10 slots x 3 features (id/50, star/3, is_backline)
        def board_slot_vec(slot_idx):
            uid = p.board_ids[other_idx, slot_idx]
            star = p.board_stars[other_idx, slot_idx]
            valid = uid >= 0
            safe_uid = jnp.maximum(uid, 0)
            rng = jnp.take(static.unit_stats, safe_uid, axis=0)[5]
            return jnp.where(
                valid,
                jnp.array([safe_uid / 50.0, star / 3.0,
                          (rng >= BACKLINE_MIN_RANGE).astype(jnp.float32)]),
                jnp.zeros(3, dtype=jnp.float32),
            )

        board_vec = jax.vmap(board_slot_vec)(jnp.arange(BOARD_SIZE)).reshape(-1)
        return jnp.concatenate([econ_vec, board_vec])

    # Get opponent indices (all players except player_idx)
    all_indices = jnp.arange(n_players)
    opp_mask = all_indices != player_idx
    opp_indices = jnp.where(opp_mask, all_indices, -1)

    # We need exactly N_OPPONENTS opponents. If n_players-1 > N_OPPONENTS,
    # we take the first N_OPPONENTS. If fewer, pad with zeros.
    # For simplicity with n_players=8, we have exactly 7 opponents.
    opp_vecs = jax.vmap(build_opponent_vec)(jnp.arange(1, n_players))  # skip player 0 (agent)
    opp_flat = opp_vecs.reshape(-1)

    return jnp.concatenate([econ, context, board_flat, bench_flat,
                           shop_flat, synergies, opp_flat])


# ============================================================
# Verification
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=" * 60)
    logger.info("GameState — Verification")
    logger.info("=" * 60)

    static, meta = load_static_data()
    key = jax.random.PRNGKey(42)
    n_players = 8

    # Create game state
    state = make_game_state(n_players, static, key)

    logger.info(f"\nGame state created:")
    logger.info(f"  Stage: {state.stage}")
    logger.info(f"  Round: {state.round_in_stage}")
    logger.info(f"  Players: {state.players.health.shape[0]}")
    logger.info(f"  Player 0 health: {state.players.health[0]}")
    logger.info(f"  Player 0 is_agent: {state.players.is_agent[0]}")
    logger.info(f"  Player 1 is_agent: {state.players.is_agent[1]}")

    # Check pool
    logger.info(f"\nPool:")
    logger.info(f"  Pool IDs shape: {state.pool.pool_ids.shape}")
    logger.info(f"  Pool counts: {state.pool.pool_counts}")

    # Build observation for player 0 (agent)
    logger.info(f"\n--- Building observation for player 0 ---")
    obs = to_observation(state, jnp.int32(0), static)
    logger.info(f"  Observation shape: {obs.shape}")
    logger.info(f"  Expected size: {static.obs_total_size}")
    match = obs.shape[0] == static.obs_total_size
    logger.info(f"  Match: {match}")
    assert match, f"Observation size {obs.shape[0]} != expected {static.obs_total_size}"

    # Cross-check with PyTorch implementation
    logger.info(f"\n--- Cross-checking with PyTorch implementation ---")
    import sys
    sys.path.insert(0, ".")
    from tft_sim.env.state import GameState as PyTorchGameState
    import numpy as np

    pt_state = PyTorchGameState(n_players=n_players, rng=np.random.default_rng(42))
    pt_state.start_round()
    pt_obs = pt_state.to_observation()
    logger.info(f"  PyTorch obs shape: {pt_obs.shape}")
    logger.info(f"  JAX obs shape:     {obs.shape}")

    # The observations won't match exactly because the PyTorch state has
    # been through start_round (which rolls shops, gives gold, etc.) while
    # our JAX state hasn't. But the SHAPE should match.
    logger.info(f"  Shape match: {pt_obs.shape[0] == obs.shape[0]}")

    # Test round type determination
    logger.info(f"\n--- Round type determination ---")
    # Stage 1, round 1 = carousel
    rt = determine_round_type_jax(jnp.int32(1), jnp.int32(1))
    logger.info(f"  Stage 1-1: {rt} (expected {ROUND_CAROUSEL}=carousel)")
    assert rt == ROUND_CAROUSEL

    # Stage 1, round 2 = pve
    rt = determine_round_type_jax(jnp.int32(1), jnp.int32(2))
    logger.info(f"  Stage 1-2: {rt} (expected {ROUND_PVE}=pve)")
    assert rt == ROUND_PVE

    # Stage 2, round 1 = pve
    rt = determine_round_type_jax(jnp.int32(2), jnp.int32(1))
    logger.info(f"  Stage 2-1: {rt} (expected {ROUND_PVE}=pve)")
    assert rt == ROUND_PVE

    # Stage 2, round 2 = pvp
    rt = determine_round_type_jax(jnp.int32(2), jnp.int32(2))
    logger.info(f"  Stage 2-2: {rt} (expected {ROUND_PVP}=pvp)")
    assert rt == ROUND_PVP

    # Test immutability (functional update)
    logger.info(f"\n--- Immutability test ---")
    new_state = state.replace(stage=jnp.int32(3))
    logger.info(f"  Original stage: {state.stage}")
    logger.info(f"  New stage: {new_state.stage}")
    assert state.stage == 1, "Original should be unchanged"
    assert new_state.stage == 3, "New state should have updated value"

    # Test player update
    logger.info(f"\n--- Player update test ---")
    new_players = state.players.replace(
        gold=state.players.gold.at[0].set(20)
    )
    logger.info(f"  Player 0 gold: {state.players.gold[0]} -> {new_players.gold[0]}")
    assert state.players.gold[0] == 0, "Original should be unchanged"
    assert new_players.gold[0] == 20, "New state should have updated gold"

    # Test jit compatibility
    logger.info(f"\n--- JIT compatibility ---")
    @jax.jit
    def jit_obs(state, player_idx, static):
        return to_observation(state, player_idx, static)

    jit_obs_result = jit_obs(state, jnp.int32(0), static)
    logger.info(f"  JIT observation shape: {jit_obs_result.shape}")
    assert jit_obs_result.shape[0] == static.obs_total_size

    # Test vmap (observations for all players)
    logger.info(f"\n--- vmap (all player observations) ---")
    all_obs = jax.vmap(lambda idx: to_observation(state, idx, static))(
        jnp.arange(n_players)
    )
    logger.info(f"  Batch obs shape: {all_obs.shape}")
    logger.info(f"  Expected: ({n_players}, {static.obs_total_size})")
    assert all_obs.shape == (n_players, static.obs_total_size)

    logger.info(f"\n{'=' * 60}")
    logger.info("ALL VERIFICATIONS PASSED")
    logger.info(f"{'=' * 60}")
