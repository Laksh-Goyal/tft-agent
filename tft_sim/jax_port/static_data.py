"""
Static data layer for the JAX TFT port.

This module replaces the Python object-oriented data layer (UnitDatabase,
Unit dataclass, trait dicts) with pre-computed JAX arrays that can be
used inside jit-compiled functions without any Python overhead.

The key insight: in PyTorch, your UnitDatabase loads JSON and creates Unit
objects on demand. In JAX, we pre-compute ALL unit stats as static arrays
at init time, then index into them during env.step(). No object creation,
no dict lookups, no Python branching inside the jit boundary.

Mirrors:
    - tft_sim/game/units.py    (UnitDatabase, Unit)
    - tft_sim/game/traits.py   (trait definitions, breakpoint logic)
    - tft_sim/game/actions.py   (action IDs, action space)
    - tft_sim/env/state.py      (normalization constants, observation layout)
    - tft_sim/game/shop.py      (pool sizes, shop odds)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import logging

logger = logging.getLogger(__name__)


# ============================================================
# Action IDs (from actions.py:10-25)
# ============================================================
# These are compile-time constants, not JAX arrays. They're used for
# Python-level branching outside jit, and for array indexing inside jit.

TOTAL_ACTIONS = 127

ACTION_PASS = 0
ACTION_BUY_XP = 1
ACTION_REROLL = 2
ACTION_BUY_UNIT_START = 3
ACTION_BUY_UNIT_END = 7
ACTION_SELL_BENCH_START = 8
ACTION_SELL_BENCH_END = 16
ACTION_SELL_BOARD_START = 17
ACTION_SELL_BOARD_END = 26
ACTION_TOGGLE_FRONTLINE_START = 27  # deprecated; always illegal
ACTION_TOGGLE_FRONTLINE_END = 36
ACTION_PLACE_UNIT_START = 37
ACTION_PLACE_UNIT_END = 126

# Action category boundaries for masking inside jit.
# We store these as JAX arrays so we can use jnp.where / jnp.logical_and
# inside compiled functions instead of Python if/elif chains.
ACTION_RANGES = jnp.array([
    [ACTION_BUY_UNIT_START, ACTION_BUY_UNIT_END],      # 0: buy_unit
    [ACTION_SELL_BENCH_START, ACTION_SELL_BENCH_END],  # 1: sell_bench
    [ACTION_SELL_BOARD_START, ACTION_SELL_BOARD_END],  # 2: sell_board
    [ACTION_PLACE_UNIT_START, ACTION_PLACE_UNIT_END],  # 3: place_unit
], dtype=jnp.int32)


# ============================================================
# Normalization constants (from state.py:30-39)
# ============================================================
MAX_GOLD = 50.0
START_HEALTH = 100.0
MAX_LEVEL = 9.0
MAX_STREAK = 5.0
MAX_STAGE = 7.0
MAX_HP = 3000.0
MAX_DAMAGE = 500.0
MAX_ARMOR = 200.0
MAX_AS = 3.0
MAX_ABILITY_COEFF = 5.0

TRAIT_BREAKPOINT_REWARD = 0.05

# Combat constants (from trait_effects.py:6-8)
BACKLINE_MIN_RANGE = 3
CRIT_DAMAGE = 0.75
MANA_REGEN_FACTOR = 2.0

# Placement reward (from metrics.py:11-16)
PLACEMENT_REWARD_SCALE = 0.5


# ============================================================
# XP and round structure (from state.py:43-53)
# ============================================================
# XP required to reach each level (1-indexed: xp_required(level) gives
# XP needed to go FROM `level` TO `level+1`).
# Stored as a JAX array for jit-compatible lookup.
XP_REQUIRED = jnp.array([0, 0, 2, 6, 10, 20, 36, 56, 80, 100], dtype=jnp.int32)


def max_rounds_in_stage(stage: int) -> int:
    """Round cap per stage (from state.py:47-53). Python-level, not jit."""
    if stage == 1:
        return 4
    if stage == 5:
        return 5
    return 6


# Action budget per stage (from state.py:157-166)
ACTION_BUDGETS = jnp.array([0, 5, 10, 15, 20, 25, 25, 25], dtype=jnp.int32)


# ============================================================
# Shop odds (from shop.py:14-24)
# ============================================================
# SHOP_ODDS[level][cost] = probability of rolling a unit of that cost.
# Levels 1-9, costs 1-5. Stored as a (10, 6) array for 1-indexed access.
SHOP_ODDS = jnp.array([
    [0, 0, 0, 0, 0, 0],        # level 0 (unused)
    [0, 1.00, 0.00, 0.00, 0.00, 0.00],  # level 1
    [0, 1.00, 0.00, 0.00, 0.00, 0.00],  # level 2
    [0, 0.75, 0.25, 0.00, 0.00, 0.00],  # level 3
    [0, 0.55, 0.30, 0.15, 0.00, 0.00],  # level 4
    [0, 0.45, 0.33, 0.20, 0.02, 0.00],  # level 5
    [0, 0.30, 0.40, 0.25, 0.05, 0.00],  # level 6
    [0, 0.19, 0.30, 0.35, 0.15, 0.01],  # level 7
    [0, 0.15, 0.20, 0.35, 0.24, 0.06],  # level 8
    [0, 0.10, 0.15, 0.30, 0.30, 0.15],  # level 9
], dtype=jnp.float32)

# Pool sizes per cost tier (from shop.py:6-12)
POOL_SIZES = jnp.array([0, 45, 30, 25, 18, 10], dtype=jnp.int32)


# ============================================================
# StaticData: all game data as JAX arrays
# ============================================================
class StaticData(NamedTuple):
    """All static game data as JAX arrays, loaded once at init.

    This replaces UnitDatabase + trait definitions + action constants.
    Every field is a JAX array or int so the entire struct can be passed
    into jit-compiled functions as a PyTree.

    Python-only metadata (unit names, trait names) is kept in a separate
    StaticMeta struct to avoid breaking jit tracing.

    Key arrays:
        unit_stats: (N_UNITS, N_STATS) — raw stats for each unit at star level 1
        unit_traits: (N_UNITS, N_TRAITS) — multi-hot trait encoding per unit
        unit_costs: (N_UNITS,) — cost of each unit
        trait_breakpoints: (N_TRAITS, MAX_BREAKPOINTS) — breakpoints per trait
        trait_effects: (N_TRAITS, MAX_BREAKPOINTS, N_EFFECT_TYPES) — effect values
        shop_odds: (10, 6) — shop roll probabilities by level
        pool_sizes: (6,) — pool count per cost tier
    """

    # --- Dimensions (Python ints, static) ---
    n_units: int
    n_traits: int
    n_ability_types: int
    unit_vec_size: int
    obs_total_size: int

    # --- Unit data (JAX arrays) ---
    unit_stats: jnp.ndarray        # (N_UNITS, 8) raw stats
    unit_costs: jnp.ndarray        # (N_UNITS,) int32
    unit_traits: jnp.ndarray       # (N_UNITS, N_TRAITS) multi-hot
    unit_ability_type: jnp.ndarray # (N_UNITS, N_ABILITY_TYPES) one-hot

    # --- Trait data (JAX arrays) ---
    trait_breakpoints: jnp.ndarray # (N_TRAITS, MAX_BPS) padded with 0
    trait_n_breakpoints: jnp.ndarray  # (N_TRAITS,) actual count
    trait_effect_matrix: jnp.ndarray  # (N_TRAITS, MAX_BPS, N_EFFECT_TYPES) effect values
    trait_max_breakpoint: jnp.ndarray  # (N_TRAITS,) max breakpoint

    # --- Shop data (JAX arrays) ---
    shop_odds: jnp.ndarray         # (10, 6)
    pool_sizes: jnp.ndarray         # (6,)


class StaticMeta(NamedTuple):
    """Python-only metadata that cannot be traced through jit.

    Kept separate from StaticData so StaticData is a pure PyTree of arrays.
    """
    unit_names: list               # unit name strings, indexed by unit_id
    trait_names: list              # sorted trait name strings
    ability_types: list            # ability type name strings


# Effect type encoding for traits
# Each trait breakpoint has an effect type (what stat it modifies) and a value.
EFFECT_TYPES = {
    "ad_multiplier": 0,
    "ability_damage_multiplier": 1,
    "hp_multiplier": 2,
    "armor_multiplier": 3,
    "mr_multiplier": 4,
    "as_multiplier": 5,
    "enemy_armor_reduction": 6,
    "hp_regen_per_sec": 7,
    "crit_bonus": 8,
    "ally_shield": 9,
    "mana_per_sec": 10,
    "execute_threshold": 11,
    "execute_bonus_damage": 12,
}
N_EFFECT_TYPES = len(EFFECT_TYPES)


def load_static_data(roster_path: str = "tft_sim/data/unit_roster.json") -> StaticData:
    """Load all game data from the roster JSON and convert to JAX arrays.

    This is called ONCE at startup (outside jit). The returned StaticData
    is then passed into jit-compiled functions as a constant.

    Mirrors UnitDatabase.__init__ (units.py:47-51) but produces arrays
    instead of Python objects.
    """
    with open(roster_path, "r") as f:
        data = json.load(f)

    units = data["units"]
    traits = data["traits"]

    # --- Units ---
    n_units = len(units)
    # Match the PyTorch order exactly (state.py:96)
    ability_types = ["damage", "heal", "shield", "cc"]
    n_ability_types = len(ability_types)
    all_trait_names = sorted(set(t["name"] for t in traits))
    n_traits = len(all_trait_names)

    # Raw stats: hp, armor, magic_resist, attack_damage, attack_speed,
    # range, ability_damage, mana_cost, (star_level=1 always), cost
    # We store 10 raw stat columns.
    stat_keys = ["hp", "armor", "magic_resist", "attack_damage",
                 "attack_speed", "range", "ability_damage", "mana_cost"]
    unit_stats = np.zeros((n_units, len(stat_keys)), dtype=np.float32)
    unit_costs = np.zeros(n_units, dtype=np.int32)
    unit_traits = np.zeros((n_units, n_traits), dtype=np.float32)
    unit_ability = np.zeros((n_units, n_ability_types), dtype=np.float32)
    unit_names = []

    for i, u in enumerate(sorted(units, key=lambda x: x["id"])):
        assert u["id"] == i, f"Unit ID mismatch: expected {i}, got {u['id']}"
        for j, key in enumerate(stat_keys):
            unit_stats[i, j] = u[key]
        unit_costs[i] = u["cost"]
        unit_names.append(u["name"])
        # Multi-hot traits
        for t in u["traits"]:
            if t in all_trait_names:
                unit_traits[i, all_trait_names.index(t)] = 1.0
        # One-hot ability type
        if u["ability_type"] in ability_types:
            unit_ability[i, ability_types.index(u["ability_type"])] = 1.0

    # --- Traits ---
    max_bps = max(len(t["breakpoints"]) for t in traits)
    trait_breakpoints = np.zeros((n_traits, max_bps), dtype=np.int32)
    trait_n_bps = np.zeros(n_traits, dtype=np.int32)
    # Dense effect matrix: (n_traits, max_bps, N_EFFECT_TYPES).
    # Each entry holds the effect value for that (trait, breakpoint, effect_type),
    # or 0 if that effect is not present at that breakpoint. This supports
    # breakpoints with multiple effects (e.g. Assassin gives both ad_multiplier
    # and crit_bonus at the same breakpoint).
    trait_effect_mat = np.zeros((n_traits, max_bps, N_EFFECT_TYPES), dtype=np.float32)
    trait_max_bp = np.zeros(n_traits, dtype=np.int32)

    for t in traits:
        ti = all_trait_names.index(t["name"])
        bps = sorted(t["breakpoints"])
        trait_n_bps[ti] = len(bps)
        trait_max_bp[ti] = max(bps) if bps else 1
        for j, bp in enumerate(bps):
            trait_breakpoints[ti, j] = bp
            effects = t["effects"][str(bp)]
            for ek, ev in effects.items():
                trait_effect_mat[ti, j, EFFECT_TYPES[ek]] = ev

    # --- Observation layout ---
    # From state.py:97: unit_vec_size = 10 + n_ability_types + n_traits
    unit_vec_size = 10 + n_ability_types + n_traits
    # From compute_obs_layout (state.py:56-75):
    econ = 8
    context = 4
    board = 10 * unit_vec_size
    bench = 9 * unit_vec_size
    shop = 5 * unit_vec_size
    synergies = n_traits * 2
    opponents = 7 * 34
    obs_total = econ + context + board + bench + shop + synergies + opponents

    return (
        StaticData(
            n_units=n_units,
            n_traits=n_traits,
            n_ability_types=n_ability_types,
            unit_vec_size=unit_vec_size,
            obs_total_size=obs_total,
            unit_stats=jnp.array(unit_stats),
            unit_costs=jnp.array(unit_costs),
            unit_traits=jnp.array(unit_traits),
            unit_ability_type=jnp.array(unit_ability),
            trait_breakpoints=jnp.array(trait_breakpoints),
            trait_n_breakpoints=jnp.array(trait_n_bps),
            trait_effect_matrix=jnp.array(trait_effect_mat),
            trait_max_breakpoint=jnp.array(trait_max_bp),
            shop_odds=SHOP_ODDS,
            pool_sizes=POOL_SIZES,
        ),
        StaticMeta(
            unit_names=unit_names,
            trait_names=all_trait_names,
            ability_types=ability_types,
        ),
    )


# ============================================================
# Unit vector construction (from state.py:344-372)
# ============================================================
def build_unit_vector(unit_id: jnp.ndarray, star_level: jnp.ndarray,
                      static: StaticData) -> jnp.ndarray:
    """Build the observation vector for a single unit.

    Replaces GameState._build_unit_vector (state.py:344-372).
    Instead of creating a Unit object and reading attributes, we index
    into pre-computed arrays and apply star-level scaling.

    Args:
        unit_id: scalar int32, -1 means empty slot
        star_level: scalar int32 (1, 2, or 3)
        static: StaticData

    Returns:
        (unit_vec_size,) float32 array
    """
    # Star-level multipliers (from units.py:36-44)
    star_mult = jnp.where(star_level == 2, 1.8,
                jnp.where(star_level == 3, 3.24, 1.0))

    # Index into unit stats (unit_id=-1 gives all zeros via masking)
    valid = unit_id >= 0
    safe_id = jnp.maximum(unit_id, 0)

    stats = jnp.take(static.unit_stats, safe_id, axis=0)  # (8,)
    hp = stats[0] * star_mult
    armor = stats[1]
    magic_resist = stats[2]
    attack_damage = stats[3] * star_mult
    attack_speed = stats[4]
    rng = stats[5]
    ability_damage = stats[6]
    mana_cost = stats[7]

    # Base 10 features (from state.py:348-359)
    base = jnp.array([
        hp / MAX_HP,
        armor / MAX_ARMOR,
        magic_resist / MAX_ARMOR,
        attack_damage / MAX_DAMAGE,
        attack_speed / MAX_AS,
        rng / 5.0,
        ability_damage / MAX_ABILITY_COEFF,
        mana_cost / 150.0,
        star_level / 3.0,
        jnp.where(rng >= BACKLINE_MIN_RANGE, 1.0, 0.0),
    ])

    # Ability type one-hot (from state.py:361-363)
    ability_vec = jnp.take(static.unit_ability_type, safe_id, axis=0)

    # Trait multi-hot (from state.py:366-369)
    trait_vec = jnp.take(static.unit_traits, safe_id, axis=0)

    vec = jnp.concatenate([base, ability_vec, trait_vec])

    # Mask to zero if empty slot
    return jnp.where(valid, vec, jnp.zeros_like(vec))


# ============================================================
# Trait computation (from traits.py)
# ============================================================
def trait_counts_jax(board_unit_ids: jnp.ndarray,
                     board_star_levels: jnp.ndarray,
                     static: StaticData) -> jnp.ndarray:
    """Count trait occurrences on a board.

    Replaces traits.py:trait_counts (which iterates over Unit objects).

    Args:
        board_unit_ids: (10,) int32, -1 for empty slots
        board_star_levels: (10,) int32
        static: StaticData

    Returns:
        (n_traits,) int32 — count of units with each trait on the board
    """
    # Get trait multi-hot for each board slot: (10, n_traits)
    valid = board_unit_ids >= 0
    safe_ids = jnp.maximum(board_unit_ids, 0)
    slot_traits = jnp.take(static.unit_traits, safe_ids, axis=0)  # (10, n_traits)
    # Zero out empty slots
    slot_traits = slot_traits * valid[:, None]
    # Sum across board slots
    return jnp.sum(slot_traits, axis=0).astype(jnp.int32)


def active_breakpoint_level_jax(counts: jnp.ndarray, static: StaticData) -> jnp.ndarray:
    """Highest active breakpoint level for each trait.

    Replaces traits.py:active_breakpoint_level.
    For each trait, finds the highest breakpoint <= count.

    Args:
        counts: (n_traits,) int32
        static: StaticData

    Returns:
        (n_traits,) int32 — active breakpoint level (0 if none active)
    """
    # trait_breakpoints: (n_traits, max_bps)
    bps = static.trait_breakpoints  # (n_traits, max_bps)
    # For each trait, check which breakpoints are <= count
    active = (bps <= counts[:, None]) & (bps > 0)  # (n_traits, max_bps)
    # Take the max active breakpoint (0 if none)
    return jnp.max(jnp.where(active, bps, 0), axis=1)


# ============================================================
# Verification
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=" * 60)
    logger.info("Static Data Layer — Verification")
    logger.info("=" * 60)

    static, meta = load_static_data()

    logger.info(f"\nUnits: {static.n_units}")
    logger.info(f"Traits: {static.n_traits}")
    logger.info(f"Ability types: {static.n_ability_types}")
    logger.info(f"Unit vec size: {static.unit_vec_size}")
    logger.info(f"Obs total size: {static.obs_total_size}")

    logger.info(f"\nUnit stats shape: {static.unit_stats.shape}")
    logger.info(f"Unit costs: {static.unit_costs}")
    logger.info(f"Unit traits shape: {static.unit_traits.shape}")

    logger.info(f"\nTrait breakpoints:\n{static.trait_breakpoints}")
    logger.info(f"Trait n_breakpoints: {static.trait_n_breakpoints}")
    logger.info(f"Trait max breakpoints: {static.trait_max_breakpoint}")

    # Test build_unit_vector for Garen (id=0, star=1)
    logger.info(f"\n--- Unit vector for Garen (id=0, star=1) ---")
    garen_vec = build_unit_vector(
        jnp.int32(0), jnp.int32(1), static
    )
    logger.info(f"Shape: {garen_vec.shape}")
    logger.info(f"Vector: {garen_vec}")

    # Verify against the PyTorch implementation
    logger.info(f"\n--- Cross-checking with PyTorch implementation ---")
    import sys
    sys.path.insert(0, ".")
    # Temporarily use numpy for the PyTorch comparison
    from tft_sim.game.units import UnitDatabase
    from tft_sim.env.state import GameState
    import numpy as np

    db = UnitDatabase("tft_sim/data/unit_roster.json")
    gs = GameState(rng=np.random.default_rng(0))
    garen = db.create_unit(0, star_level=1)
    pytorch_vec = gs._build_unit_vector(garen)
    logger.info(f"PyTorch vector: {pytorch_vec}")
    logger.info(f"JAX vector:     {np.array(garen_vec)}")
    match = np.allclose(pytorch_vec, np.array(garen_vec), atol=1e-5)
    logger.info(f"Match: {match}")
    assert match, "Unit vectors don't match!"

    # Test empty slot
    logger.info(f"\n--- Empty slot (id=-1) ---")
    empty_vec = build_unit_vector(jnp.int32(-1), jnp.int32(1), static)
    logger.info(f"All zeros: {jnp.all(empty_vec == 0.0)}")

    # Test star level scaling (id=0, star=2 should have 1.8x hp and ad)
    logger.info(f"\n--- Star level scaling (Garen star=2) ---")
    garen_star2 = build_unit_vector(jnp.int32(0), jnp.int32(2), static)
    garen_star1 = build_unit_vector(jnp.int32(0), jnp.int32(1), static)
    hp_ratio = garen_star2[0] / garen_star1[0]
    ad_ratio = garen_star2[3] / garen_star1[3]
    logger.info(f"HP ratio (star2/star1): {hp_ratio:.4f} (should be 1.8)")
    logger.info(f"AD ratio (star2/star1): {ad_ratio:.4f} (should be 1.8)")
    assert abs(hp_ratio - 1.8) < 1e-4 and abs(ad_ratio - 1.8) < 1e-4, "Star scaling wrong!"

    # Test trait_counts
    logger.info(f"\n--- Trait counts ---")
    # Board with Garen (Warlord, Bruiser) + Lux (Arcane, Mage)
    board_ids = jnp.array([0, 1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=jnp.int32)
    board_stars = jnp.ones(10, dtype=jnp.int32)
    counts = trait_counts_jax(board_ids, board_stars, static)
    logger.info(f"Board: Garen + Lux")
    logger.info(f"Trait counts: {counts}")
    logger.info(f"Trait names: {meta.trait_names}")
    # Garen: Warlord, Bruiser; Lux: Arcane, Mage
    expected = [1, 0, 1, 0, 1, 0, 0, 0, 1, 0]  # sorted: Arcane, Assassin, Bruiser, Invoker, Mage, Sentinel, Slayer, Void, Warlord, Wildborn
    logger.info(f"Expected:      {expected}")
    match = np.array_equal(np.array(counts), expected)
    logger.info(f"Match: {match}")
    assert match, "Trait counts don't match!"

    # Test active_breakpoint_level
    logger.info(f"\n--- Active breakpoint levels ---")
    warlord_ids = [u["id"] for u in json.load(open("tft_sim/data/unit_roster.json"))["units"]
                   if "Warlord" in u["traits"]]
    logger.info(f"Warlord unit IDs: {warlord_ids}")
    # Put 3 Warlords on board
    board_3wl = jnp.full(10, -1, dtype=jnp.int32)
    for i in range(3):
        board_3wl = board_3wl.at[i].set(warlord_ids[i])
    counts_3wl = trait_counts_jax(board_3wl, jnp.ones(10, dtype=jnp.int32), static)
    bps = active_breakpoint_level_jax(counts_3wl, static)
    warlord_idx = meta.trait_names.index("Warlord")
    logger.info(f"3 Warlords on board:")
    logger.info(f"  Warlord count: {counts_3wl[warlord_idx]}")
    logger.info(f"  Active breakpoint: {bps[warlord_idx]} (should be 3)")
    assert bps[warlord_idx] == 3, "Breakpoint should be 3!"

    # Test jit compatibility — StaticData is now a pure PyTree of arrays
    logger.info(f"\n--- JIT compatibility ---")
    @jax.jit
    def jit_test(unit_id, star, static):
        vec = build_unit_vector(unit_id, star, static)
        counts = trait_counts_jax(
            jnp.where(jnp.arange(10) == 0, unit_id, -1),
            jnp.ones(10, dtype=jnp.int32),
            static,
        )
        bps = active_breakpoint_level_jax(counts, static)
        return vec, bps

    vec, bps = jit_test(jnp.int32(0), jnp.int32(1), static)
    logger.info(f"JIT compiled successfully")
    logger.info(f"  vec shape: {vec.shape}")
    logger.info(f"  bps shape: {bps.shape}")

    # Test vmap (batch of unit vectors)
    logger.info(f"\n--- vmap (batch unit vectors) ---")
    all_ids = jnp.arange(static.n_units, dtype=jnp.int32)
    all_stars = jnp.ones(static.n_units, dtype=jnp.int32)
    batch_vecs = jax.vmap(build_unit_vector, in_axes=(0, 0, None))(all_ids, all_stars, static)
    logger.info(f"Batch shape: {batch_vecs.shape} (should be ({static.n_units}, {static.unit_vec_size}))")
    assert batch_vecs.shape == (static.n_units, static.unit_vec_size)

    logger.info(f"\n{'=' * 60}")
    logger.info("ALL VERIFICATIONS PASSED")
    logger.info(f"{'=' * 60}")
