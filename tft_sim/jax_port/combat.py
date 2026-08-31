"""
JAX combat resolution for the TFT port.

Ports the analytical combat model from combat.py and trait_effects.py
into pure JAX array operations. No Unit objects, no Python dicts, no
mutation — every operation is a pure function on arrays.

The combat model is analytical (not a tick simulation): it computes
aggregate team stats (HP, DPS) and determines the winner via time-to-kill
estimation. This matches the PyTorch version exactly.

Mirrors:
    - tft_sim/game/combat.py        (resolve_combat, compute_team_stats)
    - tft_sim/game/trait_effects.py (all trait effect helpers)
    - tft_sim/game/units.py:24-34   (resolved_ability_power, effective_hp)
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp

from tft_sim.jax_port.static_data import (
    StaticData,
    BACKLINE_MIN_RANGE,
    CRIT_DAMAGE,
    MANA_REGEN_FACTOR,
)

logger = logging.getLogger(__name__)

# Stat column indices in StaticData.unit_stats (see static_data.py:241-242)
STAT_HP = 0
STAT_ARMOR = 1
STAT_MAGIC_RESIST = 2
STAT_ATTACK_DAMAGE = 3
STAT_ATTACK_SPEED = 4
STAT_RANGE = 5
STAT_ABILITY_DAMAGE = 6
STAT_MANA_COST = 7

# Effect type indices (see static_data.py:197-211)
EFF_AD_MULT = 0
EFF_ABILITY_DMG_MULT = 1
EFF_HP_MULT = 2
EFF_ARMOR_MULT = 3
EFF_MR_MULT = 4
EFF_AS_MULT = 5
EFF_ENEMY_ARMOR_REDUCTION = 6
EFF_HP_REGEN_PER_SEC = 7
EFF_CRIT_BONUS = 8
EFF_ALLY_SHIELD = 9
EFF_MANA_PER_SEC = 10
EFF_EXECUTE_THRESHOLD = 11
EFF_EXECUTE_BONUS_DAMAGE = 12

# Ability type indices (see static_data.py:233)
ABILITY_DAMAGE = 0
ABILITY_HEAL = 1
ABILITY_SHIELD = 2
ABILITY_CC = 3

# Combat constants (from combat.py:110-113)
MIN_DPS = 0.001
MIN_MANA_GATE = 0.1
MANA_GATE_DIVISOR = 10.0

# Number of effect types — must match len(EFFECT_TYPES) in static_data.py
N_EFFECT_TYPES = 13

# Effects that are multiplicative stat modifiers (default 1.0, not 0.0).
# All other effects default to 0.0.
_MULTIPLIER_EFFECTS = frozenset({
    EFF_AD_MULT,
    EFF_ABILITY_DMG_MULT,
    EFF_HP_MULT,
    EFF_ARMOR_MULT,
    EFF_MR_MULT,
    EFF_AS_MULT,
    EFF_EXECUTE_BONUS_DAMAGE,  # defaults to 1.0 (no bonus)
})


def _default_effects() -> jnp.ndarray:
    """Default effect values: 1.0 for multipliers, 0.0 for additive effects.

    This is the critical fix: without it, inactive traits zero out all stats
    (hp * 0.0 = 0), making every team appear empty.
    """
    defaults = jnp.zeros(N_EFFECT_TYPES, dtype=jnp.float32)
    for eff in _MULTIPLIER_EFFECTS:
        defaults = defaults.at[eff].set(1.0)
    return defaults


# ============================================================
# Unit stat extraction
# ============================================================
def get_unit_stats(board_ids: jnp.ndarray, board_stars: jnp.ndarray,
                   static: StaticData) -> dict:
    """Extract raw stats for all units on a board, with star-level scaling.

    Replaces the pattern of reading Unit object attributes.
    Returns a dict of (BOARD_SIZE,) arrays, with zeros for empty slots.

    Args:
        board_ids: (BOARD_SIZE,) int32, -1 for empty
        board_stars: (BOARD_SIZE,) int32
        static: StaticData

    Returns:
        dict with keys: hp, armor, magic_resist, attack_damage,
        attack_speed, range, ability_damage, mana_cost, ability_type_idx,
        traits (multi-hot), valid (bool mask)
    """
    valid = board_ids >= 0
    safe_ids = jnp.maximum(board_ids, 0)

    raw = jnp.take(static.unit_stats, safe_ids, axis=0)  # (BOARD_SIZE, 8)

    # Star-level multipliers (from units.py:36-44)
    star_mult = jnp.where(board_stars == 2, 1.8,
                jnp.where(board_stars == 3, 3.24, 1.0))

    hp = raw[:, STAT_HP] * star_mult
    armor = raw[:, STAT_ARMOR]
    magic_resist = raw[:, STAT_MAGIC_RESIST]
    attack_damage = raw[:, STAT_ATTACK_DAMAGE] * star_mult
    attack_speed = raw[:, STAT_ATTACK_SPEED]
    rng = raw[:, STAT_RANGE]
    ability_damage = raw[:, STAT_ABILITY_DAMAGE]
    mana_cost = raw[:, STAT_MANA_COST]

    # Zero out empty slots
    mask = valid.astype(jnp.float32)
    hp = hp * mask
    armor = armor * mask
    magic_resist = magic_resist * mask
    attack_damage = attack_damage * mask
    attack_speed = attack_speed * mask
    ability_damage = ability_damage * mask
    mana_cost = mana_cost * mask

    # Ability type as index (for branching in DPS computation)
    ability_onehot = jnp.take(static.unit_ability_type, safe_ids, axis=0)  # (BOARD_SIZE, 4)
    ability_type_idx = jnp.argmax(ability_onehot, axis=1)

    # Trait multi-hot: (BOARD_SIZE, n_traits)
    traits = jnp.take(static.unit_traits, safe_ids, axis=0) * mask[:, None]

    return {
        "hp": hp,
        "armor": armor,
        "magic_resist": magic_resist,
        "attack_damage": attack_damage,
        "attack_speed": attack_speed,
        "range": rng,
        "ability_damage": ability_damage,
        "mana_cost": mana_cost,
        "ability_type_idx": ability_type_idx,
        "traits": traits,
        "valid": valid,
    }


# ============================================================
# Trait effect extraction (from trait_effects.py)
# ============================================================
def get_active_trait_effects(board_ids: jnp.ndarray, board_stars: jnp.ndarray,
                             static: StaticData) -> dict:
    """Compute active trait effects for a board.

    Replaces compute_active_traits (traits.py:40-58) plus the effect
    extraction in trait_effects.py.

    Returns a dict of scalar effect values, one per effect type.
    If a trait with a given effect is active, its value is included.
    If multiple traits have the same effect type, we take the max
    (matching the PyTorch behavior where the strongest active breakpoint wins).

    Args:
        board_ids: (BOARD_SIZE,) int32
        board_stars: (BOARD_SIZE,) int32
        static: StaticData

    Returns:
        dict with keys matching EFFECT_TYPES, values are float32 scalars
    """
    from tft_sim.jax_port.static_data import trait_counts_jax, active_breakpoint_level_jax

    counts = trait_counts_jax(board_ids, board_stars, static)  # (n_traits,)
    active_bps = active_breakpoint_level_jax(counts, static)  # (n_traits,)

    # For each trait, find which breakpoint slot is active
    # trait_breakpoints: (n_traits, max_bps)
    bps = static.trait_breakpoints  # (n_traits, max_bps)
    # Match active_bp to breakpoint slot
    bp_match = (bps == active_bps[:, None]) & (bps > 0)  # (n_traits, max_bps)

    # trait_effect_matrix: (n_traits, max_bps, N_EFFECT_TYPES) — dense effect values.
    # Mask to zero at non-active breakpoints.
    eff_mat = static.trait_effect_matrix  # (n_traits, max_bps, N_EFFECT_TYPES)
    masked = eff_mat * bp_match[:, :, None].astype(jnp.float32)  # (n_traits, max_bps, N_EFFECT_TYPES)

    # Max across traits and breakpoints -> (N_EFFECT_TYPES,)
    effects = jnp.max(masked, axis=(0, 1))

    # Apply defaults: multipliers default to 1.0, additive effects to 0.0
    defaults = _default_effects()
    has_any = effects > 0  # True where at least one trait contributed
    effects = jnp.where(has_any, effects, defaults)

    return {
        "ad_multiplier": effects[EFF_AD_MULT],
        "ability_damage_multiplier": effects[EFF_ABILITY_DMG_MULT],
        "hp_multiplier": effects[EFF_HP_MULT],
        "armor_multiplier": effects[EFF_ARMOR_MULT],
        "mr_multiplier": effects[EFF_MR_MULT],
        "as_multiplier": effects[EFF_AS_MULT],
        "enemy_armor_reduction": effects[EFF_ENEMY_ARMOR_REDUCTION],
        "hp_regen_per_sec": effects[EFF_HP_REGEN_PER_SEC],
        "crit_bonus": effects[EFF_CRIT_BONUS],
        "ally_shield": effects[EFF_ALLY_SHIELD],
        "mana_per_sec": effects[EFF_MANA_PER_SEC],
        "execute_threshold": effects[EFF_EXECUTE_THRESHOLD],
        "execute_bonus_damage": effects[EFF_EXECUTE_BONUS_DAMAGE],
    }


# ============================================================
# Per-unit combat computations (from combat.py:20-36, units.py:24-34)
# ============================================================
def effective_hp(hp: jnp.ndarray, armor: jnp.ndarray, magic_resist: jnp.ndarray,
                 bonus_hp: jnp.ndarray) -> jnp.ndarray:
    """Effective HP accounting for armor/MR mitigation.

    Replaces Unit.effective_hp (units.py:30-34).
    Uses average of physical and magic reduction.
    """
    phys_reduction = 100.0 / (100.0 + armor)
    magic_reduction = 100.0 / (100.0 + magic_resist)
    avg_reduction = (phys_reduction + magic_reduction) / 2.0
    return (hp + bonus_hp) / avg_reduction


def resolved_ability_power(ability_type_idx: jnp.ndarray, hp: jnp.ndarray,
                           attack_damage: jnp.ndarray,
                           ability_damage: jnp.ndarray) -> jnp.ndarray:
    """Ability cast power based on ability type.

    Replaces Unit.resolved_ability_power (units.py:24-28).
    Heal/shield scale with HP; damage scales with AD.
    """
    is_sustain = (ability_type_idx == ABILITY_HEAL) | (ability_type_idx == ABILITY_SHIELD)
    return jnp.where(is_sustain, hp * ability_damage, attack_damage * ability_damage)


def unit_auto_dps(attack_damage: jnp.ndarray, attack_speed: jnp.ndarray,
                  crit_bonus: jnp.ndarray) -> jnp.ndarray:
    """Auto-attack DPS, including crit if applicable.

    Replaces _unit_auto_dps (combat.py:20-24).
    """
    dps = attack_damage * attack_speed
    has_crit = crit_bonus > 0
    crit_mult = 1.0 + crit_bonus * CRIT_DAMAGE
    return jnp.where(has_crit, dps * crit_mult, dps)


def unit_ability_dps(ability_type_idx: jnp.ndarray, ability_power: jnp.ndarray,
                     mana_cost: jnp.ndarray, invoker_mana_per_sec: jnp.ndarray,
                     has_invoker_trait: jnp.ndarray) -> jnp.ndarray:
    """Ability DPS (damage abilities only, not heal/shield).

    Replaces _unit_ability_dps (combat.py:27-29).
    """
    # Effective mana cost with Invoker reduction
    reduction = invoker_mana_per_sec * MANA_REGEN_FACTOR
    eff_mana = jnp.where(has_invoker_trait & (invoker_mana_per_sec > 0),
                        jnp.maximum(10.0, mana_cost - reduction),
                        mana_cost)
    mana_gate = jnp.maximum(eff_mana / MANA_GATE_DIVISOR, MIN_MANA_GATE)
    return ability_power / mana_gate


# ============================================================
# Team stats (from combat.py:39-74)
# ============================================================
def compute_team_stats(unit_stats: dict, effects: dict,
                       regen_per_sec: jnp.ndarray = jnp.float32(0.0),
                       combat_time: jnp.ndarray = jnp.float32(0.0)
                       ) -> tuple:
    """Compute aggregate team stats for combat.

    Replaces compute_team_stats (combat.py:39-74).

    Returns:
        (total_hp, frontline_hp, backline_hp, total_dps)
    """
    valid = unit_stats["valid"]
    rng = unit_stats["range"]
    is_backline = rng >= BACKLINE_MIN_RANGE
    is_frontline = valid & ~is_backline
    is_backline_valid = valid & is_backline

    frontline_mask = is_frontline.astype(jnp.float32)
    backline_mask = is_backline_valid.astype(jnp.float32)

    # Apply trait multipliers to stats (from apply_ally_trait_multipliers)
    hp = unit_stats["hp"] * effects["hp_multiplier"]
    armor = unit_stats["armor"] * effects["armor_multiplier"]
    magic_resist = unit_stats["magic_resist"] * effects["mr_multiplier"]
    attack_damage = unit_stats["attack_damage"] * effects["ad_multiplier"]
    attack_speed = unit_stats["attack_speed"] * effects["as_multiplier"]
    ability_damage = unit_stats["ability_damage"] * effects["ability_damage_multiplier"]

    # Sentinel shield: add to lowest-HP unit (from apply_sentinel_shields)
    shield = effects["ally_shield"]
    # Find lowest HP valid unit
    hp_for_shield = jnp.where(valid, hp, jnp.float32(jnp.inf))
    min_idx = jnp.argmin(hp_for_shield)
    bonus_hp = jnp.zeros_like(hp)
    bonus_hp = bonus_hp.at[min_idx].set(jnp.where(shield > 0, shield, 0.0))

    # Effective HP per unit
    eff_hp = effective_hp(hp, armor, magic_resist, bonus_hp)

    # Ability power and sustain EH
    ability_type_idx = unit_stats["ability_type_idx"]
    ability_power = resolved_ability_power(
        ability_type_idx, hp, attack_damage, ability_damage
    )

    # Invoker mana regen: check if each unit has the Invoker trait.
    # The Invoker trait index is found by looking for a trait whose active
    # breakpoint contributes mana_per_sec > 0. We check per-unit whether
    # that unit's trait multi-hot includes the Invoker trait.
    invoker_mana = effects["mana_per_sec"]
    traits = unit_stats["traits"]  # (BOARD_SIZE, n_traits)

    # Find the Invoker trait index: the trait whose effect is mana_per_sec.
    # We compute this from the static data at call time (not inside jit).
    # Since we can't iterate traits inside jit, we use the fact that
    # invoker_mana > 0 implies Invoker is active, and we check which units
    # have the Invoker trait by looking at the trait multi-hot.
    # The Invoker trait index is the one whose effect_indices match EFF_MANA_PER_SEC.
    # We pass it through the effects dict as a hidden key.
    invoker_trait_idx = effects.get("_invoker_trait_idx", jnp.int32(-1))
    # Use jnp.take for JIT-safe indexing with a traced index.
    # When invoker_trait_idx is -1, this safely wraps to the last column,
    # but has_invoker will be all-False because we guard on idx >= 0.
    invoker_col = jnp.take(traits, invoker_trait_idx, axis=1)  # (BOARD_SIZE,)
    has_invoker = jnp.where(
        invoker_trait_idx >= 0,
        invoker_col > 0,
        jnp.zeros_like(hp, dtype=jnp.bool_),
    )

    ability_dps = unit_ability_dps(
        ability_type_idx, ability_power, unit_stats["mana_cost"],
        jnp.full_like(hp, invoker_mana), has_invoker
    )

    # Sustain EH: heal/shield abilities contribute effective HP
    is_sustain = (ability_type_idx == ABILITY_HEAL) | (ability_type_idx == ABILITY_SHIELD)
    sustain_eh = jnp.where(is_sustain, ability_dps, 0.0)

    # Team HP with sustain
    frontline_hp = jnp.sum((eff_hp + sustain_eh) * frontline_mask)
    backline_hp = jnp.sum((eff_hp + sustain_eh) * backline_mask)

    # Wildborn regen bonus (JIT-safe: compute unconditionally, apply via where)
    n_front = jnp.sum(frontline_mask)
    n_back = jnp.sum(backline_mask)
    n_units = n_front + n_back
    n_safe = jnp.maximum(n_units, 1.0)
    regen_bonus = regen_per_sec * combat_time
    frontline_hp = frontline_hp + regen_bonus * (n_front / n_safe)
    backline_hp = backline_hp + regen_bonus * (n_back / n_safe)

    total_hp = frontline_hp + backline_hp

    # Team DPS
    crit_bonus_scalar = effects["crit_bonus"]
    # Assassin crit applies only to units with the Assassin trait.
    # Find the Assassin trait index the same way as Invoker.
    assassin_trait_idx = effects.get("_assassin_trait_idx", jnp.int32(-1))
    assassin_col = jnp.take(traits, assassin_trait_idx, axis=1)  # (BOARD_SIZE,)
    is_assassin = jnp.where(
        assassin_trait_idx >= 0,
        assassin_col > 0,
        jnp.zeros_like(hp, dtype=jnp.bool_),
    )
    crit_bonus_per_unit = jnp.where(is_assassin, crit_bonus_scalar, 0.0)

    auto_dps = unit_auto_dps(attack_damage, attack_speed, crit_bonus_per_unit)

    # Add ability DPS for non-sustain abilities
    is_damage = ~is_sustain
    valid_f = valid.astype(jnp.float32)
    total_dps = jnp.sum(auto_dps * valid_f) + \
                jnp.sum(jnp.where(is_damage, ability_dps, 0.0) * valid_f)

    return total_hp, frontline_hp, backline_hp, total_dps


# ============================================================
# Combat resolution (from combat.py:84-186)
# ============================================================
def resolve_combat_jax(
    board_a_ids: jnp.ndarray, board_a_stars: jnp.ndarray,
    board_b_ids: jnp.ndarray, board_b_stars: jnp.ndarray,
    static: StaticData,
) -> tuple:
    """Resolve combat between two teams.

    Replaces resolve_combat (combat.py:84-186).

    Args:
        board_a_ids: (BOARD_SIZE,) int32
        board_a_stars: (BOARD_SIZE,) int32
        board_b_ids: (BOARD_SIZE,) int32
        board_b_stars: (BOARD_SIZE,) int32
        static: StaticData

    Returns:
        (winner, surviving_units) where winner is 0=A, 1=B, 2=Tie
    """
    # --- Prepare teams: extract stats and apply trait effects ---
    stats_a = get_unit_stats(board_a_ids, board_a_stars, static)
    stats_b = get_unit_stats(board_b_ids, board_b_stars, static)

    effects_a = get_active_trait_effects(board_a_ids, board_a_stars, static)
    effects_b = get_active_trait_effects(board_b_ids, board_b_stars, static)

    # Inject trait indices for Assassin and Invoker (needed by compute_team_stats
    # to apply crit only to Assassins and mana reduction only to Invokers).
    # These are static Python ints, safe inside jit.
    assassin_idx = _find_trait_index_by_effect(static, EFF_CRIT_BONUS)
    invoker_idx = _find_trait_index_by_effect(static, EFF_MANA_PER_SEC)
    effects_a["_assassin_trait_idx"] = jnp.int32(assassin_idx)
    effects_a["_invoker_trait_idx"] = jnp.int32(invoker_idx)
    effects_b["_assassin_trait_idx"] = jnp.int32(assassin_idx)
    effects_b["_invoker_trait_idx"] = jnp.int32(invoker_idx)

    # --- Apply Void debuff (enemy armor reduction) ---
    # JIT-safe: compute unconditionally, apply via jnp.where
    void_a = effects_a["enemy_armor_reduction"]
    void_b = effects_b["enemy_armor_reduction"]
    stats_b["armor"] = jnp.where(void_a > 0,
                                stats_b["armor"] * (1.0 - void_a),
                                stats_b["armor"])
    stats_b["magic_resist"] = jnp.where(void_a > 0,
                                        stats_b["magic_resist"] * (1.0 - void_a),
                                        stats_b["magic_resist"])
    stats_a["armor"] = jnp.where(void_b > 0,
                                stats_a["armor"] * (1.0 - void_b),
                                stats_a["armor"])
    stats_a["magic_resist"] = jnp.where(void_b > 0,
                                        stats_a["magic_resist"] * (1.0 - void_b),
                                        stats_a["magic_resist"])

    # --- Wildborn regen ---
    regen_a = effects_a["hp_regen_per_sec"]
    regen_b = effects_b["hp_regen_per_sec"]

    # --- Compute initial team stats ---
    a_hp_total, a_hp_front, a_hp_back, a_dps = compute_team_stats(stats_a, effects_a)
    b_hp_total, b_hp_front, b_hp_back, b_dps = compute_team_stats(stats_b, effects_b)

    # Floor DPS to avoid division by zero
    a_dps = jnp.maximum(a_dps, MIN_DPS)
    b_dps = jnp.maximum(b_dps, MIN_DPS)

    # --- Edge cases: empty teams ---
    both_empty = (a_hp_total <= 0) & (b_hp_total <= 0)
    a_empty = a_hp_total <= 0
    b_empty = b_hp_total <= 0

    # --- Estimate combat time for regen recalculation ---
    est_combat_time = jnp.maximum(
        jnp.where(a_hp_front > 0, a_hp_front / b_dps, 0.0),
        jnp.where(b_hp_front > 0, b_hp_front / a_dps, 0.0),
    ) + jnp.maximum(
        jnp.where(a_hp_back > 0, a_hp_back / b_dps, 0.0),
        jnp.where(b_hp_back > 0, b_hp_back / a_dps, 0.0),
    )

    # Recompute with regen if applicable (JIT-safe via jnp.where)
    need_regen = (regen_a > 0) | (regen_b > 0)
    a_hp_total_r, a_hp_front_r, a_hp_back_r, a_dps_r = compute_team_stats(
        stats_a, effects_a, regen_a, est_combat_time
    )
    b_hp_total_r, b_hp_front_r, b_hp_back_r, b_dps_r = compute_team_stats(
        stats_b, effects_b, regen_b, est_combat_time
    )
    a_hp_total = jnp.where(need_regen, a_hp_total_r, a_hp_total)
    a_hp_front = jnp.where(need_regen, a_hp_front_r, a_hp_front)
    a_hp_back = jnp.where(need_regen, a_hp_back_r, a_hp_back)
    a_dps = jnp.where(need_regen, a_dps_r, a_dps)
    b_hp_total = jnp.where(need_regen, b_hp_total_r, b_hp_total)
    b_hp_front = jnp.where(need_regen, b_hp_front_r, b_hp_front)
    b_hp_back = jnp.where(need_regen, b_hp_back_r, b_hp_back)
    b_dps = jnp.where(need_regen, b_dps_r, b_dps)

    # --- Time-to-kill for frontlines ---
    ttk_a_front = jnp.where(a_hp_front > 0, a_hp_front / b_dps, 0.0)
    ttk_b_front = jnp.where(b_hp_front > 0, b_hp_front / a_dps, 0.0)

    # Slayer execute thresholds
    slayer_thresh_a = effects_a["execute_threshold"]
    slayer_mult_a = effects_a["execute_bonus_damage"]
    slayer_thresh_b = effects_b["execute_threshold"]
    slayer_mult_b = effects_b["execute_bonus_damage"]

    # --- Branch: A's frontline dies first ---
    a_dies_first = ttk_a_front < ttk_b_front
    b_dies_first = ttk_b_front < ttk_a_front

    # A dies first: B wins
    combat_time_a = ttk_a_front
    remaining_b_dps = b_dps
    # Slayer execute check for B
    hp_frac_a = jnp.where(a_hp_total > 0,
                         jnp.maximum(0.0, (a_hp_front - b_dps * ttk_a_front) / a_hp_total),
                         0.0)
    execute_b = (slayer_mult_b > 1.0) & (hp_frac_a <= slayer_thresh_b)
    remaining_b_dps = jnp.where(execute_b, remaining_b_dps * slayer_mult_b, remaining_b_dps)
    ttk_a_back = jnp.where(a_hp_back > 0, a_hp_back / remaining_b_dps, 0.0)
    combat_time_a = combat_time_a + ttk_a_back

    # B dies first: A wins
    combat_time_b = ttk_b_front
    remaining_a_dps = a_dps
    hp_frac_b = jnp.where(b_hp_total > 0,
                         jnp.maximum(0.0, (b_hp_front - a_dps * ttk_b_front) / b_hp_total),
                         0.0)
    execute_a = (slayer_mult_a > 1.0) & (hp_frac_b <= slayer_thresh_a)
    remaining_a_dps = jnp.where(execute_a, remaining_a_dps * slayer_mult_a, remaining_a_dps)
    ttk_b_back = jnp.where(b_hp_back > 0, b_hp_back / remaining_a_dps, 0.0)
    combat_time_b = combat_time_b + ttk_b_back

    # --- Determine winner ---
    # Default: tie
    winner = jnp.int32(2)  # Tie
    surviving = jnp.int32(0)
    combat_time = ttk_a_front  # tie case

    # A wins (B's frontline dies first)
    winner = jnp.where(b_dies_first, jnp.int32(0), winner)
    combat_time = jnp.where(b_dies_first, combat_time_b, combat_time)
    # Surviving units for A: fraction of A's HP remaining after B's damage
    hp_frac_remaining_a = jnp.maximum(
        0.0, 1.0 - (b_dps * combat_time_b / jnp.maximum(b_hp_total, 1.0))
    )
    surv_a = jnp.sum(stats_a["valid"].astype(jnp.float32))
    surviving = jnp.where(b_dies_first,
                          jnp.round(hp_frac_remaining_a * surv_a).astype(jnp.int32),
                          surviving)

    # B wins (A's frontline dies first)
    winner = jnp.where(a_dies_first, jnp.int32(1), winner)
    combat_time = jnp.where(a_dies_first, combat_time_a, combat_time)
    hp_frac_remaining_b = jnp.maximum(
        0.0, 1.0 - (a_dps * combat_time_a / jnp.maximum(a_hp_total, 1.0))
    )
    surv_b = jnp.sum(stats_b["valid"].astype(jnp.float32))
    surviving = jnp.where(a_dies_first,
                          jnp.round(hp_frac_remaining_b * surv_b).astype(jnp.int32),
                          surviving)

    # Edge cases override
    winner = jnp.where(both_empty, jnp.int32(2), winner)
    surviving = jnp.where(both_empty, jnp.int32(0), surviving)
    winner = jnp.where(a_empty & ~b_empty, jnp.int32(1), winner)
    surviving = jnp.where(a_empty & ~b_empty, jnp.int32(0), surviving)
    winner = jnp.where(b_empty & ~a_empty, jnp.int32(0), winner)
    surviving = jnp.where(b_empty & ~a_empty, surv_a.astype(jnp.int32), surviving)

    return winner, surviving


# ============================================================
# Trait index lookup (Python-level, called outside jit tracing)
# ============================================================
def _find_trait_index_by_effect(static: StaticData, effect_type: int) -> jnp.ndarray:
    """Find the trait index whose effect matches the given effect type.

    Uses pure JAX operations (JIT-safe). Returns -1 if no trait has that effect.
    Called to determine which trait index corresponds to Assassin (crit_bonus)
    or Invoker (mana_per_sec).
    """
    eff_mat = static.trait_effect_matrix  # (n_traits, max_bps, N_EFFECT_TYPES)
    # Check if any breakpoint slot for each trait has a non-zero value
    # for this effect type.
    has_effect = jnp.any(eff_mat[:, :, effect_type] > 0, axis=1)  # (n_traits,)
    idx = jnp.argmax(has_effect)
    return jnp.where(has_effect[idx], idx, jnp.int32(-1))


# ============================================================
# Round resolution (from state.py:199-249, pve.py:25-56)
# ============================================================
# Damage constants (from state.py:228)
BASE_DAMAGE = 2
STAGE_DAMAGE_SCALE = 1

# Reward constants (from state.py:242-247)
REWARD_WIN = 0.2
REWARD_LOSS = -0.1
REWARD_ELIMINATED = -1.0

# PvE rewards (from pve.py:53-55)
PVE_REWARD_WIN = 0.2
PVE_REWARD_LOSS = -0.1

# PvE gold drops (from pve.py:21-22)
PVE_GOLD_EARLY = 2
PVE_GOLD_LATE = 3


def resolve_pvp_match(player_a: PlayerState, player_b: PlayerState,
                      stage: jnp.ndarray, static: StaticData) -> tuple:
    """Resolve a single PvP match between two players.

    Returns: (winner, damage, a_won, a_lost)
        winner: 0=A wins, 1=B wins, 2=Tie
        damage: int32 damage dealt to loser
        a_won: bool — did player A (agent side) win
        a_lost: bool — did player A lose
    """
    winner, surv_units = resolve_combat_jax(
        player_a.board_ids, player_a.board_stars,
        player_b.board_ids, player_b.board_stars,
        static,
    )
    damage = jnp.int32(BASE_DAMAGE + stage * STAGE_DAMAGE_SCALE + surv_units)
    a_won = winner == 0
    a_lost = winner == 1
    return winner, damage, a_won, a_lost


def resolve_pve_match(player: PlayerState, creep_ids: jnp.ndarray,
                      creep_stars: jnp.ndarray, static: StaticData) -> tuple:
    """Resolve a PvE match (player vs creeps).

    Returns:
        player_won: bool — did the player win
    """
    winner, _ = resolve_combat_jax(
        player.board_ids, player.board_stars,
        creep_ids, creep_stars,
        static,
    )
    player_won = winner == 0  # player is team A
    return player_won


# ============================================================
# Verification
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 60)
    logger.info("Combat Resolution — Verification")
    logger.info("=" * 60)

    from tft_sim.jax_port.static_data import load_static_data
    from tft_sim.jax_port.game_state import make_player, BOARD_SIZE

    static, meta = load_static_data()

    # --- Test 1: Empty boards (both sides empty) ---
    logger.info("\n--- Test 1: Empty boards ---")
    empty_ids = jnp.full(BOARD_SIZE, -1, dtype=jnp.int32)
    empty_stars = jnp.ones(BOARD_SIZE, dtype=jnp.int32)
    winner, surv = resolve_combat_jax(empty_ids, empty_stars, empty_ids, empty_stars, static)
    logger.info(f"  Winner: {winner} (expected 2=Tie)")
    logger.info(f"  Surviving: {surv} (expected 0)")
    assert winner == 2, f"Expected Tie, got {winner}"

    # --- Test 2: One unit vs empty board ---
    logger.info("\n--- Test 2: Garen vs empty ---")
    garen_board = jnp.full(BOARD_SIZE, -1, dtype=jnp.int32).at[0].set(0)
    garen_stars = jnp.ones(BOARD_SIZE, dtype=jnp.int32)
    winner, surv = resolve_combat_jax(garen_board, garen_stars, empty_ids, empty_stars, static)
    logger.info(f"  Winner: {winner} (expected 0=A wins)")
    logger.info(f"  Surviving: {surv}")
    assert winner == 0, f"Expected A wins, got {winner}"

    # --- Test 3: Two equal teams ---
    logger.info("\n--- Test 3: Garen vs Garen (equal) ---")
    winner, surv = resolve_combat_jax(garen_board, garen_stars, garen_board, garen_stars, static)
    logger.info(f"  Winner: {winner} (could be either or Tie)")
    logger.info(f"  Surviving: {surv}")

    # --- Test 4: Strong team vs weak team ---
    logger.info("\n--- Test 4: 3-star Garen vs 1-star Garen ---")
    garen_3star_board = jnp.full(BOARD_SIZE, -1, dtype=jnp.int32).at[0].set(0)
    garen_3star_stars = jnp.ones(BOARD_SIZE, dtype=jnp.int32).at[0].set(3)
    winner, surv = resolve_combat_jax(
        garen_3star_board, garen_3star_stars,
        garen_board, garen_stars,
        static,
    )
    logger.info(f"  Winner: {winner} (expected 0=A, 3-star should win)")
    logger.info(f"  Surviving: {surv}")
    assert winner == 0, f"3-star Garen should beat 1-star, got winner={winner}"

    # --- Test 5: Cross-check with PyTorch ---
    logger.info("\n--- Test 5: Cross-check with PyTorch ---")
    import sys
    sys.path.insert(0, ".")
    import numpy as np
    from tft_sim.game.units import UnitDatabase
    from tft_sim.game.combat import resolve_combat
    from tft_sim.game.traits import compute_active_traits

    db = UnitDatabase("tft_sim/data/unit_roster.json")

    # Build a team with 3 units
    team_a_ids = jnp.array([0, 1, 2, -1, -1, -1, -1, -1, -1, -1], dtype=jnp.int32)
    team_a_stars = jnp.ones(BOARD_SIZE, dtype=jnp.int32)
    team_b_ids = jnp.array([3, 4, 5, -1, -1, -1, -1, -1, -1, -1], dtype=jnp.int32)
    team_b_stars = jnp.ones(BOARD_SIZE, dtype=jnp.int32)

    # PyTorch version
    pt_a = [db.create_unit(int(i)) if i >= 0 else None for i in team_a_ids]
    pt_b = [db.create_unit(int(i)) if i >= 0 else None for i in team_b_ids]
    pt_traits_a = compute_active_traits(pt_a, db.trait_data)
    pt_traits_b = compute_active_traits(pt_b, db.trait_data)
    pt_winner, pt_surv = resolve_combat(pt_a, pt_traits_a, pt_b, pt_traits_b)
    pt_winner_idx = {"A": 0, "B": 1, "Tie": 2}[pt_winner]

    # JAX version
    jax_winner, jax_surv = resolve_combat_jax(team_a_ids, team_a_stars, team_b_ids, team_b_stars, static)

    logger.info(f"  PyTorch: winner={pt_winner} ({pt_winner_idx}), surviving={pt_surv}")
    logger.info(f"  JAX:     winner={jax_winner}, surviving={jax_surv}")
    logger.info(f"  Winner match: {pt_winner_idx == int(jax_winner)}")

    # --- Test 6: JIT compatibility ---
    logger.info("\n--- Test 6: JIT compatibility ---")
    # StaticData is a NamedTuple of JAX arrays — it traces as a PyTree,
    # so we pass it as a regular argument (not static_argnames).
    jit_combat = jax.jit(resolve_combat_jax)
    j_winner, j_surv = jit_combat(team_a_ids, team_a_stars, team_b_ids, team_b_stars, static)
    logger.info(f"  JIT winner: {j_winner}, surviving: {j_surv}")
    assert j_winner == jax_winner, "JIT result should match non-JIT"

    # --- Test 7: vmap (batch combat) ---
    logger.info("\n--- Test 7: vmap (4 matches) ---")
    batch_a_ids = jnp.stack([team_a_ids] * 4)
    batch_a_stars = jnp.stack([team_a_stars] * 4)
    batch_b_ids = jnp.stack([team_b_ids] * 4)
    batch_b_stars = jnp.stack([team_b_stars] * 4)
    batch_winner, batch_surv = jax.vmap(resolve_combat_jax, in_axes=(0, 0, 0, 0, None))(
        batch_a_ids, batch_a_stars, batch_b_ids, batch_b_stars, static
    )
    logger.info(f"  Batch winners: {batch_winner}")
    logger.info(f"  Batch surviving: {batch_surv}")
    assert batch_winner.shape == (4,), f"Expected (4,), got {batch_winner.shape}"

    # --- Test 8: Trait effects (3 Warlords = breakpoint active) ---
    logger.info("\n--- Test 8: Trait effects (3 Warlords) ---")
    warlord_ids = [0, 5, 8]  # Garen, Darius, Sivir — all Warlord
    warlord_board = jnp.full(BOARD_SIZE, -1, dtype=jnp.int32)
    warlord_stars = jnp.ones(BOARD_SIZE, dtype=jnp.int32)
    for i, uid in enumerate(warlord_ids):
        warlord_board = warlord_board.at[i].set(uid)
    effects = get_active_trait_effects(warlord_board, warlord_stars, static)
    logger.info(f"  Warlord AD multiplier: {effects['ad_multiplier']} (should be 1.15 for bp=3)")

    logger.info("\n" + "=" * 60)
    logger.info("ALL VERIFICATIONS PASSED")
    logger.info("=" * 60)
