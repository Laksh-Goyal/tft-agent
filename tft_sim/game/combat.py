import copy
from typing import Tuple, List, Dict, Any

from tft_sim.game.units import Unit
from tft_sim.game.trait_effects import (
    is_backline,
    apply_ally_trait_multipliers,
    apply_void_debuff,
    get_void_reduction,
    get_wildborn_regen,
    get_assassin_crit_bonus,
    get_slayer_execute,
    get_invoker_mana_per_sec,
    apply_sentinel_shields,
    effective_mana_cost,
    CRIT_DAMAGE,
)


def _unit_auto_dps(unit: Unit, crit_bonus: float) -> float:
    dps = unit.attack_damage * unit.attack_speed
    if crit_bonus > 0:
        dps *= (1.0 + crit_bonus * CRIT_DAMAGE)
    return dps


def _unit_ability_dps(unit: Unit, invoker_mana_per_sec: float) -> float:
    mana_gate = max(effective_mana_cost(unit, invoker_mana_per_sec) / 10.0, 0.1)
    return unit.resolved_ability_power() / mana_gate


def _unit_sustain_eh(unit: Unit, invoker_mana_per_sec: float) -> float:
    """Heal/shield contribute effective HP via cast throughput."""
    if unit.ability_type not in ("heal", "shield"):
        return 0.0
    return _unit_ability_dps(unit, invoker_mana_per_sec)


def compute_team_stats(
    units: List[Unit],
    active_traits: Dict[str, Any],
    regen_per_sec: float = 0.0,
    combat_time: float = 0.0,
) -> Tuple[float, float, float, float]:
    frontline = [u for u in units if u is not None and not is_backline(u)]
    backline = [u for u in units if u is not None and is_backline(u)]

    team_crit = get_assassin_crit_bonus(active_traits)
    invoker_mana = get_invoker_mana_per_sec(active_traits)

    frontline_hp = sum(
        u.effective_hp() + _unit_sustain_eh(u, invoker_mana) for u in frontline
    )
    backline_hp = sum(
        u.effective_hp() + _unit_sustain_eh(u, invoker_mana) for u in backline
    )

    if regen_per_sec > 0 and combat_time > 0:
        regen_bonus = regen_per_sec * combat_time
        n = len(frontline) + len(backline)
        if n > 0:
            frontline_hp += regen_bonus * (len(frontline) / n)
            backline_hp += regen_bonus * (len(backline) / n)

    total_hp = frontline_hp + backline_hp

    total_dps = 0.0
    for u in frontline + backline:
        cb = team_crit if "Assassin" in u.traits else 0.0
        total_dps += _unit_auto_dps(u, cb)
        if u.ability_type not in ("heal", "shield"):
            total_dps += _unit_ability_dps(u, invoker_mana)

    return total_hp, frontline_hp, backline_hp, total_dps


def _prepare_team(units: List[Unit], active_traits: Dict[str, Any]) -> List[Unit]:
    units = copy.deepcopy(units)
    apply_sentinel_shields(units, active_traits)
    apply_ally_trait_multipliers(units, active_traits)
    return units


def resolve_combat(
    team_a_units: List[Unit],
    team_a_traits: Dict[str, Any],
    team_b_units: List[Unit],
    team_b_traits: Dict[str, Any],
) -> Tuple[str, int]:
    team_a_units = _prepare_team(team_a_units, team_a_traits)
    team_b_units = _prepare_team(team_b_units, team_b_traits)

    void_a = get_void_reduction(team_a_traits)
    void_b = get_void_reduction(team_b_traits)
    if void_a > 0:
        apply_void_debuff(team_b_units, void_a)
    if void_b > 0:
        apply_void_debuff(team_a_units, void_b)

    regen_a = get_wildborn_regen(team_a_traits)
    regen_b = get_wildborn_regen(team_b_traits)

    a_hp_total, a_hp_front, a_hp_back, a_dps = compute_team_stats(
        team_a_units, team_a_traits
    )
    b_hp_total, b_hp_front, b_hp_back, b_dps = compute_team_stats(
        team_b_units, team_b_traits
    )

    if a_dps <= 0:
        a_dps = 0.001
    if b_dps <= 0:
        b_dps = 0.001

    if a_hp_total <= 0 and b_hp_total <= 0:
        return "Tie", 0
    if a_hp_total <= 0:
        return "B", 0
    if b_hp_total <= 0:
        return "A", len([u for u in team_a_units if u is not None])

    est_combat_time = max(
        a_hp_front / b_dps if a_hp_front > 0 else 0.0,
        b_hp_front / a_dps if b_hp_front > 0 else 0.0,
    ) + max(
        a_hp_back / b_dps if a_hp_back > 0 else 0.0,
        b_hp_back / a_dps if b_hp_back > 0 else 0.0,
    )
    if regen_a > 0 or regen_b > 0:
        a_hp_total, a_hp_front, a_hp_back, a_dps = compute_team_stats(
            team_a_units, team_a_traits, regen_a, est_combat_time
        )
        b_hp_total, b_hp_front, b_hp_back, b_dps = compute_team_stats(
            team_b_units, team_b_traits, regen_b, est_combat_time
        )

    time_to_kill_a_front = a_hp_front / b_dps if a_hp_front > 0 else 0.0
    time_to_kill_b_front = b_hp_front / a_dps if b_hp_front > 0 else 0.0

    combat_time = 0.0
    slayer_thresh_a, slayer_mult_a = get_slayer_execute(team_a_traits)
    slayer_thresh_b, slayer_mult_b = get_slayer_execute(team_b_traits)

    if time_to_kill_a_front < time_to_kill_b_front:
        combat_time += time_to_kill_a_front
        remaining_b_dps = b_dps
        if slayer_mult_b > 1.0 and a_hp_total > 0:
            hp_frac = (a_hp_front - b_dps * time_to_kill_a_front) / a_hp_total
            hp_frac = max(0.0, hp_frac)
            if hp_frac <= slayer_thresh_b:
                remaining_b_dps *= slayer_mult_b
        time_to_kill_a_back = a_hp_back / remaining_b_dps if a_hp_back > 0 else 0.0
        winner = "B"
        combat_time += time_to_kill_a_back
        loser_dps = a_dps
        winner_total_hp = b_hp_total
        winner_units = [u for u in team_b_units if u is not None]

    elif time_to_kill_b_front < time_to_kill_a_front:
        combat_time += time_to_kill_b_front
        remaining_a_dps = a_dps
        if slayer_mult_a > 1.0 and b_hp_total > 0:
            hp_frac = (b_hp_front - a_dps * time_to_kill_b_front) / b_hp_total
            hp_frac = max(0.0, hp_frac)
            if hp_frac <= slayer_thresh_a:
                remaining_a_dps *= slayer_mult_a
        time_to_kill_b_back = b_hp_back / remaining_a_dps if b_hp_back > 0 else 0.0
        winner = "A"
        combat_time += time_to_kill_b_back
        loser_dps = b_dps
        winner_total_hp = a_hp_total
        winner_units = [u for u in team_a_units if u is not None]

    else:
        winner = "Tie"
        loser_dps = a_dps
        winner_total_hp = a_hp_total
        combat_time += time_to_kill_a_front
        winner_units = []

    surviving_units = 0
    if winner != "Tie":
        hp_fraction_remaining = max(0.0, 1.0 - (loser_dps * combat_time / max(winner_total_hp, 1.0)))
        surviving_units = round(hp_fraction_remaining * len(winner_units))

    return winner, surviving_units
