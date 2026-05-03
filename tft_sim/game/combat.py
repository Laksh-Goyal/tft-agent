import copy
from typing import Tuple, List, Dict, Any
from tft_sim.game.units import Unit

def compute_team_stats(units: List[Unit]) -> Tuple[float, float, float, float]:
    frontline = [u for u in units if u is not None and u.is_frontline]
    backline  = [u for u in units if u is not None and not u.is_frontline]

    frontline_hp  = sum(u.effective_hp() for u in frontline)
    backline_hp   = sum(u.effective_hp() for u in backline)
    total_hp      = frontline_hp + backline_hp

    # DPS = auto attack damage * attack speed + ability damage per second
    # ability DPS = ability_damage / (mana_cost / 10)  [crude mana regen approximation]
    total_dps = sum(
        u.attack_damage * u.attack_speed + (u.ability_damage / max((u.mana_cost / 10), 0.1))
        for u in (frontline + backline)
    )

    return total_hp, frontline_hp, backline_hp, total_dps


def apply_trait_effects(units: List[Unit], active_traits: Dict[str, Any]):
    # active_traits is dict {trait_name: effects_dict}
    for unit in units:
        if unit is None: continue
        for trait_name, effects in active_traits.items():
            if trait_name in unit.traits:
                # Apply multipliers
                unit.hp *= effects.get("hp_multiplier", 1.0)
                unit.attack_damage *= effects.get("ad_multiplier", 1.0)
                unit.ability_damage *= effects.get("ability_damage_multiplier", 1.0)
                # Apply other potential effects
                unit.armor *= effects.get("armor_multiplier", 1.0)
                unit.magic_resist *= effects.get("mr_multiplier", 1.0)
                unit.attack_speed *= effects.get("as_multiplier", 1.0)


def resolve_combat(team_a_units: List[Unit], team_a_traits: Dict[str, Any], 
                   team_b_units: List[Unit], team_b_traits: Dict[str, Any]) -> Tuple[str, int]:
    # Deep copy units to apply temporary combat buffs without affecting real units
    team_a_units = copy.deepcopy(team_a_units)
    team_b_units = copy.deepcopy(team_b_units)

    apply_trait_effects(team_a_units, team_a_traits)
    apply_trait_effects(team_b_units, team_b_traits)

    a_hp_total, a_hp_front, a_hp_back, a_dps = compute_team_stats(team_a_units)
    b_hp_total, b_hp_front, b_hp_back, b_dps = compute_team_stats(team_b_units)
    
    if a_dps <= 0: a_dps = 0.001 # avoid division by zero
    if b_dps <= 0: b_dps = 0.001

    # Phase 1: frontlines trade
    time_to_kill_a_front = a_hp_front / b_dps if a_hp_front > 0 else 0
    time_to_kill_b_front = b_hp_front / a_dps if b_hp_front > 0 else 0

    combat_time = 0.0

    if time_to_kill_a_front < time_to_kill_b_front:
        # Team B kills team A's frontline first
        combat_time += time_to_kill_a_front
        remaining_b_dps = b_dps  # (simplified: no attrition on b's frontline)
        
        # Phase 2: B's full DPS vs A's backline
        time_to_kill_a_back = a_hp_back / remaining_b_dps if a_hp_back > 0 else 0
        winner = "B"
        combat_time += time_to_kill_a_back
        
        loser_dps = a_dps
        winner_total_hp = b_hp_total
        winner_units = [u for u in team_b_units if u is not None]

    elif time_to_kill_b_front < time_to_kill_a_front:
        # Team A kills team B's frontline first
        combat_time += time_to_kill_b_front
        remaining_a_dps = a_dps
        
        time_to_kill_b_back = b_hp_back / remaining_a_dps if b_hp_back > 0 else 0
        winner = "A"
        combat_time += time_to_kill_b_back
        
        loser_dps = b_dps
        winner_total_hp = a_hp_total
        winner_units = [u for u in team_a_units if u is not None]
    else:
        # Tie
        winner = "Tie"
        loser_dps = a_dps
        winner_total_hp = a_hp_total
        combat_time += time_to_kill_a_front
        winner_units = []

    # Surviving units count
    surviving_units = 0
    if winner != "Tie":
        hp_fraction_remaining = max(0.0, 1.0 - (loser_dps * combat_time / max(winner_total_hp, 1.0)))
        surviving_units = round(hp_fraction_remaining * len(winner_units))

    return winner, surviving_units
