"""Team- and unit-level trait effect helpers for combat."""
from typing import Dict, Any, List, Optional

from tft_sim.game.units import Unit

BACKLINE_MIN_RANGE = 3
CRIT_DAMAGE = 0.75
MANA_REGEN_FACTOR = 2.0


def is_backline(unit: Unit) -> bool:
    return unit.range >= BACKLINE_MIN_RANGE


def apply_ally_trait_multipliers(units: List[Unit], active_traits: Dict[str, Any]):
    for unit in units:
        if unit is None:
            continue
        for trait_name, effects in active_traits.items():
            if trait_name not in unit.traits:
                continue
            unit.hp *= effects.get("hp_multiplier", 1.0)
            unit.attack_damage *= effects.get("ad_multiplier", 1.0)
            unit.ability_damage *= effects.get("ability_damage_multiplier", 1.0)
            unit.armor *= effects.get("armor_multiplier", 1.0)
            unit.magic_resist *= effects.get("mr_multiplier", 1.0)
            unit.attack_speed *= effects.get("as_multiplier", 1.0)


def apply_void_debuff(target_units: List[Unit], reduction: float):
    if reduction <= 0:
        return
    for unit in target_units:
        if unit is None:
            continue
        unit.armor *= (1.0 - reduction)
        unit.magic_resist *= (1.0 - reduction)


def get_void_reduction(active_traits: Dict[str, Any]) -> float:
    if "Void" not in active_traits:
        return 0.0
    return float(active_traits["Void"].get("enemy_armor_reduction", 0.0))


def get_wildborn_regen(active_traits: Dict[str, Any]) -> float:
    if "Wildborn" not in active_traits:
        return 0.0
    return float(active_traits["Wildborn"].get("hp_regen_per_sec", 0.0))


def get_assassin_crit_bonus(active_traits: Dict[str, Any]) -> float:
    if "Assassin" not in active_traits:
        return 0.0
    return float(active_traits["Assassin"].get("crit_bonus", 0.0))


def get_slayer_execute(active_traits: Dict[str, Any]) -> tuple:
    if "Slayer" not in active_traits:
        return 0.0, 1.0
    effects = active_traits["Slayer"]
    return (
        float(effects.get("execute_threshold", 0.0)),
        float(effects.get("execute_bonus_damage", 1.0)),
    )


def get_invoker_mana_per_sec(active_traits: Dict[str, Any]) -> float:
    if "Invoker" not in active_traits:
        return 0.0
    return float(active_traits["Invoker"].get("mana_per_sec", 0.0))


def apply_sentinel_shields(units: List[Unit], active_traits: Dict[str, Any]):
    if "Sentinel" not in active_traits:
        return
    shield = float(active_traits["Sentinel"].get("ally_shield", 0.0))
    if shield <= 0:
        return
    board_units = [u for u in units if u is not None]
    if not board_units:
        return
    lowest = min(board_units, key=lambda u: u.hp)
    lowest.combat_bonus_hp += shield


def effective_mana_cost(unit: Unit, invoker_mana_per_sec: float) -> float:
    if invoker_mana_per_sec <= 0 or "Invoker" not in unit.traits:
        return unit.mana_cost
    reduction = invoker_mana_per_sec * MANA_REGEN_FACTOR
    return max(10.0, unit.mana_cost - reduction)
