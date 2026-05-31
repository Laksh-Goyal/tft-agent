import json
import copy
from dataclasses import dataclass, field

@dataclass
class Unit:
    id: int
    name: str
    cost: int
    hp: float
    armor: float
    magic_resist: float
    attack_damage: float
    attack_speed: float
    range: float
    ability_damage: float  # ability coefficient (multiplier) from roster JSON
    ability_type: str
    mana_cost: float
    traits: list[str]

    star_level: int = 1
    combat_bonus_hp: float = 0.0  # temporary shields (e.g. Sentinel), combat-only

    def ability_coeff(self) -> float:
        return self.ability_damage

    def resolved_ability_power(self) -> float:
        """Cast power after star scaling on AD/HP; coeff itself is not star-scaled."""
        if self.ability_type in ("heal", "shield"):
            return self.hp * self.ability_coeff()
        return self.attack_damage * self.ability_coeff()

    def effective_hp(self):
        phys_reduction = 100 / (100 + self.armor)
        magic_reduction = 100 / (100 + self.magic_resist)
        avg_reduction = (phys_reduction + magic_reduction) / 2
        return (self.hp + self.combat_bonus_hp) / avg_reduction

    def scale_stats(self):
        multiplier = 1.0
        if self.star_level == 2:
            multiplier = 1.8
        elif self.star_level == 3:
            multiplier = 3.24

        self.hp *= multiplier
        self.attack_damage *= multiplier

class UnitDatabase:
    def __init__(self, roster_path="tft_sim/data/unit_roster.json"):
        with open(roster_path, 'r') as f:
            data = json.load(f)
        self.unit_data = {u['id']: u for u in data['units']}
        self.trait_data = {t['name']: t for t in data['traits']}

    def create_unit(self, unit_id, star_level=1):
        if unit_id not in self.unit_data:
            return None
        data = copy.deepcopy(self.unit_data[unit_id])
        unit = Unit(**data, star_level=star_level)
        unit.scale_stats()
        return unit

    def get_unit_base_data(self, unit_id):
        return self.unit_data.get(unit_id)

def count_copies(units_list, unit_id, star_level):
    return sum(1 for u in units_list if u is not None and u.id == unit_id and u.star_level == star_level)

def remove_copies(player, unit_id, star_level, count=3):
    removed = 0
    for lst in [player.bench, player.board]:
        for i in range(len(lst)):
            u = lst[i]
            if u is not None and u.id == unit_id and u.star_level == star_level:
                lst[i] = None
                removed += 1
                if removed == count:
                    return

def try_combine(player, unit_id, star_level, unit_db: UnitDatabase):
    if star_level >= 3:
        return

    copies = count_copies(player.board + player.bench, unit_id, star_level)
    if copies >= 3:
        remove_copies(player, unit_id, star_level, count=3)
        new_unit = unit_db.create_unit(unit_id, star_level=star_level + 1)

        placed = False
        for i in range(len(player.bench)):
            if player.bench[i] is None:
                player.bench[i] = new_unit
                placed = True
                break
        if not placed:
            for i in range(len(player.board)):
                if player.board[i] is None:
                    player.board[i] = new_unit
                    placed = True
                    break

        if placed:
            try_combine(player, unit_id, star_level + 1, unit_db)
