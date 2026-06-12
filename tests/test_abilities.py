from tft_sim.game.units import UnitDatabase

ROSTER_PATH = "tft_sim/data/unit_roster.json"


def test_ability_coeff_unchanged_by_star():
    db = UnitDatabase(ROSTER_PATH)
    u1 = db.create_unit(0, star_level=1)
    u3 = db.create_unit(0, star_level=3)
    assert u1.ability_damage == u3.ability_damage


def test_damage_ability_scales_with_star_via_ad():
    db = UnitDatabase(ROSTER_PATH)
    u1 = db.create_unit(0, star_level=1)
    u2 = db.create_unit(0, star_level=2)
    assert u2.resolved_ability_power() > u1.resolved_ability_power()
    assert u2.attack_damage > u1.attack_damage


def test_shield_ability_scales_with_star_via_hp():
    db = UnitDatabase(ROSTER_PATH)
    u1 = db.create_unit(3, star_level=1)  # Poppy shield
    u2 = db.create_unit(3, star_level=2)
    assert u1.ability_type == "shield"
    assert u2.resolved_ability_power() > u1.resolved_ability_power()

