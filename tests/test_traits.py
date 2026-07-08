import copy

from tft_sim.game.combat import resolve_combat
from tft_sim.game.trait_effects import apply_void_debuff, apply_sentinel_shields
from tft_sim.game.rounds import determine_round_type
from tft_sim.game.traits import detect_new_breakpoints, synergy_obs_pair
from tft_sim.game.units import UnitDatabase
ROSTER_PATH = "tft_sim/data/unit_roster.json"


def test_void_reduces_effective_hp():
    db = UnitDatabase(ROSTER_PATH)
    u = db.create_unit(0)
    base_eh = u.effective_hp()
    u2 = copy.deepcopy(u)
    apply_void_debuff([u2], 0.40)
    assert u2.effective_hp() < base_eh


def test_sentinel_shield_on_lowest_hp():
    db = UnitDatabase(ROSTER_PATH)
    low = db.create_unit(9)
    low.hp = 100
    high = db.create_unit(0)
    high.hp = 900
    units = [low, high]
    traits = {"Sentinel": {"ally_shield": 200}}
    apply_sentinel_shields(units, traits)
    assert low.combat_bonus_hp == 200
    assert high.combat_bonus_hp == 0


def test_bruiser_hp_multiplier_in_combat():
    db = UnitDatabase(ROSTER_PATH)
    tank = db.create_unit(0)  # Bruiser
    traits = {"Bruiser": {"hp_multiplier": 1.15}}
    board_a = [tank]
    board_b = [None] * 10
    winner, _ = resolve_combat(board_a, traits, board_b, {})
    assert winner == "A"


def test_determine_round_types():
    assert determine_round_type(1, 1) == "carousel"
    assert determine_round_type(1, 2) == "pve_creep"
    assert determine_round_type(2, 1) == "pve_creep"
    assert determine_round_type(2, 2) == "pvp"


def test_detect_new_breakpoints():
    db = UnitDatabase(ROSTER_PATH)
    trait = next(
        t for t, data in db.trait_data.items() if 2 in data["breakpoints"]
    )
    before = {trait: 1}
    after = {trait: 2}
    new = detect_new_breakpoints(before, after, db.trait_data)
    assert trait in new


def test_synergy_obs_pair():
    db = UnitDatabase(ROSTER_PATH)
    trait = sorted(db.trait_data.keys())[0]
    cn, prog = synergy_obs_pair(trait, 0, db.trait_data)
    assert cn == 0.0
    assert prog == 0.0
    cn2, prog2 = synergy_obs_pair(trait, 2, db.trait_data)
    assert cn2 > 0.0
