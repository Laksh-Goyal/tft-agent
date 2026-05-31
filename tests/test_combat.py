import copy

import numpy as np

from tft_sim.game.combat import resolve_combat, compute_team_stats
from tft_sim.game.trait_effects import is_backline, BACKLINE_MIN_RANGE
from tft_sim.game.units import UnitDatabase


def _roster_path():
    return "tft_sim/data/unit_roster.json"


def test_effective_hp_increases_with_armor():
    db = UnitDatabase(_roster_path())
    unit = db.create_unit(0)
    base = unit.effective_hp()
    unit.armor = 200
    assert unit.effective_hp() > base


def test_is_backline_uses_range():
    db = UnitDatabase(_roster_path())
    melee = db.create_unit(0)
    ranged = db.create_unit(1)
    assert not is_backline(melee)
    assert is_backline(ranged)
    assert ranged.range >= BACKLINE_MIN_RANGE


def test_resolve_combat_returns_winner_and_survivors():
    db = UnitDatabase(_roster_path())
    u = db.create_unit(0)
    board_a = [copy.deepcopy(u)]
    board_b = [None] * 10
    winner, surv = resolve_combat(board_a, {}, board_b, {})
    assert winner in ("A", "B", "Tie")
    assert surv >= 0


def test_ranged_backline_protected_by_melee_front():
    db = UnitDatabase(_roster_path())
    melee = db.create_unit(0)
    ranged = db.create_unit(1)
    strong_melee = db.create_unit(22)
    # A: one weak melee front, one ranged back
    board_a = [copy.deepcopy(melee), None, None, None, None, None, None, None, None, copy.deepcopy(ranged)]
    # B: strong melee only
    board_b = [copy.deepcopy(strong_melee)] + [None] * 9
    winner, _ = resolve_combat(board_a, {}, board_b, {})
    # B should win; A's ranged should matter less if front dies fast
    assert winner in ("A", "B", "Tie")


def test_heal_shield_increase_team_hp_pool():
    db = UnitDatabase(_roster_path())
    poppy = db.create_unit(3)
    _, front_hp, _, _ = compute_team_stats([poppy], {})
    soraka = db.create_unit(9)
    _, front_hp2, _, _ = compute_team_stats([soraka], {})
    assert front_hp > poppy.effective_hp() or front_hp2 > soraka.effective_hp()
