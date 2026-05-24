import copy

import numpy as np

from tft_sim.game.combat import resolve_combat
from tft_sim.game.units import UnitDatabase


def _roster_path():
    return "tft_sim/data/unit_roster.json"


def test_effective_hp_increases_with_armor():
    db = UnitDatabase(_roster_path())
    unit = db.create_unit(0)
    base = unit.effective_hp()
    unit.armor = 200
    assert unit.effective_hp() > base


def test_resolve_combat_returns_winner_and_survivors():
    db = UnitDatabase(_roster_path())
    u = db.create_unit(0)
    board_a = [copy.deepcopy(u)]
    board_b = [None] * 10
    winner, surv = resolve_combat(board_a, {}, board_b, {})
    assert winner in ("A", "B", "Tie")
    assert surv >= 0


def test_resolve_combat_tie_when_symmetric():
    db = UnitDatabase(_roster_path())
    u = db.create_unit(0)
    board = [copy.deepcopy(u)]
    winner, _ = resolve_combat(board, {}, board, {})
    assert winner == "Tie"
