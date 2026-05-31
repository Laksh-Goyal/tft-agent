from tft_sim.env.action_space import (
    ACTION_PASS, ACTION_BUY_XP, ACTION_TOGGLE_FRONTLINE_START,
    ACTION_TOGGLE_FRONTLINE_END, compute_action_mask,
)
from tft_sim.game.player import Player
from tft_sim.game.units import UnitDatabase


def test_pass_always_legal():
    db = UnitDatabase("tft_sim/data/unit_roster.json")
    p = Player()
    mask = compute_action_mask(p, db)
    assert mask[ACTION_PASS] == 1


def test_buy_xp_gated_by_gold_and_level():
    db = UnitDatabase("tft_sim/data/unit_roster.json")
    p = Player(gold=3, level=1)
    mask = compute_action_mask(p, db)
    assert mask[ACTION_BUY_XP] == 0

    p.gold = 10
    mask = compute_action_mask(p, db)
    assert mask[ACTION_BUY_XP] == 1

    p.level = 9
    mask = compute_action_mask(p, db)
    assert mask[ACTION_BUY_XP] == 0


def test_place_unit_respects_level_cap():
    db = UnitDatabase("tft_sim/data/unit_roster.json")
    p = Player(level=1)
    p.bench[0] = db.create_unit(0)
    mask = compute_action_mask(p, db)
    # Can place on empty board slot 0
    assert mask[37] == 1
    p.board[0] = p.bench[0]
    p.bench[0] = None
    p.board[1] = db.create_unit(0)
    # Level 1 cap: two units on board, cannot place onto empty slot
    mask = compute_action_mask(p, db)
    assert mask[38] == 0  # bench empty anyway
    p.bench[0] = db.create_unit(0)
    mask = compute_action_mask(p, db)
    # bench 0 -> board slot 2: action 39; at level cap cannot place on empty slot
    assert mask[39] == 0


def test_toggle_frontline_always_illegal():
    db = UnitDatabase("tft_sim/data/unit_roster.json")
    p = Player()
    p.board[0] = db.create_unit(0)
    mask = compute_action_mask(p, db)
    for i in range(ACTION_TOGGLE_FRONTLINE_START, ACTION_TOGGLE_FRONTLINE_END + 1):
        assert mask[i] == 0
