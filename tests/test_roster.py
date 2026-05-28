from collections import Counter

import pytest

from tft_sim.game.units import UnitDatabase

ROSTER_PATH = "tft_sim/data/unit_roster.json"
EXPECTED_COST_COUNTS = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2}
EXPECTED_ORIGINS = {f"Origin{i}" for i in range(1, 5)}
EXPECTED_CLASSES = {f"Class{i}" for i in range(1, 7)}


@pytest.fixture
def db():
    return UnitDatabase(ROSTER_PATH)


def test_roster_loads_30_units_and_10_traits(db):
    assert len(db.unit_data) == 30
    assert len(db.trait_data) == 10


def test_unit_ids_are_unique_and_contiguous(db):
    ids = sorted(db.unit_data.keys())
    assert ids == list(range(30))


def test_cost_tier_counts(db):
    counts = Counter(u["cost"] for u in db.unit_data.values())
    assert counts == EXPECTED_COST_COUNTS


def test_trait_names_are_placeholders(db):
    names = set(db.trait_data.keys())
    assert names == EXPECTED_ORIGINS | EXPECTED_CLASSES


def test_every_unit_trait_references_defined_trait(db):
    for unit in db.unit_data.values():
        assert len(unit["traits"]) == 2
        for trait in unit["traits"]:
            assert trait in db.trait_data


def test_unit_trait_assignment_formula(db):
    for uid, unit in db.unit_data.items():
        assert unit["traits"] == [
            f"Origin{(uid % 4) + 1}",
            f"Class{(uid % 6) + 1}",
        ]


def test_create_unit_roundtrip(db):
    unit = db.create_unit(15)
    assert unit is not None
    assert unit.id == 15
    assert unit.cost == 2
