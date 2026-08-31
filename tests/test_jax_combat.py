"""Tests for the JAX combat resolution module.

Covers:
    - Edge cases (empty boards, one-sided)
    - Star-level scaling (3-star beats 1-star)
    - Trait effects (Warlord AD multiplier, Assassin crit)
    - Cross-check with PyTorch implementation
    - JIT compilation
    - vmap (batched combat)
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tft_sim.jax_port.static_data import load_static_data
from tft_sim.jax_port.game_state import BOARD_SIZE
from tft_sim.jax_port.combat import (
    resolve_combat_jax,
    get_unit_stats,
    get_active_trait_effects,
    compute_team_stats,
)


@pytest.fixture(scope="module")
def static():
    """Load static data once for all tests."""
    s, _ = load_static_data()
    return s


def _empty_board():
    return jnp.full(BOARD_SIZE, -1, dtype=jnp.int32), jnp.ones(BOARD_SIZE, dtype=jnp.int32)


def _board_with(units):
    """Build a board from a list of (unit_id, star) tuples."""
    ids = jnp.full(BOARD_SIZE, -1, dtype=jnp.int32)
    stars = jnp.ones(BOARD_SIZE, dtype=jnp.int32)
    for i, (uid, star) in enumerate(units):
        ids = ids.at[i].set(uid)
        stars = stars.at[i].set(star)
    return ids, stars


# ============================================================
# Edge cases
# ============================================================
class TestEdgeCases:
    def test_both_empty_returns_tie(self, static):
        a_ids, a_stars = _empty_board()
        b_ids, b_stars = _empty_board()
        winner, surv = resolve_combat_jax(a_ids, a_stars, b_ids, b_stars, static)
        assert int(winner) == 2, "Both empty should be a tie"
        assert int(surv) == 0

    def test_a_nonempty_b_empty_a_wins(self, static):
        a_ids, a_stars = _board_with([(0, 1)])
        b_ids, b_stars = _empty_board()
        winner, surv = resolve_combat_jax(a_ids, a_stars, b_ids, b_stars, static)
        assert int(winner) == 0, "Non-empty A should beat empty B"
        assert int(surv) >= 1

    def test_b_nonempty_a_empty_b_wins(self, static):
        a_ids, a_stars = _empty_board()
        b_ids, b_stars = _board_with([(0, 1)])
        winner, surv = resolve_combat_jax(a_ids, a_stars, b_ids, b_stars, static)
        assert int(winner) == 1, "Non-empty B should beat empty A"
        assert int(surv) == 0  # A has no survivors

    def test_winner_is_0_1_or_2(self, static):
        a_ids, a_stars = _board_with([(0, 1)])
        b_ids, b_stars = _board_with([(1, 1)])
        winner, _ = resolve_combat_jax(a_ids, a_stars, b_ids, b_stars, static)
        assert int(winner) in (0, 1, 2)


# ============================================================
# Star-level scaling
# ============================================================
class TestStarScaling:
    def test_3star_beats_1star(self, static):
        a_ids, a_stars = _board_with([(0, 3)])
        b_ids, b_stars = _board_with([(0, 1)])
        winner, _ = resolve_combat_jax(a_ids, a_stars, b_ids, b_stars, static)
        assert int(winner) == 0, "3-star should beat 1-star of same unit"

    def test_2star_beats_1star(self, static):
        a_ids, a_stars = _board_with([(0, 2)])
        b_ids, b_stars = _board_with([(0, 1)])
        winner, _ = resolve_combat_jax(a_ids, a_stars, b_ids, b_stars, static)
        assert int(winner) == 0, "2-star should beat 1-star of same unit"

    def test_star_scaling_hp_multiplier(self, static):
        """HP at star 2 should be 1.8x star 1, star 3 should be 3.24x."""
        s1_ids, s1_stars = _board_with([(0, 1)])
        s2_ids, s2_stars = _board_with([(0, 2)])
        s3_ids, s3_stars = _board_with([(0, 3)])

        stats1 = get_unit_stats(s1_ids, s1_stars, static)
        stats2 = get_unit_stats(s2_ids, s2_stars, static)
        stats3 = get_unit_stats(s3_ids, s3_stars, static)

        hp_ratio_2 = float(stats2["hp"][0] / stats1["hp"][0])
        hp_ratio_3 = float(stats3["hp"][0] / stats1["hp"][0])

        assert abs(hp_ratio_2 - 1.8) < 1e-5
        assert abs(hp_ratio_3 - 3.24) < 1e-5


# ============================================================
# Trait effects
# ============================================================
class TestTraitEffects:
    def test_warlord_ad_multiplier_at_3_units(self, static):
        """3 Warlords should activate the 3-breakpoint: ad_multiplier=1.15."""
        # Find 3 Warlord units
        import json
        with open("tft_sim/data/unit_roster.json") as f:
            data = json.load(f)
        warlord_ids = [u["id"] for u in data["units"] if "Warlord" in u["traits"]]

        ids, stars = _board_with([(warlord_ids[0], 1), (warlord_ids[1], 1), (warlord_ids[2], 1)])
        effects = get_active_trait_effects(ids, stars, static)
        ad_mult = float(effects["ad_multiplier"])
        assert abs(ad_mult - 1.15) < 1e-5, f"Expected 1.15, got {ad_mult}"

    def test_no_active_traits_gives_default_multipliers(self, static):
        """With no active trait breakpoints, multipliers should be 1.0."""
        ids, stars = _board_with([(0, 1)])
        effects = get_active_trait_effects(ids, stars, static)
        assert float(effects["ad_multiplier"]) == 1.0
        assert float(effects["hp_multiplier"]) == 1.0
        assert float(effects["armor_multiplier"]) == 1.0

    def test_bruiser_hp_multiplier_at_2_units(self, static):
        """2 Bruisers should activate the 2-breakpoint: hp_multiplier=1.15."""
        import json
        with open("tft_sim/data/unit_roster.json") as f:
            data = json.load(f)
        bruiser_ids = [u["id"] for u in data["units"] if "Bruiser" in u["traits"]]

        ids, stars = _board_with([(bruiser_ids[0], 1), (bruiser_ids[1], 1)])
        effects = get_active_trait_effects(ids, stars, static)
        hp_mult = float(effects["hp_multiplier"])
        assert abs(hp_mult - 1.15) < 1e-5, f"Expected 1.15, got {hp_mult}"


# ============================================================
# Cross-check with PyTorch
# ============================================================
class TestCrossCheckPyTorch:
    def test_winner_matches_pytorch(self, static):
        """JAX combat winner should match PyTorch for the same teams."""
        from tft_sim.game.units import UnitDatabase
        from tft_sim.game.combat import resolve_combat
        from tft_sim.game.traits import compute_active_traits

        db = UnitDatabase("tft_sim/data/unit_roster.json")

        team_a_ids = jnp.array([0, 1, 2, -1, -1, -1, -1, -1, -1, -1], dtype=jnp.int32)
        team_a_stars = jnp.ones(BOARD_SIZE, dtype=jnp.int32)
        team_b_ids = jnp.array([3, 4, 5, -1, -1, -1, -1, -1, -1, -1], dtype=jnp.int32)
        team_b_stars = jnp.ones(BOARD_SIZE, dtype=jnp.int32)

        # PyTorch
        pt_a = [db.create_unit(int(i)) if i >= 0 else None for i in team_a_ids]
        pt_b = [db.create_unit(int(i)) if i >= 0 else None for i in team_b_ids]
        pt_traits_a = compute_active_traits(pt_a, db.trait_data)
        pt_traits_b = compute_active_traits(pt_b, db.trait_data)
        pt_winner, _ = resolve_combat(pt_a, pt_traits_a, pt_b, pt_traits_b)
        pt_idx = {"A": 0, "B": 1, "Tie": 2}[pt_winner]

        # JAX
        jax_winner, _ = resolve_combat_jax(team_a_ids, team_a_stars, team_b_ids, team_b_stars, static)

        assert int(jax_winner) == pt_idx, \
            f"JAX winner {jax_winner} != PyTorch winner {pt_idx}"

    def test_cross_check_multiple_teams(self, static):
        """Cross-check several team compositions."""
        from tft_sim.game.units import UnitDatabase
        from tft_sim.game.combat import resolve_combat
        from tft_sim.game.traits import compute_active_traits

        db = UnitDatabase("tft_sim/data/unit_roster.json")

        teams = [
            ([0, 1], [3, 4]),
            ([0, 1, 2], [5, 6, 7]),
            ([0, 3, 5], [1, 4, 6]),
        ]

        for a_list, b_list in teams:
            a_ids = jnp.full(BOARD_SIZE, -1, dtype=jnp.int32)
            b_ids = jnp.full(BOARD_SIZE, -1, dtype=jnp.int32)
            for i, uid in enumerate(a_list):
                a_ids = a_ids.at[i].set(uid)
            for i, uid in enumerate(b_list):
                b_ids = b_ids.at[i].set(uid)
            stars = jnp.ones(BOARD_SIZE, dtype=jnp.int32)

            # PyTorch
            pt_a = [db.create_unit(uid) for uid in a_list]
            pt_b = [db.create_unit(uid) for uid in b_list]
            pt_ta = compute_active_traits(pt_a, db.trait_data)
            pt_tb = compute_active_traits(pt_b, db.trait_data)
            pt_winner, _ = resolve_combat(pt_a, pt_ta, pt_b, pt_tb)
            pt_idx = {"A": 0, "B": 1, "Tie": 2}[pt_winner]

            # JAX
            jax_winner, _ = resolve_combat_jax(a_ids, stars, b_ids, stars, static)

            assert int(jax_winner) == pt_idx, \
                f"Team {a_list} vs {b_list}: JAX {jax_winner} != PyTorch {pt_idx}"


# ============================================================
# JIT and vmap compatibility
# ============================================================
class TestJITVmap:
    def test_jit_matches_eager(self, static):
        a_ids, a_stars = _board_with([(0, 1), (1, 1)])
        b_ids, b_stars = _board_with([(3, 1), (4, 1)])

        eager_winner, eager_surv = resolve_combat_jax(a_ids, a_stars, b_ids, b_stars, static)

        jit_fn = jax.jit(resolve_combat_jax)
        jit_winner, jit_surv = jit_fn(a_ids, a_stars, b_ids, b_stars, static)

        assert int(jit_winner) == int(eager_winner)
        assert int(jit_surv) == int(eager_surv)

    def test_vmap_batch_shape(self, static):
        a_ids, a_stars = _board_with([(0, 1)])
        b_ids, b_stars = _board_with([(1, 1)])

        batch_size = 8
        batch_a_ids = jnp.stack([a_ids] * batch_size)
        batch_a_stars = jnp.stack([a_stars] * batch_size)
        batch_b_ids = jnp.stack([b_ids] * batch_size)
        batch_b_stars = jnp.stack([b_stars] * batch_size)

        winners, survs = jax.vmap(resolve_combat_jax, in_axes=(0, 0, 0, 0, None))(
            batch_a_ids, batch_a_stars, batch_b_ids, batch_b_stars, static
        )

        assert winners.shape == (batch_size,)
        assert survs.shape == (batch_size,)

    def test_vmap_all_same_winner(self, static):
        """Batch of identical matches should give identical results."""
        a_ids, a_stars = _board_with([(0, 3)])
        b_ids, b_stars = _board_with([(0, 1)])

        batch_a_ids = jnp.stack([a_ids] * 4)
        batch_a_stars = jnp.stack([a_stars] * 4)
        batch_b_ids = jnp.stack([b_ids] * 4)
        batch_b_stars = jnp.stack([b_stars] * 4)

        winners, _ = jax.vmap(resolve_combat_jax, in_axes=(0, 0, 0, 0, None))(
            batch_a_ids, batch_a_stars, batch_b_ids, batch_b_stars, static
        )

        # All should be 0 (3-star beats 1-star)
        assert jnp.all(winners == 0)


# ============================================================
# Team stats
# ============================================================
class TestTeamStats:
    def test_empty_board_has_zero_hp(self, static):
        ids, stars = _empty_board()
        stats = get_unit_stats(ids, stars, static)
        effects = get_active_trait_effects(ids, stars, static)
        effects["_assassin_trait_idx"] = jnp.int32(-1)
        effects["_invoker_trait_idx"] = jnp.int32(-1)
        total_hp, front_hp, back_hp, dps = compute_team_stats(stats, effects)
        assert float(total_hp) == 0.0
        assert float(dps) == 0.0

    def test_nonempty_board_has_positive_hp(self, static):
        ids, stars = _board_with([(0, 1)])
        stats = get_unit_stats(ids, stars, static)
        effects = get_active_trait_effects(ids, stars, static)
        effects["_assassin_trait_idx"] = jnp.int32(-1)
        effects["_invoker_trait_idx"] = jnp.int32(-1)
        total_hp, _, _, dps = compute_team_stats(stats, effects)
        assert float(total_hp) > 0
        assert float(dps) > 0
