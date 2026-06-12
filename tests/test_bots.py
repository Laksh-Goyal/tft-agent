import numpy as np
import pytest

from tft_sim.agents.bot import (
    ARCHETYPE_IDS,
    STRATEGY_REGISTRY,
    HyperBuyer,
    InterestSaver,
    Balanced,
    assign_bot_strategies,
    run_bot_planning_phase,
    board_unit_count,
)
from tft_sim.game.actions import (
    ACTION_PASS,
    ACTION_BUY_XP,
    ACTION_REROLL,
    ACTION_BUY_UNIT_START,
    compute_action_mask,
)
from tft_sim.env.state import GameState
from tft_sim.env.tft_env import TFTEnv
from tft_sim.game.player import Player
from tft_sim.game.units import UnitDatabase

ROSTER_PATH = "tft_sim/data/unit_roster.json"


def _mask(player: Player, db: UnitDatabase) -> np.ndarray:
    return compute_action_mask(player, db)


def _player_with_gold(gold: int, level: int = 1) -> Player:
    p = Player(gold=gold, level=level)
    return p


def test_all_archetypes_registered():
    assert set(STRATEGY_REGISTRY.keys()) == set(ARCHETYPE_IDS)


def test_bot_strategies_assigned_on_reset():
    game = GameState(n_players=8, rng=np.random.default_rng(0))
    assign_bot_strategies(game.players, game.rng)
    for player in game.players:
        if player.is_agent:
            continue
        assert player.bot_strategy_id in ARCHETYPE_IDS


def test_bot_strategies_reproducible_with_seed():
    g1 = GameState(n_players=8, rng=np.random.default_rng(42))
    assign_bot_strategies(g1.players, g1.rng)
    g2 = GameState(n_players=8, rng=np.random.default_rng(42))
    assign_bot_strategies(g2.players, g2.rng)
    ids1 = [p.bot_strategy_id for p in g1.players[1:]]
    ids2 = [p.bot_strategy_id for p in g2.players[1:]]
    assert ids1 == ids2


@pytest.mark.parametrize("strategy_id", ARCHETYPE_IDS)
def test_strategy_always_picks_legal_action(strategy_id):
    db = UnitDatabase(ROSTER_PATH)
    strategy = STRATEGY_REGISTRY[strategy_id]
    game = GameState(n_players=2, rng=np.random.default_rng(0))
    game.start_round()
    player = game.players[1]
    for _ in range(20):
        mask = _mask(player, db)
        action = strategy.choose_action(player, mask, db)
        assert mask[action] == 1, f"{strategy_id} picked illegal action {action}"


def test_hyperbuyer_never_buys_xp():
    db = UnitDatabase(ROSTER_PATH)
    strategy = HyperBuyer()
    player = _player_with_gold(gold=20, level=1)
    mask = _mask(player, db)
    mask[ACTION_BUY_XP] = 1
    for _ in range(10):
        action = strategy.choose_action(player, mask, db)
        assert action != ACTION_BUY_XP


def test_interest_saver_hoards_below_50():
    db = UnitDatabase(ROSTER_PATH)
    strategy = InterestSaver()
    player = _player_with_gold(gold=40, level=3)
    player.bench[0] = db.create_unit(0)
    mask = _mask(player, db)
    for _ in range(10):
        action = strategy.choose_action(player, mask, db)
        assert action not in (ACTION_BUY_XP, ACTION_REROLL)
        assert action < ACTION_BUY_UNIT_START or action > ACTION_BUY_UNIT_START + 4


def test_interest_saver_buys_above_50_when_under_cap():
    db = UnitDatabase(ROSTER_PATH)
    strategy = InterestSaver()
    player = _player_with_gold(gold=55, level=3)
    player.current_shop[0] = 0
    mask = _mask(player, db)
    action = strategy.choose_action(player, mask, db)
    assert action == ACTION_BUY_UNIT_START


def test_balanced_buys_xp_when_rich():
    db = UnitDatabase(ROSTER_PATH)
    strategy = Balanced()
    player = _player_with_gold(gold=20, level=1)
    player.current_shop = [None] * 5
    mask = _mask(player, db)
    action = strategy.choose_action(player, mask, db)
    assert action == ACTION_BUY_XP


def test_opponents_field_units_after_round():
    env = TFTEnv(n_players=8)
    env.reset(seed=0)
    env.step(0)  # pass immediately

    units_on_opponents = 0
    for player in env.game.players[1:]:
        if player.is_eliminated:
            continue
        on_board = board_unit_count(player)
        on_bench = sum(1 for u in player.bench if u is not None)
        units_on_opponents += on_board + on_bench

    assert units_on_opponents >= 3


def test_bot_phase_does_not_increment_agent_action_counter():
    game = GameState(n_players=4, rng=np.random.default_rng(0))
    assign_bot_strategies(game.players, game.rng)
    game.start_round()
    assert game.actions_this_round == 0
    run_bot_planning_phase(game)
    assert game.actions_this_round == 0


def test_full_round_with_bots_completes():
    env = TFTEnv(n_players=4)
    obs, _ = env.reset(seed=1)
    obs2, reward, term, trunc, info = env.step(0)
    assert obs2.shape == obs.shape
    assert isinstance(reward, float)
    assert "action_mask" in info
    assert not term or env.game.agent.health <= 0
