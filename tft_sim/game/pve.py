"""PvE creep rounds — neutral opponents and gold drops."""
from typing import List, Optional

from tft_sim.game.combat import resolve_combat
from tft_sim.game.player import Player
from tft_sim.game.traits import compute_active_traits
from tft_sim.game.units import Unit, UnitDatabase


def build_creep_board(stage: int, unit_db: UnitDatabase) -> List[Optional[Unit]]:
    """Neutral creep board scaled by stage."""
    creep_ids = [0, 1, 2] if stage <= 1 else [2, 3, 4]
    if stage >= 4:
        creep_ids = [3, 4, 5, 6]
    board: List[Optional[Unit]] = [None] * 10
    for i, uid in enumerate(creep_ids[: min(len(creep_ids), 7)]):
        board[i] = unit_db.create_unit(uid)
    return board


def pve_gold_drop(stage: int) -> int:
    return 2 if stage == 1 else 3


def resolve_pve_round(game) -> float:
    """
    Each active player fights creeps individually.
    Win grants gold; loss does not reduce HP (non-eliminating).
    Returns combat reward for the RL agent only.
    """
    creep_board = build_creep_board(game.stage, game.unit_db)
    creep_traits = compute_active_traits(creep_board, game.unit_db.trait_data)
    agent_won = False
    agent_lost = False
    drop = pve_gold_drop(game.stage)

    for player in game.players:
        if player.is_eliminated:
            continue
        traits = compute_active_traits(player.board, game.unit_db.trait_data)
        winner, _ = resolve_combat(player.board, traits, creep_board, creep_traits)
        if winner == "A":
            player.gold += drop
            if player is game.agent:
                agent_won = True
        elif winner == "B":
            if player is game.agent:
                agent_lost = True
        # Tie: no gold, no HP change

    reward = 0.0
    if agent_won:
        reward += 0.2
    if agent_lost:
        reward -= 0.1
    return reward


def resolve_carousel_round(game) -> float:
    """Carousel has no combat; picks happen during planning."""
    return 0.0


def setup_carousel_shop(game, n_options: int = 4) -> None:
    """Populate carousel unit ids and mirror into each player's shop slots."""
    all_ids = list(game.unit_db.unit_data.keys())
    picks = game.rng.choice(all_ids, size=min(n_options, len(all_ids)), replace=False)
    game.carousel_options = [int(x) for x in picks]
    for player in game.players:
        if player.is_eliminated:
            continue
        game.pool.return_units(player.current_shop, game.unit_db)
        shop = [None] * 5
        for i, uid in enumerate(game.carousel_options[:5]):
            shop[i] = uid
        player.current_shop = shop
