"""Episode metrics for training logs (see docs/tft_rl_spec.md § Logging Metrics)."""
from tft_sim.env.state import GameState
from tft_sim.game.player import Player


def board_power(player: Player) -> int:
    """Sum of unit costs on board."""
    return sum(u.cost for u in player.board if u is not None)


def agent_placement(game: GameState) -> int:
    """Agent finish rank 1 (best) through n_players (worst)."""
    alive = sorted(
        [p for p in game.players if not p.is_eliminated],
        key=lambda p: -p.health,
    )
    dead = [p for p in game.players if p.is_eliminated]
    ranked = alive + dead
    return ranked.index(game.agent) + 1


def episode_metrics(game: GameState) -> dict:
    """Snapshot metrics when an episode ends."""
    return {
        "placement": agent_placement(game),
        "rounds_survived": game.rounds_completed,
        "board_power": board_power(game.agent),
        "agent_health": game.agent.health,
    }
