import gymnasium.spaces as spaces

from tft_sim.game.actions import (  # noqa: F401 — re-exported for env callers
    TOTAL_ACTIONS,
    ACTION_PASS,
    ACTION_BUY_XP,
    ACTION_REROLL,
    ACTION_BUY_UNIT_START,
    ACTION_BUY_UNIT_END,
    ACTION_SELL_BENCH_START,
    ACTION_SELL_BENCH_END,
    ACTION_SELL_BOARD_START,
    ACTION_SELL_BOARD_END,
    ACTION_TOGGLE_FRONTLINE_START,
    ACTION_TOGGLE_FRONTLINE_END,
    ACTION_PLACE_UNIT_START,
    ACTION_PLACE_UNIT_END,
    compute_action_mask,
)


def get_action_space():
    """Discrete action space sized to TOTAL_ACTIONS (~127 masked planning actions)."""
    return spaces.Discrete(TOTAL_ACTIONS)
