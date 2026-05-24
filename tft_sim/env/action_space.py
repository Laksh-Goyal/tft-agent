import gymnasium.spaces as spaces
import numpy as np
from tft_sim.game.units import UnitDatabase

TOTAL_ACTIONS = 127

ACTION_PASS = 0
ACTION_BUY_XP = 1
ACTION_REROLL = 2
ACTION_BUY_UNIT_START = 3
ACTION_BUY_UNIT_END = 7
ACTION_SELL_BENCH_START = 8
ACTION_SELL_BENCH_END = 16
ACTION_SELL_BOARD_START = 17
ACTION_SELL_BOARD_END = 26
ACTION_TOGGLE_FRONTLINE_START = 27
ACTION_TOGGLE_FRONTLINE_END = 36
ACTION_PLACE_UNIT_START = 37
ACTION_PLACE_UNIT_END = 126

def get_action_space():
    """Returns the discrete action space for the TFT environment."""
    return spaces.Discrete(TOTAL_ACTIONS)

def compute_action_mask(player, unit_db: UnitDatabase) -> np.ndarray:
    """
    Computes a boolean action mask for a given player.
    1 means action is legal, 0 means illegal.
    """
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    
    # pass -> always legal
    mask[ACTION_PASS] = 1
    
    # buy_xp -> gold >= 4 AND level < max_level
    if player.gold >= 4 and player.level < 9:
        mask[ACTION_BUY_XP] = 1
        
    # reroll -> gold >= 2
    if player.gold >= 2:
        mask[ACTION_REROLL] = 1
        
    # buy_unit(i) -> shop slot i is occupied AND gold >= unit cost AND bench not full
    bench_full = all(u is not None for u in player.bench)
    for i, unit_id in enumerate(player.current_shop):
        if unit_id is not None and not bench_full:
            unit_data = unit_db.get_unit_base_data(unit_id)
            if unit_data and player.gold >= unit_data['cost']:
                mask[ACTION_BUY_UNIT_START + i] = 1

    # sell_bench(i) -> bench slot i is occupied
    for i, unit in enumerate(player.bench):
        if unit is not None:
            mask[ACTION_SELL_BENCH_START + i] = 1

    # sell_board(i) -> board slot i is occupied
    for i, unit in enumerate(player.board):
        if unit is not None:
            mask[ACTION_SELL_BOARD_START + i] = 1

    # toggle_frontline(i) -> board slot i is occupied
    for i, unit in enumerate(player.board):
        if unit is not None:
            mask[ACTION_TOGGLE_FRONTLINE_START + i] = 1

    # place_unit(b, d)
    board_unit_count = sum(1 for u in player.board if u is not None)
    for b in range(9):
        if player.bench[b] is not None:
            for d in range(10):
                action_idx = b * 10 + d + 37
                if player.board[d] is None:
                    # Target is empty, so we must be under the unit cap to place
                    if board_unit_count < player.level:
                        mask[action_idx] = 1
                else:
                    # Target is occupied, so it's a swap (cap doesn't change)
                    mask[action_idx] = 1

    return mask
