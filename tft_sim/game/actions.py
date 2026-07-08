"""Action IDs and legality masking for the TFT planning phase.

Lives in game/ (not env/) so bots and GameState share the same rules without
importing the Gymnasium wrapper layer.
"""
import numpy as np

from tft_sim.game.units import UnitDatabase

TOTAL_ACTIONS = 127

# Action index layout (see docs/tft_rl_spec.md)
ACTION_PASS = 0
ACTION_BUY_XP = 1
ACTION_REROLL = 2
ACTION_BUY_UNIT_START = 3   # slots 0–4 → indices 3–7
ACTION_BUY_UNIT_END = 7
ACTION_SELL_BENCH_START = 8   # bench slots 0–8
ACTION_SELL_BENCH_END = 16
ACTION_SELL_BOARD_START = 17  # board slots 0–9
ACTION_SELL_BOARD_END = 26
ACTION_TOGGLE_FRONTLINE_START = 27  # deprecated; always illegal
ACTION_TOGGLE_FRONTLINE_END = 36
ACTION_PLACE_UNIT_START = 37  # bench b × board d → index 37 + b*10 + d
ACTION_PLACE_UNIT_END = 126


def compute_action_mask(player, unit_db: UnitDatabase, *, free_shop: bool = False) -> np.ndarray:
    """
    Boolean mask for a player's legal actions this step.
    1 = legal, 0 = illegal.
    """
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)

    # pass → always legal
    mask[ACTION_PASS] = 1

    # buy_xp → gold >= 4 AND level < max level
    if player.gold >= 4 and player.level < 9:
        mask[ACTION_BUY_XP] = 1

    # reroll → gold >= 2
    if player.gold >= 2:
        mask[ACTION_REROLL] = 1

    # buy_unit(i) → shop slot occupied, affordable, bench not full
    bench_full = all(u is not None for u in player.bench)
    for i, unit_id in enumerate(player.current_shop):
        if unit_id is not None and not bench_full:
            unit_data = unit_db.get_unit_base_data(unit_id)
            if unit_data and (free_shop or player.gold >= unit_data['cost']):
                mask[ACTION_BUY_UNIT_START + i] = 1

    # sell_bench(i) → bench slot occupied
    for i, unit in enumerate(player.bench):
        if unit is not None:
            mask[ACTION_SELL_BENCH_START + i] = 1

    # sell_board(i) → board slot occupied
    for i, unit in enumerate(player.board):
        if unit is not None:
            mask[ACTION_SELL_BOARD_START + i] = 1

    # toggle_frontline deprecated: frontline/backline derived from unit.range

    # place_unit(b, d) — swap bench[b] with board[d]
    board_unit_count = sum(1 for u in player.board if u is not None)
    for b in range(9):
        if player.bench[b] is not None:
            for d in range(10):
                action_idx = b * 10 + d + ACTION_PLACE_UNIT_START
                if player.board[d] is None:
                    # Empty board slot: only legal if under level unit cap
                    if board_unit_count < player.level:
                        mask[action_idx] = 1
                else:
                    # Occupied board slot: swap always legal (cap unchanged)
                    mask[action_idx] = 1

    return mask
