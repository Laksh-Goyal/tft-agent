"""Heuristic scripted opponents with distinct economy/combat personalities."""
from typing import List, Optional, Tuple

import numpy as np

from tft_sim.game.actions import (
    ACTION_PASS,
    ACTION_BUY_XP,
    ACTION_REROLL,
    ACTION_BUY_UNIT_START,
    ACTION_BUY_UNIT_END,
    ACTION_PLACE_UNIT_START,
    compute_action_mask,
)
from tft_sim.game.player import Player
from tft_sim.game.units import Unit, UnitDatabase


ARCHETYPE_IDS = (
    "HyperBuyer",
    "InterestSaver",
    "Balanced",
    "LevelRusher",
    "Roller",
)


def unit_strength(unit: Unit) -> float:
    """Rough combat value for bot placement (cost-weighted stats)."""
    return unit.cost * 10 + unit.hp + unit.attack_damage


def board_unit_count(player: Player) -> int:
    return sum(1 for u in player.board if u is not None)


def bench_has_space(player: Player) -> bool:
    return any(u is None for u in player.bench)


def affordable_buys(
    mask: np.ndarray, player: Player, unit_db: UnitDatabase
) -> List[Tuple[int, int]]:
    """Return (action_index, cost) for legal shop purchases."""
    buys = []
    for i in range(ACTION_BUY_UNIT_END - ACTION_BUY_UNIT_START + 1):
        action = ACTION_BUY_UNIT_START + i
        if mask[action]:
            unit_id = player.current_shop[i]
            cost = unit_db.get_unit_base_data(unit_id)["cost"]
            buys.append((action, cost))
    return buys


def pick_cheapest_buy(buys: List[Tuple[int, int]]) -> Optional[int]:
    if not buys:
        return None
    return min(buys, key=lambda x: (x[1], x[0]))[0]


def pick_priciest_buy(buys: List[Tuple[int, int]]) -> Optional[int]:
    if not buys:
        return None
    return max(buys, key=lambda x: (x[1], x[0]))[0]


def first_place_on_empty_board(mask: np.ndarray, player: Player) -> Optional[int]:
    for b in range(9):
        if player.bench[b] is None:
            continue
        for d in range(10):
            action = ACTION_PLACE_UNIT_START + b * 10 + d
            if mask[action] and player.board[d] is None:
                return action
    return None


def best_place_strongest_on_empty(mask: np.ndarray, player: Player) -> Optional[int]:
    best_action = None
    best_strength = -1.0
    for b in range(9):
        unit = player.bench[b]
        if unit is None:
            continue
        strength = unit_strength(unit)
        for d in range(10):
            action = ACTION_PLACE_UNIT_START + b * 10 + d
            if mask[action] and player.board[d] is None and strength > best_strength:
                best_strength = strength
                best_action = action
    return best_action


def first_legal_place(mask: np.ndarray) -> Optional[int]:
    for action in range(ACTION_PLACE_UNIT_START, len(mask)):
        if mask[action]:
            return action
    return None


class HyperBuyer:
    """Spend on units aggressively; field strongest; never buy XP."""

    def choose_action(
        self, player: Player, mask: np.ndarray, unit_db: UnitDatabase
    ) -> int:
        buys = affordable_buys(mask, player, unit_db)
        if bench_has_space(player) and buys:
            action = pick_priciest_buy(buys)
            if action is not None:
                return action

        place = best_place_strongest_on_empty(mask, player)
        if place is not None:
            return place

        if mask[ACTION_REROLL]:
            return ACTION_REROLL

        return ACTION_PASS


class InterestSaver:
    """Hoard to INTEREST_CAP for max interest; only spend gold above that threshold."""
    INTEREST_CAP = 50  # matches max interest breakpoint (gold // 10, cap +5)
    REROLL_THRESHOLD = 60

    def choose_action(
        self, player: Player, mask: np.ndarray, unit_db: UnitDatabase
    ) -> int:
        if player.gold <= self.INTEREST_CAP:
            # Below cap: free rearrangement only (place/swap), no gold spend
            place = first_legal_place(mask)
            return place if place is not None else ACTION_PASS

        if board_unit_count(player) < player.level:
            buys = affordable_buys(mask, player, unit_db)
            action = pick_cheapest_buy(buys)
            if action is not None:
                return action

        if player.gold > self.REROLL_THRESHOLD and mask[ACTION_REROLL]:
            return ACTION_REROLL

        place = first_place_on_empty_board(mask, player)
        if place is not None:
            return place

        return ACTION_PASS


class Balanced:
    """Spec-style priority: buy → XP → reroll → place."""

    def choose_action(
        self, player: Player, mask: np.ndarray, unit_db: UnitDatabase
    ) -> int:
        buys = affordable_buys(mask, player, unit_db)
        if player.gold > 6 and bench_has_space(player) and buys:
            action = pick_cheapest_buy(buys)
            if action is not None:
                return action

        if player.level < 5 and player.gold > 8 and mask[ACTION_BUY_XP]:
            return ACTION_BUY_XP

        if player.gold > 10 and mask[ACTION_REROLL]:
            return ACTION_REROLL

        place = first_place_on_empty_board(mask, player)
        if place is not None:
            return place

        return ACTION_PASS


class LevelRusher:
    """Prioritize XP to widen board cap, then fill with units."""

    def choose_action(
        self, player: Player, mask: np.ndarray, unit_db: UnitDatabase
    ) -> int:
        if player.level < 7 and player.gold > 8 and mask[ACTION_BUY_XP]:
            return ACTION_BUY_XP

        buys = affordable_buys(mask, player, unit_db)
        if player.gold > 6 and board_unit_count(player) < player.level and buys:
            action = pick_cheapest_buy(buys)
            if action is not None:
                return action

        place = first_place_on_empty_board(mask, player)
        if place is not None:
            return place

        if (
            player.gold > 20
            and board_unit_count(player) < player.level
            and mask[ACTION_REROLL]
        ):
            return ACTION_REROLL

        return ACTION_PASS


class Roller:
    """Reroll-heavy; buy best affordable; place strongest from bench."""

    def choose_action(
        self, player: Player, mask: np.ndarray, unit_db: UnitDatabase
    ) -> int:
        if player.gold > 8 and mask[ACTION_REROLL]:
            return ACTION_REROLL

        buys = affordable_buys(mask, player, unit_db)
        if bench_has_space(player) and buys:
            action = pick_priciest_buy(buys)
            if action is not None:
                return action

        if player.level < 6 and player.gold > 12 and mask[ACTION_BUY_XP]:
            return ACTION_BUY_XP

        place = best_place_strongest_on_empty(mask, player)
        if place is not None:
            return place

        return ACTION_PASS


STRATEGY_REGISTRY = {
    "HyperBuyer": HyperBuyer(),
    "InterestSaver": InterestSaver(),
    "Balanced": Balanced(),
    "LevelRusher": LevelRusher(),
    "Roller": Roller(),
}


def assign_bot_strategies(players, rng) -> None:
    """Pick one archetype per opponent at game start (seeded, with replacement)."""
    for player in players:
        if player.is_agent:
            continue
        player.bot_strategy_id = rng.choice(list(ARCHETYPE_IDS))


def run_bot_planning_phase(game) -> None:
    """
    Run each opponent's planning loop until pass or action budget.
    Called by TFTEnv after the agent finishes planning, before combat.
    """
    budget = game.action_budget()
    for player in game.players:
        if player.is_agent or player.is_eliminated:
            continue
        strategy = STRATEGY_REGISTRY[player.bot_strategy_id]
        actions_taken = 0
        while actions_taken < budget:
            mask = compute_action_mask(player, game.unit_db)
            action = strategy.choose_action(player, mask, game.unit_db)
            if action == ACTION_PASS:
                break
            game.apply_action(action, player, count_agent_action=False)
            actions_taken += 1
