import numpy as np
from typing import List, Dict, Optional, Tuple

from tft_sim.game.player import Player
from tft_sim.game.units import UnitDatabase, Unit, try_combine
from tft_sim.game.shop import PoolManager, roll_shop, reroll
from tft_sim.game.combat import resolve_combat
from tft_sim.game.traits import compute_active_traits
from tft_sim.env.action_space import (
    compute_action_mask, ACTION_PASS, ACTION_BUY_XP, ACTION_REROLL,
    ACTION_BUY_UNIT_START, ACTION_BUY_UNIT_END,
    ACTION_SELL_BENCH_START, ACTION_SELL_BENCH_END,
    ACTION_SELL_BOARD_START, ACTION_SELL_BOARD_END,
    ACTION_TOGGLE_FRONTLINE_START, ACTION_TOGGLE_FRONTLINE_END,
    ACTION_PLACE_UNIT_START, ACTION_PLACE_UNIT_END
)

# Constants for observation normalization
MAX_GOLD = 50.0
START_HEALTH = 100.0
MAX_LEVEL = 9.0
MAX_STREAK = 5.0
MAX_STAGE = 7.0
MAX_HP = 3000.0
MAX_DAMAGE = 500.0
MAX_ARMOR = 200.0
MAX_AS = 3.0

def xp_required(level: int) -> int:
    reqs = {1: 0, 2: 2, 3: 6, 4: 10, 5: 20, 6: 36, 7: 56, 8: 80, 9: 100}
    return reqs.get(level, 100)

def max_rounds_in_stage(stage: int) -> int:
    """Round cap per stage (see docs/tft_rl_spec.md)."""
    if stage == 1:
        return 4
    if stage == 5:
        return 5
    return 6

class GameState:
    def __init__(
        self,
        n_players=8,
        unit_roster_path="tft_sim/data/unit_roster.json",
        rng: Optional[np.random.Generator] = None,
    ):
        self.n_players = n_players
        self.rng = rng if rng is not None else np.random.default_rng()
        self.unit_db = UnitDatabase(unit_roster_path)
        self.pool = PoolManager(self.unit_db)
        
        self.players = [Player() for _ in range(n_players)]
        self.agent = self.players[0]
        self.agent.is_agent = True
        
        # Determine all traits alphabetically for consistent multi-hot encoding
        self.all_traits = sorted(list(self.unit_db.trait_data.keys()))
        self.ability_types = ["damage", "heal", "shield", "cc"]
        self.unit_vec_size = 10 + len(self.ability_types) + len(self.all_traits)
        
        self.stage = 1
        self.round_in_stage = 0  # 1-indexed for rounds
        self.actions_this_round = 0

    def _max_rounds_in_stage(self, stage: int) -> int:
        return max_rounds_in_stage(stage)
        
    def start_round(self):
        self.actions_this_round = 0
        
        max_rounds = self._max_rounds_in_stage(self.stage)
        if self.round_in_stage >= max_rounds:
            self.stage += 1
            self.round_in_stage = 1
        else:
            self.round_in_stage += 1
                
        # Income and shop phase
        for p in self.players:
            if p.is_eliminated:
                continue
            
            # Passive XP
            if self.stage > 1 or self.round_in_stage > 1:  # Don't give xp on 1-1
                p.xp += 2
                self._check_level_up(p)
                
            # Income
            p.gold += self._calculate_income(p)
            
            # Return old shop and generate new
            self.pool.return_units(p.current_shop, self.unit_db)
            p.current_shop = roll_shop(p, self.pool, self.unit_db, self.rng)
            
    def _calculate_income(self, p: Player) -> int:
        if self.stage == 1 and self.round_in_stage <= 2:
            return 2  # special case for very early
        base = 5
        interest = min(5, p.gold // 10)
        streak = p.win_streak if p.win_streak > p.loss_streak else p.loss_streak
        if streak >= 5:
            sb = 3
        elif streak >= 3:
            sb = 2
        elif streak >= 2:
            sb = 1
        else:
            sb = 0
        return base + interest + sb
        
    def _check_level_up(self, p: Player):
        while p.level < 9 and p.xp >= xp_required(p.level + 1):
            p.level += 1
            
    def action_budget(self) -> int:
        if self.stage == 1:
            return 5
        elif self.stage == 2:
            return 10
        elif self.stage == 3:
            return 15
        elif self.stage == 4:
            return 20
        return 25

    def _apply_pvp_result(
        self, winner: str, p1: Player, p2: Player, damage: int
    ) -> Tuple[bool, bool]:
        """Apply PvP outcome with uniform damage for all players (no agent special-casing)."""
        agent_won = False
        agent_lost = False

        if winner == "A":
            p1.win_streak += 1
            p1.loss_streak = 0
            p2.health -= damage
            p2.win_streak = 0
            p2.loss_streak += 1
            if p1 == self.agent:
                agent_won = True
            if p2 == self.agent:
                agent_lost = True
        elif winner == "B":
            p2.win_streak += 1
            p2.loss_streak = 0
            p1.health -= damage
            p1.win_streak = 0
            p1.loss_streak += 1
            if p2 == self.agent:
                agent_won = True
            if p1 == self.agent:
                agent_lost = True
        # winner == "Tie": draw — no damage or streak changes

        return agent_won, agent_lost
        
    def resolve_round(self) -> float:
        active = [p for p in self.players if not p.is_eliminated]
        if len(active) <= 1:
            return 0.0

        order = list(active)
        self.rng.shuffle(order)
        matches = []
        for i in range(0, len(order) - 1, 2):
            matches.append((order[i], order[i + 1]))

        if len(order) % 2 == 1:
            ghost = order[self.rng.integers(0, len(order) - 1)]
            matches.append((order[-1], ghost))

        agent_won = False
        agent_lost = False

        for p1, p2 in matches:
            t1_traits = compute_active_traits(p1.board, self.unit_db.trait_data)
            t2_traits = compute_active_traits(p2.board, self.unit_db.trait_data)

            winner, surv_units = resolve_combat(p1.board, t1_traits, p2.board, t2_traits)
            damage = 2 + (self.stage * 1) + surv_units

            if winner in ("A", "B", "Tie"):
                won, lost = self._apply_pvp_result(winner, p1, p2, damage)
                agent_won = agent_won or won
                agent_lost = agent_lost or lost

        for p in active:
            if p.health <= 0:
                p.is_eliminated = True
                self.pool.return_units([u.id for u in p.board if u], self.unit_db)
                self.pool.return_units([u.id for u in p.bench if u], self.unit_db)

        reward = 0.0
        if agent_won:
            reward += 0.2
        if agent_lost:
            reward -= 0.1
        if self.agent.health <= 0:
            reward -= 1.0

        return reward
        
    def is_last_player_standing(self) -> bool:
        active = sum(1 for p in self.players if not p.is_eliminated)
        return active <= 1
        
    def action_mask(self) -> np.ndarray:
        return compute_action_mask(self.agent, self.unit_db)
        
    def apply_action(self, action: int):
        self.actions_this_round += 1
        p = self.agent
        mask = self.action_mask()
        if action < 0 or action >= len(mask) or mask[action] == 0:
            raise ValueError(
                f"Illegal action {action} at stage {self.stage}-{self.round_in_stage}"
            )

        if action == ACTION_BUY_XP:
            p.gold -= 4
            p.xp += 4
            self._check_level_up(p)

        elif action == ACTION_REROLL:
            reroll(p, self.pool, self.unit_db, self.rng)

        elif ACTION_BUY_UNIT_START <= action <= ACTION_BUY_UNIT_END:
            idx = action - ACTION_BUY_UNIT_START
            unit_id = p.current_shop[idx]
            cost = self.unit_db.get_unit_base_data(unit_id)['cost']
            p.gold -= cost

            unit = self.unit_db.create_unit(unit_id)
            for i in range(len(p.bench)):
                if p.bench[i] is None:
                    p.bench[i] = unit
                    break
            p.current_shop[idx] = None
            try_combine(p, unit_id, 1, self.unit_db)

        elif ACTION_SELL_BENCH_START <= action <= ACTION_SELL_BENCH_END:
            idx = action - ACTION_SELL_BENCH_START
            unit = p.bench[idx]
            p.gold += unit.cost
            self.pool.return_unit(unit.id, unit.cost)
            p.bench[idx] = None

        elif ACTION_SELL_BOARD_START <= action <= ACTION_SELL_BOARD_END:
            idx = action - ACTION_SELL_BOARD_START
            unit = p.board[idx]
            p.gold += unit.cost
            self.pool.return_unit(unit.id, unit.cost)
            p.board[idx] = None

        elif ACTION_TOGGLE_FRONTLINE_START <= action <= ACTION_TOGGLE_FRONTLINE_END:
            idx = action - ACTION_TOGGLE_FRONTLINE_START
            p.board[idx].is_frontline = not p.board[idx].is_frontline

        elif ACTION_PLACE_UNIT_START <= action <= ACTION_PLACE_UNIT_END:
            idx = action - ACTION_PLACE_UNIT_START
            b_idx = idx // 10
            d_idx = idx % 10
            temp = p.board[d_idx]
            p.board[d_idx] = p.bench[b_idx]
            p.bench[b_idx] = temp

    def action_reward(self, action: int) -> float:
        return 0.0

    def _build_unit_vector(self, unit: Optional[Unit]) -> List[float]:
        if unit is None:
            return [0.0] * self.unit_vec_size
            
        vec = [
            unit.hp / MAX_HP,
            unit.armor / MAX_ARMOR,
            unit.magic_resist / MAX_ARMOR,
            unit.attack_damage / MAX_DAMAGE,
            unit.attack_speed / MAX_AS,
            unit.range / 5.0,
            unit.ability_damage / MAX_DAMAGE,
            unit.mana_cost / 150.0,
            unit.star_level / 3.0,
            1.0 if unit.is_frontline else 0.0
        ]
        
        ab_type = [0.0] * len(self.ability_types)
        if unit.ability_type in self.ability_types:
            ab_type[self.ability_types.index(unit.ability_type)] = 1.0
        vec.extend(ab_type)
        
        trait_vec = [0.0] * len(self.all_traits)
        for t in unit.traits:
            if t in self.all_traits:
                trait_vec[self.all_traits.index(t)] = 1.0
        vec.extend(trait_vec)
        
        return vec
        
    def to_observation(self) -> np.ndarray:
        p = self.agent
        obs = []
        max_rounds = float(self._max_rounds_in_stage(self.stage))

        # Economy (8)
        obs.extend([
            p.gold / MAX_GOLD,
            p.health / START_HEALTH,
            p.level / MAX_LEVEL,
            (xp_required(p.level + 1) - p.xp) / 100.0,
            p.win_streak / MAX_STREAK,
            p.loss_streak / MAX_STREAK,
            self.stage / MAX_STAGE,
            self.round_in_stage / max_rounds
        ])
        
        # Context (3)
        is_pvp = 1.0 if self.stage > 1 or self.round_in_stage > 4 else 0.0
        is_pve = 1.0 - is_pvp
        players_alive = sum(1 for pl in self.players if not pl.is_eliminated) / float(self.n_players)
        obs.extend([is_pvp, is_pve, players_alive])
        
        # Board (10 * U)
        for u in p.board:
            obs.extend(self._build_unit_vector(u))
            
        # Bench (9 * U)
        for u in p.bench:
            obs.extend(self._build_unit_vector(u))
            
        # Shop (5 * U)
        for uid in p.current_shop:
            if uid is not None:
                dummy = self.unit_db.create_unit(uid)
                obs.extend(self._build_unit_vector(dummy))
            else:
                obs.extend(self._build_unit_vector(None))
                
        # Synergies (N * 2)
        active = compute_active_traits(p.board, self.unit_db.trait_data)
        for t in self.all_traits:
            if t in active:
                obs.extend([1.0, 1.0])
            else:
                obs.extend([0.0, 0.0])
                
        # Opponents (7 * 34)
        for op in self.players:
            if op == self.agent:
                continue
            if op.is_eliminated:
                obs.extend([0.0] * 34)
            else:
                op_vec = [
                    op.health / START_HEALTH,
                    op.level / MAX_LEVEL,
                    min(1.0, op.gold / MAX_GOLD),
                    max(op.win_streak, op.loss_streak) / MAX_STREAK
                ]
                for u in op.board:
                    if u is None:
                        op_vec.extend([0.0, 0.0, 0.0])
                    else:
                        op_vec.extend([u.id / 50.0, u.star_level / 3.0, 1.0 if u.is_frontline else 0.0])
                obs.extend(op_vec)
                
        return np.array(obs, dtype=np.float32)
