import numpy as np
from typing import List, Dict, Optional, Tuple

from tft_sim.game.player import Player
from tft_sim.game.units import UnitDatabase, Unit, try_combine
from tft_sim.game.shop import PoolManager, roll_shop, reroll
from tft_sim.game.combat import resolve_combat
from tft_sim.game.traits import (
    compute_active_traits,
    trait_counts,
    detect_new_breakpoints,
    synergy_obs_pair,
)
from tft_sim.game.actions import (
    compute_action_mask, ACTION_PASS, ACTION_BUY_XP, ACTION_REROLL,
    ACTION_BUY_UNIT_START, ACTION_BUY_UNIT_END,
    ACTION_SELL_BENCH_START, ACTION_SELL_BENCH_END,
    ACTION_SELL_BOARD_START, ACTION_SELL_BOARD_END,
    ACTION_PLACE_UNIT_START, ACTION_PLACE_UNIT_END,
)
from tft_sim.game.trait_effects import BACKLINE_MIN_RANGE
from tft_sim.game.rounds import determine_round_type
from tft_sim.game.pve import (
    resolve_pve_round,
    resolve_carousel_round,
    setup_carousel_shop,
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
MAX_ABILITY_COEFF = 5.0

TRAIT_BREAKPOINT_REWARD = 0.05

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


def compute_obs_layout(n_traits: int, unit_vec_size: int, n_opponents: int = 7) -> dict:
    """Fixed slice indices for parsing flat observations."""
    econ = 8
    context = 4
    board = 10 * unit_vec_size
    bench = 9 * unit_vec_size
    shop = 5 * unit_vec_size
    synergies = n_traits * 2
    opponents = n_opponents * 34
    return {
        "econ_end": econ,
        "context_end": econ + context,
        "board_end": econ + context + board,
        "bench_end": econ + context + board + bench,
        "shop_end": econ + context + board + bench + shop,
        "synergy_end": econ + context + board + bench + shop + synergies,
        "total": econ + context + board + bench + shop + synergies + opponents,
        "unit_vec_size": unit_vec_size,
        "n_traits": n_traits,
    }


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
        self.rounds_completed = 0
        self._pending_action_reward = 0.0
        self.round_type = "pvp"
        self.carousel_options: list[int] = []
        self.carousel_picked: dict[int, bool] = {}

    def start_round(self):
        self.actions_this_round = 0
        
        max_rounds = max_rounds_in_stage(self.stage)
        if self.round_in_stage >= max_rounds:
            self.stage += 1
            self.round_in_stage = 1
        else:
            self.round_in_stage += 1

        self.round_type = determine_round_type(self.stage, self.round_in_stage)
        self.carousel_picked = {}
                
        if self.round_type == "carousel":
            setup_carousel_shop(self)
        else:
            for p in self.players:
                if p.is_eliminated:
                    continue
                
                # Passive XP
                if self.stage > 1 or self.round_in_stage > 1:
                    p.xp += 2
                    self._check_level_up(p)
                    
                p.gold += self._calculate_income(p)
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
        self.rounds_completed += 1
        if self.round_type == "carousel":
            return resolve_carousel_round(self)
        if self.round_type == "pve_creep":
            return resolve_pve_round(self)

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
        """Legal actions for the RL agent (alias used by TFTEnv / MaskablePPO)."""
        mask = compute_action_mask(self.agent, self.unit_db, free_shop=self.round_type == "carousel")
        if self.round_type == "carousel":
            mask[ACTION_BUY_XP] = 0
            mask[ACTION_REROLL] = 0
            agent_key = id(self.agent)
            if self.carousel_picked.get(agent_key, False):
                mask[:] = 0
                mask[ACTION_PASS] = 1
        return mask

    def apply_action(self, action: int, player: Player, *, count_agent_action: bool = False):
        """
        Apply one planning action for any player.
        count_agent_action: only True for the RL agent (increments round action budget).
        """
        if count_agent_action:
            self.actions_this_round += 1
            before_traits = trait_counts(player.board)
        mask = compute_action_mask(
            player, self.unit_db, free_shop=self.round_type == "carousel"
        )
        if action < 0 or action >= len(mask) or mask[action] == 0:
            raise ValueError(
                f"Illegal action {action} at stage {self.stage}-{self.round_in_stage}"
            )

        if action == ACTION_BUY_XP:
            player.gold -= 4
            player.xp += 4
            self._check_level_up(player)

        elif action == ACTION_REROLL:
            reroll(player, self.pool, self.unit_db, self.rng)

        elif ACTION_BUY_UNIT_START <= action <= ACTION_BUY_UNIT_END:
            idx = action - ACTION_BUY_UNIT_START
            unit_id = player.current_shop[idx]
            cost = 0 if self.round_type == "carousel" else self.unit_db.get_unit_base_data(unit_id)['cost']
            if self.round_type != "carousel":
                player.gold -= cost
            elif self.carousel_picked.get(id(player), False):
                raise ValueError("Already picked from carousel")

            unit = self.unit_db.create_unit(unit_id)
            for i in range(len(player.bench)):
                if player.bench[i] is None:
                    player.bench[i] = unit
                    break
            player.current_shop[idx] = None
            try_combine(player, unit_id, 1, self.unit_db)
            if self.round_type == "carousel":
                self.carousel_picked[id(player)] = True

        elif ACTION_SELL_BENCH_START <= action <= ACTION_SELL_BENCH_END:
            idx = action - ACTION_SELL_BENCH_START
            unit = player.bench[idx]
            player.gold += unit.cost
            self.pool.return_unit(unit.id, unit.cost)
            player.bench[idx] = None

        elif ACTION_SELL_BOARD_START <= action <= ACTION_SELL_BOARD_END:
            idx = action - ACTION_SELL_BOARD_START
            unit = player.board[idx]
            player.gold += unit.cost
            self.pool.return_unit(unit.id, unit.cost)
            player.board[idx] = None

        elif ACTION_PLACE_UNIT_START <= action <= ACTION_PLACE_UNIT_END:
            idx = action - ACTION_PLACE_UNIT_START
            b_idx = idx // 10
            d_idx = idx % 10
            temp = player.board[d_idx]
            player.board[d_idx] = player.bench[b_idx]
            player.bench[b_idx] = temp

        if count_agent_action:
            after_traits = trait_counts(player.board)
            new_bps = detect_new_breakpoints(
                before_traits, after_traits, self.unit_db.trait_data
            )
            self._pending_action_reward = TRAIT_BREAKPOINT_REWARD * len(new_bps)

    def action_reward(self, action: int) -> float:
        reward = self._pending_action_reward
        self._pending_action_reward = 0.0
        return reward

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
            unit.ability_damage / MAX_ABILITY_COEFF,
            unit.mana_cost / 150.0,
            unit.star_level / 3.0,
            1.0 if unit.range >= BACKLINE_MIN_RANGE else 0.0
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
        return self.to_observation_for(self.agent)

    def to_observation_for(self, player: Player) -> np.ndarray:
        p = player
        obs = []
        max_rounds = float(max_rounds_in_stage(self.stage))

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
        
        # Context (4): round-type flags + players alive
        is_carousel = 1.0 if self.round_type == "carousel" else 0.0
        is_pvp = 1.0 if self.round_type == "pvp" else 0.0
        is_pve = 1.0 if self.round_type == "pve_creep" else 0.0
        players_alive = sum(1 for pl in self.players if not pl.is_eliminated) / float(self.n_players)
        obs.extend([is_carousel, is_pvp, is_pve, players_alive])
        
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
        counts = trait_counts(p.board)
        for t in self.all_traits:
            c = counts.get(t, 0)
            cn, prog = synergy_obs_pair(t, c, self.unit_db.trait_data)
            obs.extend([cn, prog])
                
        # Opponents (7 * 34) — from this player's perspective, others are opponents
        for op in self.players:
            if op is p:
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
                        op_vec.extend([
                            u.id / 50.0,
                            u.star_level / 3.0,
                            1.0 if u.range >= BACKLINE_MIN_RANGE else 0.0,
                        ])
                obs.extend(op_vec)
                
        return np.array(obs, dtype=np.float32)
