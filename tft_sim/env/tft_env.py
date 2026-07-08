import gymnasium as gym
from gymnasium import spaces
import numpy as np

from tft_sim.env.state import GameState, max_rounds_in_stage
from tft_sim.env.action_space import get_action_space, ACTION_PASS
from tft_sim.env.metrics import agent_placement, placement_reward
from tft_sim.agents.bot import assign_bot_strategies, run_bot_planning_phase

class TFTEnv(gym.Env):
    def __init__(
        self,
        n_players=8,
        unit_roster_path="tft_sim/data/unit_roster.json",
        max_stage: int | None = None,
        curriculum_mode: str = "full",
    ):
        super(TFTEnv, self).__init__()
        self.n_players = n_players
        self.unit_roster_path = unit_roster_path
        self.max_stage = max_stage
        self.curriculum_mode = curriculum_mode
        
        self.action_space = get_action_space()
        
        # Initialize a dummy game to determine the observation space shape dynamically
        dummy_game = GameState(
            n_players=self.n_players,
            unit_roster_path=self.unit_roster_path,
            rng=np.random.default_rng(0),
        )
        obs_shape = dummy_game.to_observation().shape
        
        # We allow high=1.0 for normalized, but some could exceed slightly, so high=inf is safer
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.game = None
        self._policy_bots: list = []

    def set_policy_bots(self, policy_bots: list) -> None:
        """Assign frozen policy opponents (see agents/policy_bot.py)."""
        self._policy_bots = list(policy_bots)

    def _run_policy_bot_planning(self) -> None:
        if not self._policy_bots:
            return
        from tft_sim.agents.policy_bot import run_policy_bot_planning_phase
        run_policy_bot_planning_phase(self.game, self._policy_bots)

    def _should_curriculum_truncate(self) -> bool:
        if self.game is None:
            return False
        g = self.game
        if self.curriculum_mode == "stage1_economy":
            return g.stage == 1 and g.round_in_stage >= 4
        if self.max_stage is not None:
            return (
                g.stage == self.max_stage
                and g.round_in_stage >= max_rounds_in_stage(g.stage)
            )
        return False
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = GameState(
            n_players=self.n_players,
            unit_roster_path=self.unit_roster_path,
            rng=self.np_random,
        )
        assign_bot_strategies(self.game.players, self.game.rng)
        self._assign_policy_bot_slots()
        self.game.start_round()
        obs = self.game.to_observation()
        info = {"action_mask": self.game.action_mask()}
        return obs, info

    def _assign_policy_bot_slots(self) -> None:
        """Mark designated slots as policy opponents."""
        bot_idx = 0
        for player in self.game.players:
            if player.is_agent:
                continue
            if bot_idx < len(self._policy_bots):
                player.opponent_type = "policy"
                player.policy_bot_index = bot_idx
                bot_idx += 1
        
    def step(self, action):
        if self.game is None:
            raise RuntimeError("Must call reset() before step()")
            
        if action == ACTION_PASS or self.game.actions_this_round >= self.game.action_budget():
            # Agent done planning → opponents act → combat → next round
            run_bot_planning_phase(self.game)
            self._run_policy_bot_planning()
            reward = self.game.resolve_round()
            
            terminated = bool(self.game.agent.health <= 0)
            truncated = bool(self.game.is_last_player_standing())

            if not terminated and not truncated:
                if self._should_curriculum_truncate():
                    truncated = True
                else:
                    self.game.start_round()
        else:
            # Apply planning action
            self.game.apply_action(action, self.game.agent, count_agent_action=True)
            reward = self.game.action_reward(action)
            terminated = False
            truncated = False
            
        if terminated or truncated:
            reward += placement_reward(agent_placement(self.game))

        obs = self.game.to_observation()
        info = {"action_mask": self.game.action_mask()}
        
        return obs, reward, terminated, truncated, info
        
    def action_masks(self):
        """
        Required by sb3-contrib MaskablePPO.
        """
        if self.game is None:
            return np.ones(self.action_space.n, dtype=np.int8)
        return self.game.action_mask()
        
    def render(self):
        """
        Simple text-based render for debugging.
        """
        if self.game is None:
            return
        p = self.game.agent
        print(f"--- Stage {self.game.stage}-{self.game.round_in_stage} ({self.game.round_type}) ---")
        print(f"HP: {p.health} | Gold: {p.gold} | Lvl: {p.level} ({p.xp} XP) | Streak: {max(p.win_streak, p.loss_streak)}")
        board_units = [u.name if u else "Empty" for u in p.board]
        bench_units = [u.name if u else "Empty" for u in p.bench]
        print(f"Board: {board_units}")
        print(f"Bench: {bench_units}")
        print(f"Shop: {p.current_shop}")
