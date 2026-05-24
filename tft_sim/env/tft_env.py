import gymnasium as gym
from gymnasium import spaces
import numpy as np

from tft_sim.env.state import GameState
from tft_sim.env.action_space import get_action_space, ACTION_PASS

class TFTEnv(gym.Env):
    def __init__(self, n_players=8, unit_roster_path="tft_sim/data/unit_roster.json"):
        super(TFTEnv, self).__init__()
        self.n_players = n_players
        self.unit_roster_path = unit_roster_path
        
        self.action_space = get_action_space()
        
        # Initialize a dummy game to determine the observation space shape dynamically
        dummy_game = GameState(n_players=self.n_players, unit_roster_path=self.unit_roster_path)
        obs_shape = dummy_game.to_observation().shape
        
        # We allow high=1.0 for normalized, but some could exceed slightly, so high=inf is safer
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.game = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = GameState(n_players=self.n_players, unit_roster_path=self.unit_roster_path)
        self.game.start_round()
        obs = self.game.to_observation()
        info = {"action_mask": self.game.action_mask()}
        return obs, info
        
    def step(self, action):
        if self.game is None:
            raise RuntimeError("Must call reset() before step()")
            
        if action == ACTION_PASS or self.game.actions_this_round >= self.game.action_budget():
            # End planning phase, trigger combat and advance to next round
            reward = self.game.resolve_round()
            
            terminated = bool(self.game.agent.health <= 0)
            truncated = bool(self.game.is_last_player_standing())
            
            if not terminated and not truncated:
                self.game.start_round()
        else:
            # Apply planning action
            self.game.apply_action(action)
            reward = self.game.action_reward(action)
            terminated = False
            truncated = False
            
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
        print(f"--- Stage {self.game.stage}-{self.game.round_in_stage} ---")
        print(f"HP: {p.health} | Gold: {p.gold} | Lvl: {p.level} ({p.xp} XP) | Streak: {max(p.win_streak, p.loss_streak)}")
        board_units = [u.name if u else "Empty" for u in p.board]
        bench_units = [u.name if u else "Empty" for u in p.bench]
        print(f"Board: {board_units}")
        print(f"Bench: {bench_units}")
        print(f"Shop: {p.current_shop}")
