"""Frozen-policy opponents for self-play graduation."""
from __future__ import annotations

import copy

import numpy as np
import torch

from tft_sim.agents.policy import ActorCriticNetwork, select_action, select_action_greedy
from tft_sim.game.actions import ACTION_PASS, compute_action_mask


class PolicyBot:
    def __init__(
        self,
        policy: ActorCriticNetwork,
        device: torch.device,
        deterministic: bool = False,
    ):
        self.policy = policy
        self.device = device
        self.deterministic = deterministic
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad = False

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        arch: str = "flat_mlp",
        unit_vec_size: int = 0,
        deterministic: bool = False,
    ) -> "PolicyBot":
        if arch == "structured_v1":
            from tft_sim.agents.policy import StructuredActorCritic

            policy = StructuredActorCritic(state_dim, action_dim, unit_vec_size).to(device)
        else:
            policy = ActorCriticNetwork(state_dim, action_dim).to(device)
        policy.load_state_dict(state_dict)
        return cls(policy, device, deterministic=deterministic)

    def clone_frozen(self) -> "PolicyBot":
        from tft_sim.agents.policy import StructuredActorCritic

        if isinstance(self.policy, StructuredActorCritic):
            clone = StructuredActorCritic(
                self.policy.state_dim,
                self.policy.action_dim,
                self.policy.unit_vec_size,
            ).to(self.device)
        else:
            clone = ActorCriticNetwork(
                self.policy.state_dim,
                self.policy.action_dim,
            ).to(self.device)
        clone.load_state_dict(copy.deepcopy(self.policy.state_dict()))
        return PolicyBot(clone, self.device, deterministic=self.deterministic)

    def choose_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        if self.deterministic:
            return select_action_greedy(self.policy, obs, mask, self.device)
        action, _, _ = select_action(self.policy, obs, mask, self.device)
        return action


def run_policy_bot_planning_phase(game, policy_bots: list[PolicyBot]) -> None:
    """Run planning for policy-controlled opponent slots."""
    if game.round_type == "carousel":
        from tft_sim.agents.bot import pick_carousel_unit

        for player in game.players:
            if player.opponent_type != "policy" or player.is_eliminated:
                continue
            if game.carousel_picked.get(id(player), False):
                continue
            mask = compute_action_mask(player, game.unit_db, free_shop=True)
            action = pick_carousel_unit(mask)
            if action != ACTION_PASS:
                game.apply_action(action, player, count_agent_action=False)
        return

    budget = game.action_budget()
    for player in game.players:
        if player.opponent_type != "policy" or player.is_eliminated:
            continue
        bot = policy_bots[player.policy_bot_index]
        actions_taken = 0
        while actions_taken < budget:
            obs = game.to_observation_for(player)
            mask = compute_action_mask(player, game.unit_db)
            action = bot.choose_action(obs, mask)
            if action == ACTION_PASS:
                break
            game.apply_action(action, player, count_agent_action=False)
            actions_taken += 1
