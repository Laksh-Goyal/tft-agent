"""Masked actor-critic policy for TFT PPO."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

MASK_LOGIT = -1e8


class ActorCriticNetwork(nn.Module):
    """Shared trunk with separate actor/critic heads (spec: 512→256→128)."""

    def __init__(self, state_dim: int, action_dim: int, hidden=(512, 256, 128)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.actor = nn.Linear(in_dim, action_dim)
        self.critic = nn.Linear(in_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.actor(features), self.critic(features)


def masked_distribution(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Zero out illegal actions before softmax."""
    masked_logits = logits.clone()
    masked_logits[mask == 0] = MASK_LOGIT
    return Categorical(logits=masked_logits)


def select_action(
    policy: ActorCriticNetwork,
    state: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> tuple[int, float, float]:
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device)
        logits, value = policy(state_t)
        dist = masked_distribution(logits, mask_t)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.squeeze().item()
