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
        self.state_dim = state_dim
        self.action_dim = action_dim
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


class UnitEncoder(nn.Module):
    def __init__(self, unit_vec_size: int, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(unit_vec_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class StructuredActorCritic(nn.Module):
    """Shared unit encoder over board/bench/shop slots + MLP trunk."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        unit_vec_size: int,
        hidden=(512, 256, 128),
        embed_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.unit_vec_size = unit_vec_size
        self.encoder = UnitEncoder(unit_vec_size, embed_dim)
        # 24 unit slots + remaining flat features (econ, context, synergies, opponents)
        flat_other = state_dim - 24 * unit_vec_size
        trunk_in = 24 * embed_dim + flat_other
        layers = []
        in_dim = trunk_in
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.actor = nn.Linear(in_dim, action_dim)
        self.critic = nn.Linear(in_dim, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        u = self.unit_vec_size
        # layout: econ(8) + context(4) + board(10u) + bench(9u) + shop(5u) + ...
        start = 12
        slots = x[:, start : start + 24 * u].reshape(-1, 24, u)
        encoded = self.encoder(slots).reshape(x.size(0), -1)
        other = torch.cat([x[:, :start], x[:, start + 24 * u :]], dim=1)
        features = self.backbone(torch.cat([encoded, other], dim=1))
        return self.actor(features), self.critic(features)


def masked_distribution(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Zero out illegal actions before softmax."""
    masked_logits = logits.clone()
    if masked_logits.dim() == 1:
        masked_logits = masked_logits.unsqueeze(0)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    masked_logits[mask == 0] = MASK_LOGIT
    if logits.dim() == 1:
        masked_logits = masked_logits.squeeze(0)
    return Categorical(logits=masked_logits)


def select_action_greedy(
    policy: nn.Module,
    state: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> int:
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device)
        logits, _ = policy(state_t)
        dist = masked_distribution(logits, mask_t)
        return dist.probs.argmax().item()


def select_action(
    policy: nn.Module,
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
