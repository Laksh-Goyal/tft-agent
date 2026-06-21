"""
Masked PPO training entry point for TFTEnv.

Ported from CartPole PPO with action masking and TFT-sized MLP trunk.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tft_sim.agents.policy import ActorCriticNetwork, masked_distribution, select_action
from tft_sim.env.metrics import episode_metrics
from tft_sim.env.tft_env import TFTEnv

# --- Hyperparameters (spec starting point) ---
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 10
MINI_BATCH_SIZE = 64
N_STEPS = 2048
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
DEFAULT_TIMESTEPS = 500_000


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class RolloutBuffer:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.values.clear()
        self.rewards.clear()
        self.masks.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)


class MaskedPPO:
    def __init__(self, state_dim: int, action_dim: int, device: torch.device):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy = ActorCriticNetwork(state_dim, action_dim).to(device)
        self.policy_old = ActorCriticNetwork(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.mse = nn.MSELoss()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    def load(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_old.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def update(self, buffer: RolloutBuffer):
        if len(buffer) == 0:
            return

        old_states = torch.FloatTensor(np.array(buffer.states)).to(self.device)
        old_actions = torch.LongTensor(np.array(buffer.actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(buffer.logprobs)).to(self.device)
        old_values = torch.FloatTensor(np.array(buffer.values)).to(self.device)
        old_masks = torch.FloatTensor(np.array(buffer.masks)).to(self.device)

        advantages = []
        gae = 0.0
        next_value = 0.0
        for i in reversed(range(len(buffer.rewards))):
            mask = 1.0 - float(buffer.dones[i])
            delta = buffer.rewards[i] + GAMMA * next_value * mask - buffer.values[i]
            gae = delta + GAMMA * GAE_LAMBDA * mask * gae
            advantages.insert(0, gae)
            next_value = buffer.values[i]

        returns = torch.tensor(advantages, dtype=torch.float32, device=self.device) + old_values
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        dataset_size = old_states.size(0)
        for _ in range(K_EPOCHS):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, MINI_BATCH_SIZE):
                batch_idx = indices[start : start + MINI_BATCH_SIZE]
                batch_states = old_states[batch_idx]
                batch_actions = old_actions[batch_idx]
                batch_logprobs = old_logprobs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_masks = old_masks[batch_idx]

                logits, state_values = self.policy(batch_states)
                state_values = state_values.squeeze(-1)
                dist = masked_distribution(logits, batch_masks)
                logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy()

                ratios = torch.exp(logprobs - batch_logprobs.detach())
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * batch_advantages
                loss = (
                    -torch.min(surr1, surr2)
                    + VALUE_LOSS_COEF * self.mse(state_values, batch_returns)
                    - ENTROPY_COEF * entropy
                )

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


def _log_update(
    update_num: int,
    timesteps: int,
    episode_rewards: list[float],
    metrics_log: list[dict],
    window: int = 10,
) -> dict:
    recent_rewards = episode_rewards[-window:]
    recent_metrics = metrics_log[-window:]
    summary = {
        "update": update_num,
        "timesteps": timesteps,
        "avg_ep_reward": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
        "placement": float(np.mean([m["placement"] for m in recent_metrics])) if recent_metrics else 0.0,
        "rounds_survived": float(np.mean([m["rounds_survived"] for m in recent_metrics])) if recent_metrics else 0.0,
        "board_power": float(np.mean([m["board_power"] for m in recent_metrics])) if recent_metrics else 0.0,
    }
    print(
        f"update {summary['update']} | steps {summary['timesteps']} | "
        f"avg_ep_reward({window})={summary['avg_ep_reward']:.3f} | "
        f"placement={summary['placement']:.1f} | "
        f"rounds={summary['rounds_survived']:.1f} | "
        f"board_power={summary['board_power']:.1f}"
    )
    return summary


def train(
    total_timesteps: int = DEFAULT_TIMESTEPS,
    n_players: int = 8,
    seed: int = 0,
    log_interval: int = 1,
    save_dir: Path | None = None,
    checkpoint_interval: int = 10,
    resume_from: Path | None = None,
):
    device = get_device()
    env = TFTEnv(n_players=n_players)
    obs, info = env.reset(seed=seed)
    state_dim = obs.shape[0]
    action_dim = env.action_space.n

    ppo = MaskedPPO(state_dim, action_dim, device)
    if resume_from is not None:
        ppo.load(resume_from)
        print(f"Resumed from {resume_from}")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "total_timesteps": total_timesteps,
            "n_players": n_players,
            "seed": seed,
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "n_steps": N_STEPS,
            "learning_rate": LEARNING_RATE,
        }
        (save_dir / "config.json").write_text(json.dumps(config, indent=2))

    buffer = RolloutBuffer()
    timesteps = 0
    update_num = 0
    episode_rewards: list[float] = []
    ep_reward = 0.0
    metrics_log: list[dict] = []
    update_summaries: list[dict] = []

    while timesteps < total_timesteps:
        mask = info["action_mask"]
        action, log_prob, value = select_action(ppo.policy_old, obs, mask, device)

        buffer.states.append(obs)
        buffer.actions.append(action)
        buffer.logprobs.append(log_prob)
        buffer.values.append(value)
        buffer.masks.append(mask.copy())

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.rewards.append(reward)
        buffer.dones.append(done)
        ep_reward += reward
        timesteps += 1

        if done:
            episode_rewards.append(ep_reward)
            metrics_log.append(episode_metrics(env.game))
            ep_reward = 0.0
            obs, info = env.reset()

        if len(buffer) >= N_STEPS:
            ppo.update(buffer)
            buffer.clear()
            update_num += 1
            if update_num % log_interval == 0 and episode_rewards:
                update_summaries.append(
                    _log_update(update_num, timesteps, episode_rewards, metrics_log)
                )
            if save_dir and update_num % checkpoint_interval == 0:
                ckpt = save_dir / f"checkpoint_{update_num:04d}.pt"
                ppo.save(ckpt)
                print(f"  saved {ckpt}")

    if len(buffer) > 0:
        ppo.update(buffer)
        buffer.clear()
        update_num += 1
        if episode_rewards:
            update_summaries.append(
                _log_update(update_num, timesteps, episode_rewards, metrics_log)
            )

    if save_dir:
        ppo.save(save_dir / "final.pt")
        history = {
            "episode_rewards": episode_rewards,
            "metrics_log": metrics_log,
            "update_summaries": update_summaries,
        }
        (save_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"Saved final checkpoint and history to {save_dir}")

    return {
        "timesteps": timesteps,
        "updates": update_num,
        "episodes": len(episode_rewards),
        "episode_rewards": episode_rewards,
        "metrics_log": metrics_log,
        "update_summaries": update_summaries,
    }


def main():
    parser = argparse.ArgumentParser(description="Train masked PPO on TFTEnv")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--n-players", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="runs/default")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt")
    args = parser.parse_args()

    print(f"Device: {get_device()}")
    stats = train(
        total_timesteps=args.timesteps,
        n_players=args.n_players,
        seed=args.seed,
        log_interval=args.log_interval,
        save_dir=Path(args.save_dir),
        checkpoint_interval=args.checkpoint_interval,
        resume_from=Path(args.resume) if args.resume else None,
    )
    print(
        f"Done: {stats['timesteps']} steps, {stats['updates']} updates, "
        f"{stats['episodes']} episodes"
    )


if __name__ == "__main__":
    main()
