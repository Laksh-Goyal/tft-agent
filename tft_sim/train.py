"""
Masked PPO training entry point for TFTEnv.

Ported from CartPole PPO with action masking and TFT-sized MLP trunk.
"""
from __future__ import annotations

import argparse
import json
import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tft_sim.agents.policy import (
    ActorCriticNetwork,
    StructuredActorCritic,
    masked_distribution,
    select_action,
)
from tft_sim.agents.policy_bot import PolicyBot
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
ENTROPY_COEF_START = 0.01
ENTROPY_COEF_END = 0.001
VALUE_LOSS_COEF = 0.5
LEARNING_RATE_END = 1e-4
MAX_GRAD_NORM = 0.5
DEFAULT_TIMESTEPS = 500_000
WIN_RATE_WINDOW = 500
GRADUATION_THRESHOLD = 0.60
MAX_POLICY_BOTS = 3


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    next_value: float,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
) -> list[float]:
    """Generalized advantage estimation with optional bootstrap at rollout end."""
    advantages: list[float] = []
    gae = 0.0
    bootstrap = next_value
    for i in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[i])
        delta = rewards[i] + gamma * bootstrap * mask - values[i]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
        bootstrap = values[i]
    return advantages


def _schedule(progress: float, start: float, end: float) -> float:
    """Linear interpolate; progress in [0, 1]."""
    progress = max(0.0, min(1.0, progress))
    return start + (end - start) * progress


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
    last_obs: np.ndarray | None = None
    last_mask: np.ndarray | None = None
    last_done: bool = True

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.values.clear()
        self.rewards.clear()
        self.masks.clear()
        self.dones.clear()
        self.last_obs = None
        self.last_mask = None
        self.last_done = True

    def __len__(self):
        return len(self.states)


class LeagueManager:
    """Rolling win-rate tracker and policy-bot graduation (spec § Self-Play)."""

    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        arch: str = "flat_mlp",
        unit_vec_size: int = 0,
        save_dir: Path | None = None,
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.arch = arch
        self.unit_vec_size = unit_vec_size
        self.save_dir = save_dir
        self.policy_bots: list[PolicyBot] = []
        self.placement_window: list[int] = []

    def record_episode(self, metrics: dict) -> None:
        self.placement_window.append(metrics["placement"])
        if len(self.placement_window) > WIN_RATE_WINDOW:
            self.placement_window.pop(0)

    def win_rate(self) -> float:
        if not self.placement_window:
            return 0.0
        return sum(1 for p in self.placement_window if p == 1) / len(self.placement_window)

    def maybe_graduate(self, ppo: "MaskedPPO") -> bool:
        if len(self.policy_bots) >= MAX_POLICY_BOTS:
            return False
        if len(self.placement_window) < WIN_RATE_WINDOW:
            return False
        if self.win_rate() <= GRADUATION_THRESHOLD:
            return False
        bot = PolicyBot.from_state_dict(
            copy.deepcopy(ppo.policy.state_dict()),
            self.state_dim,
            self.action_dim,
            self.device,
            arch=self.arch,
            unit_vec_size=self.unit_vec_size,
        )
        self.policy_bots.append(bot)
        self.placement_window.clear()
        self._persist()
        return True

    def _persist(self) -> None:
        if self.save_dir is None:
            return
        league = {
            "policy_bot_count": len(self.policy_bots),
            "win_rate_window": WIN_RATE_WINDOW,
            "graduation_threshold": GRADUATION_THRESHOLD,
        }
        (self.save_dir / "league.json").write_text(json.dumps(league, indent=2))


class MaskedPPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        *,
        arch: str = "flat_mlp",
        unit_vec_size: int = 0,
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.arch = arch
        self.unit_vec_size = unit_vec_size
        if arch == "structured_v1":
            self.policy = StructuredActorCritic(
                state_dim, action_dim, unit_vec_size
            ).to(device)
        else:
            self.policy = ActorCriticNetwork(state_dim, action_dim).to(device)
        self.policy_old = copy.deepcopy(self.policy)
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
                "arch": self.arch,
                "unit_vec_size": self.unit_vec_size,
            },
            path,
        )

    def load(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.arch = checkpoint.get("arch", "flat_mlp")
        self.unit_vec_size = checkpoint.get("unit_vec_size", 0)
        if self.arch == "structured_v1":
            self.policy = StructuredActorCritic(
                self.state_dim, self.action_dim, self.unit_vec_size
            ).to(self.device)
            self.policy_old = StructuredActorCritic(
                self.state_dim, self.action_dim, self.unit_vec_size
            ).to(self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_old.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def update(self, buffer: RolloutBuffer, *, ent_coef: float = ENTROPY_COEF_START):
        if len(buffer) == 0:
            return

        old_states = torch.FloatTensor(np.array(buffer.states)).to(self.device)
        old_actions = torch.LongTensor(np.array(buffer.actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(buffer.logprobs)).to(self.device)
        old_values = torch.FloatTensor(np.array(buffer.values)).to(self.device)
        old_masks = torch.FloatTensor(np.array(buffer.masks)).to(self.device)

        if buffer.last_done or buffer.last_obs is None:
            next_value = 0.0
        else:
            with torch.no_grad():
                state_t = torch.as_tensor(
                    buffer.last_obs, dtype=torch.float32, device=self.device
                )
                _, v = self.policy_old(state_t)
                next_value = v.squeeze().item()

        adv_list = compute_gae(
            buffer.rewards, buffer.values, buffer.dones, next_value
        )
        advantages = torch.tensor(adv_list, dtype=torch.float32, device=self.device)
        returns = advantages + old_values
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
                    - ent_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


def _log_update(
    update_num: int,
    timesteps: int,
    episode_rewards: list[float],
    metrics_log: list[dict],
    window: int = 10,
    policy_bot_count: int = 0,
    win_rate: float = 0.0,
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
        "policy_bot_count": policy_bot_count,
        "win_rate": win_rate,
    }
    print(
        f"update {summary['update']} | steps {summary['timesteps']} | "
        f"avg_ep_reward({window})={summary['avg_ep_reward']:.3f} | "
        f"placement={summary['placement']:.1f} | "
        f"rounds={summary['rounds_survived']:.1f} | "
        f"board_power={summary['board_power']:.1f} | "
        f"policy_bots={policy_bot_count} | win_rate={win_rate:.2f}"
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
    curriculum: str = "full",
    curriculum_switch_updates: int | None = None,
    arch: str = "flat_mlp",
):
    device = get_device()
    curriculum_mode = "stage1_economy" if curriculum == "stage1" else "full"
    env = TFTEnv(n_players=n_players, curriculum_mode=curriculum_mode)
    obs, info = env.reset(seed=seed)
    state_dim = obs.shape[0]
    action_dim = env.action_space.n
    unit_vec_size = env.game.unit_vec_size if env.game else 0

    ppo = MaskedPPO(
        state_dim, action_dim, device, arch=arch, unit_vec_size=unit_vec_size
    )
    if resume_from is not None:
        ppo.load(resume_from)
        print(f"Resumed from {resume_from}")

    league = LeagueManager(
        device, state_dim, action_dim, arch=arch, unit_vec_size=unit_vec_size, save_dir=save_dir
    )
    env.set_policy_bots(league.policy_bots)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "total_timesteps": total_timesteps,
            "n_players": n_players,
            "seed": seed,
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "unit_vec_size": int(unit_vec_size),
            "arch": arch,
            "n_steps": N_STEPS,
            "learning_rate": LEARNING_RATE,
            "curriculum": curriculum,
            "curriculum_switch_updates": curriculum_switch_updates,
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
            m = episode_metrics(env.game)
            episode_rewards.append(ep_reward)
            metrics_log.append(m)
            league.record_episode(m)
            ep_reward = 0.0
            obs, info = env.reset()

        if len(buffer) >= N_STEPS:
            buffer.last_obs = obs.copy()
            buffer.last_mask = info["action_mask"].copy()
            buffer.last_done = done
            progress = timesteps / total_timesteps
            ent_coef = _schedule(progress, ENTROPY_COEF_START, ENTROPY_COEF_END)
            lr = _schedule(progress, LEARNING_RATE, LEARNING_RATE_END)
            for pg in ppo.optimizer.param_groups:
                pg["lr"] = lr
            ppo.update(buffer, ent_coef=ent_coef)
            buffer.clear()
            update_num += 1
            if (
                curriculum == "stage1"
                and curriculum_switch_updates is not None
                and update_num >= curriculum_switch_updates
                and env.curriculum_mode != "full"
            ):
                env.curriculum_mode = "full"
                print(f"  curriculum switched to full at update {update_num}")
            if league.maybe_graduate(ppo):
                env.set_policy_bots(league.policy_bots)
                print(f"  graduated policy bot (total={len(league.policy_bots)})")
            if update_num % log_interval == 0 and episode_rewards:
                update_summaries.append(
                    _log_update(
                        update_num,
                        timesteps,
                        episode_rewards,
                        metrics_log,
                        policy_bot_count=len(league.policy_bots),
                        win_rate=league.win_rate(),
                    )
                )
            if save_dir and update_num % checkpoint_interval == 0:
                ckpt = save_dir / f"checkpoint_{update_num:04d}.pt"
                ppo.save(ckpt)
                print(f"  saved {ckpt}")

    if len(buffer) > 0:
        buffer.last_obs = obs.copy()
        buffer.last_mask = info["action_mask"].copy()
        buffer.last_done = done
        progress = min(1.0, timesteps / total_timesteps)
        ent_coef = _schedule(progress, ENTROPY_COEF_START, ENTROPY_COEF_END)
        ppo.update(buffer, ent_coef=ent_coef)
        buffer.clear()
        update_num += 1
        if episode_rewards:
            update_summaries.append(
                _log_update(
                    update_num,
                    timesteps,
                    episode_rewards,
                    metrics_log,
                    policy_bot_count=len(league.policy_bots),
                    win_rate=league.win_rate(),
                )
            )

    if save_dir:
        ppo.save(save_dir / "final.pt")
        league._persist()
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
    parser.add_argument(
        "--curriculum",
        type=str,
        default="full",
        choices=["full", "stage1"],
        help="stage1 = economy-only episodes first",
    )
    parser.add_argument(
        "--curriculum-switch-updates",
        type=int,
        default=None,
        help="Switch from stage1 to full after N PPO updates",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="flat_mlp",
        choices=["flat_mlp", "structured_v1"],
        help="Policy architecture",
    )
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
        curriculum=args.curriculum,
        curriculum_switch_updates=args.curriculum_switch_updates,
        arch=args.arch,
    )
    print(
        f"Done: {stats['timesteps']} steps, {stats['updates']} updates, "
        f"{stats['episodes']} episodes"
    )


if __name__ == "__main__":
    main()
