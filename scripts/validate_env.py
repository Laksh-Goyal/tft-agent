"""Masked random rollout validation (spec § Testing the Environment)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tft_sim.env.tft_env import TFTEnv


def sample_masked_action(rng: np.random.Generator, mask: np.ndarray) -> int:
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        raise RuntimeError("Empty action mask")
    return int(rng.choice(legal))


def run_masked_rollout(
    n_steps: int = 1000,
    seed: int = 0,
    n_players: int = 8,
) -> dict:
    env = TFTEnv(n_players=n_players)
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)

    rewards = []
    terminated_count = 0
    truncated_count = 0
    episodes = 0

    for step in range(n_steps):
        mask = info["action_mask"]
        action = sample_masked_action(rng, mask)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            episodes += 1
            if terminated:
                terminated_count += 1
            if truncated:
                truncated_count += 1
            obs, info = env.reset()

    return {
        "steps": n_steps,
        "episodes": episodes,
        "terminated": terminated_count,
        "truncated": truncated_count,
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "reward_mean": float(np.mean(rewards)),
        "obs_shape": obs.shape,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate TFTEnv with masked random policy")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-players", type=int, default=8)
    args = parser.parse_args()

    stats = run_masked_rollout(args.steps, args.seed, args.n_players)
    print(f"Masked rollout ({stats['steps']} steps, seed={args.seed})")
    print(f"  episodes finished: {stats['episodes']} "
          f"(terminated={stats['terminated']}, truncated={stats['truncated']})")
    print(f"  reward range: [{stats['reward_min']:.3f}, {stats['reward_max']:.3f}], "
          f"mean={stats['reward_mean']:.4f}")
    print(f"  obs shape: {stats['obs_shape']}")


if __name__ == "__main__":
    main()
