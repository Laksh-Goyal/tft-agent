#!/usr/bin/env python3
"""Evaluate a trained policy checkpoint against scripted bots."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tft_sim.agents.policy import ActorCriticNetwork, select_action, select_action_greedy
from tft_sim.env.metrics import episode_metrics
from tft_sim.env.tft_env import TFTEnv
from tft_sim.train import get_device


def load_policy(checkpoint_path: Path, device: torch.device) -> ActorCriticNetwork:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "flat_mlp")
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    if arch == "structured_v1":
        from tft_sim.agents.policy import StructuredActorCritic

        policy = StructuredActorCritic(state_dim, action_dim, ckpt["unit_vec_size"]).to(device)
    else:
        policy = ActorCriticNetwork(state_dim, action_dim).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


def opponent_archetypes(game) -> list[str]:
    return [
        p.bot_strategy_id
        for p in game.players
        if not p.is_agent
        and getattr(p, "opponent_type", "scripted") == "scripted"
        and p.bot_strategy_id
    ]


def run_eval(
    checkpoint: Path,
    episodes: int = 100,
    seed: int = 42,
    n_players: int = 8,
    deterministic: bool = False,
) -> dict:
    device = get_device()
    policy = load_policy(checkpoint, device)
    env = TFTEnv(n_players=n_players)
    rng = np.random.default_rng(seed)

    placements: list[int] = []
    metrics_list: list[dict] = []
    by_archetype: dict[str, list[int]] = defaultdict(list)

    for ep in range(episodes):
        ep_seed = int(rng.integers(0, 2**31))
        obs, info = env.reset(seed=ep_seed)
        done = False
        while not done:
            mask = info["action_mask"]
            if deterministic:
                action = select_action_greedy(policy, obs, mask, device)
            else:
                action, _, _ = select_action(policy, obs, mask, device)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        m = episode_metrics(env.game)
        placements.append(m["placement"])
        metrics_list.append(m)
        for arch in set(opponent_archetypes(env.game)):
            by_archetype[arch].append(m["placement"])

    wins = sum(1 for p in placements if p == 1)
    archetype_summary = {
        arch: {
            "games": len(pls),
            "avg_placement": mean(pls),
            "win_rate": sum(1 for p in pls if p == 1) / len(pls),
        }
        for arch, pls in sorted(by_archetype.items())
    }

    return {
        "episodes": episodes,
        "seed": seed,
        "deterministic": deterministic,
        "win_rate": wins / episodes,
        "avg_placement": mean(placements),
        "median_placement": median(placements),
        "avg_rounds_survived": mean(m["rounds_survived"] for m in metrics_list),
        "avg_board_power": mean(m["board_power"] for m in metrics_list),
        "by_archetype": archetype_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TFT policy checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-players", type=int, default=8)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = run_eval(
        Path(args.checkpoint),
        episodes=args.episodes,
        seed=args.seed,
        n_players=args.n_players,
        deterministic=args.deterministic,
    )
    print(json.dumps(results, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
