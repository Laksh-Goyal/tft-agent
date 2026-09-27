#!/usr/bin/env python3
"""Evaluate a JAX TFT policy checkpoint against scripted bots."""
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

import jax
import jax.numpy as jnp
import numpy as np

from tft_sim.jax_port.static_data import load_static_data, TOTAL_ACTIONS
from tft_sim.jax_port.ppo import (
    ActorCriticNetwork, StructuredActorCritic, PPOConfig,
    create_train_state, select_action,
)
from tft_sim.jax_port.checkpoint import restore_train_state
from tft_sim.jax_port.step import reset_jax, step_jax, compute_player_mask
from tft_sim.jax_port.game_state import to_observation


ARCHETYPE_NAMES = ("HyperBuyer", "InterestSaver", "Balanced", "LevelRusher", "Roller")


def _build_model(arch: str, static):
    if arch == "structured_v1":
        return StructuredActorCritic(
            action_dim=TOTAL_ACTIONS, unit_vec_size=static.unit_vec_size
        )
    return ActorCriticNetwork(action_dim=TOTAL_ACTIONS)


def board_power(state, static) -> int:
    ids = state.players.board_ids[0]
    valid = ids >= 0
    costs = jnp.take(static.unit_costs, jnp.maximum(ids, 0))
    return int(jnp.sum(jnp.where(valid, costs, 0)))


def run_eval(
    checkpoint: Path,
    episodes: int = 50,
    seed: int = 42,
    arch: str = "flat_mlp",
    deterministic: bool = False,
) -> dict:
    static, _ = load_static_data()
    model = _build_model(arch, static)
    config = PPOConfig(n_steps=8)
    dummy = create_train_state(model, static, jax.random.PRNGKey(0), config)
    ts = restore_train_state(checkpoint, dummy)
    params = ts.params

    rng = np.random.default_rng(seed)
    placements: list[int] = []
    rounds: list[int] = []
    powers: list[int] = []
    by_archetype: dict[str, list[int]] = defaultdict(list)

    jit_step = jax.jit(lambda s, a: step_jax(s, a, static))

    for _ in range(episodes):
        ep_key = jax.random.PRNGKey(int(rng.integers(0, 2**31)))
        state = reset_jax(static, ep_key)
        arch_ids = [int(x) for x in np.array(state.players.bot_strategy_id)[1:]]
        done = False
        act_key = jax.random.PRNGKey(int(rng.integers(0, 2**31)))
        while not done:
            obs = to_observation(state, jnp.int32(0), static)
            mask = compute_player_mask(state, jnp.int32(0), static)
            action, _, _, act_key = select_action(
                params, model, obs, mask, act_key, deterministic=deterministic
            )
            result = jit_step(state, action)
            state = result.state
            done = bool(result.terminated | result.truncated)
        from tft_sim.jax_port.step import agent_placement_jax
        place = int(agent_placement_jax(state))
        placements.append(place)
        rounds.append(int(state.rounds_completed))
        powers.append(board_power(state, static))
        for aid in set(arch_ids):
            by_archetype[ARCHETYPE_NAMES[aid]].append(place)

    wins = sum(1 for p in placements if p == 1)
    return {
        "episodes": episodes,
        "seed": seed,
        "deterministic": deterministic,
        "win_rate": wins / episodes,
        "avg_placement": mean(placements),
        "median_placement": median(placements),
        "avg_rounds_survived": mean(rounds),
        "avg_board_power": mean(powers),
        "by_archetype": {
            name: {
                "games": len(pls),
                "avg_placement": mean(pls),
                "win_rate": sum(1 for p in pls if p == 1) / len(pls),
            }
            for name, pls in sorted(by_archetype.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate JAX TFT checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arch", type=str, default="flat_mlp")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    results = run_eval(
        Path(args.checkpoint),
        episodes=args.episodes,
        seed=args.seed,
        arch=args.arch,
        deterministic=args.deterministic,
    )
    print(json.dumps(results, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
