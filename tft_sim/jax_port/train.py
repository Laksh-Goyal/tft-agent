"""
JAX PPO training entry point.

Usage:
    PYTHONPATH=. python -m tft_sim.jax_port.train --timesteps 10000 --save-dir runs/jax_smoke
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tft_sim.jax_port.static_data import load_static_data, TOTAL_ACTIONS
from tft_sim.jax_port.ppo import (
    ActorCriticNetwork, StructuredActorCritic, PPOConfig,
    create_train_state, train_step, insert_frozen_snapshot,
)
from tft_sim.jax_port.league import LeagueManager
from tft_sim.jax_port.checkpoint import save_train_state, restore_train_state

logger = logging.getLogger(__name__)


def _build_model(arch: str, static):
    if arch == "structured_v1":
        return StructuredActorCritic(
            action_dim=TOTAL_ACTIONS, unit_vec_size=static.unit_vec_size
        )
    return ActorCriticNetwork(action_dim=TOTAL_ACTIONS)


def _maybe_plot(history: dict, save_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_svg_curves(history, save_dir)
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(history["mean_reward"])
    axes[0, 0].set_title("mean reward")
    axes[0, 1].plot(history["loss"])
    axes[0, 1].set_title("loss")
    axes[1, 0].plot(history["mean_placement"])
    axes[1, 0].set_title("mean placement (rollout)")
    axes[1, 1].plot(history["entropy"])
    axes[1, 1].set_title("entropy")
    for ax in axes.ravel():
        ax.set_xlabel("update")
    fig.tight_layout()
    out = save_dir / "learning_curves.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Wrote %s", out)


def _write_svg_curves(history: dict, save_dir: Path) -> None:
    """Fallback curves when matplotlib is not installed."""
    keys = ["mean_reward", "loss", "mean_placement", "entropy"]
    width, height, pad = 320, 120, 10

    def polyline(vals: list) -> str:
        if not vals:
            return ""
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        n = max(len(vals) - 1, 1)
        pts = []
        for i, v in enumerate(vals):
            x = pad + (width - 2 * pad) * i / n
            y = height - pad - (height - 2 * pad) * (v - lo) / span
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    panels = []
    for i, key in enumerate(keys):
        x0 = (i % 2) * (width + 20)
        y0 = (i // 2) * (height + 30)
        panels.append(
            f'<g transform="translate({x0},{y0})">'
            f'<text x="{pad}" y="12" font-size="12">{key}</text>'
            f'<polyline fill="none" stroke="#333" stroke-width="1.5" '
            f'points="{polyline(history.get(key, []))}"/>'
            f'</g>'
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="680" height="290">'
        + "".join(panels) + "</svg>"
    )
    out = save_dir / "learning_curves.svg"
    out.write_text(svg)
    logger.info("Wrote %s (matplotlib not installed)", out)


def train(
    timesteps: int = 10_000,
    seed: int = 0,
    save_dir: Path | None = None,
    arch: str = "flat_mlp",
    checkpoint_interval: int = 10,
    resume: Path | None = None,
    n_steps: int = 256,
) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    static, _ = load_static_data()
    config = PPOConfig(n_steps=n_steps)
    model = _build_model(arch, static)
    rng = jax.random.PRNGKey(seed)
    train_state = create_train_state(model, static, rng, config)

    if resume is not None:
        train_state = restore_train_state(resume, train_state)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    league = LeagueManager(save_dir=save_dir)
    league.n_policy_bots = int(train_state.n_policy_bots)

    jit_step = jax.jit(lambda ts: train_step(ts, model, static, config, timesteps))

    n_updates = max(1, timesteps // config.n_steps)
    history = {
        "mean_reward": [], "loss": [], "policy_loss": [],
        "value_loss": [], "entropy": [], "mean_placement": [],
        "win_rate": [], "policy_bot_count": [],
    }

    logger.info("JAX train: %s updates, %s steps/update, arch=%s", n_updates, config.n_steps, arch)
    ts = train_state
    for update in range(1, n_updates + 1):
        ts, metrics = jit_step(ts)
        dones = np.asarray(metrics["dones"])
        placements = np.asarray(metrics["placements"])
        for done, place in zip(dones, placements):
            if done:
                league.record_episode(int(place))
        if league.maybe_graduate():
            ts = ts.replace(
                frozen_params=insert_frozen_snapshot(
                    ts.frozen_params, ts.params, league.n_policy_bots - 1
                ),
                n_policy_bots=jnp.int32(league.n_policy_bots),
            )
            jit_step = jax.jit(lambda ts: train_step(ts, model, static, config, timesteps))

        history["mean_reward"].append(float(metrics["mean_reward"]))
        history["loss"].append(float(metrics["loss"]))
        history["policy_loss"].append(float(metrics["policy_loss"]))
        history["value_loss"].append(float(metrics["value_loss"]))
        history["entropy"].append(float(metrics["entropy"]))
        history["mean_placement"].append(float(metrics["mean_placement"]))
        history["win_rate"].append(league.win_rate())
        history["policy_bot_count"].append(league.n_policy_bots)

        if update == 1 or update % 5 == 0 or update == n_updates:
            logger.info(
                "update %s/%s loss=%.4f reward=%.4f place=%.2f wr=%.2f bots=%s",
                update, n_updates, history["loss"][-1], history["mean_reward"][-1],
                history["mean_placement"][-1], history["win_rate"][-1],
                league.n_policy_bots,
            )

        if save_dir is not None and update % checkpoint_interval == 0:
            save_train_state(save_dir / f"checkpoint_{update:04d}", ts)

    if save_dir is not None:
        save_train_state(save_dir / "final", ts)
        (save_dir / "history.json").write_text(json.dumps(history))
        (save_dir / "config.json").write_text(json.dumps({
            "timesteps": timesteps, "seed": seed, "arch": arch,
            "n_steps": config.n_steps,
        }, indent=2))
        league.persist()
        _maybe_plot(history, save_dir)
        logger.info("Saved run artifacts to %s", save_dir)
    return history


def main():
    parser = argparse.ArgumentParser(description="Train JAX TFT PPO")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="runs/jax_smoke")
    parser.add_argument("--arch", type=str, default="flat_mlp",
                        choices=["flat_mlp", "structured_v1"])
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--n-steps", type=int, default=256)
    args = parser.parse_args()
    train(
        timesteps=args.timesteps,
        seed=args.seed,
        save_dir=Path(args.save_dir),
        arch=args.arch,
        checkpoint_interval=args.checkpoint_interval,
        resume=Path(args.resume) if args.resume else None,
        n_steps=args.n_steps,
    )


if __name__ == "__main__":
    main()
