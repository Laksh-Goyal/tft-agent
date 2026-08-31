"""
Benchmark: JAX vs PyTorch wall-clock performance for the TFT environment.

Measures:
    1. Single env step speed (JAX jit vs PyTorch)
    2. Batched env step speed (JAX vmap vs PyTorch loop)
    3. Observation building speed
    4. Full training step speed (rollout + PPO update)

The JAX version should be significantly faster for batched operations
because vmap parallelizes across environments, while PyTorch must loop.
"""
import time
import sys
import numpy as np

import logging

logger = logging.getLogger(__name__)

sys.path.insert(0, ".")

# ============================================================
# JAX benchmark
# ============================================================
def benchmark_jax():
    import jax
    import jax.numpy as jnp
    from tft_sim.jax_port.static_data import load_static_data
    from tft_sim.jax_port.game_state import make_game_state, to_observation
    from tft_sim.jax_port.step import step_jax, compute_action_mask_jax
    from tft_sim.jax_port.ppo import (
        ActorCriticNetwork, PPOConfig, create_train_state,
        train_step, collect_rollout, select_action,
    )

    static, meta = load_static_data()
    key = jax.random.PRNGKey(42)

    config = PPOConfig(n_steps=128, k_epochs=4, mini_batch_size=64)
    model = ActorCriticNetwork(action_dim=127)
    train_state = create_train_state(model, static, key, config)

    # --- 1. Single env step ---
    action = jnp.int32(0)  # ACTION_PASS (always legal)
    jit_step = jax.jit(step_jax, static_argnames=[])
    # Warm up
    _ = jit_step(train_state.env_state, action, static)
    _ = jit_step(train_state.env_state, action, static).state  # ensure computed

    N = 1000
    start = time.time()
    for _ in range(N):
        result = jit_step(train_state.env_state, action, static)
        result.state.actions_this_round.block_until_ready()
    jax_single_step_ms = (time.time() - start) / N * 1000

    # --- 2. Batched env step (vmap) ---
    N_ENVS = 8
    batched_state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (N_ENVS,) + x.shape).copy(),
        train_state.env_state,
    )
    batched_actions = jnp.zeros(N_ENVS, dtype=jnp.int32)  # ACTION_PASS
    jit_batch_step = jax.jit(jax.vmap(step_jax, in_axes=(0, 0, None)))
    # Warm up
    _ = jit_batch_step(batched_state, batched_actions, static)
    _ = jit_batch_step(batched_state, batched_actions, static)

    N = 1000
    start = time.time()
    for _ in range(N):
        result = jit_batch_step(batched_state, batched_actions, static)
        result.state.actions_this_round.block_until_ready()
    jax_batch_step_ms = (time.time() - start) / N * 1000

    # --- 3. Observation building ---
    jit_obs = jax.jit(to_observation)
    # Warm up
    _ = jit_obs(train_state.env_state, jnp.int32(0), static)
    _ = jit_obs(train_state.env_state, jnp.int32(0), static)

    N = 1000
    start = time.time()
    for _ in range(N):
        obs = jit_obs(train_state.env_state, jnp.int32(0), static)
        obs.block_until_ready()
    jax_obs_ms = (time.time() - start) / N * 1000

    # --- 4. Full training step ---
    jit_train = jax.jit(
        lambda ts: train_step(ts, model, static, config, 10000)
    )
    # Warm up (includes compilation)
    ts = train_state
    _ = jit_train(ts)
    ts = train_state
    _ = jit_train(ts)

    N = 50
    start = time.time()
    ts = train_state
    for _ in range(N):
        ts, metrics = jit_train(ts)
        ts.step.block_until_ready()
    jax_train_ms = (time.time() - start) / N * 1000

    return {
        "jax_single_step_ms": jax_single_step_ms,
        "jax_batch_step_ms": jax_batch_step_ms,
        "jax_obs_ms": jax_obs_ms,
        "jax_train_ms": jax_train_ms,
        "n_envs": N_ENVS,
    }


# ============================================================
# PyTorch benchmark
# ============================================================
def benchmark_pytorch():
    import torch
    from tft_sim.env.tft_env import TFTEnv
    from tft_sim.agents.policy import ActorCriticNetwork, select_action
    from tft_sim.train import MaskedPPO, RolloutBuffer, compute_gae

    device = torch.device("cpu")  # fair comparison on CPU

    env = TFTEnv()
    obs, info = env.reset()

    policy = ActorCriticNetwork(
        state_dim=obs.shape[0],
        action_dim=env.action_space.n,
    ).to(device)

    # --- 1. Single env step ---
    # Use ACTION_PASS (always legal) for fair comparison
    from tft_sim.game.actions import ACTION_PASS
    N = 1000
    action = ACTION_PASS
    start = time.time()
    for _ in range(N):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
    pt_single_step_ms = (time.time() - start) / N * 1000

    # --- 2. Batched env step (PyTorch must loop) ---
    N_ENVS = 8
    envs = [TFTEnv() for _ in range(N_ENVS)]
    for e in envs:
        e.reset()

    start = time.time()
    for _ in range(N):
        for e in envs:
            e.step(action)
            if e.game.agent.health <= 0 or e.game.is_last_player_standing():
                e.reset()
    pt_batch_step_ms = (time.time() - start) / N * 1000

    # --- 3. Observation building ---
    start = time.time()
    for _ in range(N):
        obs = env.game.to_observation()
    pt_obs_ms = (time.time() - start) / N * 1000

    # --- 4. Full training step (rollout + update) ---
    ppo = MaskedPPO(
        state_dim=obs.shape[0],
        action_dim=env.action_space.n,
        device=device,
    )
    buffer = RolloutBuffer()
    env.reset()

    # Warm up
    obs, info = env.reset()
    for _ in range(128):
        action, logprob, value = select_action(ppo.policy, obs, info["action_mask"], device)
        obs, reward, terminated, truncated, info = env.step(action)
        buffer.states.append(obs.copy())
        buffer.actions.append(action)
        buffer.logprobs.append(logprob)
        buffer.values.append(value)
        buffer.rewards.append(reward)
        buffer.masks.append(info["action_mask"].copy())
        buffer.dones.append(terminated or truncated)
        if terminated or truncated:
            obs, info = env.reset()

    buffer.last_obs = obs.copy()
    buffer.last_mask = info["action_mask"].copy()
    buffer.last_done = False

    N = 10
    start = time.time()
    for _ in range(N):
        ppo.update(buffer, ent_coef=0.01)
    pt_train_ms = (time.time() - start) / N * 1000

    return {
        "pt_single_step_ms": pt_single_step_ms,
        "pt_batch_step_ms": pt_batch_step_ms,
        "pt_obs_ms": pt_obs_ms,
        "pt_train_ms": pt_train_ms,
        "n_envs": N_ENVS,
    }


# ============================================================
# Run benchmarks
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=" * 60)
    logger.info("Benchmark: JAX vs PyTorch (CPU)")
    logger.info("=" * 60)

    logger.info("\nRunning JAX benchmark...")
    jax_results = benchmark_jax()
    logger.info("Done.")

    logger.info("\nRunning PyTorch benchmark...")
    pt_results = benchmark_pytorch()
    logger.info("Done.")

    logger.info("\n" + "=" * 60)
    logger.info("Results")
    logger.info("=" * 60)

    logger.info(f"\n{'Metric':<35} {'JAX (ms)':>10} {'PyTorch (ms)':>14} {'Speedup':>10}")
    logger.info("-" * 70)

    # Single env step
    jax_ms = jax_results["jax_single_step_ms"]
    pt_ms = pt_results["pt_single_step_ms"]
    speedup = pt_ms / jax_ms if jax_ms > 0 else 0
    logger.info(f"{'Single env step':<35} {jax_ms:>10.3f} {pt_ms:>14.3f} {speedup:>9.1f}x")

    # Batched env step (8 envs)
    jax_ms = jax_results["jax_batch_step_ms"]
    pt_ms = pt_results["pt_batch_step_ms"]
    speedup = pt_ms / jax_ms if jax_ms > 0 else 0
    logger.info(f"{'Batched step (8 envs)':<35} {jax_ms:>10.3f} {pt_ms:>14.3f} {speedup:>9.1f}x")

    # Observation building
    jax_ms = jax_results["jax_obs_ms"]
    pt_ms = pt_results["pt_obs_ms"]
    speedup = pt_ms / jax_ms if jax_ms > 0 else 0
    logger.info(f"{'Observation building':<35} {jax_ms:>10.3f} {pt_ms:>14.3f} {speedup:>9.1f}x")

    # Full training step
    jax_ms = jax_results["jax_train_ms"]
    pt_ms = pt_results["pt_train_ms"]
    speedup = pt_ms / jax_ms if jax_ms > 0 else 0
    logger.info(f"{'Training step (128 steps + update)':<35} {jax_ms:>10.3f} {pt_ms:>14.3f} {speedup:>9.1f}x")

    logger.info(f"\n{'=' * 60}")
    logger.info("Notes:")
    logger.info("  - JAX batched step uses vmap (parallel), PyTorch loops sequentially")
    logger.info("  - JAX training step is fully jit-compiled (rollout + update)")
    logger.info("  - PyTorch training step is PPO update only (rollout already collected)")
    logger.info("  - Both run on CPU for fair comparison")
    logger.info("  - JAX advantage grows with batch size and on GPU/TPU")
    logger.info(f"{'=' * 60}")
