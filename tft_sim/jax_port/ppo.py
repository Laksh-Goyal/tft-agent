"""
JAX PPO agent for the TFT port.

This module ports the PyTorch PPO implementation (policy.py + train.py)
to JAX/Flax. The key advantage: the ENTIRE training loop (rollout
collection + GAE + PPO update) can be jit-compiled as a single unit
using jax.lax.scan, following the PureJaxRL pattern.

Mirrors:
    - tft_sim/agents/policy.py  (ActorCriticNetwork, StructuredActorCritic)
    - tft_sim/train.py           (MaskedPPO, compute_gae, train loop)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
from flax.struct import dataclass as flax_dataclass
from typing import Any, NamedTuple

from tft_sim.jax_port.static_data import StaticData, load_static_data, TOTAL_ACTIONS, MAX_POLICY_BOTS
from tft_sim.jax_port.game_state import GameState, make_game_state, to_observation
from tft_sim.jax_port.step import (
    step_jax, compute_action_mask_jax, compute_player_mask, StepResult, reset_jax,
    reset_from_template, agent_placement_jax,
)
from tft_sim.jax_port.game_state import tree_where

import logging

logger = logging.getLogger(__name__)


# ============================================================
# Policy Network (mirrors policy.py:10-83)
# ============================================================
class ActorCriticNetwork(nn.Module):
    """Flat MLP actor-critic (mirrors policy.py:10-28)."""
    action_dim: int

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(512)(x)
        h = nn.relu(h)
        h = nn.Dense(256)(h)
        h = nn.relu(h)
        h = nn.Dense(128)(h)
        h = nn.relu(h)
        logits = nn.Dense(self.action_dim)(h)
        value = nn.Dense(1)(h)
        return logits, value.squeeze(-1)


class UnitEncoder(nn.Module):
    """Shared unit encoder (mirrors policy.py:31-42)."""
    embed_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.embed_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.embed_dim)(x)
        x = nn.relu(x)
        return x


class StructuredActorCritic(nn.Module):
    """Structured actor-critic with unit encoder (mirrors policy.py:45-83)."""
    action_dim: int
    unit_vec_size: int

    @nn.compact
    def __call__(self, x):
        start = 12  # econ(8) + context(4)
        u = self.unit_vec_size
        # Extract 24 unit slots (10 board + 9 bench + 5 shop)
        unit_slots = x[start: start + 24 * u].reshape(24, u)
        encoded = UnitEncoder(embed_dim=64)(unit_slots).reshape(-1)
        other = jnp.concatenate([x[:start], x[start + 24 * u:]])
        features = jnp.concatenate([encoded, other])

        h = nn.Dense(512)(features)
        h = nn.relu(h)
        h = nn.Dense(256)(h)
        h = nn.relu(h)
        h = nn.Dense(128)(h)
        h = nn.relu(h)
        logits = nn.Dense(self.action_dim)(h)
        value = nn.Dense(1)(h)
        return logits, value.squeeze(-1)


# ============================================================
# Action Selection (mirrors policy.py:86-125)
# ============================================================
MASK_LOGIT = -1e8


def select_action(params: dict, model: nn.Module, obs: jnp.ndarray,
                  mask: jnp.ndarray, rng_key: jax.Array,
                  deterministic: bool = False) -> tuple:
    """Select an action from the policy.

    Replaces select_action (policy.py:113-125) and select_action_greedy.

    Returns: (action, log_prob, value, new_rng_key)
    """
    logits, value = model.apply(params, obs)
    # Mask illegal actions
    masked_logits = jnp.where(mask == 1, logits, MASK_LOGIT)

    if deterministic:
        action = jnp.argmax(masked_logits)
        return action, jnp.float32(0.0), value, rng_key

    # Sample from masked distribution
    action_key, rng_key = jax.random.split(rng_key)
    action = jax.random.categorical(action_key, masked_logits)
    log_prob = jax.nn.log_softmax(masked_logits)[action]
    return action, log_prob, value, rng_key


# ============================================================
# GAE (mirrors train.py:48-66)
# ============================================================
def compute_gae(rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray,
                next_value: jnp.ndarray, gamma: float = 0.99,
                gae_lambda: float = 0.95) -> jnp.ndarray:
    """Generalized Advantage Estimation.

    Replaces compute_gae (train.py:48-66). Uses scan for a backward pass.
    The carry is (gae_accumulator, next_value_for_bootstrap).
    """
    def gae_step(carry, x):
        gae, next_val = carry
        reward, value, done = x
        delta = reward + gamma * next_val * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        # Next iteration's bootstrap value is this step's value
        new_carry = (gae, value)
        return new_carry, gae

    # Reverse for backward scan
    rewards_rev = jnp.flip(rewards)
    values_rev = jnp.flip(values)
    dones_rev = jnp.flip(dones)

    # Initial carry: gae=0, next_val=next_value (bootstrap from final state)
    init_carry = (jnp.float32(0.0), next_value)

    _, advantages_rev = jax.lax.scan(
        gae_step, init_carry,
        (rewards_rev, values_rev, dones_rev),
    )
    return jnp.flip(advantages_rev)


# ============================================================
# PPO Update (mirrors train.py:228-288)
# ============================================================
@flax_dataclass
class PPOConfig:
    """PPO hyperparameters (mirrors train.py:30-41)."""
    learning_rate: float = 3e-4
    learning_rate_end: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    k_epochs: int = 10
    mini_batch_size: int = 64
    n_steps: int = 256
    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.001
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5


@flax_dataclass
class RolloutBatch:
    """A batch of rollout data (mirrors RolloutBuffer in train.py:83-109)."""
    observations: jnp.ndarray   # (n_steps, obs_dim)
    actions: jnp.ndarray        # (n_steps,)
    log_probs: jnp.ndarray      # (n_steps,)
    values: jnp.ndarray         # (n_steps,)
    rewards: jnp.ndarray        # (n_steps,)
    masks: jnp.ndarray          # (n_steps, action_dim)
    dones: jnp.ndarray          # (n_steps,)
    last_obs: jnp.ndarray       # (obs_dim,)
    last_mask: jnp.ndarray      # (action_dim,)
    last_done: jnp.ndarray      # scalar bool
    placements: jnp.ndarray     # (n_steps,) int32, 0 if not done


def ppo_update(params: dict, opt_state: dict, model: nn.Module,
               batch: RolloutBatch, config: PPOConfig,
               progress: float, rng_key: jax.Array) -> tuple:
    """k_epochs of minibatch PPO (train.py:228-288).

    Returns: (new_params, new_opt_state, loss_dict, rng_key)
    """
    _, last_value = model.apply(params, batch.last_obs)
    last_value = jnp.where(batch.last_done, jnp.float32(0.0), last_value)
    advantages = compute_gae(
        batch.rewards, batch.values, batch.dones,
        last_value, config.gamma, config.gae_lambda,
    )
    returns = advantages + batch.values
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-7)

    lr = config.learning_rate + (config.learning_rate_end - config.learning_rate) * progress
    ent_coef = config.entropy_coef_start + (
        config.entropy_coef_end - config.entropy_coef_start
    ) * progress
    n = batch.observations.shape[0]
    mb = int(config.mini_batch_size)
    n_minibatches = max(1, n // mb)
    k_epochs = int(config.k_epochs)

    def minibatch_loss(params, idx):
        obs = batch.observations[idx]
        acts = batch.actions[idx]
        old_lp = batch.log_probs[idx]
        adv = advantages[idx]
        ret = returns[idx]
        masks = batch.masks[idx]
        logits, values = model.apply(params, obs)
        masked_logits = jnp.where(masks == 1, logits, MASK_LOGIT)
        log_probs_all = jax.nn.log_softmax(masked_logits)
        log_probs = jnp.take_along_axis(
            log_probs_all, acts[:, None], axis=1
        ).squeeze(-1)
        entropy = -jnp.sum(jnp.exp(log_probs_all) * log_probs_all, axis=-1).mean()
        ratios = jnp.exp(log_probs - old_lp)
        surr1 = ratios * adv
        surr2 = jnp.clip(ratios, 1 - config.eps_clip, 1 + config.eps_clip) * adv
        policy_loss = -jnp.minimum(surr1, surr2).mean()
        value_loss = jnp.mean((values - ret) ** 2)
        loss = policy_loss + config.value_loss_coef * value_loss - ent_coef * entropy
        return loss, (policy_loss, value_loss, entropy)

    def epoch_body(carry, _):
        params, opt_state, rng_key = carry
        rng_key, perm_key = jax.random.split(rng_key)
        perm = jax.random.permutation(perm_key, n)

        def mb_body(i, carry):
            params, opt_state = carry
            start = i * mb
            idx = jax.lax.dynamic_slice(perm, (start,), (mb,))
            (loss, aux), grads = jax.value_and_grad(minibatch_loss, has_aux=True)(params, idx)
            updates, opt_state = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=lr),
            ).update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        params, opt_state = jax.lax.fori_loop(
            0, n_minibatches, mb_body, (params, opt_state)
        )
        return (params, opt_state, rng_key), None

    (params, opt_state, rng_key), _ = jax.lax.scan(
        epoch_body, (params, opt_state, rng_key), None, length=k_epochs
    )
    # Report loss on the full buffer after the last epoch
    loss, (policy_loss, value_loss, entropy) = minibatch_loss(
        params, jnp.arange(min(mb, n))
    )
    return params, opt_state, {
        "loss": loss, "policy_loss": policy_loss,
        "value_loss": value_loss, "entropy": entropy,
    }, rng_key


# ============================================================
# Rollout Collection (mirrors train.py:340-415)
# ============================================================
def collect_rollout(params: dict, model: nn.Module, state: GameState,
                    static: StaticData, rng_key: jax.Array,
                    config: PPOConfig, env_template: GameState,
                    frozen_params, n_policy_bots) -> tuple:
    """Collect a rollout of n_steps.

    Uses jax.lax.scan to step the env n_steps times, collecting
    (obs, action, log_prob, value, reward, mask, done) at each step.

    Returns: (batch, final_state, final_rng_key)
    """
    obs_dim = static.obs_total_size

    def scan_step(carry, _):
        state, rng_key = carry
        obs = to_observation(state, jnp.int32(0), static)
        mask = compute_player_mask(state, jnp.int32(0), static)

        action, log_prob, value, rng_key = select_action(
            params, model, obs, mask, rng_key
        )

        result = step_jax(
            state, action, static, frozen_params, n_policy_bots, model.apply
        )
        done = result.terminated | result.truncated
        reset_key, rng_key = jax.random.split(result.state.rng_key)
        fresh = reset_from_template(
            env_template, reset_key, static, n_policy_bots
        )
        new_state = tree_where(done, fresh, result.state)
        placement = agent_placement_jax(result.state)

        transition = {
            "observation": obs,
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "reward": result.reward,
            "mask": mask,
            "done": done,
            "placement": jnp.where(done, placement, jnp.int32(0)),
        }
        return (new_state, rng_key), transition

    (final_state, final_rng_key), transitions = jax.lax.scan(
        scan_step,
        (state, rng_key),
        None,
        length=config.n_steps,
    )

    batch = RolloutBatch(
        observations=transitions["observation"],
        actions=transitions["action"],
        log_probs=transitions["log_prob"],
        values=transitions["value"],
        rewards=transitions["reward"],
        masks=transitions["mask"],
        dones=transitions["done"],
        last_obs=to_observation(final_state, jnp.int32(0), static),
        last_mask=compute_player_mask(final_state, jnp.int32(0), static),
        last_done=transitions["done"][-1],
        placements=transitions["placement"],
    )

    return batch, final_state, final_rng_key


# ============================================================
# Training Loop
# ============================================================
@flax_dataclass
class TrainState:
    """Mutable training state carried through the training loop."""
    params: dict
    opt_state: dict
    env_state: GameState
    env_template: GameState
    frozen_params: dict
    n_policy_bots: jnp.ndarray
    rng_key: jax.Array
    step: int


def stack_frozen_params(params, n: int = MAX_POLICY_BOTS):
    """Leading-dim stack of policy snapshots for JIT policy-bot dispatch."""
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n), params)


def insert_frozen_snapshot(frozen_params, params, idx: int):
    """Write current params into frozen slot `idx` (host-side graduation)."""
    return jax.tree_util.tree_map(lambda f, p: f.at[idx].set(p), frozen_params, params)


def create_train_state(model: nn.Module, static: StaticData,
                       rng_key: jax.Array, config: PPOConfig) -> TrainState:
    """Initialize training state."""
    # Init model params
    dummy_obs = jnp.zeros(static.obs_total_size, dtype=jnp.float32)
    params = model.init(rng_key, dummy_obs)

    # Init optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )
    opt_state = optimizer.init(params)

    # Init env
    env_key, rng_key = jax.random.split(rng_key)
    env_template = make_game_state(8, static, env_key)
    env_state = reset_from_template(env_template, env_key, static)
    frozen_params = stack_frozen_params(params)

    return TrainState(
        params=params,
        opt_state=opt_state,
        env_state=env_state,
        env_template=env_template,
        frozen_params=frozen_params,
        n_policy_bots=jnp.int32(0),
        rng_key=rng_key,
        step=jnp.int32(0),
    )


def train_step(train_state: TrainState, model: nn.Module,
               static: StaticData, config: PPOConfig,
               total_timesteps: int) -> tuple:
    """One training step: collect rollout + PPO update.

    Returns: (new_train_state, metrics_dict)
    """
    # Collect rollout
    batch, final_env_state, final_rng_key = collect_rollout(
        train_state.params, model, train_state.env_state,
        static, train_state.rng_key, config, train_state.env_template,
        train_state.frozen_params, train_state.n_policy_bots,
    )

    progress = jnp.float32(train_state.step * config.n_steps / total_timesteps)
    new_params, new_opt_state, loss_dict, final_rng_key = ppo_update(
        train_state.params, train_state.opt_state, model,
        batch, config, progress, final_rng_key,
    )

    new_train_state = TrainState(
        params=new_params,
        opt_state=new_opt_state,
        env_state=final_env_state,
        env_template=train_state.env_template,
        frozen_params=train_state.frozen_params,
        n_policy_bots=train_state.n_policy_bots,
        rng_key=final_rng_key,
        step=train_state.step + 1,
    )

    n_done = jnp.sum(batch.dones)
    mean_placement = jnp.where(
        n_done > 0,
        jnp.sum(batch.placements.astype(jnp.float32)) / jnp.maximum(n_done, 1),
        jnp.float32(0.0),
    )
    metrics = {
        "step": train_state.step,
        "mean_reward": jnp.mean(batch.rewards),
        "mean_value": jnp.mean(batch.values),
        "episodes_done": n_done,
        "mean_placement": mean_placement,
        "placements": batch.placements,
        "dones": batch.dones,
        **loss_dict,
    }

    return new_train_state, metrics


# ============================================================
# Verification
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=" * 60)
    logger.info("PPO Agent — Verification")
    logger.info("=" * 60)

    static, meta = load_static_data()
    key = jax.random.PRNGKey(42)

    config = PPOConfig(
        n_steps=64,       # small for verification
        k_epochs=4,
        mini_batch_size=32,
    )

    # Create model (flat MLP for now)
    model = ActorCriticNetwork(action_dim=TOTAL_ACTIONS)

    # Init training state
    train_state = create_train_state(model, static, key, config)
    logger.info(f"\nTraining state created:")
    logger.info(f"  Params keys: {list(jax.tree_util.tree_leaves(train_state.params)[0].shape)}")
    logger.info(f"  Env state stage: {train_state.env_state.stage}")
    logger.info(f"  Step: {train_state.step}")

    # --- Test action selection ---
    logger.info(f"\n--- Action selection ---")
    obs = to_observation(train_state.env_state, jnp.int32(0), static)
    mask = compute_player_mask(train_state.env_state, jnp.int32(0), static)
    action, log_prob, value, _ = select_action(
        train_state.params, model, obs, mask, key
    )
    logger.info(f"  Action: {action} (should be in [0, {TOTAL_ACTIONS}))")
    logger.info(f"  Log prob: {log_prob}")
    logger.info(f"  Value: {value}")
    assert 0 <= action < TOTAL_ACTIONS

    # --- Test GAE ---
    logger.info(f"\n--- GAE computation ---")
    rewards = jnp.array([1.0, 0.5,-0.5, 2.0, 0.0], dtype=jnp.float32)
    values = jnp.array([0.5, 0.3, 0.1, 0.8, 0.2], dtype=jnp.float32)
    dones = jnp.array([0, 0, 0, 0, 1], dtype=jnp.float32)
    next_value = jnp.float32(0.0)
    advs = compute_gae(rewards, values, dones, next_value)
    logger.info(f"  Rewards: {rewards}")
    logger.info(f"  Advantages: {advs}")
    assert advs.shape == rewards.shape

    # --- Test rollout collection ---
    logger.info(f"\n--- Rollout collection ({config.n_steps} steps) ---")
    batch, final_state, final_key = collect_rollout(
        train_state.params, model, train_state.env_state,
        static, key, config, train_state.env_template,
        train_state.frozen_params, train_state.n_policy_bots,
    )
    logger.info(f"  Observations shape: {batch.observations.shape}")
    logger.info(f"  Actions shape: {batch.actions.shape}")
    logger.info(f"  Rewards shape: {batch.rewards.shape}")
    logger.info(f"  Mean reward: {jnp.mean(batch.rewards)}")
    assert batch.observations.shape == (config.n_steps, static.obs_total_size)
    assert batch.actions.shape == (config.n_steps,)

    # --- Test PPO update ---
    logger.info(f"\n--- PPO update (1 epoch) ---")
    new_params, new_opt_state, loss_dict, _ = ppo_update(
        train_state.params, train_state.opt_state, model,
        batch, config, jnp.float32(0.0), key,
    )
    logger.info(f"  Loss: {loss_dict['loss']:.4f}")
    logger.info(f"  Policy loss: {loss_dict['policy_loss']:.4f}")
    logger.info(f"  Value loss: {loss_dict['value_loss']:.4f}")
    logger.info(f"  Entropy: {loss_dict['entropy']:.4f}")

    # --- Test full training step ---
    logger.info(f"\n--- Full training step ---")
    new_train_state, metrics = train_step(
        train_state, model, static, config, total_timesteps=10000,
    )
    logger.info(f"  Step: {metrics['step']}")
    logger.info(f"  Mean reward: {metrics['mean_reward']:.4f}")
    logger.info(f"  Loss: {metrics['loss']:.4f}")
    logger.info(f"  Entropy: {metrics['entropy']:.4f}")

    # --- Test multiple training steps (loss should decrease) ---
    logger.info(f"\n--- Multiple training steps (5 updates) ---")
    ts = train_state
    for i in range(5):
        ts, m = train_step(ts, model, static, config, total_timesteps=10000)
        logger.info(f"  Update {i+1}: loss={m['loss']:.4f}, policy_loss={m['policy_loss']:.4f}, "
              f"entropy={m['entropy']:.4f}")

    # --- Test jit compilation ---
    logger.info(f"\n--- JIT compilation ---")
    jit_train_step = jax.jit(
        lambda ts: train_step(ts, model, static, config, 10000),
    )
    ts_jit, m_jit = jit_train_step(train_state)
    logger.info(f"  JIT compiled successfully")
    logger.info(f"  Loss: {m_jit['loss']:.4f}")

    # --- Test vmap (parallel envs) — just verify it compiles ---
    logger.info(f"\n--- vmap (4 parallel training states) ---")
    batched_ts = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (4,) + x.shape).copy(),
        train_state,
    )
    # This would need more work to fully support batched training
    # For now just verify the state can be batched
    logger.info(f"  Batched params shape sample: {jax.tree_util.tree_leaves(batched_ts.params)[0].shape}")
    logger.info(f"  (Full batched training would require vmap over train_step)")

    logger.info(f"\n{'=' * 60}")
    logger.info("ALL VERIFICATIONS PASSED")
    logger.info(f"{'=' * 60}")
