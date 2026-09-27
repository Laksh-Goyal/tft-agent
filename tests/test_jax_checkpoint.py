"""Tests for Orbax TrainState checkpointing."""
import jax
import jax.numpy as jnp
import pytest

from tft_sim.jax_port.static_data import load_static_data, TOTAL_ACTIONS
from tft_sim.jax_port.ppo import ActorCriticNetwork, PPOConfig, create_train_state
from tft_sim.jax_port.checkpoint import save_train_state, restore_train_state


def test_save_restore_roundtrip(tmp_path):
    static, _ = load_static_data()
    model = ActorCriticNetwork(action_dim=TOTAL_ACTIONS)
    config = PPOConfig(n_steps=8, k_epochs=1, mini_batch_size=8)
    ts = create_train_state(model, static, jax.random.PRNGKey(0), config)
    path = tmp_path / "ckpt"
    save_train_state(path, ts)
    ts2 = create_train_state(model, static, jax.random.PRNGKey(1), config)
    restored = restore_train_state(path, ts2)
    orig = jax.tree_util.tree_leaves(ts.params)
    got = jax.tree_util.tree_leaves(restored.params)
    for a, b in zip(orig, got):
        assert jnp.allclose(a, b)
    assert int(restored.step) == int(ts.step)
    assert int(restored.n_policy_bots) == 0
