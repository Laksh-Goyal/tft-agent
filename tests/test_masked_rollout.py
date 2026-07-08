import numpy as np

from scripts.validate_env import run_masked_rollout


def test_masked_rollout_1000_steps_no_crash():
    stats = run_masked_rollout(n_steps=1000, seed=42, n_players=4)
    assert stats["steps"] == 1000
    assert stats["reward_min"] >= -1.5
    assert stats["reward_max"] <= 5.0
    assert len(stats["obs_shape"]) == 1
