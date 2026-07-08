"""Unit tests for GAE computation."""
from tft_sim.train import compute_gae, GAMMA, GAE_LAMBDA


def test_gae_terminal_episode_zero_bootstrap():
    rewards = [1.0, 1.0]
    values = [0.5, 0.5]
    dones = [False, True]
    adv = compute_gae(rewards, values, dones, next_value=0.0)
    assert len(adv) == 2
    # Last step: delta = 1 + 0 - 0.5 = 0.5
    assert abs(adv[1] - 0.5) < 1e-5


def test_gae_mid_episode_bootstrap():
    rewards = [0.0, 1.0]
    values = [0.2, 0.4]
    dones = [False, False]
    adv_no_boot = compute_gae(rewards, values, dones, next_value=0.0)
    adv_boot = compute_gae(rewards, values, dones, next_value=2.0)
    assert adv_boot[0] > adv_no_boot[0]


def test_gae_three_step_buffer():
    rewards = [0.0, 0.5, 1.0]
    values = [0.1, 0.2, 0.3]
    dones = [False, False, True]
    adv = compute_gae(rewards, values, dones, next_value=0.0, gamma=GAMMA, gae_lambda=GAE_LAMBDA)
    assert len(adv) == 3
    assert all(isinstance(a, float) for a in adv)
