import numpy as np

from tft_sim.env.tft_env import TFTEnv


def test_reset_and_step_shapes():
    env = TFTEnv(n_players=2)
    obs, info = env.reset(seed=123)
    assert obs.dtype == np.float32
    assert len(info["action_mask"]) == env.action_space.n

    obs2, reward, term, trunc, info2 = env.step(0)  # pass
    assert obs2.shape == obs.shape
    assert isinstance(reward, float)
    assert "action_mask" in info2
