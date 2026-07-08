"""Smoke test for eval_policy script."""
import pytest

torch = pytest.importorskip("torch")

from pathlib import Path

from tft_sim.env.tft_env import TFTEnv
from tft_sim.train import MaskedPPO, get_device


def test_eval_smoke(tmp_path):
    from scripts.eval_policy import run_eval

    env = TFTEnv(n_players=4)
    obs, _ = env.reset(seed=0)
    device = get_device()
    ppo = MaskedPPO(obs.shape[0], env.action_space.n, device)
    ckpt = tmp_path / "random.pt"
    ppo.save(ckpt)

    results = run_eval(ckpt, episodes=2, seed=99, n_players=4)
    assert results["episodes"] == 2
    assert "win_rate" in results
    assert "by_archetype" in results
