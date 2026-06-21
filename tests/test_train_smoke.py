"""Smoke test: short PPO run completes without error."""
import pytest

torch = pytest.importorskip("torch")

from tft_sim.train import train


def test_train_smoke_run(tmp_path):
    stats = train(
        total_timesteps=512,
        n_players=4,
        seed=0,
        log_interval=999,
        save_dir=tmp_path,
    )
    assert stats["timesteps"] == 512
    assert stats["episodes"] >= 1
    assert (tmp_path / "final.pt").exists()
    assert (tmp_path / "history.json").exists()
