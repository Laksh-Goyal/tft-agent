"""Tests for frozen policy opponents."""
import pytest

torch = pytest.importorskip("torch")

from tft_sim.agents.policy_bot import PolicyBot
from tft_sim.env.tft_env import TFTEnv
from tft_sim.train import LeagueManager, MaskedPPO, get_device, WIN_RATE_WINDOW


def test_policy_bot_takes_legal_actions():
    env = TFTEnv(n_players=4)
    obs, _ = env.reset(seed=0)
    device = get_device()
    ppo = MaskedPPO(obs.shape[0], env.action_space.n, device)
    bot = PolicyBot.from_state_dict(
        ppo.policy.state_dict(),
        obs.shape[0],
        env.action_space.n,
        device,
    )
    mask = env.game.action_mask()
    action = bot.choose_action(obs, mask)
    assert mask[action] == 1


def test_league_graduation_threshold(tmp_path):
    device = get_device()
    env = TFTEnv(n_players=4)
    obs, _ = env.reset(seed=0)
    ppo = MaskedPPO(obs.shape[0], env.action_space.n, device)
    league = LeagueManager(
        device, obs.shape[0], env.action_space.n, save_dir=tmp_path
    )
    for _ in range(WIN_RATE_WINDOW):
        league.record_episode({"placement": 1})
    assert league.win_rate() == 1.0
    assert league.maybe_graduate(ppo)
    assert len(league.policy_bots) == 1
    assert (tmp_path / "league.json").exists()
