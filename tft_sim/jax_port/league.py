"""
Host-side self-play graduation for the JAX port.

Mirrors tft_sim/train.py LeagueManager. Frozen policy params live on
TrainState; this module only tracks win rate and decides when to snapshot.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from tft_sim.jax_port.static_data import (
    WIN_RATE_WINDOW, GRADUATION_THRESHOLD, MAX_POLICY_BOTS,
)

logger = logging.getLogger(__name__)


class LeagueManager:
    """Rolling win-rate tracker and policy-bot graduation (train.py:112-171)."""

    def __init__(self, save_dir: Path | None = None):
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.n_policy_bots = 0
        self.placement_window: list[int] = []

    def record_episode(self, placement: int) -> None:
        self.placement_window.append(int(placement))
        if len(self.placement_window) > WIN_RATE_WINDOW:
            self.placement_window.pop(0)

    def win_rate(self) -> float:
        if not self.placement_window:
            return 0.0
        return sum(1 for p in self.placement_window if p == 1) / len(self.placement_window)

    def maybe_graduate(self) -> bool:
        if self.n_policy_bots >= MAX_POLICY_BOTS:
            return False
        if len(self.placement_window) < WIN_RATE_WINDOW:
            return False
        if self.win_rate() <= GRADUATION_THRESHOLD:
            return False
        self.n_policy_bots += 1
        self.placement_window.clear()
        self.persist()
        logger.info("Graduated policy bot (total=%s)", self.n_policy_bots)
        return True

    def persist(self) -> None:
        if self.save_dir is None:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_bot_count": self.n_policy_bots,
            "win_rate_window": WIN_RATE_WINDOW,
            "graduation_threshold": GRADUATION_THRESHOLD,
        }
        (self.save_dir / "league.json").write_text(json.dumps(payload, indent=2))
