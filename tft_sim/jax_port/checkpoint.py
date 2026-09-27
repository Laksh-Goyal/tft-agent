"""
Orbax save/restore for JAX TrainState.

Mirrors tft_sim/train.py MaskedPPO.save/load. League bot weights live
in the TrainState frozen_params pytree; league.json holds counts only.
"""
from __future__ import annotations

import logging
from pathlib import Path

import jax
import orbax.checkpoint as ocp

logger = logging.getLogger(__name__)


def save_train_state(path: Path | str, train_state) -> None:
    """Save TrainState pytree (params, opt, env, rng, step, frozen policies)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        import shutil
        shutil.rmtree(path)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path.resolve(), train_state)
    checkpointer.wait_until_finished()
    logger.info("Saved checkpoint to %s", path)


def restore_train_state(path: Path | str, target) -> object:
    """Restore TrainState into an identically-structured target pytree."""
    path = Path(path)
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(path.resolve(), target)
    logger.info("Restored checkpoint from %s", path)
    return restored
