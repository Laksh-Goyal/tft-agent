# tft-agent

A reinforcement-learning project for learning to play a stripped-down Teamfight Tactics (TFT)–style autobattler. The simulator and environment design are documented in [docs/tft_rl_spec.md](docs/tft_rl_spec.md).

## Project layout

```
tft_sim/
  game/          # Pure game engine (no RL dependencies)
    player.py    # Gold, health, board, bench, shop
    shop.py      # Shared pool and shop rolls
    combat.py    # DPS-based combat resolution
    units.py     # Roster load, star scaling, auto-combine
    traits.py    # Trait breakpoint bonuses
  env/           # Gymnasium wrapper
    tft_env.py   # TFTEnv (reset, step, action_masks)
    state.py     # GameState, observations, round loop
    action_space.py
  data/
    unit_roster.json
tests/           # Pytest unit and smoke tests
docs/
  tft_rl_spec.md
```

## Setup

Create and activate a virtual environment at the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running tests

From the repo root (with the venv active):

```bash
pytest -q
```

Tests cover combat, action masking, stage round limits, reproducible RNG seeding, uniform PvP damage, roster loading, and a minimal Gymnasium smoke check.

## Quick smoke check

```bash
python -c "
from tft_sim.env.tft_env import TFTEnv
env = TFTEnv(n_players=2)
obs, info = env.reset(seed=42)
obs, r, term, trunc, info = env.step(0)
print('obs shape:', obs.shape, 'reward:', r)
"
```

## Current status

**Implemented**

- Gymnasium environment with masked discrete actions (~127)
- Shop pool, leveling, auto-combine, DPS combat with traits
- Correctness fixes: uniform PvP damage, combat ties, stage-5 round cap, seeded RNG, illegal actions raise `ValueError`
- Placeholder roster: 30 units (10/8/6/4/2 by cost) and 10 traits (`Origin1`–`Origin4`, `Class1`–`Class6`) in [`tft_sim/data/unit_roster.json`](tft_sim/data/unit_roster.json) — names, stats, and effects are for you to replace

**Not yet implemented**

- Scripted opponent bots (opponents do not plan or build boards)
- Carousel / PvE creep round types (all rounds use PvP-style pairing today)
- RL training (`train.py`, MaskablePPO)
