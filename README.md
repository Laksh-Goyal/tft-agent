# tft-agent

A reinforcement-learning project for learning to play a stripped-down Teamfight Tactics (TFT)–style autobattler. The simulator and environment design are documented in [docs/tft_rl_spec.md](docs/tft_rl_spec.md).

## Project layout

```
tft_sim/
  game/
    player.py, shop.py, combat.py, units.py, traits.py, trait_effects.py, actions.py
    rounds.py, pve.py
  agents/
    bot.py, policy.py, policy_bot.py
  env/
    tft_env.py, state.py, action_space.py, metrics.py
  train.py
scripts/
  validate_env.py
  eval_policy.py
  data/
    unit_roster.json
tests/
docs/
  tft_rl_spec.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Training stack (current)

| Feature | Location |
|---------|----------|
| Terminal placement reward `(9 - place) * 0.5` | [`tft_sim/env/tft_env.py`](tft_sim/env/tft_env.py) |
| Trait breakpoint dense reward (+0.05) | [`tft_sim/env/state.py`](tft_sim/env/state.py) |
| Carousel / PvE / PvP round types | [`tft_sim/game/rounds.py`](tft_sim/game/rounds.py), [`pve.py`](tft_sim/game/pve.py) |
| Synergy obs `[count_norm, breakpoint_progress]` | [`GameState.to_observation()`](tft_sim/env/state.py) |
| Masked PPO + GAE bootstrap + grad clip | [`tft_sim/train.py`](tft_sim/train.py) |
| Self-play graduation (60% win rate → PolicyBot) | [`LeagueManager`](tft_sim/train.py), [`policy_bot.py`](tft_sim/agents/policy_bot.py) |
| Flat MLP or structured unit encoder | [`tft_sim/agents/policy.py`](tft_sim/agents/policy.py) |

## Validate, train, and evaluate

```bash
# Masked random rollout (~1000 steps)
python scripts/validate_env.py --steps 1000 --seed 0

# Short smoke run
python -m tft_sim.train --timesteps 10000 --seed 0 --save-dir runs/smoke

# Recommended full run (5M steps, stage-1 curriculum → full game, structured encoder)
python -m tft_sim.train \
  --timesteps 5000000 \
  --seed 0 \
  --save-dir runs/exp1 \
  --arch structured_v1 \
  --curriculum stage1 \
  --curriculum-switch-updates 50 \
  --checkpoint-interval 25

# Flat MLP baseline (faster, fewer params)
python -m tft_sim.train --timesteps 500000 --seed 0 --save-dir runs/exp0

# Resume from checkpoint
python -m tft_sim.train --timesteps 5000000 --save-dir runs/exp1 \
  --resume runs/exp1/checkpoint_0100.pt --arch structured_v1

# Evaluate held-out performance (win rate, placement, per-archetype breakdown)
python scripts/eval_policy.py --checkpoint runs/exp1/final.pt --episodes 500 --seed 42
python scripts/eval_policy.py --checkpoint runs/exp1/final.pt --episodes 200 --deterministic
```

Training logs include `placement`, `rounds_survived`, `board_power`, `policy_bot_count`, and rolling `win_rate`. League state is saved to `runs/<exp>/league.json` when policy bots graduate.

## Round types

| Type | When | Behavior |
|------|------|----------|
| `carousel` | Stage 1-1 | Free unit pick from shared pool; no combat |
| `pve_creep` | Stage 1 rounds 2–4, inter-stage round 1 | Fight neutral creeps; gold on win, no HP loss on loss |
| `pvp` | All other rounds | Paired combat with damage |

## Scripted opponent bots

Seven opponents plan after the agent passes (or hits the action budget), using the same masked action space. Each opponent is assigned a random **archetype** on `reset(seed=…)`.

| Archetype | Playstyle |
|-----------|-----------|
| **HyperBuyer** | Spend on units aggressively, field strongest, never level |
| **InterestSaver** | Hoard to 50g for max interest; only spend gold above 50 |
| **Balanced** | Spec-style buy → XP → reroll → place |
| **LevelRusher** | Prioritize XP to widen board, then fill units |
| **Roller** | Reroll-heavy shop fishing, then buy and place |

When rolling win rate exceeds **60%** over the last 500 games, one scripted slot is replaced by a frozen copy of the current policy (up to 3 policy bots).

## Combat and roster semantics

See [docs/tft_rl_spec.md](docs/tft_rl_spec.md) for ability coefficients, trait effects, and roster schema.
