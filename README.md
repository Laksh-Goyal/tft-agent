# tft-agent

A reinforcement-learning project for learning to play a stripped-down Teamfight Tactics (TFT)–style autobattler. The simulator and environment design are documented in [docs/tft_rl_spec.md](docs/tft_rl_spec.md).

## Project layout

```
tft_sim/
  game/
    player.py, shop.py, combat.py, units.py, traits.py, trait_effects.py, actions.py
  agents/
    bot.py, policy.py
  env/
    tft_env.py, state.py, action_space.py, metrics.py
  train.py
scripts/
  validate_env.py
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

## Combat and roster semantics

### Positioning (range-based)

- `range` 1–2: **frontline** (targeted first in combat)
- `range` 3+: **backline** (protected until frontline is eliminated)
- `toggle_frontline` actions (IDs 27–36) are deprecated and always masked illegal

### Abilities (`ability_damage` field)

The JSON field `ability_damage` is an **ability coefficient**, not flat damage:

| `ability_type` | Resolved power |
|----------------|----------------|
| `damage`, `cc` | `attack_damage × coeff` |
| `heal`, `shield` | `hp × coeff` |

Star levels scale `hp` and `attack_damage` only; coefficients stay fixed so ability power grows with stars automatically.

### Trait effects in combat

| Effect key | Status |
|------------|--------|
| `hp_multiplier`, `ad_multiplier`, `ability_damage_multiplier`, `armor_multiplier`, `mr_multiplier`, `as_multiplier` | Applied to units with matching trait |
| `enemy_armor_reduction` (Void) | Applied to enemy team before effective HP |
| `hp_regen_per_sec` (Wildborn) | Bonus effective HP over estimated combat time |
| `crit_bonus` (Assassin) | Expected-value bonus on auto-attack DPS |
| `execute_threshold`, `execute_bonus_damage` (Slayer) | DPS multiplier when enemy HP is low after frontline phase |
| `ally_shield` (Sentinel) | Shield on lowest-HP ally (combat copy only) |
| `mana_per_sec` (Invoker) | Lowers effective mana cost for ability throughput |

## Scripted opponent bots

Seven opponents plan after the agent passes (or hits the action budget), using the same masked action space. Each opponent is assigned a random **archetype** on `reset(seed=…)` (reproducible with seed).

| Archetype | Playstyle |
|-----------|-----------|
| **HyperBuyer** | Spend on units aggressively, field strongest, never level |
| **InterestSaver** | Hoard to 50g for max interest; only spend gold above 50 |
| **Balanced** | Spec-style buy → XP → reroll → place |
| **LevelRusher** | Prioritize XP to widen board, then fill units |
| **Roller** | Reroll-heavy shop fishing, then buy and place |

Policy / frozen-weight opponents are reserved for later self-play graduation.

## Current status

**Implemented**

- Gymnasium env with masked actions (~127), shop, combine, traits, range-based combat, ability coefficients
- Full champion roster in [`tft_sim/data/unit_roster.json`](tft_sim/data/unit_roster.json) (30 units, 10 traits)
- Multi-strategy scripted bots ([`tft_sim/agents/bot.py`](tft_sim/agents/bot.py))
- Masked PPO training ([`tft_sim/train.py`](tft_sim/train.py), [`tft_sim/agents/policy.py`](tft_sim/agents/policy.py))
- Env validation script ([`scripts/validate_env.py`](scripts/validate_env.py))

**Not yet implemented**

- Carousel / PvE round types
- Placement / trait dense rewards, self-play graduation

## Validate and train

```bash
# Masked random rollout (~1000 steps)
python scripts/validate_env.py --steps 1000 --seed 0

# Short PPO smoke run (logs placement, rounds survived, board power)
python -m tft_sim.train --timesteps 10000 --seed 0 --save-dir runs/smoke

# Full training run (500k steps default, checkpoints every 10 updates)
python -m tft_sim.train --timesteps 500000 --seed 0 --save-dir runs/exp0

# Resume from checkpoint
python -m tft_sim.train --timesteps 500000 --save-dir runs/exp0 --resume runs/exp0/checkpoint_0020.pt
```
