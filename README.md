# tft-agent

A reinforcement-learning project for learning to play a stripped-down Teamfight Tactics (TFT)–style autobattler. The simulator and environment design are documented in [docs/tft_rl_spec.md](docs/tft_rl_spec.md).

The project has two implementations:

- **PyTorch (original):** Mutable, object-oriented environment with Gymnasium wrapper. Full combat resolution, bot planning, self-play graduation.
- **JAX (port):** Functional, immutable environment using flax.struct PyTrees. JIT-compiled with `vmap` for parallel environments. Follows the PureJaxRL pattern. 200x faster on batched env steps.

## Project layout

```
tft_sim/
  game/                          # PyTorch game logic (original)
    player.py, shop.py, combat.py, units.py, traits.py, trait_effects.py, actions.py
    rounds.py, pve.py
  agents/                        # PyTorch agents (original)
    bot.py, policy.py, policy_bot.py
  env/                           # PyTorch env (original)
    tft_env.py, state.py, action_space.py, metrics.py
  train.py                       # PyTorch PPO training (original)
  jax_port/                      # JAX port (new)
    static_data.py               # Unit roster, traits, constants as JAX arrays
    game_state.py                # PlayerState, GameState as flax.struct PyTrees
    step.py                      # Action mask, apply_action, env step, round resolution
    combat.py                    # Analytical combat resolution (team stats, traits, TTK)
    ppo.py                       # Flax actor-critic, GAE, PPO update, training loop
    benchmark.py                 # JAX vs PyTorch wall-clock comparison
scripts/
  validate_env.py
  eval_policy.py
  data/
    unit_roster.json
tests/
tutorials/
  01_jax_fundamentals.py         # 10-lesson JAX tutorial grounded in TFT code
docs/
  tft_rl_spec.md
```

## Setup

### PyTorch (original)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

### JAX port

```bash
pyenv virtualenv 3.10.18 tft-jax
pyenv local tft-jax
pip install "jax[cpu]" flax optax rlax orbax-checkpoint numpy gymnasium
# For benchmarking against PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify the JAX port:

```bash
PYTHONPATH=. python tft_sim/jax_port/static_data.py
PYTHONPATH=. python tft_sim/jax_port/game_state.py
PYTHONPATH=. python tft_sim/jax_port/step.py
PYTHONPATH=. python tft_sim/jax_port/combat.py
PYTHONPATH=. python tft_sim/jax_port/ppo.py
PYTHONPATH=. python tft_sim/jax_port/benchmark.py
```

Run the JAX fundamentals tutorial:

```bash
PYTHONPATH=. python tutorials/01_jax_fundamentals.py
```

## Training stack

### PyTorch (original)

| Feature | Location |
|---------|----------|
| Terminal placement reward `(9 - place) * 0.5` | [`tft_sim/env/tft_env.py`](tft_sim/env/tft_env.py) |
| Trait breakpoint dense reward (+0.05) | [`tft_sim/env/state.py`](tft_sim/env/state.py) |
| Carousel / PvE / PvP round types | [`tft_sim/game/rounds.py`](tft_sim/game/rounds.py), [`pve.py`](tft_sim/game/pve.py) |
| Synergy obs `[count_norm, breakpoint_progress]` | [`GameState.to_observation()`](tft_sim/env/state.py) |
| Masked PPO + GAE bootstrap + grad clip | [`tft_sim/train.py`](tft_sim/train.py) |
| Self-play graduation (60% win rate → PolicyBot) | [`LeagueManager`](tft_sim/train.py), [`policy_bot.py`](tft_sim/agents/policy_bot.py) |
| Flat MLP or structured unit encoder | [`tft_sim/agents/policy.py`](tft_sim/agents/policy.py) |

### JAX port

| Feature | Location |
|---------|----------|
| Static data as JAX arrays (unit roster, traits, shop odds) | [`tft_sim/jax_port/static_data.py`](tft_sim/jax_port/static_data.py) |
| Immutable game state (flax.struct PyTrees) | [`tft_sim/jax_port/game_state.py`](tft_sim/jax_port/game_state.py) |
| Action mask + apply_action (all 5 action types) | [`tft_sim/jax_port/step.py`](tft_sim/jax_port/step.py) |
| Analytical combat (team stats, trait effects, time-to-kill) | [`tft_sim/jax_port/combat.py`](tft_sim/jax_port/combat.py) |
| Flax actor-critic (flat MLP + structured encoder) | [`tft_sim/jax_port/ppo.py`](tft_sim/jax_port/ppo.py) |
| GAE via `jax.lax.scan` | [`tft_sim/jax_port/ppo.py`](tft_sim/jax_port/ppo.py) |
| PPO update with Optax (grad clip + Adam + LR schedule) | [`tft_sim/jax_port/ppo.py`](tft_sim/jax_port/ppo.py) |
| JIT-compiled rollout collection via `jax.lax.scan` | [`tft_sim/jax_port/ppo.py`](tft_sim/jax_port/ppo.py) |
| `vmap` parallel environments | [`tft_sim/jax_port/step.py`](tft_sim/jax_port/step.py) |

### Benchmark: JAX vs PyTorch (CPU)

| Metric | JAX | PyTorch | Speedup |
|--------|-----|---------|---------|
| Single env step | 0.050 ms | 2.060 ms | 40.8x |
| Batched step (8 envs) | 0.077 ms | 15.999 ms | 208.2x |
| Observation building | 0.059 ms | 0.074 ms | 1.3x |
| Training step (128 steps + update) | 7.441 ms | 40.387 ms | 5.4x |

The 208x speedup on batched env steps comes from `vmap` parallelizing 8 environments into a single JIT-compiled kernel, while PyTorch must loop sequentially.

## Validate, train, and evaluate (PyTorch)

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

## JAX port status

The JAX port is a work in progress. What's done:

- [x] Static data layer (unit roster, traits, action IDs as JAX arrays, verified against PyTorch)
- [x] Game state (PlayerState, GameState as flax.struct PyTrees, observation verified at 846 dims)
- [x] Step function (action mask, all 5 action types, round transition, JIT + vmap compatible)
- [x] Combat resolution (analytical team stats, trait effects, time-to-kill, cross-checked vs PyTorch)
- [x] PPO agent (Flax actor-critic, GAE via scan, PPO update, rollout collection, training loop)
- [x] Benchmark (JAX vs PyTorch wall-clock comparison)

What's next:

- [ ] Bot planning phase (scripted opponents in JAX)
- [ ] Self-play graduation (LeagueManager in JAX)
- [ ] Pool management (shop reroll with proper pool tracking)
- [ ] Unit combining (star level upgrades)
- [ ] Orbax checkpointing integration
- [ ] Full training run with learning curves

## Round types

| Type | When | Behavior |
|------|------|----------|
| `carousel` | Stage 1-1 | Free unit pick from shared pool; no combat |
| `pve_creep` | Stage 1 rounds 2-4, inter-stage round 1 | Fight neutral creeps; gold on win, no HP loss on loss |
| `pvp` | All other rounds | Paired combat with damage |

## Scripted opponent bots

Seven opponents plan after the agent passes (or hits the action budget), using the same masked action space. Each opponent is assigned a random **archetype** on `reset(seed=...)`.

| Archetype | Playstyle |
|-----------|-----------|
| **HyperBuyer** | Spend on units aggressively, field strongest, never level |
| **InterestSaver** | Hoard to 50g for max interest; only spend gold above 50 |
| **Balanced** | Spec-style buy -> XP -> reroll -> place |
| **LevelRusher** | Prioritize XP to widen board, then fill units |
| **Roller** | Reroll-heavy shop fishing, then buy and place |

When rolling win rate exceeds **60%** over the last 500 games, one scripted slot is replaced by a frozen copy of the current policy (up to 3 policy bots).

## JAX port architecture

The JAX port follows the PureJaxRL pattern: the entire training loop (rollout collection + PPO update) is JIT-compiled as a single unit using `jax.lax.scan`.

Key architectural differences from the PyTorch version:

| Concept | PyTorch | JAX |
|---------|---------|-----|
| State representation | Mutable Python objects (`Player`, `GameState`) | Immutable flax.struct PyTrees (`PlayerState`, `GameState`) |
| Unit data | `Unit` objects with attributes | Pre-computed JAX arrays, indexed by unit ID |
| Board/bench | `List[Optional[Unit]]` | `jnp.ndarray` of unit IDs + star levels |
| Action application | `player.board[idx] = unit` (mutation) | `state.replace(board_ids=...)` (functional update) |
| Control flow | Python `if/elif` | `jnp.where`, `jax.lax.cond`, `jax.lax.scan` |
| Randomness | `np.random.default_rng()` (global state) | Explicit `jax.random.PRNGKey` (split and passed) |
| Parallelism | Sequential loop over envs | `jax.vmap` (auto-batched) |
| Training loop | Python `for` loop | `jax.lax.scan` (compiled) |
| Logging | `print()` | `logging.getLogger(__name__).info()` |

## Combat and roster semantics

See [docs/tft_rl_spec.md](docs/tft_rl_spec.md) for ability coefficients, trait effects, and roster schema.
