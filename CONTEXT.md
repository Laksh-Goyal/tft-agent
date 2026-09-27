# CONTEXT.md — Read This First

> **STOP — new agents must read this file before starting any task.**
>
> This is the leading context document for the `tft-agent` repository. It records
> what has been accomplished, the current state of the code, the conventions that
> must be followed, and what remains to be done. Read it end to end, then read
> `RULES.md` (coding rules) and `docs/tft_rl_spec.md` (game/environment spec)
> before writing any code.
>
> Last updated: 2026-09-06. All 126 tests passing at time of writing.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Layout](#repository-layout)
3. [Implementation 1: PyTorch (Original, Complete)](#implementation-1-pytorch-original-complete)
4. [Implementation 2: JAX Port](#implementation-2-jax-port)
5. [Work Completed — Chronological Summary](#work-completed--chronological-summary)
6. [Bugs Found and Fixed (Lessons for New Agents)](#bugs-found-and-fixed-lessons-for-new-agents)
7. [Current State and Verification Status](#current-state-and-verification-status)
8. [What Remains To Be Done](#what-remains-to-be-done)
9. [Conventions and Rules](#conventions-and-rules)
10. [How to Run Things](#how-to-run-things)
11. [Gotchas](#gotchas)

---

## Project Overview

A reinforcement-learning environment for a stripped-down Teamfight Tactics
(TFT) autobattler. The game is an 8-player economy + combat loop: each round
you buy units from a shop, place them on a board, form trait synergies, then
fight another player's board. You lose HP when you lose fights; last player
standing wins. The RL agent controls player 0 and learns to manage gold,
build a board, and win combats.

The full design is documented in `docs/tft_rl_spec.md` (748 lines covering
game rules, state/action spaces, combat model, reward structure, bot system,
and training setup).

**The repo has two parallel implementations of the same game:**

| | PyTorch (original) | JAX (port) |
|---|---|---|
| Status | Complete, trainable, two training runs finished | Trainable; 8-player pairing in JAX |
| Style | Mutable Python objects, Gymnasium wrapper | Functional, immutable flax.struct PyTrees |
| Speed | Baseline | ~208x faster on batched env steps (vmap) |
| Training | Masked PPO, 5M-step run completed (runs/exp1) | PPO + league + Orbax; smoke/medium JAX run (runs/jax_exp0) |

The strategic direction: the JAX port is the future. It follows the PureJaxRL
pattern (entire training loop JIT-compiled as one unit) and exists because the
PyTorch version is too slow for the scale of training this project wants.

---

## Repository Layout

```
tft-agent/
├── CONTEXT.md                   # THIS FILE — read first
├── RULES.md                     # Coding rules (10 principles, mandatory)
├── README.md                    # User-facing docs, setup, benchmark numbers
├── docs/
│   └── tft_rl_spec.md           # Full game + RL environment spec (748 lines)
├── tft_sim/
│   ├── data/
│   │   └── unit_roster.json     # 30 units, 10 traits, 5 cost tiers (single source of truth)
│   ├── game/                    # PyTorch game engine (zero RL knowledge)
│   │   ├── player.py            # Player dataclass (gold, health, board, bench, shop)
│   │   ├── shop.py              # PoolManager: cost-tiered pool, roll_shop, reroll
│   │   ├── units.py             # UnitDatabase, Unit, star-level scaling (2★=1.8x, 3★=3.24x)
│   │   ├── traits.py            # Trait counting, breakpoint activation
│   │   ├── trait_effects.py     # All trait effect helpers (Void, Wildborn, Assassin, etc.)
│   │   ├── combat.py            # Analytical combat: team stats, time-to-kill, Slayer execute
│   │   ├── actions.py           # 127 discrete actions + action masking
│   │   ├── rounds.py            # Round type logic
│   │   └── pve.py               # PvE creep rounds, gold drops
│   ├── env/                     # PyTorch Gymnasium wrapper
│   │   ├── tft_env.py           # TFTEnv: reset/step/action_masks (sb3-contrib compatible)
│   │   ├── state.py             # GameState, observation builder (846-dim), resolve_round
│   │   ├── action_space.py      # Action space definition
│   │   └── metrics.py           # Placement reward (9 - place) * 0.5
│   ├── agents/                  # PyTorch agents
│   │   ├── bot.py               # 5 scripted archetypes (HyperBuyer, InterestSaver, Balanced, LevelRusher, Roller)
│   │   ├── policy.py            # ActorCriticNetwork (flat MLP), StructuredActorCritic (unit encoder)
│   │   └── policy_bot.py        # Frozen policy copy for self-play graduation
│   ├── train.py                 # Masked PPO + LeagueManager (self-play graduation)
│   └── jax_port/                # JAX port (the future)
│       ├── static_data.py       # All game data as JAX arrays (unit stats, traits, shop odds)
│       ├── game_state.py        # PlayerState, GameState as flax.struct PyTrees
│       ├── step.py              # Action mask, apply_action, step_jax, resolve_round_jax
│       ├── combat.py            # resolve_combat_jax: analytical combat, JIT/vmap compatible
│       ├── ppo.py               # Flax actor-critic, GAE via scan, PPO update, train_step
│       ├── bots.py              # 5 scripted archetypes, JIT action selection
│       ├── league.py            # Host-side LeagueManager (win-rate graduation)
│       ├── checkpoint.py        # Orbax save/restore of TrainState
│       ├── train.py             # JAX training CLI
│       └── benchmark.py         # JAX vs PyTorch wall-clock comparison
├── scripts/
│   ├── validate_env.py          # Masked random rollout validation (PyTorch env)
│   ├── eval_policy.py           # Evaluate PyTorch checkpoint
│   └── eval_policy_jax.py       # Evaluate JAX checkpoint vs scripted bots
├── tests/                       # 126 tests, all passing
├── tutorials/
│   └── 01_jax_fundamentals.py   # 10-lesson JAX tutorial grounded in TFT code
└── runs/
    ├── exp0/                    # Flat MLP baseline (500K steps), checkpoints + history.json
    └── exp1/                    # Structured encoder (5M steps), league.json (policy bots graduated)
```

---

## Implementation 1: PyTorch (Original, Complete)

This is the working, fully trainable version. Everything below is done and
tested.

### Game engine (`tft_sim/game/`)

- **`units.py`** — `UnitDatabase` loads `tft_sim/data/unit_roster.json`
  (30 units across 5 cost tiers, 10 traits). Creates `Unit` objects on demand.
  Star-level scaling: 2-star = 1.8x HP/AD, 3-star = 3.24x.
- **`traits.py`** — Counts trait occurrences on a board, finds the highest
  active breakpoint per trait (e.g. 3 Warlords activates the 3-breakpoint).
- **`trait_effects.py`** — Applies trait effects: ally stat multipliers,
  Void armor/MR reduction on enemies, Wildborn HP regen, Assassin crit bonus
  (Assassins only), Slayer execute below HP threshold, Invoker mana regen
  (Invokers only), Sentinel shield on lowest-HP ally.
- **`combat.py`** — Analytical combat model (not a tick simulation): computes
  aggregate team stats (frontline/backline HP with armor mitigation, total DPS
  from auto-attacks + abilities), estimates time-to-kill, applies Slayer
  execute, returns winner and surviving-unit count.
- **`actions.py`** — 127 discrete actions: pass, buy XP, reroll, buy unit
  (5 slots), sell bench (9), sell board (10), place unit (9x10 bench x board
  grid). Full action masking.
- **`shop.py`** — `PoolManager` with proper pool tracking: units are removed
  from the shared pool when bought and returned when sold. Cost-tiered pool
  sizes (45/30/25/18/10 per copy for costs 1-5).
- **`pve.py`** — Creep rounds: win grants gold (+2 stage 1, +3 later), loss
  deals no HP damage (non-eliminating).

### Environment (`tft_sim/env/`)

- **`state.py`** — `GameState` with the full round loop: `start_round()`
  (income, interest, streak bonuses, shop roll), `apply_action()`,
  `resolve_round()` (PvP matching, damage `2 + stage + survivors`,
  elimination), observation builder producing a fixed 846-dim float vector:
  economy (8) + context (4) + board (10 units x 24) + bench (9 x 24) +
  shop (5 x 24) + synergies (10 x 2) + opponents (7 x 34).
- **`tft_env.py`** — Gymnasium wrapper. `step()` runs the bot planning phase
  after the agent passes, resolves the round, adds placement reward at
  episode end. `action_masks()` for sb3-contrib MaskablePPO.
- **`metrics.py`** — Terminal placement reward `(9 - place) * 0.5`.

### Agents (`tft_sim/agents/`)

- **`bot.py`** — Five scripted opponent archetypes with distinct economies:
  HyperBuyer (aggressive spend), InterestSaver (hoard to 50g), Balanced
  (buy -> XP -> reroll -> place), LevelRusher (XP first), Roller (reroll-heavy).
  Each opponent gets a random archetype at `reset(seed=...)`.
- **`policy.py`** — `ActorCriticNetwork` (flat MLP 846 -> 512 -> 256 -> 128 ->
  127 actions + 1 value) and `StructuredActorCritic` (shared unit encoder over
  the 24 board/bench/shop slots before the trunk). Masked action selection.
- **`policy_bot.py`** — Frozen copy of the policy used as an opponent after
  graduation.

### Training (`tft_sim/train.py`)

- Masked PPO: GAE with bootstrap, grad clipping (0.5), entropy coefficient
  schedule (0.01 -> 0.001), LR schedule (3e-4 -> 1e-4), 10 epochs, clip 0.2.
- **`LeagueManager`** — self-play graduation: tracks rolling win rate over
  500 episodes; when it exceeds 60%, one scripted bot slot is replaced with a
  frozen copy of the current policy (up to 3 policy bots). League state
  persisted to `runs/<exp>/league.json`.
- Curriculum: `--curriculum stage1` trains economy-only episodes first, then
  switches to the full game after N updates.
- Checkpointing every N updates + `history.json` with episode rewards, metrics,
  and update summaries.

### Completed training runs

- `runs/exp0` — flat MLP baseline, 500K steps. 24 checkpoints + final.pt.
- `runs/exp1` — structured encoder, 5M steps, stage1 curriculum. League state
  shows policy bots graduated. 10 checkpoints + final.pt + history.json.

Evaluate a checkpoint:

```bash
python scripts/eval_policy.py --checkpoint runs/exp1/final.pt --episodes 500 --seed 42
```

---

## Implementation 2: JAX Port

Functional, immutable reimplementation following the PureJaxRL pattern. The
goal is a single JIT-compiled training loop using `jax.lax.scan` and `vmap`
for parallel environments.

### What's done

- **`static_data.py`** — All game data as JAX arrays, loaded once at startup:
  unit stats `(30, 8)`, unit traits multi-hot `(30, 10)`, unit ability type
  one-hot `(30, 4)`, dense trait effect matrix `(10 traits, 3 bps, 13 effect
  types)`, shop odds `(10, 6)`, pool sizes, XP table, action budgets. A
  `StaticData` NamedTuple is a pure PyTree of arrays (passes through jit as a
  traced argument, NOT `static_argnames`). `StaticMeta` holds Python-only
  metadata (unit/trait names) separately so it never enters jit.
- **`game_state.py`** — `PlayerState` and `GameState` as `flax.struct`
  dataclasses (PyTrees). Board/bench/shop are fixed-size int arrays of unit
  IDs (-1 = empty) plus star levels. All updates are functional via
  `.replace()` / `.at[].set()`. Observation builder verified to match the
  PyTorch 846-dim layout. `PoolState` with `reserve_from_pool` /
  `return_to_pool` / `roll_shop_jax` (level odds, remaining-pool sampling).
- **`step.py`** — `compute_action_mask_jax`, `apply_action_jax` (all 5 action
  types plus pool-aware reroll/sell, combine-on-buy, trait-breakpoint
  reward), `start_round_jax` (income, interest, streak, shop refresh,
  carousel), `resolve_round_jax` (8-player shuffle/pair/ghost PvP, PvE for
  every living player; stage/round advance lives in `start_round_jax`),
  `step_jax` (bots → combat → placement reward → next
  round or reset). JIT and vmap compatible. Episode auto-reset via
  `reset_from_template` (pool init is NumPy, so a host-built template is
  copied under jit).
- **`combat.py`** — `resolve_combat_jax`: full port of the analytical combat
  model. Cross-checked against PyTorch. 17 dedicated tests in
  `tests/test_jax_combat.py`.
- **`bots.py`** — Five archetypes as integer-ID `lax.switch` dispatch
  (HyperBuyer, InterestSaver, Balanced, LevelRusher, Roller). Sequential
  scan over opponents because they share the pool.
- **`ppo.py`** — Flax actor-critic, GAE via scan, PPO update with `k_epochs`
  and minibatches, `collect_rollout` with episode reset, frozen-policy
  opponent slots on `TrainState`.
- **`league.py`** — Host-side `LeagueManager`: 500-game window, graduate at
  >60% first-place rate, cap 3 frozen policies.
- **`checkpoint.py`** — Orbax `StandardCheckpointer` save/restore of
  `TrainState`. Call `wait_until_finished()` so async commits complete.
- **`train.py`** — CLI: `python -m tft_sim.jax_port.train`. Writes
  `history.json`, `league.json`, Orbax checkpoints, learning-curve SVG
  (PNG if matplotlib is installed).
- **`benchmark.py`** — Wall-clock comparison. Results (CPU, M-series Mac,
  measured before pool/bots were added — re-run after env changes):

  | Metric | JAX | PyTorch | Speedup |
  |--------|-----|---------|---------|
  | Single env step | 0.050 ms | 2.060 ms | 40.8x |
  | Batched step (8 envs) | 0.077 ms | 15.999 ms | 208.2x |
  | Observation building | 0.059 ms | 0.074 ms | 1.3x |
  | Training step (128 steps + update) | 7.441 ms | 40.387 ms | 5.4x |

- **`tutorials/01_jax_fundamentals.py`** — 10-lesson tutorial grounded in the
  TFT codebase.

---

## Work Completed — Chronological Summary

1. **PyTorch game engine built** — units, traits, combat, shop, actions,
   rounds, PvE. Commit history: "Initial TFT game environment build",
   "Environment space done", "Base done".
2. **Unit roster authored** — 30 units, 10 traits with multi-effect
   breakpoints, class-specific abilities. "Added unit roster as well as class
   specific abilities".
3. **Scripted opponents** — 5 bot archetypes with distinct economies.
   "Added agent personalities".
4. **PyTorch training complete** — Masked PPO + LeagueManager self-play
   graduation + curriculum. Two full runs (exp0, exp1). "Stage 1 of training
   complete", "Current state".
5. **JAX port built** (commit `b71d3d0`, "Add JAX port of TFT simulator for
   fast parallel RL training") — static data layer, functional game state,
   step function, combat resolution, PPO loop, benchmark, tutorial, README
   docs. This commit includes work done across several sessions:
   - Static data, game state, step, PPO, benchmark (initial port)
   - Combat resolution (`resolve_combat_jax`) written and debugged
   - Six real bugs found and fixed (see next section)
   - Combat integrated into `step_jax` via `resolve_round_jax`
   - 17-test JAX combat suite written
   - README updated (combat moved to "done", layout + feature table + verify
     command added)
6. **RULES.md created** — 10 coding rules that all future code must follow.
7. **JAX env completed to training-ready** — pool tracking, `start_round`
   economy, unit combining, scripted bots, episode reset, placement reward,
   LeagueManager, Orbax checkpoints, `k_epochs` minibatches, train CLI, JAX
   eval script. Smoke + medium run at `runs/jax_exp0` (1024 steps). 113 tests.
8. **Full 8-player pairing** — `resolve_round_jax` shuffles living players,
   pairs them, leftover fights a ghost (sequential damage), then eliminates
   anyone with HP ≤ 0. PvE grants creep gold to every living player. Placement
   is 1–8 (alive by health, then dead in index order). Last standing is
   `n_alive <= 1`. Tests in `tests/test_jax_pairing.py`.

---

## Bugs Found and Fixed (Lessons for New Agents)

These were real bugs found during the JAX combat port. Read them — they are
the traps this codebase has already sprung once.

1. **Multiplier effects defaulted to 0.0** — When no trait breakpoint was
   active, every stat multiplier read as 0.0, so `hp * 0.0 = 0` and every
   team appeared empty (all combats resolved as ties). Fix: multipliers
   (ad/ability/hp/armor/mr/as, execute_bonus) default to **1.0**; additive
   effects default to 0.0. See `_default_effects()` in `jax_port/combat.py`.
2. **Python `if` on traced values broke JIT** — `if void_a > 0:` on a traced
   array raises `TracerBoolConversionError` under jit. Fix: compute both
   branches unconditionally and select with `jnp.where`. Same for the
   Wildborn regen recalculation.
3. **Assassin crit applied to all units** — The crit bonus was broadcast to
   every unit instead of only units carrying the Assassin trait. Fix: find
   the Assassin trait index and mask per-unit via the trait multi-hot.
4. **Invoker mana reduction never applied** — `has_invoker` was hardcoded to
   all-False. Fix: same trait-index lookup and masking approach as Assassin.
5. **Multi-effect breakpoints silently dropped** — The static data loader
   stored only the FIRST effect per breakpoint (`break` after the first dict
   entry), so Assassin's `crit_bonus` and Slayer's `execute_bonus_damage`
   were lost. Fix: replaced `(n_traits, max_bps)` index/value arrays with a
   dense `trait_effect_matrix` of shape `(n_traits, max_bps, 13)`. This was
   found by an adversarial test, not by the original suite.
6. **Surviving-unit HP denominator wrong** — The surviving-units fraction
   divided by the wrong team's HP pool. Fix: divide by the winning team's
   total HP.
7. **`static_argnames=["static"]` fails** — `StaticData` is a NamedTuple of
   JAX arrays; it must be passed as a regular (traced) argument to jit, not
   declared static. Passing Python-level numpy conversions inside traced code
   (`np.asarray` on a tracer) also fails — use pure JAX ops.

---

## Current State and Verification Status

- **Git**: `CONTEXT.md` untracked; `README.md` has a pointer to it. Latest
  commit `b71d3d0` plus the JAX remaining-work implementation on the working
  tree. Remote: `origin/main`.
- **Tests**: 126 passed in ~30s (`PYTHONPATH=. python -m pytest tests/ -q`).
  - 65 PyTorch tests (unchanged)
  - 17 JAX combat tests
  - Pool, combine, bots, league, Orbax checkpoint roundtrip
  - 13 pairing tests (`tests/test_jax_pairing.py`)
- **Self-verification scripts**:
  `static_data.py`, `game_state.py`, `step.py`, `combat.py`, `ppo.py`,
  `benchmark.py` (run with `PYTHONPATH=. python tft_sim/jax_port/<module>.py`).
- **Environment**: Python 3.14.2 venv at `.venv` (note: `.python-version`
  says `tft-jax` pyenv 3.10.18 — the venv in use is newer and works). JAX
  0.11.0, Flax 0.12.8, Optax 0.2.8, torch 2.12.0, orbax-checkpoint installed.
- **JAX training artifacts** (gitignored under `runs/`):
  - `runs/jax_exp0` — 1024-step medium run, Orbax `final` + `history.json` +
    `learning_curves.svg`
- **Known remaining simplifications in the JAX port**:
  - Benchmark numbers predate pool + bot planning + 8-player pairing; re-run
    `benchmark.py` before quoting speedups for the current env.

---

## What Remains To Be Done

The original six-item JAX checklist and 8-player pairing are done. Still open:

1. **Long JAX training run** — Multi-million-step job (the PyTorch exp1
   analogue) now that pairing is in. `runs/jax_exp0` is a medium smoke, not a
   finished learning curve.

### Beyond the checklist (what would make this a strong RL project)

- **Ablation studies** — flat MLP vs structured encoder, stage1 curriculum vs
  full game from the start, entropy coefficient sweep.
- **Learned combat simulator** — replace the analytical combat model with a
  small neural net predicting win probability from board states.
- **Full self-play** — all 8 slots sharing a live policy (AlphaZero-style),
  as the endpoint after graduation.
- **Experiment tracking** — WandB or TensorBoard integration for learning
  curves.

---

## Conventions and Rules

**Read `RULES.md` — it is mandatory.** Summary (full text in the file):

1. KISS — simple solutions, no confusing tricks
2. DRY — reuse code, never copy-paste logic
3. Meaningful names — names must tell you what things do
4. Small functions — one job each
5. No magic numbers — named constants with clear labels
6. Early returns — handle errors/exit conditions first, avoid deep nesting
7. Comment the "why" — the code shows the how; comments explain choices
8. Follow style guides — standard formatting for the language
9. Test your code — automated tests before shipping
10. Leave code better — clean up messes step by step

**Project-specific conventions:**

- **Logging**: use `logging.getLogger(__name__)` with `logger.info(...)`,
  never `print()` in library code. The PyTorch `train.py` predates this rule
  and still uses `print` — do not copy that pattern into new code.
- **JAX port mirrors PyTorch**: every JAX function's docstring cites the
  PyTorch source it mirrors (file + line range). Keep this convention when
  porting more code.
- **Static data is read-only**: game data lives in
  `tft_sim/data/unit_roster.json` and is loaded into arrays at startup. Never
  hard-code unit-specific logic in the engine.
- **Immutability in the JAX port**: no mutation, no Python dicts inside jit
  boundaries, explicit `jax.random.PRNGKey` threading instead of global RNG
  state.
- **Every module gets a `__main__` verification block** with assertions and
  (where applicable) cross-checks against the PyTorch implementation.
- **Every new behavior gets tests** in `tests/`. JAX combat tests live in
  `tests/test_jax_combat.py`; follow that file's structure (pytest classes by
  category, module-scoped `static` fixture).

---

## How to Run Things

```bash
cd tft-agent
source .venv/bin/activate

# Full test suite (126 tests)
PYTHONPATH=. python -m pytest tests/ -q

# JAX port module self-verification (each should end with ALL VERIFICATIONS PASSED)
PYTHONPATH=. python tft_sim/jax_port/static_data.py
PYTHONPATH=. python tft_sim/jax_port/game_state.py
PYTHONPATH=. python tft_sim/jax_port/step.py
PYTHONPATH=. python tft_sim/jax_port/combat.py
PYTHONPATH=. python tft_sim/jax_port/ppo.py
PYTHONPATH=. python tft_sim/jax_port/benchmark.py

# JAX tutorial
PYTHONPATH=. python tutorials/01_jax_fundamentals.py

# PyTorch: validate env, train, evaluate
python scripts/validate_env.py --steps 1000 --seed 0
python -m tft_sim.train --timesteps 10000 --seed 0 --save-dir runs/smoke
python scripts/eval_policy.py --checkpoint runs/exp1/final.pt --episodes 500 --seed 42

# JAX train / resume / eval
PYTHONPATH=. python -m tft_sim.jax_port.train --timesteps 10000 --seed 0 \
  --save-dir runs/jax_smoke --n-steps 256
PYTHONPATH=. python -m tft_sim.jax_port.train --timesteps 10000 \
  --save-dir runs/jax_smoke --resume runs/jax_smoke/final
PYTHONPATH=. python scripts/eval_policy_jax.py --checkpoint runs/jax_exp0/final \
  --episodes 50 --seed 42

# PyTorch full run (as done for exp1)
python -m tft_sim.train --timesteps 5000000 --seed 0 --save-dir runs/exp1 \
  --arch structured_v1 --curriculum stage1 --curriculum-switch-updates 50 \
  --checkpoint-interval 25
```

---

## Gotchas

- **`PYTHONPATH=.` is required** for the JAX self-verification scripts and
  tests (imports are `tft_sim.*` from the repo root).
- **`~/.zshenv` has a bad assignment** (line 1) that prints a warning on
  every shell command. It is noise, not an error; ignore it.
- **`runs/` contains training artifacts** (checkpoints, history) — do not
  commit new run artifacts casually; they are large.
- **`StaticData` must never be passed as `static_argnames`** to `jax.jit` —
  it is a PyTree of arrays and must be traced (see bug #7).
- **Trait multipliers default to 1.0, not 0.0** — if you touch effect
  extraction, re-run `tests/test_jax_combat.py::TestTraitEffects` (see bug
  #1).
- **Reset under jit copies a host-built template** — `make_pool` uses NumPy
  and cannot run inside `jit`. `reset_from_template` copies `env_template`
  stored on `TrainState`.
- **Orbax saves are async** — always `wait_until_finished()` after
  `StandardCheckpointer.save` or the process can exit before the commit.
- **Two Python environments exist**: `.venv` (Python 3.14.2, has JAX +
  torch, currently used) and the pyenv `tft-jax` (3.10.18) named in
  `.python-version` and the README setup. Use `.venv`; it is verified
  working.
