# TFT Reinforcement Learning Simulator — Full Specification

> This document is a complete spec for building a stripped-down TFT simulator and RL training environment. It is intended to be fed to an AI coding agent. All design decisions are recorded here with rationale.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Game Rules & Stages](#game-rules--stages)
3. [State Space](#state-space)
4. [Action Space](#action-space)
5. [Combat Simulator](#combat-simulator)
6. [Reward Structure](#reward-structure)
7. [Environment Loop](#environment-loop)
8. [Shop & Pool System](#shop--pool-system)
9. [Player System](#player-system)
10. [Trait Engine](#trait-engine)
11. [Bot System & Self-Play](#bot-system--self-play)
12. [Unit Roster & Trait System](#unit-roster--trait-system)
13. [Training Setup](#training-setup)
14. [Implementation Notes](#implementation-notes)

---

## Project Structure

```
tft_sim/
├── env/
│   ├── tft_env.py            # Main Gym environment (step, reset, render)
│   ├── state.py              # GameState dataclass + observation builder
│   └── action_space.py       # Action definitions + masking logic
├── game/
│   ├── player.py             # Player class (gold, health, board, bench)
│   ├── shop.py               # Shop generation + pool manager
│   ├── combat.py             # DPS-based battle simulator
│   ├── units.py              # Unit stat lookup + auto-combine logic
│   └── traits.py             # Trait engine + synergy activation
├── data/
│   └── unit_roster.json      # All unit stats, traits, ability info (modular)
├── agents/
│   ├── bot.py                # Scripted bot agent
│   └── policy.py             # RL policy wrapper
└── train.py                  # RL training entry point
```

**Design principle:** The `game/` layer has zero knowledge of RL. It is a pure game engine. The `env/` layer wraps it into a Gymnasium-compatible interface. This separation makes the game engine testable independently and keeps the RL interface clean.

---

## Game Rules & Stages

### Stage Structure

```
Stage 1: Rounds 1–4     (PvE carousel + creep rounds, no PvP)
Stage 2: Rounds 1–6     (PvP begins round 2-1)
Stage 3: Rounds 1–6
Stage 4: Rounds 1–6
Stage 5: Rounds 1–5
Stage 6+: Until one player remains
```

### Round Types

| Type | Description |
|---|---|
| `carousel` | Stage 1 opener. All players pick a free unit. Encode as a special state flag; agent picks from available units. |
| `pve_creep` | Rounds 1-1 through 1-4, and occasional inter-stage rounds. Fight neutral creeps. On win, drop gold/items (no items in v1, just gold). |
| `pvp` | Standard round. Fight a randomly selected opponent's board. |

### PvE Gold Drops (Creep Rounds)
- Give every player a flat **gold bonus** on completing a creep round (e.g. +3 gold).
- This simulates loot orbs without needing an item system.
- Amount can scale slightly with stage (stage 1: +2, stage 2+: +3).

### Action Budget Per Round (Scales with Stage)

The agent signals end of planning with the `pass` action. A hard cap prevents infinite loops:

```
Stage 1: max 5  actions per round
Stage 2: max 10 actions per round
Stage 3: max 15 actions per round
Stage 4: max 20 actions per round
Stage 5+: max 25 actions per round
```

If the cap is hit before `pass`, the planning phase ends automatically and combat begins.

### Damage to Players (PvP Rounds)

When a player loses a PvP round, they take damage:

```
base_damage     = 2
per_stage_bonus = current_stage * 1
surviving_units = number of enemy units still alive after combat

total_damage = base_damage + per_stage_bonus + surviving_units
```

This mirrors real TFT's damage formula and creates urgency to field a strong board.

### Player Elimination
- A player is eliminated when their health reaches 0.
- Eliminated players are removed from the opponent pool.
- The game ends when one player remains (or for training, when the agent is eliminated).

---

## State Space

All observations are **fixed-size padded float vectors**. Empty slots are zero-padded.

### Per-Unit Feature Vector (~32 floats)

```
hp                  # float, normalized to [0, 1] by max hp in roster
armor               # float
magic_resist        # float
attack_damage       # float
attack_speed        # float (attacks per second)
range               # float (1 = melee, 2+ = ranged)
ability_damage      # float
mana_cost           # float
star_level          # float: 0.33 / 0.66 / 1.0 for 1★ / 2★ / 3★
is_frontline        # bool (0 or 1)
ability_type        # one-hot, 4 dims: [damage, heal, shield, cc]
traits              # multi-hot, N dims (one per trait in roster)
```

Star level scales stats: 2★ = 1.8× base stats, 3★ = 3.24× base stats.

### Full Observation Vector

```python
observation = {

  # --- Economy (8 floats) ---
  "gold":               float,   # normalized by max gold cap (e.g. 50)
  "health":             float,   # normalized by starting health (e.g. 100)
  "level":              float,   # normalized by max level (e.g. 9)
  "xp_to_next_level":   float,
  "win_streak":         float,   # normalized by max streak (e.g. 5)
  "loss_streak":        float,
  "stage":              float,   # normalized by max stage (e.g. 7)
  "round_in_stage":     float,

  # --- Round Context (3 floats) ---
  "is_pvp_round":       float,   # 0 or 1
  "is_pve_round":       float,   # 0 or 1
  "players_remaining":  float,   # normalized by 8

  # --- My Board (10 × 32 = 320 floats) ---
  "board": [unit_vector × 10],

  # --- My Bench (9 × 32 = 288 floats) ---
  "bench": [unit_vector × 9],

  # --- Shop (5 × 32 = 160 floats) ---
  "shop": [unit_vector × 5],

  # --- Active Synergies (N_traits × 2 floats each) ---
  # For each trait: [active_count_normalized, breakpoint_progress]
  "synergies": [float × (N_traits * 2)],

  # --- Opponents (7 × opponent_vector) ---
  # opponent_vector:
  #   health (1), level (1), approx_gold_bin (1), streak (1),
  #   board units (10 × 3: unit_id_normalized, star_level, is_frontline)
  "opponents": [opponent_vector × 7],
}
```

**Total approximate size:** ~1100–1200 floats depending on trait count. Flat concatenation for MLP input, or process board/bench/shop/opponents through a shared unit encoder first.

---

## Action Space

Total: **~127 discrete actions** (after simplification).

```
ID Range    Action
─────────────────────────────────────────────
0           pass  (end planning phase, trigger combat)
1           buy_xp
2           reroll_shop
3–7         buy_unit(slot 0–4)
8–16        sell_bench(slot 0–8)
17–26       sell_board(slot 0–9)
27–36       toggle_frontline(board_slot 0–9)
37–126      place_unit(bench_slot 0–8, board_slot 0–9)
            → bench_slot * 10 + board_slot + 37
```

### Action Masking

A boolean mask vector of length 127 must be computed every step and passed to the policy. An action is **illegal** if:

```
pass          → always legal
buy_xp        → gold >= 4 AND level < max_level
reroll        → gold >= 2
buy_unit(i)   → shop slot i is occupied AND gold >= unit cost AND bench not full
sell_bench(i) → bench slot i is occupied
sell_board(i) → board slot i is occupied
toggle(i)     → board slot i is occupied
place_unit(b, d) → bench slot b is occupied AND board slot d is empty
                   OR bench slot b is occupied AND board not yet at level cap
```

The policy should **only ever sample from unmasked actions**. Pass the mask directly to your RL library (Stable-Baselines3 and CleanRL both support `action_masks` natively via MaskablePPO or equivalent).

---

## Combat Simulator

### Design Philosophy
No turn-by-turn simulation. Combat is resolved as a **DPS race** — whoever destroys the enemy team's total HP first wins. This is fast, differentiable in principle, and produces a clean win/loss/damage signal.

### Step 1: Compute Team Stats

For each team, separate units into frontline and backline:

```python
def compute_team_stats(units):
    frontline = [u for u in units if u.is_frontline]
    backline  = [u for u in units if not u.is_frontline]

    frontline_hp  = sum(u.effective_hp() for u in frontline)
    backline_hp   = sum(u.effective_hp() for u in backline)
    total_hp      = frontline_hp + backline_hp

    # DPS = auto attack damage * attack speed + ability damage per second
    # ability DPS = ability_damage / (mana_cost / 10)  [crude mana regen approximation]
    total_dps = sum(
        u.attack_damage * u.attack_speed + (u.ability_damage / (u.mana_cost / 10))
        for u in units
    )

    return total_hp, frontline_hp, backline_hp, total_dps
```

Effective HP accounts for armor/magic resist via a damage reduction factor:

```python
def effective_hp(unit):
    phys_reduction  = 100 / (100 + unit.armor)
    magic_reduction = 100 / (100 + unit.magic_resist)
    # assume 50/50 physical/magic damage split
    avg_reduction   = (phys_reduction + magic_reduction) / 2
    return unit.hp / avg_reduction
```

### Step 2: Frontline Buffer Rule

Backline units cannot be targeted until the frontline is destroyed. This is modeled by splitting the DPS race into two phases:

```python
def resolve_combat(team_a, team_b):
    a_hp_front, a_hp_back, a_dps = team_a
    b_hp_front, b_hp_back, b_dps = team_b

    # Phase 1: frontlines trade
    time_to_kill_a_front = a_hp_front / b_dps
    time_to_kill_b_front = b_hp_front / a_dps

    if time_to_kill_a_front < time_to_kill_b_front:
        # Team B kills team A's frontline first
        remaining_b_dps = b_dps  # (simplified: no attrition on b's frontline)
        time_remaining  = time_to_kill_b_front - time_to_kill_a_front
        damage_to_b_back = a_dps * time_to_kill_a_front  # A dealt this before dying

        # Phase 2: B's full DPS vs A's backline
        time_to_kill_a_back = a_hp_back / remaining_b_dps
        winner = "B"
    else:
        # symmetric, team A wins frontline
        winner = "A"

    return winner
```

This is intentionally simplified. Attrition within the frontline is ignored for now. Add it later if the combat signal feels too noisy.

### Step 3: Surviving Unit Count

After determining the winner, estimate surviving units for damage calculation:

```python
# Crude: proportion of HP remaining maps to proportion of units surviving
hp_fraction_remaining = max(0, 1 - (loser_dps * combat_time / winner_total_hp))
surviving_units = round(hp_fraction_remaining * len(winner_units))
```

### Step 4: Apply Results

```python
if agent_lost:
    damage = base_damage + stage_bonus + surviving_enemy_units
    agent.health -= damage
    agent.loss_streak += 1
    agent.win_streak = 0
else:
    agent.win_streak += 1
    agent.loss_streak = 0
```

### Trait Modifiers on Combat

Apply trait bonuses as multipliers on relevant stats before computing DPS/HP:

```python
# Example trait effect lookup (defined in traits.py)
trait_effects = {
    "Bruiser": {"hp_multiplier": 1.2},      # at 2-count breakpoint
    "Sniper":  {"ad_multiplier": 1.15},
    "Mage":    {"ability_damage_multiplier": 1.3},
}
```

The trait engine computes active bonuses and applies them to unit stats before combat resolves.

---

## Reward Structure

Rewards are discussed separately and are intentionally left flexible. A suggested baseline:

```python
reward = 0.0

# Dense signals (each step)
if action == buy_unit and creates_new_trait_breakpoint:
    reward += 0.05

# End of round signals
if won_combat:
    reward += 0.2
if lost_combat:
    reward -= 0.1

# Terminal signal
final_placement = get_placement()  # 1st through 8th
reward += (9 - final_placement) * 0.5   # 1st = +4.0, 8th = +0.5
if eliminated:
    reward -= 1.0
```

Tune this aggressively. Sparse rewards (terminal only) are cleaner but slower to train.

---

## Environment Loop

```python
class TFTEnv(gymnasium.Env):

    def __init__(self, n_players=8):
        self.n_players = n_players
        self.action_space = spaces.Discrete(127)
        self.observation_space = spaces.Box(low=0, high=1, shape=(OBS_DIM,))

    def reset(self, seed=None):
        self.game = GameState(n_players=self.n_players)
        self.game.start_round()
        obs  = self.game.to_observation()
        mask = self.game.action_mask()
        return obs, {"action_mask": mask}

    def step(self, action):
        if action == PASS or self.game.actions_this_round >= self.game.action_budget():
            # End planning phase
            reward    = self.game.resolve_round()   # runs combat, updates streaks/health/gold
            self.game.start_round()                 # advances stage/round, generates shop
            terminated = self.game.agent.health <= 0
            truncated  = self.game.is_last_player_standing()
        else:
            # Apply planning action
            self.game.apply_action(action)
            reward     = self.game.action_reward(action)  # small dense reward or 0
            terminated = False
            truncated  = False

        obs  = self.game.to_observation()
        mask = self.game.action_mask()
        return obs, reward, terminated, truncated, {"action_mask": mask}

    def action_masks(self):
        return self.game.action_mask()
```

### Round Lifecycle (inside GameState)

```
start_round()
    → determine round type (carousel / pve / pvp)
    → generate shop for agent (and bots)
    → calculate and grant income (base + interest + streak bonus)
    → reset actions_this_round counter

resolve_round()
    → for each player: run combat against assigned opponent
    → apply damage to losers
    → eliminate players at 0 hp
    → calculate and return reward
    → advance round/stage counters
```

### Income Calculation (per round start)

```python
def calculate_income(player):
    base_income    = 5
    interest       = min(5, player.gold // 10)       # 1g per 10g saved, cap 5
    streak_bonus   = streak_gold(player.win_streak or player.loss_streak)
    return base_income + interest + streak_bonus

def streak_gold(streak):
    if streak >= 5: return 3
    if streak >= 3: return 2
    if streak >= 2: return 1
    return 0
```

---

## Shop & Pool System

### Pool Sizes (by unit cost tier)

```
1-cost: 45 copies per unit
2-cost: 30 copies per unit
3-cost: 25 copies per unit
4-cost: 18 copies per unit
5-cost: 10 copies per unit
```

These are shared across all players. Buying removes from pool; selling returns to pool.

### Shop Odds (by player level)

```
Level │ 1-cost │ 2-cost │ 3-cost │ 4-cost │ 5-cost
──────┼────────┼────────┼────────┼────────┼───────
  1   │  100%  │   0%   │   0%   │   0%   │   0%
  2   │  100%  │   0%   │   0%   │   0%   │   0%
  3   │   75%  │  25%   │   0%   │   0%   │   0%
  4   │   55%  │  30%   │  15%   │   0%   │   0%
  5   │   45%  │  33%   │  20%   │   2%   │   0%
  6   │   30%  │  40%   │  25%   │   5%   │   0%
  7   │   19%  │  30%   │  35%   │  15%   │   1%
  8   │   15%  │  20%   │  35%   │  24%   │   6%
  9   │   10%  │  15%   │  30%   │  30%   │  15%
```

### Shop Generation

```python
def roll_shop(player, pool):
    shop = []
    for _ in range(5):
        cost_tier = sample_cost_tier(player.level)
        available = pool.get_available(cost_tier)
        if available:
            unit = random.choice(available)
            pool.reserve(unit)   # don't remove yet; only remove on purchase
            shop.append(unit)
    return shop

def reroll(player, pool):
    if player.gold < 2: return
    pool.return_units(player.current_shop)
    player.gold -= 2
    player.current_shop = roll_shop(player, pool)
```

### Auto-Combine Logic

When a unit is purchased, check for 3-copies upgrade:

```python
def try_combine(player, unit_id, star_level):
    copies = count_copies(player.board + player.bench, unit_id, star_level)
    if copies >= 3:
        remove_copies(player, unit_id, star_level, count=3)
        add_unit(player, unit_id, star_level + 1)
        try_combine(player, unit_id, star_level + 1)  # check for 3★ chain
```

---

## Player System

```python
@dataclass
class Player:
    health:       int   = 100
    gold:         int   = 0
    level:        int   = 1
    xp:           int   = 0
    win_streak:   int   = 0
    loss_streak:  int   = 0
    board:        list  = field(default_factory=lambda: [None] * 10)
    bench:        list  = field(default_factory=lambda: [None] * 9)
    shop:         list  = field(default_factory=lambda: [None] * 5)
    is_agent:     bool  = False
    is_eliminated: bool = False
```

### Leveling

```
Level │ XP Required (cumulative)
──────┼──────────────────────────
  2   │  2
  3   │  6
  4   │  10
  5   │  20
  6   │  36
  7   │  56
  8   │  80
  9   │  100  (max)
```

Buying XP costs 4 gold and grants 4 XP. XP is also granted each round (+2 per round automatically).

### Board Size = Player Level

The number of units a player can field equals their level (max 9 at level 9, board has 10 slots but slot 10 is never used until level 9 is reached). Enforce this in the action mask: `place_unit` is illegal if `board_unit_count >= player.level`.

---

## Trait Engine

The trait engine is **fully data-driven** — trait definitions live in `unit_roster.json`, making the system modular.

```python
def compute_active_traits(board_units, trait_definitions):
    trait_counts = defaultdict(int)
    for unit in board_units:
        if unit is None: continue
        for trait in unit.traits:
            trait_counts[trait] += 1

    active_bonuses = {}
    for trait, count in trait_counts.items():
        breakpoints = trait_definitions[trait]["breakpoints"]
        effects     = trait_definitions[trait]["effects"]
        # find highest active breakpoint
        active_bp = max((bp for bp in breakpoints if bp <= count), default=None)
        if active_bp:
            active_bonuses[trait] = effects[active_bp]

    return active_bonuses

def next_breakpoint(trait, count, trait_definitions):
    breakpoints = trait_definitions[trait]["breakpoints"]
    upcoming = [bp for bp in breakpoints if bp > count]
    return min(upcoming) if upcoming else None
```

Trait bonuses are applied in `combat.py` before DPS/HP calculations. Each bonus is a stat multiplier stored in the JSON:

```json
{
  "Bruiser": {
    "breakpoints": [2, 4, 6],
    "effects": {
      "2": {"hp_multiplier": 1.15},
      "4": {"hp_multiplier": 1.35},
      "6": {"hp_multiplier": 1.6}
    }
  }
}
```

---

## Bot System & Self-Play

### Phase 1: Scripted Bots

Bots start as simple heuristic agents occupying all 7 opponent slots:

```python
class ScriptedBot:
    def act(self, state):
        # Priority order:
        # 1. Buy units if gold > 6 and bench not full
        # 2. Buy XP if level < 5 and gold > 8
        # 3. Reroll if gold > 10
        # 4. Place units on board greedily (fill empty slots)
        # 5. Pass
```

Bots do not need to play optimally — they just need to field a plausible board to generate a real combat signal.

### Phase 2: Bot → Agent Graduation

Track agent win rate over a rolling window (e.g. last 500 games). When win rate exceeds **60% against the current bot pool**, replace one bot slot with a copy of the current agent policy (frozen weights). Repeat as the agent continues to improve. This is a simplified **league-style self-play**.

```python
def maybe_upgrade_bots(env, agent, win_rate):
    if win_rate > 0.60 and env.bot_slots_remaining > 0:
        env.replace_one_bot_with_agent(agent.get_frozen_policy())
```

Full self-play (all 8 players sharing a live policy) can be introduced once the environment is stable.

---

## Unit Roster & Trait System

**The roster is intentionally left for the user to define.** The system is fully modular — all unit and trait data lives in `data/unit_roster.json`. The engine reads this file at startup and never hard-codes unit-specific logic.

### unit_roster.json Schema

```json
{
  "units": [
    {
      "id": 0,
      "name": "UnitName",
      "cost": 1,
      "hp": 600,
      "armor": 40,
      "magic_resist": 40,
      "attack_damage": 55,
      "attack_speed": 0.7,
      "range": 1,
      "ability_damage": 150,
      "ability_type": "damage",
      "mana_cost": 60,
      "traits": ["TraitA", "TraitB"]
    }
  ],
  "traits": [
    {
      "name": "TraitA",
      "breakpoints": [2, 4, 6],
      "effects": {
        "2": {"hp_multiplier": 1.15},
        "4": {"hp_multiplier": 1.35},
        "6": {"hp_multiplier": 1.60}
      }
    }
  ]
}
```

### Design Guidelines for Roster

- Aim for **15–25 units** across 5 cost tiers (3–5 units per tier) for a good v1
- Aim for **6–10 traits** with 2–3 traits per unit
- Ensure each trait has at least 4–6 units that carry it (so 2/4 breakpoints are reachable)
- Balance trait effects so no single trait dominates (test with combat sim directly)
- Differentiate roles clearly: pure tanks (high HP/armor, low DPS), pure carries (low HP, high DPS/ability), and flex units

---

## Training Setup

### Recommended Algorithm

**MaskablePPO** from `sb3-contrib` (Stable-Baselines3 extension). It natively supports action masking via the `action_masks()` method on the environment.

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

env = ActionMasker(TFTEnv(), lambda env: env.action_masks())
model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5_000_000)
```

### Network Architecture (starting point)

```
Input (~1200 floats)
    → Linear(1200, 512) + ReLU
    → Linear(512, 256) + ReLU
    → Linear(256, 128) + ReLU
    ↓                    ↓
Actor head           Critic head
Linear(128, 127)     Linear(128, 1)
Softmax + mask       (value estimate)
```

A more principled architecture would use a **shared unit encoder** (small MLP or attention block) over board/bench/shop slots before the main trunk, but the flat MLP above is sufficient to start.

### Hyperparameters (starting point)

```python
MaskablePPO(
    policy         = "MlpPolicy",
    learning_rate  = 3e-4,
    n_steps        = 2048,
    batch_size     = 64,
    n_epochs       = 10,
    gamma          = 0.99,
    gae_lambda     = 0.95,
    clip_range     = 0.2,
    ent_coef       = 0.01,   # encourage exploration early
)
```

### Curriculum Suggestion

1. **Stage 1 only** — teach basic economy (buy/sell/reroll) with dummy combat
2. **Full game vs bots** — add real combat and stage progression
3. **Mixed bot/agent** — begin self-play graduation
4. **Full self-play** — all 8 agent slots

---

## Implementation Notes

### Normalization
All observation values should be normalized to roughly `[0, 1]`. Use known max values (max gold = 50, max health = 100, max level = 9, etc.).

### Padding Convention
Empty board/bench/shop slots are represented as zero vectors. The network can learn to ignore these via the masking on the action side, but a zero vector is a safe representation.

### Pool Contention (Multi-Agent)
In self-play, all agents share one pool object. The pool must be thread-safe or access must be serialized. Use a single game loop that steps each player in sequence rather than parallel threads to avoid this.

### Reproducibility
Always pass a seed to `reset()` and to the pool's random sampler. Log seeds during training so interesting runs can be reproduced.

### Testing the Environment
Before training, run `stable_baselines3.common.env_checker.check_env(env)` to catch shape/dtype issues. Also run a random policy for 1000 steps and verify:
- No illegal actions are ever sampled (mask is working)
- Rewards are in expected range
- Episodes terminate correctly

### Logging Metrics to Track
Beyond reward, log these each episode:
- Final placement (1–8)
- Highest trait breakpoint reached
- Gold efficiency (units purchased / gold spent)
- Average board power (sum of unit costs on board)
- Rounds survived

These give interpretable signals about *what the agent is learning*, not just whether reward is going up.
