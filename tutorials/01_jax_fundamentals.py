"""
JAX Fundamentals for the TFT Port — Tutorial 1
===============================================

This script teaches JAX by connecting every concept to your existing TFT code.
Run it with: python tutorials/01_jax_fundamentals.py

It is structured as a series of "lessons" — each one introduces a JAX concept,
shows the PyTorch/numpy equivalent from your TFT codebase, and gives you an
exercise to verify understanding.

The exercises are ASSERTED — the script will fail if your answers are wrong.
Fill in the TODO sections, then run the script to check your work.

Prerequisites:
    - You've read the JAX docs "JAX for the impatient" (jax.readthedocs.io)
    - You've skimmed the PureJaxRL walkthrough notebook
    - This venv is active (tft-jax with jax, flax, optax, rlax installed)
"""

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
from flax.struct import dataclass as flax_dataclass
from functools import partial

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ============================================================
# LESSON 1: JAX Arrays vs NumPy Arrays
# ============================================================
# Your TFT code uses numpy everywhere: np.array, np.zeros, np.random.default_rng.
# JAX has its own array type (jax.Array) and its own numpy-like API (jax.numpy).
#
# Key differences from numpy:
#   1. JAX arrays are IMMUTABLE — you cannot do arr[0] = 5. You use arr.at[0].set(5).
#   2. JAX operations are dispatched to accelerators (GPU/TPU) if available.
#   3. JAX arrays have an explicit dtype — jnp.zeros(10) gives float32, not float64.

logger.info("=" * 60)
logger.info("LESSON 1: JAX Arrays vs NumPy Arrays")
logger.info("=" * 60)

# Your TFT code creates action masks like this (actions.py:33):
#   mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
# In JAX:
mask_numpy = np.zeros(127, dtype=np.int8)
mask_jax = jnp.zeros(127, dtype=jnp.int8)

logger.info(f"numpy mask type: {type(mask_numpy)}, dtype: {mask_numpy.dtype}")
logger.info(f"jax mask type:   {type(mask_jax)}, dtype: {mask_jax.dtype}")

# Immutability demo — this is the biggest mental shift
arr = jnp.arange(5)
logger.info(f"\nOriginal array: {arr}")

# numpy way (your TFT code does this in state.py:329):
#   player.board[d_idx] = player.bench[b_idx]
# In JAX, this DOESN'T WORK:
#   arr[2] = 99  # TypeError: JAX arrays are immutable

# JAX way — functional update, returns a NEW array:
new_arr = arr.at[2].set(99)
logger.info(f"After arr.at[2].set(99): {new_arr}  (original unchanged: {arr})")

# EXERCISE 1: Create a JAX array of shape (10, 5) filled with ones (float32),
# then set the element at row 3, column 2 to 0.0 using the .at[] syntax.
# This mirrors your TFT board representation: 10 board slots, each with a unit vector.

# TODO: Replace None with your code
board = None  # jnp.ones((10, 5), dtype=jnp.float32), then set [3, 2] to 0.0
board = jnp.ones((10, 5), dtype=jnp.float32).at[3, 2].set(0.0)

assert board is not None, "EXERCISE 1: Fill in the board array"
assert board.shape == (10, 5), f"EXERCISE 1: Expected shape (10, 5), got {board.shape}"
assert board[3, 2] == 0.0, f"EXERCISE 1: Expected board[3,2] == 0.0, got {board[3, 2]}"
assert board[0, 0] == 1.0, f"EXERCISE 1: Other elements should still be 1.0"
logger.info(f"EXERCISE 1 PASSED: board shape {board.shape}, board[3,2]={board[3,2]}")

# ============================================================
# LESSON 2: PRNG — The Key Splitting Pattern
# ============================================================
# Your TFT code uses numpy RNG: rng = np.random.default_rng(seed)
# Then rng.random(), rng.choice(), rng.shuffle(), etc.
#
# JAX RNG is COMPLETELY DIFFERENT. There is no global RNG state.
# Instead, you pass explicit PRNG keys to every stochastic function,
# and you SPLIT keys to get independent sub-keys.
#
# Why? Because JAX is functional — no hidden state. This makes everything
# reproducible and parallelizable. But it means you must manage keys carefully.

logger.info("\n" + "=" * 60)
logger.info("LESSON 2: PRNG — The Key Splitting Pattern")
logger.info("=" * 60)

# Your TFT code (state.py:86):
#   self.rng = rng if rng is not None else np.random.default_rng()
# Then in shop.py:76:
#   unit_id = int(rng.choice(available))

# JAX equivalent:
master_key = jax.random.PRNGKey(42)
logger.info(f"Master key: {master_key}")

# Split into sub-keys for independent random operations
key1, key2 = jax.random.split(master_key)
logger.info(f"Key 1: {key1}  (used for, say, shop roll)")
logger.info(f"Key 2: {key2}  (used for, say, combat RNG)")

# Split into N keys for N parallel environments
# This is what you'll use with vmap — one key per parallel env
N_ENVS = 8
env_keys = jax.random.split(master_key, N_ENVS)
logger.info(f"Split into {N_ENVS} env keys, shape: {env_keys.shape}")

# Using a key to sample — equivalent to rng.choice(available)
# Your TFT shop rolls 5 units. Here's how you'd sample 5 random unit IDs:
all_unit_ids = jnp.arange(30)  # 30 units in your roster
shop_key, next_master = jax.random.split(master_key)
shop_units = jax.random.choice(shop_key, all_unit_ids, shape=(5,), replace=False)
logger.info(f"\nShop roll (5 units): {shop_units}")

# EXERCISE 2: Write a function that takes a PRNG key and returns
# (a) a random action from the TFT action space (0-126), and
# (b) a new PRNG key for the next operation.
# This is the pattern you'll use everywhere in the JAX port —
# every stochastic function takes a key and returns a new one.

def sample_random_action(key):
    """Sample a random TFT action and return (action, next_key)."""
    # TODO: Split the key, use one half to sample an action in [0, 127),
    # return (action, other_half)
    action_key, next_key = jax.random.split(key)
    action = jax.random.randint(action_key, (), 0, 127)
    return action, next_key

key = jax.random.PRNGKey(0)
action, next_key = sample_random_action(key)
assert 0 <= action < 127, f"EXERCISE 2: Action {action} out of range [0, 127)"
assert next_key is not key, "EXERCISE 2: Must return a DIFFERENT key (split it)"
# Verify reproducibility — same key gives same action
action2, _ = sample_random_action(key)
assert action == action2, "EXERCISE 2: Same key must give same result (reproducibility)"
logger.info(f"EXERCISE 2 PASSED: action={int(action)}, reproducible=True")

# ============================================================
# LESSON 3: jit — Just-In-Time Compilation
# ============================================================
# jit compiles a Python function into optimized XLA code.
# This is where JAX gets its speed.
#
# Rules:
#   1. The function must be PURE — no side effects, no mutation.
#   2. Arguments must be JAX arrays or PyTrees (nested dicts/lists of arrays).
#   3. Shapes must be STATIC (known at compile time). Dynamic shapes recompile.
#   4. Python control flow (if/else, for) runs at TRACE TIME, not runtime.
#      Use jnp.where, jax.lax.cond, jax.lax.scan for runtime control flow.

logger.info("\n" + "=" * 60)
logger.info("LESSON 3: jit — Just-In-Time Compilation")
logger.info("=" * 60)

# Your TFT observation builder (state.py:344) builds a flat numpy array.
# Here's a simplified version showing the jit pattern:

def build_unit_vector_jax(unit_vec):
    """Normalize a unit's stats (simplified from state.py:344)."""
    max_hp = 3000.0
    max_damage = 500.0
    return jnp.array([
        unit_vec[0] / max_hp,      # hp
        unit_vec[1] / max_damage,  # attack_damage
    ])

# Without jit — runs in Python, slow
unit = jnp.array([650.0, 55.0])  # Garen's stats
result = build_unit_vector_jax(unit)
logger.info(f"Without jit: {result}")

# With jit — compiled to optimized XLA, fast
build_unit_vector_jit = jax.jit(build_unit_vector_jax)
result_jit = build_unit_vector_jit(unit)
logger.info(f"With jit:    {result_jit}")

# Timing comparison
import time

import logging

logger = logging.getLogger(__name__)
big_input = jnp.arange(10000, dtype=jnp.float32)

def slow_function(x):
    """Simulate a computation-heavy function."""
    for _ in range(100):
        x = jnp.sin(x) + jnp.cos(x)
    return x

# Without jit
start = time.time()
for _ in range(10):
    _ = slow_function(big_input)
_ = slow_function(big_input).block_until_ready()  # force computation
no_jit_time = (time.time() - start) / 10

# With jit
fast_function = jax.jit(slow_function)
_ = fast_function(big_input).block_until_ready()  # warm up (compile)
start = time.time()
for _ in range(10):
    _ = fast_function(big_input)
_ = fast_function(big_input).block_until_ready()
jit_time = (time.time() - start) / 10

logger.info(f"\nTiming (10000 elements, 100 sin+cos):")
logger.info(f"  Without jit: {no_jit_time*1000:.2f} ms")
logger.info(f"  With jit:    {jit_time*1000:.2f} ms")
logger.info(f"  Speedup:     {no_jit_time/jit_time:.1f}x")

# EXERCISE 3: Write a function that computes the masked logits for TFT actions.
# Given raw logits (shape [127]) and a mask (shape [127], 1=legal, 0=illegal),
# set illegal action logits to -1e8 and return the result.
# Then jit it and verify the output matches.
#
# This is exactly what your policy.py:86 (masked_distribution) does in PyTorch.

def mask_logits(logits, mask):
    """Set logits of illegal actions to -1e8."""
    # TODO: Use jnp.where to select between logits and -1e8 based on mask
    return jnp.where(mask == 1, logits, -1e8)

mask_logits_jit = jax.jit(mask_logits)

test_logits = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
test_mask = jnp.array([1, 0, 1, 0, 1])
result = mask_logits_jit(test_logits, test_mask)
assert result[0] == 1.0, f"EXERCISE 3: Legal action 0 should keep its logit"
assert result[1] == -1e8, f"EXERCISE 3: Illegal action 1 should be -1e8"
assert result[2] == 3.0, f"EXERCISE 3: Legal action 2 should keep its logit"
assert result[4] == 5.0, f"EXERCISE 3: Legal action 4 should keep its logit"
logger.info(f"EXERCISE 3 PASSED: masked logits = {result}")

# ============================================================
# LESSON 4: vmap — Automatic Vectorization (Parallel Envs)
# ============================================================
# vmap takes a function that operates on a single example and automatically
# batches it. This is how you run N environments in parallel in JAX.
#
# In PyTorch, you'd write a batch dimension into every operation manually.
# In JAX, you write the single-env function, then vmap it.

logger.info("\n" + "=" * 60)
logger.info("LESSON 4: vmap — Automatic Vectorization")
logger.info("=" * 60)

# Your TFT env.step takes a single action and returns a single result.
# With vmap, you write step(state, action) for ONE env, then vmap it
# to step N envs in parallel.

# Example: normalize a single unit vector (from state.py:344)
def normalize_unit(unit_vec):
    """Normalize a single unit's stats."""
    max_hp = 3000.0
    max_damage = 500.0
    max_as = 3.0
    return jnp.array([
        unit_vec[0] / max_hp,
        unit_vec[1] / max_damage,
        unit_vec[2] / max_as,
    ])

# Single unit
single_unit = jnp.array([650.0, 55.0, 0.65])  # Garen
logger.info(f"Single unit normalized: {normalize_unit(single_unit)}")

# Batch of units — vmap handles the batch dimension automatically
batch_units = jnp.array([
    [650.0, 55.0, 0.65],   # Garen
    [800.0, 70.0, 0.70],   # stronger unit
    [0.0, 0.0, 0.0],       # empty slot
    [500.0, 40.0, 0.60],   # weaker unit
])

# vmap over axis 0 (the batch dimension)
normalize_batch = jax.vmap(normalize_unit)
batch_result = normalize_batch(batch_units)
logger.info(f"Batch normalized:\n{batch_result}")

# EXERCISE 4: Write a function that computes the TFT placement reward
# for a single placement (from metrics.py:14):
#   placement_reward(placement) = (9 - placement) * 0.5
# Then vmap it to compute rewards for a batch of placements.
# This is what you'll use to compute rewards across parallel envs.

def placement_reward(placement):
    """Terminal bonus: 1st = +4.0, 8th = +0.5."""
    # TODO: Implement using the formula above
    return (9 - placement) * 0.5

# Test single
assert placement_reward(jnp.array(1)) == 4.0, f"1st place should be +4.0"
assert placement_reward(jnp.array(8)) == 0.5, f"8th place should be +0.5"

# Vmap it
batch_placements = jnp.array([1, 2, 3, 4, 5, 6, 7, 8])
placement_reward_batch = jax.vmap(placement_reward)
batch_rewards = placement_reward_batch(batch_placements)

assert batch_rewards[0] == 4.0, f"1st place should be +4.0"
assert batch_rewards[7] == 0.5, f"8th place should be +0.5"
assert batch_rewards.shape == (8,), f"Should return 8 rewards"
logger.info(f"EXERCISE 4 PASSED: batch rewards = {batch_rewards}")

# ============================================================
# LESSON 5: grad — Automatic Differentiation
# ============================================================
# grad computes the gradient of a scalar-valued function.
# This replaces PyTorch's autograd (loss.backward()).
#
# Key difference: JAX grad is FUNCTIONAL. You pass the function to grad,
# and it returns a NEW function that computes gradients.
# No computational graph, no backward pass, no .grad attributes.

logger.info("\n" + "=" * 60)
logger.info("LESSON 5: grad — Automatic Differentiation")
logger.info("=" * 60)

# Simple example: gradient of x^2 is 2x
def f(x):
    return x ** 2

df_dx = jax.grad(f)  # This is a NEW function
logger.info(f"f(3.0) = {f(3.0)}")
logger.info(f"f'(3.0) = {df_dx(3.0)}  (should be 6.0)")

# value_and_grad — get both value and gradient in one pass
f_val_and_grad = jax.value_and_grad(f)
val, grad = f_val_and_grad(3.0)
logger.info(f"value_and_grad: val={val}, grad={grad}")

# EXERCISE 5: Compute the gradient of the PPO clipped surrogate objective
# with respect to the ratio. This is the core of your PPO update
# (train.py:276-278 in PyTorch).
#
# The clipped objective is:
#   L(ratio, advantage) = min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv)
# Its gradient w.r.t. ratio is:
#   - adv, if ratio is within [1-eps, 1+eps]
#   - 0, if ratio is outside the clip range (gradient is clipped)

def ppo_surrogate(ratio, advantage, eps=0.2):
    """PPO clipped surrogate objective (negated for minimization)."""
    # TODO: Implement the clipped surrogate
    surr1 = ratio * advantage
    surr2 = jnp.clip(ratio, 1 - eps, 1 + eps) * advantage
    return jnp.minimum(surr1, surr2)

# Test: when ratio is within clip range, gradient = advantage
ratio = jnp.array(1.0)  # within [0.8, 1.2]
advantage = jnp.array(2.0)
grad_fn = jax.grad(ppo_surrogate)
grad_val = grad_fn(ratio, advantage)
logger.info(f"\nratio=1.0 (within clip), advantage=2.0")
logger.info(f"  surrogate value: {ppo_surrogate(ratio, advantage)}")
logger.info(f"  gradient: {grad_val}  (should be 2.0 = advantage)")

# Test: when ratio is outside clip range, gradient = 0
ratio_clipped = jnp.array(2.0)  # outside [0.8, 1.2] when eps=0.2
grad_clipped = grad_fn(ratio_clipped, advantage)
logger.info(f"\nratio=2.0 (outside clip), advantage=2.0")
logger.info(f"  surrogate value: {ppo_surrogate(ratio_clipped, advantage)}")
logger.info(f"  gradient: {grad_clipped}  (should be 0.0 — gradient is clipped)")

assert abs(grad_val - 2.0) < 1e-6, f"EXERCISE 5: Gradient should be 2.0, got {grad_val}"
assert abs(grad_clipped - 0.0) < 1e-6, f"EXERCISE 5: Clipped gradient should be 0.0, got {grad_clipped}"
logger.info(f"EXERCISE 5 PASSED: PPO gradient behavior verified")

# ============================================================
# LESSON 6: Flax — Neural Networks in JAX
# ============================================================
# Flax is JAX's neural network library. Your TFT policy (policy.py:10)
# is an nn.Module in PyTorch. Here's how to write it in Flax.
#
# Key differences from PyTorch:
#   1. Flax modules are defined with @nn.compact or setup()
#   2. Parameters are NOT stored in the module. They are separate.
#   3. You call module.apply(params, x) to run the forward pass.
#   4. No .forward(), no __call__ with parameters baked in.

logger.info("\n" + "=" * 60)
logger.info("LESSON 6: Flax — Neural Networks in JAX")
logger.info("=" * 60)

# Your PyTorch policy (policy.py:10-28):
#   class ActorCriticNetwork(nn.Module):
#       def __init__(self, state_dim, action_dim, hidden=(512, 256, 128)):
#           self.backbone = nn.Sequential(nn.Linear(state_dim, 512), nn.ReLU(), ...)
#           self.actor = nn.Linear(128, action_dim)
#           self.critic = nn.Linear(128, 1)
#       def forward(self, x):
#           features = self.backbone(x)
#           return self.actor(features), self.critic(features)

# Flax equivalent:
class ActorCriticFlax(nn.Module):
    action_dim: int  # config passed as class attribute

    @nn.compact
    def __call__(self, x):
        # Backbone: 512 -> 256 -> 128 (same as your PyTorch version)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        # Actor and critic heads
        actor_logits = nn.Dense(self.action_dim)(x)
        critic_value = nn.Dense(1)(x)
        return actor_logits, critic_value

# Initialize the model — this is different from PyTorch
# In PyTorch: model = ActorCriticNetwork(state_dim, action_dim) and params are inside
# In Flax: model = ActorCriticFlax(action_dim=127), then you init params separately
model = ActorCriticFlax(action_dim=127)
key = jax.random.PRNGKey(0)
dummy_obs = jnp.zeros(314, dtype=jnp.float32)  # your TFT obs is ~314 dims
params = model.init(key, dummy_obs)  # returns the parameters as a PyTree

logger.info(f"Model: {model}")
logger.info(f"Param tree structure:")
jax.tree_util.tree_map(lambda x: logger.info(f"  shape={x.shape}, dtype={x.dtype}"), params)

# Forward pass — note params are passed explicitly
logits, value = model.apply(params, dummy_obs)
logger.info(f"\nForward pass: logits shape={logits.shape}, value shape={value.shape}")

# EXERCISE 6: Create a Flax model that mirrors your StructuredActorCritic
# (policy.py:45-83). It should:
#   1. Take an observation of shape (state_dim,)
#   2. Extract 24 unit slots (each of size unit_vec_size) starting at index 12
#   3. Encode each unit through a shared UnitEncoder (Dense -> relu -> Dense -> relu)
#   4. Concatenate encoded units with the remaining flat features
#   5. Pass through an MLP trunk (512 -> 256 -> 128)
#   6. Output actor logits (action_dim) and critic value (1)
#
# This is a simplified version — just get the structure right.

class UnitEncoderFlax(nn.Module):
    embed_dim: int = 64

    @nn.compact
    def __call__(self, x):
        # x shape: (..., unit_vec_size)
        x = nn.Dense(self.embed_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.embed_dim)(x)
        x = nn.relu(x)
        return x

class StructuredActorCriticFlax(nn.Module):
    action_dim: int
    unit_vec_size: int

    @nn.compact
    def __call__(self, x):
        # TODO: Implement the structured actor-critic
        # 1. Extract 24 unit slots starting at index 12
        start = 12
        u = self.unit_vec_size
        unit_slots = x[start: start + 24 * u].reshape(24, u)
        # 2. Encode each unit through the shared encoder
        encoder = UnitEncoderFlax(embed_dim=64)
        encoded = encoder(unit_slots)  # (24, 64)
        encoded = encoded.reshape(-1)  # (24 * 64,)
        # 3. Concatenate with remaining flat features
        other = jnp.concatenate([x[:start], x[start + 24 * u:]])
        features = jnp.concatenate([encoded, other])
        # 4. MLP trunk
        h = nn.Dense(512)(features)
        h = nn.relu(h)
        h = nn.Dense(256)(h)
        h = nn.relu(h)
        h = nn.Dense(128)(h)
        h = nn.relu(h)
        # 5. Heads
        logits = nn.Dense(self.action_dim)(h)
        value = nn.Dense(1)(h)
        return logits, value

# Test it
unit_vec_size = 14  # 10 base + 4 ability types (simplified)
state_dim = 12 + 24 * unit_vec_size + 50  # simplified obs
structured_model = StructuredActorCriticFlax(action_dim=127, unit_vec_size=unit_vec_size)
dummy_state = jnp.zeros(state_dim, dtype=jnp.float32)
s_params = structured_model.init(jax.random.PRNGKey(0), dummy_state)
s_logits, s_value = structured_model.apply(s_params, dummy_state)

assert s_logits.shape == (127,), f"EXERCISE 6: logits shape should be (127,), got {s_logits.shape}"
assert s_value.shape == (1,), f"EXERCISE 6: value shape should be (1,), got {s_value.shape}"
logger.info(f"EXERCISE 6 PASSED: structured model logits={s_logits.shape}, value={s_value.shape}")

# ============================================================
# LESSON 7: Optax — Optimizers
# ============================================================
# Optax replaces torch.optim. Instead of optimizer = Adam(params, lr=1e-3)
# and optimizer.step(), you compose gradient transformations.

logger.info("\n" + "=" * 60)
logger.info("LESSON 7: Optax — Optimizers")
logger.info("=" * 60)

# Your PyTorch code (train.py:196):
#   self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
# And in the update loop (train.py:283-286):
#   self.optimizer.zero_grad()
#   loss.mean().backward()
#   torch.nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
#   self.optimizer.step()

# Optax equivalent — chain of transformations
optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),  # MAX_GRAD_NORM from your train.py:41
    optax.adam(learning_rate=3e-4),   # LEARNING_RATE from your train.py:30
)

# Initialize optimizer state (separate from model params)
opt_state = optimizer.init(params)

# Training step in JAX:
def loss_fn(params, model, x, target):
    logits, _ = model.apply(params, x)
    return jnp.mean((logits - target) ** 2)  # dummy loss

def train_step(params, opt_state, model, x, target):
    # Compute loss AND gradients in one pass
    loss, grads = jax.value_and_grad(loss_fn)(params, model, x, target)
    # Apply gradient transformation
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

# JIT the training step for speed.
# NOTE: The Flax `model` is a Python object, not a JAX array, so jit would
# try to trace it as an array and fail. We mark it as a STATIC argument so
# jit treats it as a compile-time constant. This is the standard Flax pattern.
train_step_jit = jax.jit(train_step, static_argnames=["model"])

# Run a few steps
key = jax.random.PRNGKey(42)
x = jax.random.normal(key, (314,))
target = jax.random.normal(key, (127,))

logger.info("Training step (dummy loss to verify optimizer works):")
for i in range(5):
    params, opt_state, loss = train_step_jit(params, opt_state, model, x, target)
    logger.info(f"  Step {i+1}: loss = {loss:.4f}")

# EXERCISE 7: Create an optimizer that matches your PPO training setup.
# From train.py:30-41, your PPO uses:
#   - Learning rate: 3e-4 (with linear decay to 1e-4)
#   - Gradient clipping: max_grad_norm = 0.5
#   - Adam optimizer
# Create an Optax chain that does all three.
# For the LR schedule, use optax.linear_schedule.

# TODO: Create the optimizer
lr_schedule = optax.linear_schedule(
    init_value=3e-4,
    end_value=1e-4,
    transition_steps=1000,
)
ppo_optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(learning_rate=lr_schedule),
)

# Verify it works
ppo_opt_state = ppo_optimizer.init(params)
assert ppo_opt_state is not None, "EXERCISE 7: Optimizer state should not be None"
logger.info(f"EXERCISE 7 PASSED: PPO optimizer created with LR schedule + grad clipping")

# ============================================================
# LESSON 8: PyTrees — Nested State Structures
# ============================================================
# PyTrees are how JAX handles nested structures of arrays.
# Your TFT GameState has nested Player objects with boards, benches, etc.
# In JAX, this becomes a PyTree — a nested dict/dataclass of arrays.
#
# JAX can jit/vmap/grad over entire PyTrees automatically.

logger.info("\n" + "=" * 60)
logger.info("LESSON 8: PyTrees — Nested State Structures")
logger.info("=" * 60)

# Your TFT Player (player.py:6-20) is a dataclass with lists of Units.
# In JAX, use flax.struct.dataclass (NOT dataclasses.dataclass) to make
# it a PyTree that JAX can trace through.

@flax_dataclass
class PlayerState:
    """JAX-compatible player state (mirrors player.py:6-20)."""
    health: jnp.ndarray      # scalar int32
    gold: jnp.ndarray        # scalar int32
    level: jnp.ndarray       # scalar int32
    xp: jnp.ndarray          # scalar int32
    win_streak: jnp.ndarray  # scalar int32
    loss_streak: jnp.ndarray # scalar int32
    board: jnp.ndarray       # (10, unit_vec_size) float32
    bench: jnp.ndarray       # (9, unit_vec_size) float32
    shop: jnp.ndarray        # (5,) int32 (unit IDs)
    is_agent: jnp.ndarray   # scalar bool
    is_eliminated: jnp.ndarray  # scalar bool

# Create a player
unit_vec_size = 14
player = PlayerState(
    health=jnp.int32(100),
    gold=jnp.int32(0),
    level=jnp.int32(1),
    xp=jnp.int32(0),
    win_streak=jnp.int32(0),
    loss_streak=jnp.int32(0),
    board=jnp.zeros((10, unit_vec_size), dtype=jnp.float32),
    bench=jnp.zeros((9, unit_vec_size), dtype=jnp.float32),
    shop=jnp.full((5,), -1, dtype=jnp.int32),  # -1 = empty
    is_agent=jnp.bool_(False),
    is_eliminated=jnp.bool_(False),
)

# It's a PyTree — JAX can inspect its structure
leaves, treedef = jax.tree_util.tree_flatten(player)
logger.info(f"Player PyTree has {len(leaves)} leaves (arrays)")
logger.info(f"Leaf shapes: {[l.shape for l in leaves]}")

# You can modify it functionally (no mutation!)
# This is how you "update" player gold after income:
new_player = player.replace(gold=player.gold + 5)
logger.info(f"\nPlayer gold before income: {player.gold}")
logger.info(f"Player gold after income:  {new_player.gold}  (original unchanged: {player.gold})")

# EXERCISE 8: Create a GameState PyTree that holds N players.
# This mirrors your GameState (state.py:78) but in JAX form.
# Use jnp.stack to hold N players in a batched array.

@flax_dataclass
class GameStateJax:
    """JAX-compatible game state (mirrors state.py:78)."""
    players: PlayerState  # This will be vmapped to hold N players
    stage: jnp.ndarray
    round_in_stage: jnp.ndarray
    actions_this_round: jnp.ndarray
    rounds_completed: jnp.ndarray
    rng_key: jax.Array

# Create a game state
game = GameStateJax(
    players=player,
    stage=jnp.int32(1),
    round_in_stage=jnp.int32(0),
    actions_this_round=jnp.int32(0),
    rounds_completed=jnp.int32(0),
    rng_key=jax.random.PRNGKey(42),
)

# Verify it's a PyTree
game_leaves, game_treedef = jax.tree_util.tree_flatten(game)
logger.info(f"\nGameState PyTree has {len(game_leaves)} leaves")

# EXERCISE: Write a function that advances the round counter
def advance_round(state):
    """Increment round_in_stage (simplified from state.py:108)."""
    # TODO: Return a new state with round_in_stage incremented by 1
    return state.replace(round_in_stage=state.round_in_stage + 1)

new_game = advance_round(game)
assert new_game.round_in_stage == 1, f"EXERCISE 8: round should be 1, got {new_game.round_in_stage}"
assert game.round_in_stage == 0, f"EXERCISE 8: original should be unchanged"
logger.info(f"EXERCISE 8 PASSED: round {game.round_in_stage} -> {new_game.round_in_stage}")

# ============================================================
# LESSON 9: scan — The Training Loop Primitive
# ============================================================
# scan is JAX's replacement for Python for loops inside jit.
# It carries state through iterations without recompiling.
# This is how PureJaxRL runs entire training loops under jit.
#
# scan(fn, init, xs) where:
#   fn(carry, x) -> (carry, y)
#   init is the initial carry
#   xs is a sequence to iterate over

logger.info("\n" + "=" * 60)
logger.info("LESSON 9: scan — The Training Loop Primitive")
logger.info("=" * 60)

# Example: accumulate rewards over a rollout
# This mirrors your GAE computation (train.py:48-66) which loops
# backward over rewards.

def accumulate_reward(carry, reward):
    """Accumulate rewards with discount factor."""
    total = carry + reward
    return total, total

rewards = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
final_total, all_totals = jax.lax.scan(accumulate_reward, 0.0, rewards)
logger.info(f"Rewards: {rewards}")
logger.info(f"Running totals: {all_totals}")
logger.info(f"Final total: {final_total}")

# EXERCISE 9: Use scan to compute discounted returns.
# This is the core of your GAE computation (train.py:48-66).
# discounted_return(t) = reward(t) + gamma * discounted_return(t+1)
# But scan goes forward, so we compute it as:
#   carry = reward + gamma * carry
# Process rewards in REVERSE order.

def discounted_return_step(carry, reward_and_done):
    """One step of discounted return computation."""
    reward, done = reward_and_done
    # TODO: Compute the new carry.
    # If done=1, the return is just the reward (episode ended).
    # If done=0, the return is reward + gamma * carry.
    new_carry = jnp.where(done, reward, reward + 0.99 * carry)
    return new_carry, new_carry

# Test with a simple sequence
rewards = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
dones = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0])  # episode ends on last step

# Reverse for scan (we compute returns backward)
rewards_reversed = rewards[::-1]
dones_reversed = dones[::-1]

# Run scan
final_return, returns_reversed = jax.lax.scan(
    discounted_return_step,
    0.0,  # initial carry (bootstrap value = 0 for terminal)
    (rewards_reversed, dones_reversed),
)

# Reverse back to get returns in forward order
returns = returns_reversed[::-1]
logger.info(f"Rewards: {rewards}")
logger.info(f"Dones:   {dones}")
logger.info(f"Returns: {returns}")
logger.info(f"  (last return should be 1.0, first should be ~4.9)")

# Verify: with gamma=0.99 and episode ending at step 4:
# return[4] = 1.0
# return[3] = 1.0 + 0.99 * 1.0 = 1.99
# return[2] = 1.0 + 0.99 * 1.99 = 2.9701
# return[1] = 1.0 + 0.99 * 2.9701 = 3.940...
# return[0] = 1.0 + 0.99 * 3.940... = 4.901...
assert abs(returns[4] - 1.0) < 1e-6, f"Last return should be 1.0, got {returns[4]}"
assert abs(returns[0] - 4.901) < 0.01, f"First return should be ~4.901, got {returns[0]}"
logger.info(f"EXERCISE 9 PASSED: discounted returns computed correctly")

# ============================================================
# LESSON 10: Putting It Together — A Minimal JIT-Compiled Step
# ============================================================
# Now combine jit + vmap + PyTrees to create a batched, compiled
# environment step function. This is the pattern PureJaxRL uses.

logger.info("\n" + "=" * 60)
logger.info("LESSON 10: Putting It Together — Batched JIT Step")
logger.info("=" * 60)

# Minimal example: a batched "apply action" that updates player gold
# This mirrors your state.py:283-286 (buy_xp action)

@flax_dataclass
class MiniState:
    gold: jnp.ndarray
    level: jnp.ndarray

def apply_buy_xp(state):
    """Apply buy_xp action: costs 4 gold, gives 4 XP."""
    # Simplified — just deduct gold
    new_gold = state.gold - 4
    return state.replace(gold=new_gold)

# Single env
state = MiniState(gold=jnp.int32(20), level=jnp.int32(1))
new_state = apply_buy_xp(state)
logger.info(f"Single env: gold {state.gold} -> {new_state.gold}")

# Batch over 8 envs with vmap
batch_states = MiniState(
    gold=jnp.array([20, 15, 10, 8, 30, 5, 12, 18], dtype=jnp.int32),
    level=jnp.array([1, 2, 1, 3, 2, 1, 1, 2], dtype=jnp.int32),
)

# vmap the function — it now works on batches automatically
batch_apply = jax.vmap(apply_buy_xp)
batch_new = batch_apply(batch_states)
logger.info(f"\nBatch envs: gold {batch_states.gold}")
logger.info(f"After buy_xp:      {batch_new.gold}")

# Now JIT the batched function — compiled + batched
batch_apply_jit = jax.jit(batch_apply)
batch_new_jit = batch_apply_jit(batch_states)
logger.info(f"JIT batched:       {batch_new_jit.gold}")

# EXERCISE 10: Write a function that computes the TFT income for a player
# (from state.py:137-151), then vmap and jit it for batch processing.
#
# Income = base(5) + interest(min(5, gold//10)) + streak_bonus
# streak_bonus: 3 if streak>=5, 2 if streak>=3, 1 if streak>=2, else 0

def compute_income(gold, win_streak, loss_streak):
    """Compute player income (from state.py:137-151)."""
    # TODO: Implement the income calculation
    base = 5
    interest = jnp.minimum(5, gold // 10)
    streak = jnp.maximum(win_streak, loss_streak)
    streak_bonus = jnp.where(streak >= 5, 3,
                   jnp.where(streak >= 3, 2,
                   jnp.where(streak >= 2, 1, 0)))
    return base + interest + streak_bonus

# Test single
# 20 gold, 5 win streak: 5 base + min(5, 20//10)=2 interest + 3 streak = 10
assert compute_income(jnp.int32(20), jnp.int32(5), jnp.int32(0)) == 10, \
    f"20 gold, 5 streak: 5 base + 2 interest + 3 streak = 10"
# 10 gold, 2 win streak: 5 base + min(5, 10//10)=1 interest + 1 streak = 7
assert compute_income(jnp.int32(10), jnp.int32(2), jnp.int32(0)) == 7, \
    f"10 gold, 2 streak: 5 base + 1 interest + 1 streak = 7"

# Vmap and jit
batch_income = jax.jit(jax.vmap(compute_income))
golds = jnp.array([20, 10, 50, 0, 30], dtype=jnp.int32)
streaks = jnp.array([5, 2, 0, 0, 3], dtype=jnp.int32)
losses = jnp.array([0, 0, 4, 0, 0], dtype=jnp.int32)
incomes = batch_income(golds, streaks, losses)
logger.info(f"\nBatch income: {incomes}")
logger.info(f"  (20g/5 streak=10, 10g/2 streak=7, 50g/4 loss=12, 0g/0=5, 30g/3 streak=10)")

assert incomes[0] == 10, f"Expected 10, got {incomes[0]}"
assert incomes[1] == 7, f"Expected 7, got {incomes[1]}"
assert incomes[3] == 5, f"Expected 5, got {incomes[3]}"
logger.info(f"EXERCISE 10 PASSED: batched income computation works")

# ============================================================
# SUMMARY
# ============================================================

logger.info("\n" + "=" * 60)
logger.info("TUTORIAL 1 COMPLETE — All exercises passed!")
logger.info("=" * 60)
logger.info("""
You now understand the core JAX primitives needed for the TFT port:
  1. JAX arrays (immutable, .at[].set())
  2. PRNG keys (split, pass explicitly)
  3. jit (compile pure functions)
  4. vmap (auto-batch single-env functions)
  5. grad (functional differentiation)
  6. Flax modules (params separate from model)
  7. Optax (chain of gradient transformations)
  8. PyTrees (nested state via flax.struct.dataclass)
  9. scan (training loops under jit)
  10. Combining jit + vmap + PyTrees for batched env steps

Next: Tutorial 2 will port the TFT static data layer (unit roster, constants)
to JAX arrays, then Tutorial 3 will port the full GameState and step function.
""")
