# Actor-Critic — A2C across Discrete, Continuous, and Pixel-Based Environments

> CSE 4/546: Reinforcement Learning · Spring 2025 · University at Buffalo
> Course instructor: Prof. Alina Vereshchaka

This repository is the third assignment for my graduate Reinforcement Learning course. I implemented a synchronous Advantage Actor-Critic (A2C) agent from scratch in PyTorch and trained it on three environments with very different observation and action spaces — a vector-state discrete control task, a pixel-based Atari game, and a continuous-control locomotion task.

📄 **[Read the full PDF report](./a3_report_ssaurav_asinha25.pdf)**

---

## What I built

A single A2C trainer, written from the ground up, then re-shaped per-environment around three policy families:

| Part | Environment | Obs | Action | Policy head | Backbone |
| --- | --- | --- | --- | --- | --- |
| 2 | `LunarLander-v3` | 8-dim vector | 4 discrete | Categorical | MLP (256, 256) — Tanh |
| 3 (Env 1) | `ALE/Pong-v5` | 84×84×4 stacked grayscale frames | 6 discrete | Categorical | 4-layer CNN → MLP (512, 512) → LayerNorm |
| 3 (Env 2) | `BipedalWalker-v3` | 24-dim vector | 4 continuous in `[-1, 1]` | Gaussian (learnable log-std) | MLP (256, 256) — ReLU → LayerNorm |

Part 1 is the pedagogical scaffolding from the course — separate vs shared actor-critic networks, an auto-adaptive `create_shared_network(env)` that inspects an environment's spaces, observation normalization, and gradient-clipping experiments. Part 2 and Part 3 are where the trained agents live.

---

## A2C, the way I implemented it

The trainer is the classical synchronous A2C: N worker processes each unroll `T` steps in their own copy of the environment, push the rollout onto a shared `multiprocessing.Queue`, and the main process aggregates, computes returns, and does a single gradient update against the global network.

**Per worker (`worker.py` in each env's module):**
1. Reset the env (seed = `proc_id`, so workers explore differently).
2. For `rollout_steps` steps, sample an action from the current global model, step the env, log `(state, action, reward, done)`.
3. Bootstrap the value of the final state and push the trajectory to the queue.

**Per update (notebook driver):**
1. Drain one rollout from each worker.
2. Compute discounted returns with the bootstrap value (`compute_returns.py`):
   `R_t = r_t + γ · R_{t+1} · (1 - done_t)`
3. Advantage = returns − values (no GAE in this version — plain n-step advantage).
4. Loss:
   ```
   L = policy_loss + value_loss_coef · value_loss − entropy_coef · entropy
   ```
5. Backprop, clip gradients to a global norm of `0.5`, step the optimizer.

**Hyperparameters (LunarLander, others scale similarly):**
```
learning_rate    = 7e-4
gamma            = 0.99
rollout_steps    = 20
num_workers      = cpu_count() // 2
entropy_coef     = 0.01
value_loss_coef  = 0.5
max_grad_norm    = 0.5
max_updates      = 200_000
```

---

## What changes per environment

The orchestrator is the same. The interesting work is at the network and wrapper boundary.

**`LunarLander-v3` (Part 2)** — discrete, vector state. Plain MLP with Tanh activations, orthogonal init scaled with `calculate_gain('tanh')` for the trunk and `gain=0.01` for the policy head (small policy-head init is a classic stability trick — it keeps the initial policy close to uniform so early gradients don't blow up). Categorical distribution over four discrete actions.

**`ALE/Pong-v5` (Part 3, Env 1)** — discrete, pixel observations. The wrapper stack matters as much as the network here:
- `AtariPreprocessing` for grayscale + 84×84 resize + frame skip handling.
- A custom `FrameStack(k=4)` wrapper to give the policy temporal context (otherwise the agent can't tell which direction the ball is moving).
- A 4-layer convolutional feature extractor (`8×8/s4 → 4×4/s2 → 3×3/s1 → 3×3/s1`) followed by a 2-layer 512-unit MLP and LayerNorm.
- The flat feature size is inferred at construction time by running a zero tensor through the conv stack — no hard-coded dimensions.

**`BipedalWalker-v3` (Part 3, Env 2)** — continuous, vector state. The shape of the policy head changes from logits to (mean, std):
- `mean` comes from a linear head (orthogonal-init at gain 0.01).
- `log_std` is a learnable `nn.Parameter`, clamped to `[-3.0, 0.5]` before exponentiation so the standard deviation stays in a sane range.
- Sample from `Normal(mean, std)` and clamp the action to `[-1, 1]` before stepping the env.
- A small temperature parameter (`temp`) divides the mean at inference time, which I used during evaluation to sharpen the deterministic-ish policy.

---

## Repo layout

```
.
├── README.md
├── a3_report_ssaurav_asinha25.pdf         Full write-up: setup, results, training curves, ablations
├── spring25_rl_assignment_3.pdf           Original assignment spec (for reference)
├── assignment_3_part_1.ipynb              Part 1 — pedagogical exercises (templates)
├── a3_part_1_ssaurav_asinha25.ipynb       Part 1 — my submitted solutions
├── a3_part_2_ssaurav_asinha25.ipynb       Part 2 — short companion notebook
└── final/                                 Final training notebooks + trained weights
    ├── a3_part_1_ssaurav_asinha25.ipynb
    ├── a3_part_2_ssaurav_asinha25.ipynb              LunarLander A2C training driver
    ├── a3_part_3_ENV1_ssaurav_asinha25.ipynb         Pong A2C training driver
    ├── a3_part_3_ENV2_ssaurav_asinha25.ipynb         BipedalWalker A2C training driver
    ├── a3_part_2_a2c_ENV1_ssaurav_asinha25.pth       LunarLander weights
    ├── a3_part_3_a2c_ENV1_ssaurav_asinha25.pth       Pong weights
    ├── a3_part_3_a2c_ENV2_ssaurav_asinha25.pth       BipedalWalker weights
    ├── part_2_LunarLander_modules/                   ActorCritic, worker, compute_returns
    ├── part_3__Pong_v5_modules/                      + FrameStack, record_stats
    └── part_3__BipedalWalker_V3_modules/             Gaussian-policy variant
```

The `.pth` files are the snapshots I evaluated against in the report — drop them into the notebook's `evaluate(...)` call to skip training and just watch the policy run.

---

## Running it

```bash
# Python 3.10+ recommended
pip install torch gymnasium "gymnasium[box2d]" "gymnasium[atari]" ale-py matplotlib numpy
```

Then open any notebook in `final/` and run top-to-bottom. The trainers use `torch.multiprocessing`, so on Windows make sure the entry-point cell is inside `if __name__ == "__main__":` if you tear them out of the notebook.

To watch a trained agent without re-training, jump to the `evaluate(...)` cell at the bottom of each notebook — it loads the matching `.pth` file and renders the episode in human-render mode.

---

## What I learned

- **The orchestrator is almost trivial; the wrapper and head choices are everything.** Swapping LunarLander for Pong meant adding a frame-stack wrapper and a CNN; swapping for BipedalWalker meant replacing the Categorical with a Gaussian and adding a learnable log-std. The A2C update equation itself didn't change a line.
- **Policy-head initialization matters more than I expected.** Initializing the policy head at `gain=0.01` (orthogonal) means the initial policy is close to uniform, which keeps early-episode advantages well-conditioned and avoids the agent locking into a bad action distribution before it has any value signal.
- **Synchronous A2C is a much easier first build than A3C.** No HOGWILD!-style lock-free updates, no per-worker optimizer state — just a queue and a single backward pass per update. The training-throughput cost is real but the debugging time you save is worth it for an assignment.
- **Atari preprocessing is half the work.** I spent more time getting the `AtariPreprocessing → FrameStack` pipeline right than I spent on the actual network.

---

## Stack

PyTorch · Gymnasium · ALE (Atari Learning Environment) · `torch.multiprocessing` · Box2D (LunarLander, BipedalWalker) · NumPy · Matplotlib

---

*Spring 2025 · Graduate coursework in Reinforcement Learning.*
