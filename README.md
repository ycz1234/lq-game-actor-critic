# Zero-Sum LQ Game — Model-Free Actor-Critic Solver

A model-free Actor-Critic framework for computing Nash equilibrium mixed strategies in zero-sum linear-quadratic (LQ) games, extending the NeurIPS 2024 submission *Learning Mixed Strategies in Zero-Sum Linear Quadratic Games*.

---

## Overview

The original paper proves that the optimization landscape of zero-sum LQ games under mixed strategies can be reduced to the first two conditional moments (mean Γ, covariance Σ) of the control variables, and designs stacked per-step policy networks exploiting full knowledge of the system dynamics.

This work makes two architectural changes:

### Change 1: Model-Free (Actor-Critic)
Replace dynamics-layer backpropagation with a **Double Q-Critic** trained via TD learning. The system model (A, B₁, B₂) is treated as a black box — only payoff samples are used.

### Change 2: Sinusoidal Time Embedding (replaces per-step subnetworks)
The original paper uses **N separate subnetworks**, one per timestep. This work uses a **single shared network** conditioned on a sinusoidal time embedding:

$$e(k)_{2j} = \sin\!\left(\bar{k} \cdot 10^{j/d_t}\right), \quad e(k)_{2j+1} = \cos\!\left(\bar{k} \cdot 10^{j/d_t}\right)$$

where $\bar{k} = k/(N-1)$ is the normalised time. Benefits:
- Parameter count reduced by ~N×
- Shared representations across timesteps
- Smooth interpolation between time steps

### Method Comparison

| Aspect | Original Paper | This Work |
|--------|---------------|-----------|
| Model knowledge | Full (A, B₁, B₂ known) | **Unknown (black-box)** |
| Gradient source | Backprop through dynamics | **Critic-based policy gradient** |
| Policy network | N stacked subnetworks | **Single network + time embedding** |
| Opponent information | Full trajectory | Moments (Γ₋ᵢ, Σ₋ᵢ) only |
| Theoretical basis | Direct minimax | SDG moment reduction + TD learning |

---

## Benchmark: 1D Pursuit-Evasion Game

$$x(k+1) = x(k) + u_1(k) + u_2(k), \quad x(0) = 2$$

| Parameter | Value |
|-----------|-------|
| Pursuer bound c₁ | 3 |
| Evader bound c₂ | 1 |
| Horizon N | 5 |
| Terminal cost Q_N | 1.0 |

**Analytical equilibrium (Q=0):**
- Player 1 (minimiser): pure strategy $u_1^\star(k,x) = -\text{sign}(x)\min(|x|, c_1)$
- Player 2 (maximiser): two-point mixed strategy $u_2 \in \{-1, +1\}$ with equal probability
- Game value: $V^\star(x_0=2) = c_2^2 = 1.0$

---

## Architecture

```
Actor (shared across all timesteps):
  Input:  [x, z, e(k)]   x ∈ ℝ, z ~ N(0,I₄), e(k) ∈ ℝ⁸
  Hidden: Linear(13,64) → Mish → Linear(64,64) → Mish
  Output: Linear(64,1) → tanh × cᵢ    (guarantees u ∈ [−cᵢ, cᵢ])

Double Q-Critic:
  Input:  [x, uᵢ, Γ₋ᵢ, Σ₋ᵢ, e(k)]
  Hidden: Linear(12,64) → LayerNorm → Mish → Linear(64,64) → LayerNorm → Mish
  Output: Linear(64,1)  (×2 independent critics)
  Q_cons = (Q1+Q2)/2 − β|Q1−Q2|   (conservative estimate, β=0.2)
```

**Why opponent moments instead of opponent actions?**  
By the SDG approximation theorem (Theorem 1-2 in the original paper), the optimization landscape depends on the opponent's strategy only through (Γ₋ᵢ, Σ₋ᵢ). This reduces the input dimension from an infinite-dimensional distribution to 2 scalars, and improves training stability.

---

## Results

Training configuration: Q=0, 5000 episodes.

| Metric | Value | Target |
|--------|-------|--------|
| ValErr = \|E[J] − V★\| | **< 0.01** | < 0.1 ✅ |
| E[u₂(0)] | ≈ 0 | ≈ 0 ✅ |
| Std[u₂(0)] | ≈ 0.955 | ≈ c₂ = 1.0 ✅ |

**Player 2** successfully learns the two-point mixed strategy (maximum variance ≈ c₂²).  
**Player 1** converges to a valid Nash equilibrium — note that at Q=0, Player 1's equilibrium is non-unique (any u₁(0) ∈ [−3,3] is valid by Proposition 2.8), so u₁(0) ≈ −1 is correct behaviour, not a failure.

---

## Repository Structure

```
lq-game-actor-critic/
├── README.md
├── notebooks/
│   └── 03_pursuit_evasion_lq_game.ipynb
├── report/
│   └── LQ_Game_AC_Framework_Report.pdf   # Technical report
└── requirements.txt
```

---

## Requirements

```
torch>=2.0
numpy>=1.24
matplotlib>=3.7
```

---

## Reference

Original paper: *Learning Mixed Strategies in Zero-Sum Linear Quadratic Games* (NeurIPS 2024 submission).
