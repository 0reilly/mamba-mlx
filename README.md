# Mamba: Selective State Space Sequence Modeling in MLX

MLX implementation of **"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"** (Gu & Dao, arXiv:2312.00752) — with a Transformer baseline for direct comparison.

## Overview

Mamba replaces the attention mechanism with a **selective state space model (S6)** that achieves linear-time sequence modeling by making the SSM parameters input-dependent:

$$\begin{aligned} A &= \text{diag}(-\exp(A_{\log})) & \text{(diagonal state matrix, learned)} \\ B &= s_B(x) & \text{(input-dependent)} \\ C &= s_C(x) & \text{(input-dependent)} \\ \Delta &= \text{softplus}(\delta_{\text{bias}} + s_\Delta(x)) & \text{(selective timescale, per-channel)} \end{aligned}$$

The key insight: unlike prior SSMs (S4, DSS, etc.) where A, B, C, Δ are *fixed* for all inputs, Mamba makes them **selective** — the model learns to ignore or attend to specific inputs by modulating Δ per-channel and per-timestep.

### MambaBlock Architecture

```
x ──► RMSNorm ──► Linear(D→4D) ──► split ──┐
             │                               │
             │          ┌── left (SiLU gate)  │
             │          │                    │
             │          └── right ──► Depthwise Conv1d(k=4) ──► SiLU ──► SelectiveSSM ──┐
             │                                                                         │
             │                                  (left ⊙ right) ◄────────────────────────┘
             │                                      │
             │                            Linear(2D→D) ──► (+) ◄── residual
             │
             └────────────────────────────────────────────────┘
```

###  SelectiveSSM (S6) — ZOH Discretization

Uses the full zero-order hold (ZOH), not the Euler approximation:

$$\begin{aligned} \bar{A} &= \exp(\Delta \cdot A) \\ \bar{B} &= (\exp(\Delta A) - I) \cdot A^{-1} \cdot B = \frac{\bar{A} - 1}{A} \cdot B \\ h_t &= \bar{A}_t \odot h_{t-1} + \bar{B}_t \cdot u_t \\ y_t &= C_t^\top h_t + D \odot u_t \end{aligned}$$

where $D$ is a learned skip-connection parameter. The scan runs sequentially over the sequence dimension — correct but simple (no parallel scan like the CUDA kernel in the paper).

## Models

| Model | Params | Description |
|-------|--------|-------------|
| **MambaLM** | 1,817,860 | 4× MambaBlock, d_model=256, SSM state=16, conv kernel=4 |
| **StandardTransformer** | 4,327,680 | 4× decoder layers, 4 heads (dim 64), RoPE, SwiGLU, RMSNorm |

Mamba achieves similar representational power with **2.4× fewer parameters**.

## Tasks

Three synthetic next-token prediction tasks evaluating different reasoning capabilities:

| Task | What it tests | Why it matters for Mamba |
|------|--------------|--------------------------|
| **Retrieval (MQAR)** | Copy a token from N positions back | Selective SSM must learn to "store and retrieve" — the selective mechanism should outperform fixed SSMs |
| **Decimal Addition** | Sum two multi-digit numbers | Algorithmic reasoning requires state tracking across positions |
| **Dyck-1 (Parentheses)** | Predict next token in balanced parentheses | Stack-like state — natural fit for SSM hidden state dynamics |

## Usage

```bash
# Install dependencies
pip install mlx numpy

# Run all experiments (Transformer + Mamba × 3 tasks)
python run.py

# Quick test
python -c "
from models import create_model
from train import train_model
model = create_model('mamba')
train_model(model, 'retrieval', n_steps=500)
"
```

## Structure

```
mamba_experiment/
├── models.py       # SelectiveSSM, MambaBlock, MambaLM, StandardTransformer
├── tasks.py        # Task generators (retrieval, addition, dyck)
├── train.py        # Training loop + evaluation
├── run.py          # Orchestrator — run all experiments
└── results.json    # Output after running run.py
```

## Implementation Notes

- **ZOH B discretization**: uses $(e^{\Delta A} - I)A^{-1}B$, not the Euler $\Delta B$ approximation
- **Δ parameterization**: matches paper Eq. (3): $\Delta = \text{softplus}(\delta_{\text{bias}} + s_\Delta(x))$ with per-channel learned bias
- **Conv1d**: causal depthwise convolution with manual left-padding (MLX doesn't support asymmetric padding)
- **A initialization**: $A = -\exp(A_{\log})$ with $A_{\log}$ initialized to $-\log(1), -\log(2), ..., -\log(N)$ giving geometrically spaced eigenvalues in $(0,1)$
- **Selective scan**: sequential Python for-loop over positions — not the CUDA parallel scan from the paper (MLX has no native scan primitive)

## Dependencies

- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon array framework
- NumPy

## Paper

> Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752, 2023.

## License

MIT
