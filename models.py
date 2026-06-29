"""
MLX implementation of Mamba and Standard Transformer architectures.

Based on:
- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (arXiv:2312.00752)

Provides:
1. SelectiveSSM  — the core S6 selective state space model
2. MambaBlock   — a single Mamba layer (RMSNorm, expand, conv, SSM, gate, project)
3. MambaLM      — stacked MambaBlocks for language modelling
4. StandardTransformer — Llama-style decoder-only baseline (RMSNorm, RoPE, SwiGLU)
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    """RMS normalisation — equivalent to Llama's RMSNorm."""
    return mx.fast.rms_norm(x, weight, eps)


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x, self.weight, self.eps)


# ---------------------------------------------------------------------------
# Selective SSM (S6) — Section 3.3 of the Mamba paper
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """
    The S6 selective state space model.

    Parameters
    ----------
    d_model : int
        Model dimension D (number of input/output channels).
    ssm_state_dim : int
        Inner state dimension N (default 16, as used in Mamba).

    Forward signature
    -----------------
    Input:  u  shape (B, L, D)
    Output: y  shape (B, L, D)
    """

    def __init__(
        self,
        d_model: int,
        ssm_state_dim: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.ssm_state_dim = ssm_state_dim

        # -- Learned log of diagonal A matrix (D, N) --
        # Initialise A_log so that after neg-exp it favours values in (0, 1).
        # The paper uses a geometric spacing: A_log = log(range(1, N+1)).
        A_init = -mx.log(mx.arange(1, ssm_state_dim + 1, dtype=mx.float32))  # (N,)
        # Repeat across D channels
        self.A_log = mx.expand_dims(A_init, axis=0)  # (1, N)
        self.A_log = mx.tile(self.A_log, (d_model, 1))   # (D, N)

        # -- Skip/residual connection parameter D (D,) --
        self.D_param = mx.ones((d_model,))

        # -- Input → Δ projection: Linear(D → 1) (no bias; bias handled by dt_bias) --
        self.dt_proj = nn.Linear(d_model, 1, bias=False)

        # -- Learned broadcast bias for Δ (Section 3.3: Δ = softplus(Parameter + s_Δ(x))) --
        # dt_bias is broadcast over B and L dimensions; shape (D,)
        self.dt_bias = mx.zeros((d_model,))

        # -- Input → B projection: Linear(D → N) --
        self.B_proj = nn.Linear(d_model, ssm_state_dim, bias=False)

        # -- Input → C projection: Linear(D → N) --
        self.C_proj = nn.Linear(d_model, ssm_state_dim, bias=False)

    def __call__(self, u: mx.array) -> mx.array:
        """
        Forward pass of the selective SSM.

        Parameters
        ----------
        u : mx.array, shape (B, L, D)

        Returns
        -------
        y : mx.array, shape (B, L, D)
        """
        B, L, D = u.shape
        N = self.ssm_state_dim
        assert D == self.d_model, f"Expected D={self.d_model}, got {D}"

        # ---- 1. Δ (delta) — Section 3.3: Δ = softplus(Parameter + s_Δ(x)) ----
        # dt_proj: (B, L, D) → (B, L, 1) — per-position scalar
        delta_raw = self.dt_proj(u)                 # (B, L, 1)
        # Add learned per-channel bias (D, 1) broadcast → (B, L, D)
        delta_raw = mx.broadcast_to(delta_raw, (B, L, D)) + mx.expand_dims(self.dt_bias, axis=(0, 1))  # (B, L, D)
        delta = nn.softplus(delta_raw)              # (B, L, D)

        # ---- 2. A (per-channel diagonal state matrix) ----
        A = -mx.exp(self.A_log)                     # (D, N), all negative

        # ---- 3. B (input-dependent projection) ----
        B_proj = self.B_proj(u)                     # (B, L, N)

        # ---- 4. C (input-dependent projection) ----
        C_proj = self.C_proj(u)                     # (B, L, N)

        # ---- 5. ZOH discretisation ----
        # A_bar = exp(Δ ⊗ A)  —  (B, L, D) × (D, N) → (B, L, D, N)
        # Δ.unsqueeze(-1): (B, L, D, 1)
        # A:               (1, 1, D, N)  [after broadcast]
        delta_expanded = mx.expand_dims(delta, axis=-1)          # (B, L, D, 1)
        A_expanded = mx.expand_dims(A, axis=(0, 1))               # (1, 1, D, N)
        A_bar = mx.exp(delta_expanded * A_expanded)               # (B, L, D, N)

        # ---- 5b. B_bar via ZOH: (exp(ΔA) - I)·A^{-1}·B ----
        # For diagonal A, this is: (A_bar - 1) / A * B
        # A_bar: (B, L, D, N), A_expanded: (1, 1, D, N), B_proj_expanded: (B, L, 1, N)
        B_proj_expanded = mx.expand_dims(B_proj, axis=2)          # (B, L, 1, N)
        # (A_bar - 1) / A  — elementwise, handled safely near A ≈ 0
        # A is negative, so division is safe (no zeros for properly initialised A)
        A_safe = mx.where(mx.abs(A_expanded) < 1e-12,
                          mx.ones_like(A_expanded),
                          A_expanded)
        B_bar = (A_bar - 1.0) / A_safe * B_proj_expanded          # (B, L, D, N)

        # ---- 6. Selective scan (sequential recurrence) ----
        # We maintain h of shape (B, D, N) — a state per batch item and channel.
        h = mx.zeros((B, D, N), dtype=u.dtype)

        y_list = []
        for t in range(L):
            # A_bar_t: (B, D, N)
            # B_bar_t: (B, D, N)
            # u_t:     (B, D)   → unsqueeze to (B, D, 1)
            # C_t:     (B, N)

            u_t = u[:, t, :]                              # (B, D)
            u_t_exp = mx.expand_dims(u_t, axis=-1)         # (B, D, 1)

            A_bar_t = A_bar[:, t, :, :]                   # (B, D, N)
            B_bar_t = B_bar[:, t, :, :]                   # (B, D, N)
            C_t = C_proj[:, t, :]                          # (B, N)

            # Recurrence: h_t = A_bar_t * h + B_bar_t * u_t (elementwise)
            h = A_bar_t * h + B_bar_t * u_t_exp           # (B, D, N)

            # Output: y_t = Σ_n C_t[:, n] * h[:, :, n]
            # C_t_exp: (B, 1, N); h: (B, D, N); sum over N → (B, D)
            C_t_exp = mx.expand_dims(C_t, axis=1)          # (B, 1, N)
            y_t = mx.sum(C_t_exp * h, axis=-1)             # (B, D)
            y_list.append(y_t)

        # Stack → (B, L, D)
        y = mx.stack(y_list, axis=1)                       # (B, L, D)

        # ---- 7. Residual skip with D ----
        # D_param: (D,) → broadcast to (1, 1, D) → (B, L, D)
        y = y + u * mx.expand_dims(self.D_param, axis=(0, 1))

        return y


# ---------------------------------------------------------------------------
# Mamba Block — Section 3.4
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    A single Mamba block.

    Architecture (from Sec 3.4, Fig. 3):
        x ──► RMSNorm ──► Linear(D → 2*E*D) ──► split ──┐
            │                                            │
            │              ┌─ left (SiLU)                │
            │              │                              │
            │              └─ right ──► Conv1d ──► SiLU ──► SSM ──┐
            │                                                      │
            │                                    (left * right) ◄──┘
            │                                        │
            │                              Linear(2D → D) ──► (+) ◄── residual
            │
            └────────────────────────────────────────────────┘

    Parameters
    ----------
    d_model : int
        Model dimension D.
    ssm_state_dim : int
        Inner state dimension N for the SSM.
    expand_factor : int
        Expansion factor E (default 2). The block expands D → 2*E*D.
    conv_kernel_size : int
        Kernel size for the 1D causal convolution on the right branch.
    """

    def __init__(
        self,
        d_model: int,
        ssm_state_dim: int = 16,
        expand_factor: int = 2,
        conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        inner_dim = expand_factor * d_model  # = 2 * D  (when E=2)

        # Pre-norm
        self.norm = RMSNorm(d_model)

        # Expansion: D → 2 * inner_dim = 2 * 2 * D = 4 * D
        self.in_proj = nn.Linear(d_model, 2 * inner_dim, bias=False)

        # 1D causal (depthwise) convolution on the right branch.
        # MLX Conv1d expects (B, L, C) layout and only supports symmetric integer
        # padding, so we manually left-pad (causal) and use padding=0.
        self.conv_kernel_size = conv_kernel_size
        self.conv1d = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=conv_kernel_size,
            padding=0,                     # we handle padding manually
            groups=inner_dim,               # depthwise (one group per channel)
            bias=True,
        )

        # Selective SSM on the right branch
        self.ssm = SelectiveSSM(
            d_model=inner_dim,
            ssm_state_dim=ssm_state_dim,
        )

        # Output projection: inner_dim → d_model
        self.out_proj = nn.Linear(inner_dim, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Parameters
        ----------
        x : mx.array, shape (B, L, D)

        Returns
        -------
        out : mx.array, shape (B, L, D)
        """
        B, L, D = x.shape
        residual = x

        # Pre-norm
        x_norm = self.norm(x)                                   # (B, L, D)

        # Expansion: D → 2 * inner_dim
        projected = self.in_proj(x_norm)                         # (B, L, 2*inner_dim)

        # Split into left (gate) and right (SSM) branches, each (B, L, inner_dim)
        z, x_proj = mx.split(projected, 2, axis=-1)

        # ---- Left branch: SiLU gate ----
        z = nn.silu(z)                                           # (B, L, inner_dim)

        # ---- Right branch: Conv1d → SiLU → SSM ----
        # MLX Conv1d expects (B, L, C) layout.  We pad the L axis (axis=1)
        # on the left with (kernel_size-1) zeros for causality.
        pad_left = self.conv_kernel_size - 1
        x_conv = mx.pad(x_proj, [(0, 0), (pad_left, 0), (0, 0)])  # (B, L+pad, inner_dim)
        x_conv = self.conv1d(x_conv)                               # (B, L, inner_dim) — same len

        # SiLU activation
        x_conv = nn.silu(x_conv)                                  # (B, L, inner_dim)

        # Selective SSM
        x_ssm = self.ssm(x_conv)                                  # (B, L, inner_dim)

        # ---- Gate: left * right ----
        gated = z * x_ssm                                         # (B, L, inner_dim)

        # ---- Output projection ----
        out = self.out_proj(gated)                                # (B, L, D)

        # ---- Residual connection ----
        out = out + residual

        return out


# ---------------------------------------------------------------------------
# Mamba Language Model
# ---------------------------------------------------------------------------

class MambaLM(nn.Module):
    """
    Stack of MambaBlocks for causal language modelling.

    Architecture:
        tokens → Embedding → MambaBlock × N_layers → RMSNorm → Linear head → logits
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        n_layers: int = 4,
        ssm_state_dim: int = 16,
        expand_factor: int = 2,
        conv_kernel_size: int = 4,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # Stack of Mamba blocks
        self.layers = [
            MambaBlock(
                d_model=d_model,
                ssm_state_dim=ssm_state_dim,
                expand_factor=expand_factor,
                conv_kernel_size=conv_kernel_size,
            )
            for _ in range(n_layers)
        ]

        # Final norm
        self.final_norm = RMSNorm(d_model)

        # Output head
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embedding and output weights
        self.output.weight = self.embed.weight

    def __call__(self, tokens: mx.array) -> mx.array:
        """
        Forward pass.

        Parameters
        ----------
        tokens : mx.array, shape (B, L) with integer token ids.

        Returns
        -------
        logits : mx.array, shape (B, L, vocab_size)
        """
        h = self.embed(tokens)               # (B, L, D)

        for layer in self.layers:
            h = layer(h)                      # (B, L, D)

        h = self.final_norm(h)               # (B, L, D)
        logits = self.output(h)              # (B, L, vocab_size)
        return logits

    def _flatten_params(self, d, prefix: str = "") -> dict:
        """Flatten nested parameter dict/list into a flat dict keyed by dotted path."""
        flat = {}
        if isinstance(d, list):
            for i, item in enumerate(d):
                flat.update(self._flatten_params(item, f"{prefix}{i}."))
        elif isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (dict, list)):
                    flat.update(self._flatten_params(v, f"{prefix}{k}."))
                elif isinstance(v, mx.array):
                    flat[f"{prefix}{k}"] = v
                elif hasattr(v, "size"):
                    flat[f"{prefix}{k}"] = v
        return flat

    def count_params(self) -> tuple[int, int]:
        """Return (total_params, trainable_params)."""
        all_params = self._flatten_params(self.parameters())
        total = sum(v.size for v in all_params.values())
        return total, total


# ---------------------------------------------------------------------------
# Standard Transformer (Llama-style decoder-only baseline)
# ---------------------------------------------------------------------------

def precompute_rotary_frequencies(dim: int, max_seq_len: int) -> mx.array:
    """Precompute RoPE cosine/sine tables.

    Returns
    -------
    freqs : mx.array, shape (max_seq_len, dim // 2, 2)
        freqs[..., 0] = cos, freqs[..., 1] = sin
    """
    freqs = 1.0 / (10000.0 ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(max_seq_len, dtype=mx.float32)
    freqs = mx.outer(t, freqs)                               # (max_seq_len, dim//2)
    return mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1)  # (max_seq_len, dim//2, 2)


def apply_rotary_embedding(x: mx.array, freqs: mx.array) -> mx.array:
    """Apply rotary position embeddings to queries or keys.

    Parameters
    ----------
    x : mx.array, shape (B, n_heads, L, head_dim)
    freqs : mx.array, shape (max_seq_len, head_dim//2, 2)

    Returns
    -------
    x_rotated : mx.array, same shape as x
    """
    B, n_heads, L, head_dim = x.shape

    # Reshape into complex pairs
    x_reshaped = x.reshape(*x.shape[:-1], head_dim // 2, 2)  # (B, H, L, hd/2, 2)

    # Truncate freqs to sequence length
    cos = freqs[:L, :, 0]  # (L, hd/2)
    sin = freqs[:L, :, 1]  # (L, hd/2)

    # Broadcast for batch & heads
    cos = cos[None, None, :, :, None]  # (1, 1, L, hd/2, 1)
    sin = sin[None, None, :, :, None]  # (1, 1, L, hd/2, 1)

    # Rotate: (a+bi) * (cos+isin) = (a*cos - b*sin) + (a*sin + b*cos)i
    x_rotated = mx.concatenate([
        x_reshaped[..., 0:1] * cos - x_reshaped[..., 1:2] * sin,
        x_reshaped[..., 0:1] * sin + x_reshaped[..., 1:2] * cos,
    ], axis=-1)

    return x_rotated.reshape(*x.shape)


def create_causal_mask(seq_len: int) -> mx.array:
    """Create causal attention mask: lower-tri = 0, upper-tri = -inf."""
    rows = mx.arange(seq_len)[:, None]   # (S, 1)
    cols = mx.arange(seq_len)[None, :]   # (1, S)
    mask_bool = rows >= cols
    return mx.where(mask_bool, 0.0, -float("inf"))


class GatedMLP(nn.Module):
    """SwiGLU MLP (SiLU-gated linear unit)."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class StandardAttention(nn.Module):
    """Standard multi-head causal self-attention with RoPE."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        head_dim: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_dim, bias=False)

        # Precomputed RoPE frequencies
        self._freqs = precompute_rotary_frequencies(head_dim, max_seq_len)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, S, D = x.shape

        q = self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # RoPE
        q = apply_rotary_embedding(q, self._freqs)
        k = apply_rotary_embedding(k, self._freqs)

        # Scaled dot-product
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q * scale) @ k.transpose(0, 1, 3, 2)    # (B, H, S, S)

        if mask is not None:
            scores = scores + mask[None, None, :S, :S]

        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """A single Transformer block: pre-norm attention + pre-norm SwiGLU MLP."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        head_dim: int,
        intermediate_dim: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim)
        self.attention = StandardAttention(hidden_dim, n_heads, head_dim, max_seq_len)
        self.mlp_norm = RMSNorm(hidden_dim)
        self.mlp = GatedMLP(hidden_dim, intermediate_dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Attention with residual
        x = x + self.attention(self.attn_norm(x), mask)
        # MLP with residual
        x = x + self.mlp(self.mlp_norm(x))
        return x


class StandardTransformer(nn.Module):
    """
    Llama-style decoder-only transformer.

    Architecture:
        tokens → Embedding → TransformerBlock × N_layers → RMSNorm → Linear head → logits
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        intermediate_dim = d_model * 4

        # Embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # Transformer blocks
        self.layers = [
            TransformerBlock(
                hidden_dim=d_model,
                n_heads=n_heads,
                head_dim=head_dim,
                intermediate_dim=intermediate_dim,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ]

        # Final norm and output head
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embedding and output weights
        self.output.weight = self.embed.weight

        # Precomputed causal mask
        self._causal_mask = create_causal_mask(max_seq_len)

    def __call__(self, tokens: mx.array) -> mx.array:
        """
        Forward pass.

        Parameters
        ----------
        tokens : mx.array, shape (B, L) with integer token ids.

        Returns
        -------
        logits : mx.array, shape (B, L, vocab_size)
        """
        B, S = tokens.shape
        h = self.embed(tokens)                               # (B, L, D)

        mask = self._causal_mask[:S, :S]

        for layer in self.layers:
            h = layer(h, mask)

        h = self.norm(h)
        logits = self.output(h)                              # (B, L, vocab_size)
        return logits

    def _flatten_params(self, d, prefix: str = "") -> dict:
        """Flatten nested parameter dict/list into a flat dict keyed by dotted path."""
        flat = {}
        if isinstance(d, list):
            for i, item in enumerate(d):
                flat.update(self._flatten_params(item, f"{prefix}{i}."))
        elif isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (dict, list)):
                    flat.update(self._flatten_params(v, f"{prefix}{k}."))
                elif isinstance(v, mx.array):
                    flat[f"{prefix}{k}"] = v
                elif hasattr(v, "size"):
                    flat[f"{prefix}{k}"] = v
        return flat

    def count_params(self) -> tuple[int, int]:
        """Return (total_params, trainable_params)."""
        all_params = self._flatten_params(self.parameters())
        total = sum(v.size for v in all_params.values())
        return total, total


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_model(
    model_type: str,
    vocab_size: int = 256,
    d_model: int = 256,
    n_layers: int = 4,
    **kwargs,
) -> nn.Module:
    """
    Create a Mamba or StandardTransformer model.

    Parameters
    ----------
    model_type : str
        "mamba" or "transformer".
    vocab_size : int
        Vocabulary size.
    d_model : int
        Model dimension.
    n_layers : int
        Number of layers.
    **kwargs
        Passed through to the constructor:
        - Mamba: ssm_state_dim, expand_factor, conv_kernel_size, max_seq_len
        - Transformer: n_heads, head_dim, max_seq_len

    Returns
    -------
    model : nn.Module
    """
    if model_type == "mamba":
        return MambaLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            ssm_state_dim=kwargs.get("ssm_state_dim", 16),
            expand_factor=kwargs.get("expand_factor", 2),
            conv_kernel_size=kwargs.get("conv_kernel_size", 4),
            max_seq_len=kwargs.get("max_seq_len", 128),
        )
    elif model_type == "transformer":
        return StandardTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=kwargs.get("n_heads", 4),
            head_dim=kwargs.get("head_dim", 64),
            max_seq_len=kwargs.get("max_seq_len", 128),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'mamba' or 'transformer'.")
