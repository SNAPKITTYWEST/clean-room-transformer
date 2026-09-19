"""Transformer layer primitives: LayerNorm, MHSA with KV-cache, SwiGLU FFN.

Conventions
-----------
Single-stream (batch-free) tensors are used throughout so that every shape in
the forward pass is unambiguous and inspectable:

    hidden states      (T, d_model)
    q / k / v per head (n_heads, T, d_head)
    attention weights  (n_heads, T_q, T_kv)

All arithmetic is float32. Softmax is computed with max-subtraction so that
no intermediate exp() can overflow to inf for finite inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import DTYPE, ModelConfig
from .determinism import generator_for
from .shapes import expect


def _normal(gen: np.random.Generator, shape: tuple[int, ...], std: float) -> np.ndarray:
    return gen.normal(loc=0.0, scale=std, size=shape).astype(DTYPE)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    return (normed * gamma + beta).astype(DTYPE)


def silu(x: np.ndarray) -> np.ndarray:
    """x * sigmoid(x), computed without overflow for large |x|."""
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = x[pos] / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = x[~pos] * exp_x / (1.0 + exp_x)
    return out.astype(DTYPE)


def softmax_last_axis(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(DTYPE)


@dataclass
class KVCache:
    """Pre-allocated, deterministically zero-filled key/value cache.

    Capacity is fixed at construction: the cache never reallocates, so the
    memory footprint of a run is known before the first token is emitted.
    """

    n_heads: int
    capacity: int
    d_head: int
    keys: np.ndarray = field(init=False)
    values: np.ndarray = field(init=False)
    length: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.keys = np.zeros((self.n_heads, self.capacity, self.d_head), dtype=DTYPE)
        self.values = np.zeros((self.n_heads, self.capacity, self.d_head), dtype=DTYPE)

    def reset(self) -> None:
        self.keys.fill(0.0)
        self.values.fill(0.0)
        self.length = 0

    def rollback_to(self, length: int) -> None:
        """Rewind to `length` positions, zeroing the abandoned region.

        Zeroing is not required for correctness -- nothing reads past
        `self.length` -- but it makes the rolled-back cache byte-identical to
        its pre-step state, which is what lets a test assert that a quarantined
        step leaves no residue. See tests/test_rollback.py.
        """
        if not 0 <= length <= self.length:
            raise ValueError(f"cannot roll back to {length} from {self.length}")
        self.keys[:, length : self.length, :] = 0.0
        self.values[:, length : self.length, :] = 0.0
        self.length = length

    def append(self, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Append `T` new positions and return views over the live prefix."""
        t_new = k.shape[1]
        if self.length + t_new > self.capacity:
            raise ValueError(
                f"KV cache overflow: length {self.length} + {t_new} exceeds "
                f"capacity {self.capacity}"
            )
        lo, hi = self.length, self.length + t_new
        self.keys[:, lo:hi, :] = k
        self.values[:, lo:hi, :] = v
        self.length = hi
        return self.keys[:, :hi, :], self.values[:, :hi, :]


class MultiHeadSelfAttention:
    def __init__(self, cfg: ModelConfig, layer_index: int) -> None:
        self.cfg = cfg
        self.layer_index = layer_index
        d, h, dh = cfg.d_model, cfg.n_heads, cfg.d_head
        std = (1.0 / d) ** 0.5
        tag = f"layer{layer_index}.attn"
        self.w_q = _normal(generator_for(cfg.seed, f"{tag}.w_q"), (d, d), std)
        self.w_k = _normal(generator_for(cfg.seed, f"{tag}.w_k"), (d, d), std)
        self.w_v = _normal(generator_for(cfg.seed, f"{tag}.w_v"), (d, d), std)
        # Output projection is down-scaled by depth, the usual residual-growth
        # control; it also keeps activation norms inside a predictable band,
        # which the membrane's calibrated bounds rely on.
        self.w_o = _normal(
            generator_for(cfg.seed, f"{tag}.w_o"), (d, d), std / (2 * cfg.n_layers) ** 0.5
        )
        self.n_heads, self.d_head = h, dh
        self.scale = DTYPE(1.0 / np.sqrt(dh))

    def _split_heads(self, x: np.ndarray, t: int) -> np.ndarray:
        return x.reshape(t, self.n_heads, self.d_head).transpose(1, 0, 2)

    def forward(
        self, x: np.ndarray, cache: KVCache | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (output (T, d_model), attention weights (n_heads, T, T_kv))."""
        t, d = x.shape
        expect(x, (t, self.cfg.d_model), f"attn{self.layer_index}.input")

        q = self._split_heads(x @ self.w_q, t)
        k = self._split_heads(x @ self.w_k, t)
        v = self._split_heads(x @ self.w_v, t)
        expect(q, (self.n_heads, t, self.d_head), f"attn{self.layer_index}.q")

        if cache is not None:
            offset = cache.length
            k, v = cache.append(k, v)
        else:
            offset = 0
        t_kv = k.shape[1]

        scores = (q @ k.transpose(0, 2, 1)) * self.scale
        expect(scores, (self.n_heads, t, t_kv), f"attn{self.layer_index}.scores")

        # Causal mask: query at absolute position (offset + i) may attend to
        # key positions <= offset + i. Correct for both prefill and decode.
        q_pos = np.arange(offset, offset + t)[:, None]
        k_pos = np.arange(t_kv)[None, :]
        scores = np.where(k_pos <= q_pos, scores, DTYPE(-np.inf))

        weights = softmax_last_axis(scores)
        context = weights @ v
        merged = context.transpose(1, 0, 2).reshape(t, d)
        out = (merged @ self.w_o).astype(DTYPE)
        expect(out, (t, self.cfg.d_model), f"attn{self.layer_index}.output")
        return out, weights


class SwiGLUFeedForward:
    def __init__(self, cfg: ModelConfig, layer_index: int) -> None:
        self.cfg = cfg
        self.layer_index = layer_index
        d, f = cfg.d_model, cfg.d_ff
        tag = f"layer{layer_index}.ffn"
        std = (1.0 / d) ** 0.5
        self.w_gate = _normal(generator_for(cfg.seed, f"{tag}.w_gate"), (d, f), std)
        self.w_up = _normal(generator_for(cfg.seed, f"{tag}.w_up"), (d, f), std)
        self.w_down = _normal(
            generator_for(cfg.seed, f"{tag}.w_down"),
            (f, d),
            (1.0 / f) ** 0.5 / (2 * cfg.n_layers) ** 0.5,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        t = x.shape[0]
        expect(x, (t, self.cfg.d_model), f"ffn{self.layer_index}.input")
        gate = silu(x @ self.w_gate)
        up = x @ self.w_up
        expect(gate, (t, self.cfg.d_ff), f"ffn{self.layer_index}.gate")
        hidden = (gate * up).astype(DTYPE)
        out = (hidden @ self.w_down).astype(DTYPE)
        expect(out, (t, self.cfg.d_model), f"ffn{self.layer_index}.output")
        return out


class DecoderLayer:
    """Pre-norm decoder block: x + attn(ln1(x)), then x + ffn(ln2(x))."""

    def __init__(self, cfg: ModelConfig, layer_index: int) -> None:
        self.cfg = cfg
        self.layer_index = layer_index
        self.attn = MultiHeadSelfAttention(cfg, layer_index)
        self.ffn = SwiGLUFeedForward(cfg, layer_index)
        d = cfg.d_model
        self.ln1_gamma = np.ones((d,), dtype=DTYPE)
        self.ln1_beta = np.zeros((d,), dtype=DTYPE)
        self.ln2_gamma = np.ones((d,), dtype=DTYPE)
        self.ln2_beta = np.zeros((d,), dtype=DTYPE)

    def forward(
        self, x: np.ndarray, cache: KVCache | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        normed = layer_norm(x, self.ln1_gamma, self.ln1_beta, self.cfg.norm_eps)
        attn_out, weights = self.attn.forward(normed, cache)
        x = (x + attn_out).astype(DTYPE)
        normed2 = layer_norm(x, self.ln2_gamma, self.ln2_beta, self.cfg.norm_eps)
        x = (x + self.ffn.forward(normed2)).astype(DTYPE)
        expect(x, (x.shape[0], self.cfg.d_model), f"layer{self.layer_index}.output")
        return x, weights
