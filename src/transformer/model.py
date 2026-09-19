"""Decoder-only transformer with deterministic initialization and state hooks."""

from __future__ import annotations

import numpy as np

from .config import DTYPE, ModelConfig
from .determinism import fingerprint, generator_for
from .hooks import StateEvent, StateObserver
from .layers import DecoderLayer, KVCache, layer_norm
from .shapes import expect


class CleanRoomTransformer:
    """A small, fully inspectable decoder-only transformer.

    Weights are untrained: they are deterministically initialized from
    `cfg.seed`. The point of this object is the execution and instrumentation
    path, not language modelling quality.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        std = (1.0 / cfg.d_model) ** 0.5
        self.token_embedding = (
            generator_for(cfg.seed, "embed.token")
            .normal(0.0, std, size=(cfg.vocab_size, cfg.d_model))
            .astype(DTYPE)
        )
        self.position_embedding = (
            generator_for(cfg.seed, "embed.position")
            .normal(0.0, std, size=(cfg.max_seq_len, cfg.d_model))
            .astype(DTYPE)
        )
        self.layers = [DecoderLayer(cfg, i) for i in range(cfg.n_layers)]
        self.final_gamma = np.ones((cfg.d_model,), dtype=DTYPE)
        self.final_beta = np.zeros((cfg.d_model,), dtype=DTYPE)
        # Untied output head.
        self.w_out = (
            generator_for(cfg.seed, "head.w_out")
            .normal(0.0, std, size=(cfg.d_model, cfg.vocab_size))
            .astype(DTYPE)
        )

    # -- identity -----------------------------------------------------------

    def weight_fingerprint(self) -> str:
        """Content hash over every parameter tensor, in a fixed order.

        Two runtimes that report the same fingerprint are executing bit-identical
        weights; this is what "no unverified weight execution" reduces to in
        practice.
        """
        parts = [
            fingerprint(self.token_embedding),
            fingerprint(self.position_embedding),
        ]
        for layer in self.layers:
            parts += [
                fingerprint(layer.attn.w_q),
                fingerprint(layer.attn.w_k),
                fingerprint(layer.attn.w_v),
                fingerprint(layer.attn.w_o),
                fingerprint(layer.ffn.w_gate),
                fingerprint(layer.ffn.w_up),
                fingerprint(layer.ffn.w_down),
                fingerprint(layer.ln1_gamma),
                fingerprint(layer.ln1_beta),
                fingerprint(layer.ln2_gamma),
                fingerprint(layer.ln2_beta),
            ]
        parts += [
            fingerprint(self.final_gamma),
            fingerprint(self.final_beta),
            fingerprint(self.w_out),
        ]
        import hashlib

        return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()

    def new_caches(self) -> list[KVCache]:
        return [
            KVCache(self.cfg.n_heads, self.cfg.max_seq_len, self.cfg.d_head)
            for _ in range(self.cfg.n_layers)
        ]

    # -- execution ----------------------------------------------------------

    def forward(
        self,
        tokens: np.ndarray,
        caches: list[KVCache] | None = None,
        observers: list[StateObserver] | None = None,
        step: int = 0,
    ) -> np.ndarray:
        """Run the stack over `tokens` and return logits of shape (T, vocab).

        `caches` carries KV state across calls for incremental decoding. Every
        instrumented boundary is published to `observers` before execution
        continues, so an observer may abort the pass by raising.
        """
        observers = observers or []
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.ndim != 1:
            raise ValueError(f"tokens must be rank 1, got shape {tokens.shape}")
        if tokens.size == 0:
            raise ValueError("tokens must be non-empty")
        if tokens.min() < 0 or tokens.max() >= self.cfg.vocab_size:
            raise ValueError(
                f"token id out of range [0, {self.cfg.vocab_size}): "
                f"min={tokens.min()} max={tokens.max()}"
            )

        t = tokens.shape[0]
        offset = caches[0].length if caches else 0
        if offset + t > self.cfg.max_seq_len:
            raise ValueError(
                f"context overflow: {offset} + {t} > max_seq_len "
                f"{self.cfg.max_seq_len}"
            )

        positions = np.arange(offset, offset + t)
        x = (self.token_embedding[tokens] + self.position_embedding[positions]).astype(DTYPE)
        expect(x, (t, self.cfg.d_model), "embedding.output")
        self._publish(observers, StateEvent("embedding", x, step))

        for index, layer in enumerate(self.layers):
            cache = caches[index] if caches else None
            x, weights = layer.forward(x, cache)
            self._publish(
                observers,
                StateEvent("attention", weights, step, index, {"t_kv": weights.shape[-1]}),
            )
            self._publish(observers, StateEvent("layer_output", x, step, index))

        x = layer_norm(x, self.final_gamma, self.final_beta, self.cfg.norm_eps)
        self._publish(observers, StateEvent("final_hidden", x, step))

        logits = (x @ self.w_out).astype(DTYPE)
        expect(logits, (t, self.cfg.vocab_size), "head.logits")
        self._publish(observers, StateEvent("logits", logits, step))
        return logits

    @staticmethod
    def _publish(observers: list[StateObserver], event: StateEvent) -> None:
        for observer in observers:
            observer.on_state(event)
