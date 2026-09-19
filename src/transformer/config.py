"""Model configuration with a content-addressed schema hash.

The schema hash is the SHA-256 of the canonical JSON encoding of the config.
It is recorded in every audit entry so a log can be tied to the exact model
geometry that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

# The single dtype used across the entire runtime. Mixed precision is
# deliberately not supported: it is a source of non-reproducibility.
DTYPE = np.float32


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256  # byte-level vocabulary; no external tokenizer
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 176  # SwiGLU inner width; ~= 8/3 * d_model rounded to mult of 8
    max_seq_len: int = 64
    norm_eps: float = 1e-5
    seed: int = 20260918

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        if self.d_model <= 0 or self.n_layers <= 0 or self.vocab_size <= 0:
            raise ValueError("dimensions must be positive")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def schema_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
