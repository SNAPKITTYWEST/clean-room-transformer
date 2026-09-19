"""Calibration of statistical activation envelopes.

A calibration pass runs the model over a corpus of clean inputs, records the
largest element magnitude and row norm seen at each boundary, and multiplies by
a safety margin. The result is a `CalibrationProfile`: a pinned, hashable
artifact that can be signed and shipped alongside the weights.

Limits, stated plainly: an envelope fitted on a corpus bounds only what that
corpus exercised. Inputs unlike the corpus can exceed it without anything being
wrong (false positive), and a corruption that stays inside the envelope will
not be caught by this tier (false negative). Structural and mathematical
invariants carry the load that actually has to hold; this tier is a drift
detector, and widening the margin trades sensitivity for false positives.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from transformer.config import ModelConfig
from transformer.hooks import StateEvent
from transformer.model import CleanRoomTransformer

from .audit import canonical_json
from .invariants import MagnitudeBounds


@dataclass
class CalibrationProfile:
    bounds: dict[str, MagnitudeBounds]
    margin: float
    samples: int
    schema_hash: str
    weight_fingerprint: str
    generalize_layers: bool = True

    def for_path(self, path: str) -> MagnitudeBounds | None:
        found = self.bounds.get(path)
        if found is not None or not self.generalize_layers:
            return found
        # A boundary seen at one layer index shares its contract with the same
        # boundary at another layer; fall back to the widest observed envelope
        # for that boundary kind so an uncalibrated layer is not unguarded.
        kind = path.split(".", 1)[-1]
        candidates = [b for p, b in self.bounds.items() if p.split(".", 1)[-1] == kind]
        if not candidates:
            return None
        return MagnitudeBounds(
            max_abs=max(c.max_abs for c in candidates),
            max_row_norm=max(c.max_row_norm for c in candidates),
        )

    def as_record(self) -> dict[str, object]:
        return {
            "margin": self.margin,
            "samples": self.samples,
            "schema_hash": self.schema_hash,
            "weight_fingerprint": self.weight_fingerprint,
            "bounds": {p: b.as_record() for p, b in sorted(self.bounds.items())},
        }

    def pin_hash(self) -> str:
        """Content hash of the profile, recorded in the audit chain."""
        return hashlib.sha256(canonical_json(self.as_record()).encode()).hexdigest()

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_record(), sort_keys=True, indent=indent)

    @classmethod
    def from_record(cls, record: dict) -> "CalibrationProfile":
        return cls(
            bounds={
                p: MagnitudeBounds(v["max_abs"], v["max_row_norm"])
                for p, v in record["bounds"].items()
            },
            margin=float(record["margin"]),
            samples=int(record["samples"]),
            schema_hash=str(record["schema_hash"]),
            weight_fingerprint=str(record["weight_fingerprint"]),
        )


@dataclass
class _Collector:
    observed: dict[str, list[float]] = field(default_factory=dict)

    def on_state(self, event: StateEvent) -> None:
        tensor = event.tensor
        if not isinstance(tensor, np.ndarray) or not np.isfinite(tensor).all():
            raise ValueError(
                f"calibration corpus produced a non-finite tensor at {event.path}; "
                "the corpus or the model is already unsound"
            )
        max_abs = float(np.abs(tensor).max())
        row_norm = float(
            np.linalg.norm(tensor.reshape(-1, tensor.shape[-1]), axis=-1).max()
        )
        current = self.observed.setdefault(event.path, [0.0, 0.0])
        current[0] = max(current[0], max_abs)
        current[1] = max(current[1], row_norm)


def calibrate(
    model: CleanRoomTransformer,
    cfg: ModelConfig,
    corpus: Iterable[Sequence[int]],
    margin: float = 1.5,
) -> CalibrationProfile:
    if margin < 1.0:
        raise ValueError("margin must be >= 1.0 or clean runs will be quarantined")

    collector = _Collector()
    samples = 0
    for tokens in corpus:
        token_array = np.asarray(list(tokens), dtype=np.int64)
        model.forward(token_array, observers=[collector])
        samples += 1

    if samples == 0:
        raise ValueError("calibration corpus is empty")

    bounds = {
        path: MagnitudeBounds(max_abs=values[0] * margin, max_row_norm=values[1] * margin)
        for path, values in collector.observed.items()
    }
    return CalibrationProfile(
        bounds=bounds,
        margin=margin,
        samples=samples,
        schema_hash=cfg.schema_hash(),
        weight_fingerprint=model.weight_fingerprint(),
    )


def default_corpus(cfg: ModelConfig, n: int = 24, seed: int = 7) -> list[list[int]]:
    """Deterministic synthetic corpus of byte sequences for calibration."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    lengths = rng.integers(4, min(cfg.max_seq_len, 32), size=n)
    return [
        rng.integers(0, cfg.vocab_size, size=int(length)).tolist() for length in lengths
    ]
