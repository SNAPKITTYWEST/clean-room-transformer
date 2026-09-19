"""Invariants checked at every instrumented boundary.

Three tiers, kept separate on purpose because they carry very different
epistemic weight:

STRUCTURAL   Declared contract of the boundary: rank, extents, dtype,
             finiteness. Violation means the tensor is not the object the
             next layer expects. Exact, no tolerance, no false positives.

MATHEMATICAL Properties that follow from the operation itself -- softmax rows
             sum to 1, attention weights lie in [0,1], masked positions are
             exactly 0. A violation is a defect or a tampered tensor, never a
             matter of degree. Checked with an explicit float tolerance.

STATISTICAL  Calibrated bounds on activation magnitude, derived by observing
             clean runs. These are heuristics: they detect gross drift and
             injected corruption, and they can both miss anomalies inside the
             band and fire on legitimate out-of-distribution inputs. They are
             not proofs of anything and are labelled accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class Tier(str, Enum):
    STRUCTURAL = "structural"
    MATHEMATICAL = "mathematical"
    STATISTICAL = "statistical"
    POLICY = "policy"


@dataclass(frozen=True)
class Violation:
    path: str
    rule: str
    tier: Tier
    detail: str
    measured: float | None = None
    bound: float | None = None

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "path": self.path,
            "rule": self.rule,
            "tier": self.tier.value,
            "detail": self.detail,
        }
        if self.measured is not None:
            record["measured"] = float(self.measured)
        if self.bound is not None:
            record["bound"] = float(self.bound)
        return record

    def __str__(self) -> str:
        suffix = ""
        if self.measured is not None and self.bound is not None:
            suffix = f" (measured {self.measured:.6g}, bound {self.bound:.6g})"
        return f"[{self.tier.value}] {self.path}: {self.rule}: {self.detail}{suffix}"


# ---------------------------------------------------------------------------
# structural
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralContract:
    """Declared shape/dtype contract for one boundary.

    `shape` entries may be `None` (any extent) or a string naming a dimension
    that must match a value supplied at check time (e.g. "t_kv").
    """

    shape: tuple[int | str | None, ...]
    dtype: str = "float32"

    def check(self, path: str, tensor: Any, bindings: dict[str, int]) -> list[Violation]:
        if not isinstance(tensor, np.ndarray):
            return [
                Violation(
                    path,
                    "is_ndarray",
                    Tier.STRUCTURAL,
                    f"expected ndarray, got {type(tensor).__name__}",
                )
            ]

        violations: list[Violation] = []
        if tensor.ndim != len(self.shape):
            return [
                Violation(
                    path,
                    "rank",
                    Tier.STRUCTURAL,
                    f"expected rank {len(self.shape)} {self.shape}, got rank "
                    f"{tensor.ndim} {tensor.shape}",
                )
            ]

        for axis, declared in enumerate(self.shape):
            actual = tensor.shape[axis]
            if declared is None:
                continue
            expected = bindings.get(declared) if isinstance(declared, str) else declared
            if expected is None:
                continue
            if actual != expected:
                violations.append(
                    Violation(
                        path,
                        "extent",
                        Tier.STRUCTURAL,
                        f"axis {axis} expected {expected}, got {actual} "
                        f"(full shape {tensor.shape})",
                        measured=float(actual),
                        bound=float(expected),
                    )
                )

        if str(tensor.dtype) != self.dtype:
            violations.append(
                Violation(
                    path,
                    "dtype",
                    Tier.STRUCTURAL,
                    f"expected {self.dtype}, got {tensor.dtype}",
                )
            )

        if not np.isfinite(tensor).all():
            n_nan = int(np.isnan(tensor).sum())
            n_inf = int(np.isinf(tensor).sum())
            violations.append(
                Violation(
                    path,
                    "finite",
                    Tier.STRUCTURAL,
                    f"{n_nan} NaN and {n_inf} Inf element(s) present",
                    measured=float(n_nan + n_inf),
                    bound=0.0,
                )
            )

        return violations


# ---------------------------------------------------------------------------
# mathematical
# ---------------------------------------------------------------------------


def check_attention_weights(
    path: str, weights: np.ndarray, q_offset: int, tol: float = 1e-4
) -> list[Violation]:
    """Properties that any correct causal-softmax attention map must satisfy."""
    violations: list[Violation] = []
    if weights.ndim != 3:
        return violations  # structural tier already reported this

    lo, hi = float(weights.min()), float(weights.max())
    if lo < -tol:
        violations.append(
            Violation(path, "attn_nonneg", Tier.MATHEMATICAL,
                      "attention weight below zero", measured=lo, bound=0.0)
        )
    if hi > 1.0 + tol:
        violations.append(
            Violation(path, "attn_max_one", Tier.MATHEMATICAL,
                      "attention weight above one", measured=hi, bound=1.0)
        )

    row_sums = weights.sum(axis=-1)
    worst = float(np.abs(row_sums - 1.0).max())
    if worst > tol:
        violations.append(
            Violation(path, "attn_rows_sum_to_one", Tier.MATHEMATICAL,
                      "softmax rows do not sum to 1", measured=worst, bound=tol)
        )

    # Causality: query at absolute position q_offset + i must place exactly
    # zero mass on key positions > q_offset + i.
    t_q, t_kv = weights.shape[1], weights.shape[2]
    q_pos = np.arange(q_offset, q_offset + t_q)[:, None]
    future = np.arange(t_kv)[None, :] > q_pos
    if future.any():
        leaked = float(np.abs(weights[:, future]).max())
        if leaked > tol:
            violations.append(
                Violation(path, "attn_causal_mask", Tier.MATHEMATICAL,
                          "non-zero attention mass on future positions",
                          measured=leaked, bound=tol)
            )

    return violations


# ---------------------------------------------------------------------------
# statistical
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MagnitudeBounds:
    """Calibrated activation-magnitude envelope for one boundary."""

    max_abs: float
    max_row_norm: float

    def as_record(self) -> dict[str, float]:
        return {"max_abs": self.max_abs, "max_row_norm": self.max_row_norm}


def check_magnitude(path: str, tensor: np.ndarray, bounds: MagnitudeBounds) -> list[Violation]:
    violations: list[Violation] = []
    if not np.isfinite(tensor).all():
        return violations  # structural tier owns this; magnitudes are meaningless

    measured_abs = float(np.abs(tensor).max())
    if measured_abs > bounds.max_abs:
        violations.append(
            Violation(path, "max_abs", Tier.STATISTICAL,
                      "element magnitude outside calibrated envelope",
                      measured=measured_abs, bound=bounds.max_abs)
        )

    row_norms = np.linalg.norm(tensor.reshape(-1, tensor.shape[-1]), axis=-1)
    measured_norm = float(row_norms.max())
    if measured_norm > bounds.max_row_norm:
        violations.append(
            Violation(path, "max_row_norm", Tier.STATISTICAL,
                      "row L2 norm outside calibrated envelope",
                      measured=measured_norm, bound=bounds.max_row_norm)
        )

    return violations
