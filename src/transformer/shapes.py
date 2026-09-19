"""Explicit tensor boundary checks.

Every tensor crossing a layer boundary is checked for rank, shape, dtype and
C-contiguity. These are cheap assertions on metadata, not on element values --
value-level invariants are the membrane's job (see src/membrane).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import DTYPE


class ShapeViolation(AssertionError):
    """Raised when a tensor does not match its declared boundary contract."""


def expect(
    array: np.ndarray,
    shape: Sequence[int | None],
    name: str,
    dtype: np.dtype | type = DTYPE,
) -> np.ndarray:
    """Assert that `array` matches `shape` and `dtype`; return it unchanged.

    A `None` entry in `shape` means "any extent on this axis".
    """
    if not isinstance(array, np.ndarray):
        raise ShapeViolation(f"{name}: expected ndarray, got {type(array).__name__}")
    if array.ndim != len(shape):
        raise ShapeViolation(
            f"{name}: expected rank {len(shape)} {tuple(shape)}, got rank "
            f"{array.ndim} {array.shape}"
        )
    for axis, (actual, declared) in enumerate(zip(array.shape, shape)):
        if declared is not None and actual != declared:
            raise ShapeViolation(
                f"{name}: axis {axis} expected {declared}, got {actual} "
                f"(full shape {array.shape}, declared {tuple(shape)})"
            )
    if array.dtype != np.dtype(dtype):
        raise ShapeViolation(
            f"{name}: expected dtype {np.dtype(dtype)}, got {array.dtype}"
        )
    return array
