"""Deterministic RNG plumbing.

Design rule: no global RNG state. Every consumer of randomness receives an
explicitly derived generator, so execution order changes cannot silently
change results. Sub-generators are derived by *name*, not by draw order, via
SeedSequence entropy mixing -- so adding a new parameter tensor does not
perturb the initialization of existing ones.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _name_to_entropy(name: str) -> int:
    """Map a stable string key to a 128-bit integer for SeedSequence mixing."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()[:16]
    return int.from_bytes(digest, "big")


def generator_for(seed: int, name: str) -> np.random.Generator:
    """Return the generator for a named tensor. Pure function of (seed, name)."""
    seq = np.random.SeedSequence(entropy=[int(seed), _name_to_entropy(name)])
    return np.random.Generator(np.random.PCG64(seq))


def fingerprint(array: np.ndarray) -> str:
    """Stable content hash of an array, including shape and dtype.

    Used for reproducibility assertions and audit records. C-contiguity is
    forced so that the byte view does not depend on memory layout history.
    """
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(str(contiguous.shape).encode("utf-8"))
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()
