"""Meta-test: the invariants must be load-bearing, not decorative.

If every fault in the corpus were still "caught" with the membrane's checks
removed, the suite would be measuring something other than the membrane --
NumPy's own shape rules, say. These tests disable tiers and require that faults
start getting through.

`scripts/mutation_probe.py` prints the full per-fault table; this file asserts
the properties that must hold.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pytest
from conftest import PROMPT, inject_into_attention, inject_into_layer_output

import membrane.interceptor as interceptor
from membrane.invariants import StructuralContract
from runtime import Status
from transformer import CleanRoomTransformer, ModelConfig


@pytest.fixture(autouse=True)
def _quiet_numpy():
    """NaN propagation is the point of some of these mutations, not a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        old = np.seterr(all="ignore")
        try:
            yield
        finally:
            np.seterr(**old)


@pytest.fixture
def without_structural_rules():
    original = StructuralContract.check
    applied: list[set[str]] = []

    def disable(rules: set[str]) -> None:
        applied.append(rules)

        def patched(self, path, tensor, bindings):
            return [
                v for v in original(self, path, tensor, bindings) if v.rule not in rules
            ]

        StructuralContract.check = patched  # type: ignore[method-assign]

    yield disable
    StructuralContract.check = original  # type: ignore[method-assign]


@pytest.fixture
def without_check():
    originals: dict[str, object] = {}

    def disable(name: str) -> None:
        originals[name] = getattr(interceptor, name)
        setattr(interceptor, name, lambda *a, **k: [])

    yield disable
    for name, original in originals.items():
        setattr(interceptor, name, original)


def run(cfg: ModelConfig, build_engine, inject: Callable) -> Status | None:
    model = CleanRoomTransformer(cfg)
    inject(model)
    try:
        return build_engine(model).generate(PROMPT, max_new_tokens=4).status
    except Exception:
        return None  # crashed downstream rather than being intercepted


def test_finite_check_is_what_catches_nan(cfg, build_engine, without_structural_rules) -> None:
    inject = lambda m: inject_into_layer_output(  # noqa: E731
        m, 1, lambda x: _poison(x, np.nan)
    )
    assert run(cfg, build_engine, inject) is Status.QUARANTINED

    without_structural_rules({"finite"})
    assert run(cfg, build_engine, inject) is not Status.QUARANTINED


def test_attention_math_is_what_catches_a_causal_leak(cfg, build_engine, without_check) -> None:
    inject = lambda m: inject_into_attention(  # noqa: E731
        m, 1, lambda w: np.full_like(w, 1.0 / w.shape[-1])
    )
    assert run(cfg, build_engine, inject) is Status.QUARANTINED

    without_check("check_attention_weights")
    assert run(cfg, build_engine, inject) is not Status.QUARANTINED


def test_magnitude_envelope_is_what_catches_drift(cfg, build_engine, without_check) -> None:
    inject = lambda m: inject_into_layer_output(  # noqa: E731
        m, 3, lambda x: (x * 1e4).astype(np.float32)
    )
    assert run(cfg, build_engine, inject) is Status.QUARANTINED

    without_check("check_magnitude")
    assert run(cfg, build_engine, inject) is not Status.QUARANTINED


def test_dtype_check_is_what_catches_promotion(cfg, build_engine, without_structural_rules) -> None:
    """float64 hidden states would otherwise flow all the way to emission."""
    inject = lambda m: inject_into_layer_output(m, 1, lambda x: x.astype(np.float64))  # noqa: E731
    assert run(cfg, build_engine, inject) is Status.QUARANTINED

    without_structural_rules({"rank", "extent", "dtype"})
    assert run(cfg, build_engine, inject) is not Status.QUARANTINED


def test_with_every_tier_disabled_corruption_reaches_emission(
    cfg, build_engine, without_structural_rules, without_check
) -> None:
    without_structural_rules({"finite", "rank", "extent", "dtype"})
    without_check("check_attention_weights")
    without_check("check_magnitude")

    faults = [
        lambda m: inject_into_layer_output(m, 1, lambda x: _poison(x, np.nan)),
        lambda m: inject_into_layer_output(m, 3, lambda x: (x * 1e4).astype(np.float32)),
        lambda m: inject_into_attention(m, 1, lambda w: np.full_like(w, 1.0 / w.shape[-1])),
    ]
    statuses = [run(cfg, build_engine, inject) for inject in faults]
    assert all(status is not Status.QUARANTINED for status in statuses), (
        "faults were still intercepted with every tier disabled, so the suite is "
        "not measuring the membrane"
    )


def _poison(x: np.ndarray, value: float) -> np.ndarray:
    corrupted = x.copy()
    corrupted[0, 0] = value
    return corrupted
