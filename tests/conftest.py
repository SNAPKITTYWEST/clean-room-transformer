"""Shared fixtures and the fault-injection toolkit used by the suite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from membrane import (  # noqa: E402
    AuditLog,
    CalibrationProfile,
    IntegrityMembrane,
    Mode,
    TokenPolicy,
    TraceLevel,
    calibrate,
    default_corpus,
    signing_key_from_seed,
)
from runtime import CleanRoomEngine  # noqa: E402
from transformer import CleanRoomTransformer, ModelConfig  # noqa: E402

TEST_SEED = b"\x2a" * 32
PROMPT = [72, 101, 108, 108, 111]


@pytest.fixture(scope="session")
def cfg() -> ModelConfig:
    return ModelConfig()


@pytest.fixture(scope="session")
def model(cfg: ModelConfig) -> CleanRoomTransformer:
    return CleanRoomTransformer(cfg)


@pytest.fixture(scope="session")
def profile(model: CleanRoomTransformer, cfg: ModelConfig) -> CalibrationProfile:
    return calibrate(model, cfg, default_corpus(cfg), margin=1.5)


@pytest.fixture
def fresh_model(cfg: ModelConfig) -> CleanRoomTransformer:
    """An un-patched model, for tests that inject faults into layer objects."""
    return CleanRoomTransformer(cfg)


@pytest.fixture
def audit_log(tmp_path: Path) -> AuditLog:
    log = AuditLog(tmp_path / "audit.jsonl", signing_key_from_seed(TEST_SEED))
    yield log
    log.close()


@pytest.fixture
def build_engine(
    cfg: ModelConfig, profile: CalibrationProfile
) -> Callable[..., CleanRoomEngine]:
    """Factory: build an engine with a chosen model, policy, mode and audit log."""

    def _build(
        model: CleanRoomTransformer,
        policy: TokenPolicy | None = None,
        audit: AuditLog | None = None,
        mode: Mode = Mode.ENFORCE,
        trace: TraceLevel = TraceLevel.SUMMARY,
        use_profile: bool = True,
    ) -> CleanRoomEngine:
        membrane = IntegrityMembrane(
            cfg=cfg,
            policy=policy or TokenPolicy(cfg.vocab_size),
            profile=profile if use_profile else None,
            audit=audit,
            mode=mode,
            trace=trace,
            weight_fingerprint=model.weight_fingerprint(),
        )
        return CleanRoomEngine(model=model, membrane=membrane, audit=audit)

    return _build


# ---------------------------------------------------------------------------
# fault injection
# ---------------------------------------------------------------------------


def inject_into_layer_output(
    model: CleanRoomTransformer, index: int, corrupt: Callable[[np.ndarray], np.ndarray]
) -> None:
    """Corrupt the hidden state a decoder layer returns.

    The corrupted tensor is published to the membrane by `model.forward` exactly
    as a genuine one would be, so the interception path under test is the real
    one and not a stub.
    """
    layer = model.layers[index]
    original = layer.forward

    def patched(x, cache=None):
        out, weights = original(x, cache)
        return corrupt(out), weights

    layer.forward = patched  # type: ignore[method-assign]


def inject_into_attention(
    model: CleanRoomTransformer, index: int, corrupt: Callable[[np.ndarray], np.ndarray]
) -> None:
    """Corrupt the attention weight matrix a layer reports."""
    layer = model.layers[index]
    original = layer.forward

    def patched(x, cache=None):
        out, weights = original(x, cache)
        return out, corrupt(weights)

    layer.forward = patched  # type: ignore[method-assign]


# Each entry: (id, injector) where injector(model) installs one fault.
FAULT_CORPUS: list[tuple[str, Callable[[CleanRoomTransformer], None]]] = [
    (
        "nan_in_hidden_state",
        lambda m: inject_into_layer_output(
            m, 1, lambda x: _with_element(x, (0, 0), np.nan)
        ),
    ),
    (
        "inf_in_hidden_state",
        lambda m: inject_into_layer_output(
            m, 2, lambda x: _with_element(x, (0, 3), np.inf)
        ),
    ),
    (
        "negative_inf_in_hidden_state",
        lambda m: inject_into_layer_output(
            m, 0, lambda x: _with_element(x, (0, 1), -np.inf)
        ),
    ),
    (
        "truncated_sequence_axis",
        lambda m: inject_into_layer_output(m, 0, lambda x: x[:-1].copy()),
    ),
    (
        "widened_feature_axis",
        lambda m: inject_into_layer_output(
            m, 3, lambda x: np.concatenate([x, x[:, :1]], axis=1)
        ),
    ),
    (
        "dtype_promotion",
        lambda m: inject_into_layer_output(m, 1, lambda x: x.astype(np.float64)),
    ),
    (
        "rank_collapse",
        lambda m: inject_into_layer_output(m, 2, lambda x: x.reshape(-1)),
    ),
    (
        "magnitude_explosion",
        lambda m: inject_into_layer_output(m, 3, lambda x: (x * 1e4).astype(np.float32)),
    ),
    (
        "magnitude_explosion_single_element",
        lambda m: inject_into_layer_output(
            m, 0, lambda x: _with_element(x, (0, 0), 1e6)
        ),
    ),
    (
        "causal_mask_leak",
        lambda m: inject_into_attention(m, 1, _uniform_over_all_positions),
    ),
    (
        "attention_mass_not_normalized",
        lambda m: inject_into_attention(m, 2, lambda w: (w * 0.5).astype(np.float32)),
    ),
    (
        "negative_attention_weight",
        lambda m: inject_into_attention(
            m, 0, lambda w: _with_element(w, (0, 0, 0), -0.5)
        ),
    ),
    (
        "attention_weight_above_one",
        lambda m: inject_into_attention(
            m, 3, lambda w: _with_element(w, (0, 0, 0), 4.0)
        ),
    ),
    (
        "nan_in_attention",
        lambda m: inject_into_attention(
            m, 2, lambda w: _with_element(w, (0, 0, 0), np.nan)
        ),
    ),
]


def _with_element(array: np.ndarray, index: tuple[int, ...], value: float) -> np.ndarray:
    corrupted = array.copy()
    corrupted[index] = value
    return corrupted


def _uniform_over_all_positions(weights: np.ndarray) -> np.ndarray:
    """Replace a causal attention map with a uniform one over all key positions.

    Rows still sum to 1 and stay in [0, 1], so only the causality invariant can
    detect this. It is the interesting case: a corruption that looks healthy by
    every aggregate statistic.
    """
    uniform = np.full_like(weights, 1.0 / weights.shape[-1])
    return uniform.astype(np.float32)
