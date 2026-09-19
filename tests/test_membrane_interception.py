"""Fault injection through the real execution path.

Every fault here is injected into a live layer object, so the corrupted tensor
reaches the membrane by the same route a genuine one does. Nothing is stubbed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from conftest import FAULT_CORPUS, PROMPT, inject_into_layer_output
from membrane import (
    CalibrationMismatch,
    IntegrityMembrane,
    Mode,
    QuarantineError,
    TokenPolicy,
    Tier,
)
from membrane.calibration import calibrate, default_corpus
from runtime import Status
from transformer import CleanRoomTransformer, ModelConfig

CLEAN_PROMPTS = [
    [72, 101, 108, 108, 111],
    [0],
    [255, 254, 1, 2, 3, 4, 5, 6],
    list(range(20, 40)),
    [7, 7, 7, 7],
]


@pytest.mark.parametrize("fault_id,inject", FAULT_CORPUS, ids=[f[0] for f in FAULT_CORPUS])
def test_every_injected_fault_is_quarantined(
    fault_id: str,
    inject: Callable[[CleanRoomTransformer], None],
    fresh_model: CleanRoomTransformer,
    build_engine,
) -> None:
    inject(fresh_model)
    engine = build_engine(fresh_model)
    result = engine.generate(PROMPT, max_new_tokens=8)

    assert result.status is Status.QUARANTINED, f"{fault_id} was not intercepted"
    assert result.quarantine is not None
    assert result.quarantine["violations"], "quarantine carried no violation detail"
    assert result.report.quarantines == 1
    assert len(result.quarantine["signature"]) == 64


def test_catch_rate_over_fault_corpus_is_total(cfg: ModelConfig, build_engine) -> None:
    """The headline claim, measured rather than asserted in prose.

    Scope: 100% of the faults enumerated in `FAULT_CORPUS`. This is a
    detection rate over a fixed corpus, not a proof that no undetectable
    corruption exists -- see README "Honest scoping".
    """
    caught = 0
    for fault_id, inject in FAULT_CORPUS:
        model = CleanRoomTransformer(cfg)
        inject(model)
        result = build_engine(model).generate(PROMPT, max_new_tokens=8)
        caught += int(result.status is Status.QUARANTINED)

    assert caught == len(FAULT_CORPUS)


@pytest.mark.parametrize("prompt", CLEAN_PROMPTS, ids=lambda p: f"len{len(p)}")
def test_clean_traffic_is_not_quarantined(
    prompt: list[int], fresh_model: CleanRoomTransformer, build_engine
) -> None:
    """False-positive check: the envelope must not fire on uncorrupted runs."""
    result = build_engine(fresh_model).generate(prompt, max_new_tokens=8)
    assert result.status is not Status.QUARANTINED
    assert result.report.violations == 0
    assert result.report.inspections > 0


def test_monitor_mode_records_but_does_not_block(
    fresh_model: CleanRoomTransformer, build_engine
) -> None:
    inject_into_layer_output(fresh_model, 1, lambda x: (x * 1e4).astype(np.float32))
    engine = build_engine(fresh_model, mode=Mode.MONITOR)
    result = engine.generate(PROMPT, max_new_tokens=4)

    assert result.status is not Status.QUARANTINED
    assert result.report.violations > 0
    assert result.report.quarantines == 0
    assert result.report.by_tier[Tier.STATISTICAL.value] > 0


def test_denylisted_emission_is_gated(
    cfg: ModelConfig, fresh_model: CleanRoomTransformer, build_engine
) -> None:
    """Deny the token the clean runtime actually emits, then require it be stopped."""
    baseline = build_engine(fresh_model).generate(PROMPT, max_new_tokens=1)
    assert baseline.emitted, "baseline produced no token to deny"
    first_token = baseline.emitted[0]

    policy = TokenPolicy(cfg.vocab_size, forbidden_tokens=frozenset({first_token}))
    result = build_engine(fresh_model, policy=policy).generate(PROMPT, max_new_tokens=4)

    assert result.status is Status.QUARANTINED
    assert result.emitted == []
    assert result.quarantine["violations"][0]["rule"] == "token_denylist"
    assert result.quarantine["violations"][0]["tier"] == Tier.POLICY.value


def test_step_budget_stops_generation(fresh_model: CleanRoomTransformer, build_engine) -> None:
    result = build_engine(fresh_model).generate(PROMPT, max_new_tokens=3)
    assert result.status is Status.BUDGET_EXHAUSTED
    assert len(result.emitted) == 3


def test_context_exhaustion_is_a_clean_stop(
    cfg: ModelConfig, fresh_model: CleanRoomTransformer, build_engine
) -> None:
    prompt = list(range(cfg.max_seq_len - 2))
    result = build_engine(fresh_model).generate(prompt, max_new_tokens=32)
    assert result.status is Status.CONTEXT_FULL
    assert result.report.violations == 0


def test_same_fault_yields_a_stable_signature(cfg: ModelConfig, build_engine) -> None:
    signatures = []
    for _ in range(2):
        model = CleanRoomTransformer(cfg)
        inject_into_layer_output(model, 1, lambda x: (x * 1e4).astype(np.float32))
        signatures.append(build_engine(model).generate(PROMPT, 4).quarantine["signature"])
    assert signatures[0] == signatures[1]


def test_different_faults_yield_different_signatures(cfg: ModelConfig, build_engine) -> None:
    results = {}
    for fault_id, inject in FAULT_CORPUS[:6]:
        model = CleanRoomTransformer(cfg)
        inject(model)
        results[fault_id] = build_engine(model).generate(PROMPT, 4).quarantine["signature"]
    assert len(set(results.values())) == len(results)


def test_quarantine_error_carries_structured_detail(
    fresh_model: CleanRoomTransformer, cfg: ModelConfig, profile
) -> None:
    from transformer.hooks import StateEvent

    membrane = IntegrityMembrane(cfg, TokenPolicy(cfg.vocab_size), profile)
    membrane.begin_step(0, 4, 0)
    bad = np.zeros((4, cfg.d_model), dtype=np.float32)
    bad[0, 0] = np.nan

    with pytest.raises(QuarantineError) as caught:
        membrane.on_state(StateEvent("layer_output", bad, 0, 1))

    error = caught.value
    assert error.step == 0
    assert {v.rule for v in error.violations} == {"finite"}
    assert "layer1.layer_output" in str(error)


def test_profile_from_other_config_is_refused(cfg: ModelConfig, profile) -> None:
    other = ModelConfig(n_layers=cfg.n_layers + 1)
    with pytest.raises(CalibrationMismatch, match="different model"):
        IntegrityMembrane(other, TokenPolicy(other.vocab_size), profile)


def test_profile_from_other_weights_is_refused(cfg: ModelConfig) -> None:
    trained_elsewhere = CleanRoomTransformer(ModelConfig(seed=cfg.seed + 99))
    profile = calibrate(trained_elsewhere, cfg, default_corpus(cfg, n=4))
    with pytest.raises(CalibrationMismatch, match="different weights"):
        IntegrityMembrane(
            cfg,
            TokenPolicy(cfg.vocab_size),
            profile,
            weight_fingerprint=CleanRoomTransformer(cfg).weight_fingerprint(),
        )


def test_calibration_refuses_margin_below_one(cfg: ModelConfig, model) -> None:
    with pytest.raises(ValueError, match="margin"):
        calibrate(model, cfg, default_corpus(cfg, n=2), margin=0.9)


def test_calibration_refuses_empty_corpus(cfg: ModelConfig, model) -> None:
    with pytest.raises(ValueError, match="empty"):
        calibrate(model, cfg, [], margin=1.5)


def test_uncalibrated_layer_falls_back_to_widest_envelope(profile) -> None:
    """A boundary absent from the profile is guarded, not waved through."""
    bounds = profile.for_path("layer999.layer_output")
    assert bounds is not None
    known = [b.max_abs for p, b in profile.bounds.items() if p.endswith("layer_output")]
    assert bounds.max_abs == max(known)
