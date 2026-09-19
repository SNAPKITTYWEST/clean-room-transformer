"""MONITOR-mode semantics: record everything, block nothing that can continue.

A shadow deployment is only useful if it follows the same trajectory the
unguarded runtime would take. If a recorded violation silently changed the
emitted token, the shadow run would diverge from the thing it is meant to
predict.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import PROMPT, inject_into_layer_output

from membrane import IntegrityMembrane, Mode, QuarantineError, Tier, TokenPolicy
from runtime import Status
from transformer import CleanRoomTransformer, ModelConfig


def test_policy_violation_is_recorded_without_changing_the_trajectory(
    cfg: ModelConfig, build_engine
) -> None:
    unguarded = build_engine(CleanRoomTransformer(cfg)).generate(PROMPT, max_new_tokens=6)
    denied = frozenset(unguarded.emitted[:2])

    shadow = build_engine(
        CleanRoomTransformer(cfg),
        policy=TokenPolicy(cfg.vocab_size, forbidden_tokens=denied),
        mode=Mode.MONITOR,
    ).generate(PROMPT, max_new_tokens=6)

    assert shadow.status is not Status.QUARANTINED
    assert shadow.emitted == unguarded.emitted, "monitor mode altered the trajectory"
    # One violation per *occurrence*, not per denied id: a denied token that
    # recurs later in the trajectory is reported each time it is emitted.
    expected_hits = sum(1 for token in unguarded.emitted if token in denied)
    assert expected_hits >= len(denied)
    assert shadow.report.by_tier[Tier.POLICY.value] == expected_hits
    assert shadow.report.by_rule["token_denylist"] == expected_hits
    assert shadow.report.quarantines == 0


def test_monitor_mode_records_every_violating_step_not_just_the_first(
    cfg: ModelConfig, build_engine
) -> None:
    """ENFORCE halts at the first divergence, so it can only ever report one.
    MONITOR is what measures how often a fault actually fires."""
    model = CleanRoomTransformer(cfg)
    inject_into_layer_output(model, 2, lambda x: (x * 1e4).astype(np.float32))
    result = build_engine(model, mode=Mode.MONITOR).generate(PROMPT, max_new_tokens=5)

    assert result.status is Status.BUDGET_EXHAUSTED
    assert result.report.violations >= 5
    assert len(result.report.signatures) >= 5


def test_unrecoverable_fault_quarantines_even_in_monitor_mode(
    cfg: ModelConfig, profile
) -> None:
    """There is no token to decode from malformed logits, so there is nothing
    to let through: this raises in both modes."""
    membrane = IntegrityMembrane(
        cfg, TokenPolicy(cfg.vocab_size), profile, mode=Mode.MONITOR
    )
    membrane.begin_step(0, 4, 0)

    with pytest.raises(QuarantineError, match="logits_shape"):
        membrane.gate_emission(np.zeros((3,), dtype=np.float32), [], 0)

    assert membrane.report.quarantines == 1


def test_monitor_mode_emits_a_denylisted_token_and_says_so(
    cfg: ModelConfig, build_engine
) -> None:
    baseline = build_engine(CleanRoomTransformer(cfg)).generate(PROMPT, max_new_tokens=1)
    token = baseline.emitted[0]

    shadow = build_engine(
        CleanRoomTransformer(cfg),
        policy=TokenPolicy(cfg.vocab_size, forbidden_tokens=frozenset({token})),
        mode=Mode.MONITOR,
    ).generate(PROMPT, max_new_tokens=3)

    assert shadow.emitted[0] == token
    assert shadow.report.by_rule["token_denylist"] == 1
