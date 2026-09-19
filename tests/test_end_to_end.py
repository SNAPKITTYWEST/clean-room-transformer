"""Whole-system behaviour: interception ordering, logging thread, reporting."""

from __future__ import annotations

import threading
from pathlib import Path

from conftest import FAULT_CORPUS, PROMPT, TEST_SEED, inject_into_layer_output

from membrane import AuditLog, Mode, read_entries, signing_key_from_seed, verify_log
from runtime import Status
from transformer import CleanRoomTransformer, ModelConfig

CLEAN_CORPUS = [
    [72, 101, 108, 108, 111],
    [1, 2, 3],
    [200, 201, 202, 203, 204, 205],
    list(range(10, 30)),
]


def test_clean_run_completes_and_verifies(tmp_path: Path, cfg: ModelConfig, build_engine) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, signing_key_from_seed(TEST_SEED))
    engine = build_engine(CleanRoomTransformer(cfg), audit=log)
    result = engine.generate(PROMPT, max_new_tokens=8)
    log.close()

    assert result.status is Status.BUDGET_EXHAUSTED
    assert len(result.emitted) == 8
    assert all(0 <= t < cfg.vocab_size for t in result.emitted)
    assert result.report.violations == 0
    assert result.report.quarantines == 0
    # 8 steps x (embedding + 4x(attention + layer_output) + final_hidden + logits)
    assert result.report.inspections == 8 * (1 + 2 * cfg.n_layers + 2)
    assert verify_log(path, log.public_key, expected_head=result.audit_head).ok


def test_interception_precedes_downstream_layers(cfg: ModelConfig, build_engine) -> None:
    """The membrane must stop the pass, not merely notice afterwards.

    A fault at layer 1 must prevent layers 2 and 3 from ever running. Counting
    inspections is how that is observable from outside.
    """
    model = CleanRoomTransformer(cfg)
    inject_into_layer_output(model, 1, lambda x: (x * 1e6).astype("float32"))
    result = build_engine(model).generate(PROMPT, max_new_tokens=4)

    assert result.status is Status.QUARANTINED
    # embedding, layer0 attention+output, layer1 attention+output = 5 inspections.
    assert result.report.inspections == 5
    assert result.report.inspections < 1 + 2 * cfg.n_layers + 2


def test_emission_gate_runs_after_the_full_stack(cfg: ModelConfig, build_engine) -> None:
    from membrane import TokenPolicy

    model = CleanRoomTransformer(cfg)
    baseline = build_engine(model).generate(PROMPT, max_new_tokens=1)
    policy = TokenPolicy(cfg.vocab_size, forbidden_tokens=frozenset(baseline.emitted[:1]))

    result = build_engine(model, policy=policy).generate(PROMPT, max_new_tokens=4)
    assert result.status is Status.QUARANTINED
    # The whole stack ran and passed; only the emission gate objected.
    assert result.report.inspections == 1 + 2 * cfg.n_layers + 2
    assert result.report.by_tier["policy"] == 1


def test_audit_writes_happen_off_the_calling_thread(tmp_path: Path) -> None:
    """Non-blocking logging claim: hashing, signing and I/O run on the writer."""
    log = AuditLog(tmp_path / "audit.jsonl", signing_key_from_seed(TEST_SEED))
    writer_idents: set[int] = set()
    original = log._write_one

    def spy(handle, record):
        writer_idents.add(threading.get_ident())
        return original(handle, record)

    log._write_one = spy  # type: ignore[method-assign]
    for i in range(50):
        log.append({"event": "state", "i": i})
    log.flush()
    log.close()

    assert writer_idents, "writer never ran"
    assert threading.get_ident() not in writer_idents


def test_false_positive_rate_on_clean_corpus_is_zero(cfg: ModelConfig, build_engine) -> None:
    """Measured in MONITOR mode so that a single violation cannot mask later ones."""
    total_violations = 0
    for prompt in CLEAN_CORPUS:
        engine = build_engine(CleanRoomTransformer(cfg), mode=Mode.MONITOR)
        result = engine.generate(prompt, max_new_tokens=6)
        total_violations += result.report.violations
    assert total_violations == 0


def test_detection_rate_report(cfg: ModelConfig, build_engine) -> None:
    """Produces the numbers quoted in the README, and fails if they regress."""
    detected = {}
    for fault_id, inject in FAULT_CORPUS:
        model = CleanRoomTransformer(cfg)
        inject(model)
        result = build_engine(model).generate(PROMPT, max_new_tokens=6)
        rules = (
            {v["rule"] for v in result.quarantine["violations"]}
            if result.quarantine
            else set()
        )
        detected[fault_id] = (result.status is Status.QUARANTINED, sorted(rules))

    missed = [name for name, (caught, _) in detected.items() if not caught]
    assert not missed, f"undetected faults: {missed}"
    # Each fault must be caught by a rule, not by an unrelated crash.
    assert all(rules for _, rules in detected.values())


def test_session_records_are_complete_for_a_quarantined_run(
    tmp_path: Path, cfg: ModelConfig, build_engine
) -> None:
    model = CleanRoomTransformer(cfg)
    inject_into_layer_output(model, 0, lambda x: x.astype("float64"))

    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, signing_key_from_seed(TEST_SEED))
    result = build_engine(model, audit=log).generate(PROMPT, max_new_tokens=4)
    log.close()

    records = [e["record"] for e in read_entries(path)]
    events = [r["event"] for r in records]
    assert events[0] == "session_open"
    assert "divergence" in events
    assert events[-1] == "session_close"

    closing = records[-1]
    assert closing["status"] == "quarantined"
    assert closing["emitted"] == []
    assert closing["membrane"]["quarantines"] == 1
    assert verify_log(path, log.public_key, expected_head=result.audit_head).ok
