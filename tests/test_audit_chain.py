"""Audit chain: signing, linkage, and detection of every edit class."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import PROMPT, TEST_SEED

from membrane import (
    GENESIS_HASH,
    AuditLog,
    canonical_json,
    entry_hash,
    generate_signing_key,
    read_entries,
    signing_key_from_seed,
    verify_log,
)


def write_log(path: Path, n: int = 6) -> AuditLog:
    log = AuditLog(path, signing_key_from_seed(TEST_SEED))
    for i in range(n):
        log.append({"event": "state", "path": f"layer{i}.layer_output", "i": i})
    log.close()
    return log


def rewrite(path: Path, lines: list[str]) -> None:
    path.write_text("".join(line if line.endswith("\n") else line + "\n" for line in lines))


def test_clean_log_verifies(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = write_log(path)
    report = verify_log(path, log.public_key, expected_head=log.head)
    assert report.ok
    assert report.entries == 6
    assert report.head == log.head
    assert report.failures == []
    assert bool(report) is True


def test_first_entry_links_to_genesis(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    write_log(path, n=1)
    first = next(read_entries(path))
    assert first["prev"] == GENESIS_HASH
    assert first["seq"] == 0


def test_hashes_chain_forward(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    write_log(path, n=4)
    entries = list(read_entries(path))
    for previous, current in zip(entries, entries[1:]):
        assert current["prev"] == previous["hash"]
        assert current["hash"] == entry_hash(
            current["seq"], current["prev"], current["record"]
        )


def test_modified_record_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = write_log(path)
    lines = path.read_text().splitlines()
    entry = json.loads(lines[3])
    entry["record"]["i"] = 999  # hash and signature left intact
    lines[3] = canonical_json(entry)
    rewrite(path, lines)

    report = verify_log(path, log.public_key)
    assert not report.ok
    assert "hash mismatch" in report.failures[0]


def test_deleted_entry_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = write_log(path)
    lines = path.read_text().splitlines()
    del lines[2]
    rewrite(path, lines)

    report = verify_log(path, log.public_key)
    assert not report.ok
    assert "seq is" in report.failures[0]


def test_reordered_entries_are_detected(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = write_log(path)
    lines = path.read_text().splitlines()
    lines[2], lines[3] = lines[3], lines[2]
    rewrite(path, lines)

    report = verify_log(path, log.public_key)
    assert not report.ok


def test_forged_entry_signed_by_another_key_is_detected(tmp_path: Path) -> None:
    """An attacker who can rewrite the file but lacks the signing key cannot
    produce an entry that verifies."""
    path = tmp_path / "audit.jsonl"
    log = write_log(path)
    attacker_key = generate_signing_key()

    lines = path.read_text().splitlines()
    entry = json.loads(lines[4])
    entry["record"]["i"] = -1
    forged_hash = entry_hash(entry["seq"], entry["prev"], entry["record"])
    entry["hash"] = forged_hash
    entry["sig"] = attacker_key.sign(bytes.fromhex(forged_hash)).hex()
    lines[4] = canonical_json(entry)
    rewrite(path, lines)

    report = verify_log(path, log.public_key)
    assert not report.ok
    assert "invalid signature" in report.failures[0]
    # Nor does swapping in the attacker's public key rescue the log: the
    # entries before and after the forgery were signed by the genuine key and
    # now fail instead. Verification is only meaningful against a key pinned
    # out of band -- a key read from the log's own directory proves nothing.
    assert not verify_log(path, attacker_key.public_key()).ok


def test_truncation_is_only_detectable_against_a_pinned_head(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = write_log(path)
    lines = path.read_text().splitlines()
    rewrite(path, lines[:-2])

    # Linkage alone cannot see a missing tail: the prefix is internally valid.
    assert verify_log(path, log.public_key).ok

    report = verify_log(path, log.public_key, expected_head=log.head)
    assert not report.ok
    assert "truncated" in report.failures[0]


def test_malformed_line_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = write_log(path, n=2)
    path.write_text(path.read_text() + "{not json\n")
    with pytest.raises(ValueError, match="malformed JSON"):
        verify_log(path, log.public_key)


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = write_log(path, n=3)
    path.write_text(path.read_text() + "\n\n")
    assert verify_log(path, log.public_key).ok


def test_existing_nonempty_log_is_never_reopened(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    write_log(path, n=2)
    with pytest.raises(FileExistsError, match="append-only"):
        AuditLog(path, signing_key_from_seed(TEST_SEED))


def test_append_after_close_is_refused(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", signing_key_from_seed(TEST_SEED))
    log.append({"event": "state"})
    log.close()
    with pytest.raises(RuntimeError, match="closed"):
        log.append({"event": "state"})


def test_writes_are_durable_after_flush(tmp_path: Path) -> None:
    """The queue is asynchronous, so flush() is what makes a read-back valid."""
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, signing_key_from_seed(TEST_SEED))
    for i in range(200):
        log.append({"event": "state", "i": i})
    log.flush()
    assert len(list(read_entries(path))) == 200
    log.close()


def test_writer_errors_surface_on_flush(tmp_path: Path) -> None:
    """A record that cannot be canonically encoded must not vanish silently."""
    log = AuditLog(tmp_path / "audit.jsonl", signing_key_from_seed(TEST_SEED))
    log.append({"event": "state", "bad": float("nan")})
    with pytest.raises(RuntimeError, match="audit writer failed"):
        log.flush()


def test_canonical_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("inf")})


def test_canonical_json_is_key_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_signing_key_seed_length_is_checked() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        signing_key_from_seed(b"short")


def test_generation_produces_a_verifiable_chain(tmp_path: Path, fresh_model, build_engine) -> None:
    """End to end: a real run's audit log verifies against its reported head."""
    path = tmp_path / "run.jsonl"
    log = AuditLog(path, signing_key_from_seed(TEST_SEED))
    engine = build_engine(fresh_model, audit=log)
    result = engine.generate(PROMPT, max_new_tokens=6)
    log.close()

    report = verify_log(path, log.public_key, expected_head=result.audit_head)
    assert report.ok, report.failures

    events = [e["record"]["event"] for e in read_entries(path)]
    assert events[0] == "session_open"
    assert events[-1] == "session_close"
    assert "state" in events and "emission" in events


def test_session_open_pins_model_identity(tmp_path: Path, fresh_model, build_engine) -> None:
    path = tmp_path / "run.jsonl"
    log = AuditLog(path, signing_key_from_seed(TEST_SEED))
    engine = build_engine(fresh_model, audit=log)
    result = engine.generate(PROMPT, max_new_tokens=2)
    log.close()

    opening = next(read_entries(path))["record"]
    assert opening["weight_fingerprint"] == result.weight_fingerprint
    assert opening["schema_hash"] == result.schema_hash
    assert opening["profile_pin"] is not None
    assert opening["mode"] == "enforce"


def test_divergence_is_recorded_with_its_signature(tmp_path: Path, cfg, build_engine) -> None:
    from conftest import inject_into_layer_output
    from transformer import CleanRoomTransformer

    model = CleanRoomTransformer(cfg)
    inject_into_layer_output(model, 1, lambda x: x * 0 + 1e9)

    path = tmp_path / "diverge.jsonl"
    log = AuditLog(path, signing_key_from_seed(TEST_SEED))
    result = build_engine(model, audit=log).generate(PROMPT, max_new_tokens=4)
    log.close()

    assert verify_log(path, log.public_key, expected_head=result.audit_head).ok
    divergences = [
        e["record"] for e in read_entries(path) if e["record"]["event"] == "divergence"
    ]
    assert len(divergences) == 1
    assert divergences[0]["signature"] == result.quarantine["signature"]
    assert divergences[0]["violations"]
