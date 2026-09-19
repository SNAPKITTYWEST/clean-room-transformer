"""Reproducibility of execution and of the audit trail it produces."""

from __future__ import annotations

from pathlib import Path

from conftest import PROMPT, TEST_SEED

from membrane import AuditLog, read_entries, signing_key_from_seed
from runtime import Status
from transformer import CleanRoomTransformer, ModelConfig


def run_with_log(path: Path, cfg: ModelConfig, build_engine, steps: int = 5):
    log = AuditLog(path, signing_key_from_seed(TEST_SEED))
    engine = build_engine(CleanRoomTransformer(cfg), audit=log)
    result = engine.generate(PROMPT, max_new_tokens=steps)
    log.close()
    return result


def stable_records(path: Path) -> list[dict]:
    """Audit records with the wall-clock field removed.

    `ts_ns` is the one field that legitimately differs between two identical
    runs; everything else must match exactly, which is what makes the trail a
    usable reproducibility artifact.
    """
    records = []
    for entry in read_entries(path):
        record = dict(entry["record"])
        record.pop("ts_ns", None)
        records.append(record)
    return records


def test_two_runs_emit_the_same_tokens(tmp_path: Path, cfg: ModelConfig, build_engine) -> None:
    first = run_with_log(tmp_path / "a.jsonl", cfg, build_engine)
    second = run_with_log(tmp_path / "b.jsonl", cfg, build_engine)

    assert first.status is second.status is Status.BUDGET_EXHAUSTED
    assert first.emitted == second.emitted
    assert first.weight_fingerprint == second.weight_fingerprint
    assert first.schema_hash == second.schema_hash


def test_audit_trail_is_reproducible_except_for_timestamps(
    tmp_path: Path, cfg: ModelConfig, build_engine
) -> None:
    path_a, path_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_with_log(path_a, cfg, build_engine)
    run_with_log(path_b, cfg, build_engine)
    assert stable_records(path_a) == stable_records(path_b)


def test_chain_head_differs_only_because_of_timestamps(
    tmp_path: Path, cfg: ModelConfig, build_engine
) -> None:
    """Honest detail: the head hash is not reproducible, and should not be.

    Records carry `ts_ns`, so two identical runs produce different heads. The
    reproducible artifact is the record sequence; the head identifies one
    concrete execution.
    """
    first = run_with_log(tmp_path / "a.jsonl", cfg, build_engine)
    second = run_with_log(tmp_path / "b.jsonl", cfg, build_engine)
    assert first.audit_head != second.audit_head


def test_schema_hash_tracks_config_changes(cfg: ModelConfig) -> None:
    assert cfg.schema_hash() == ModelConfig().schema_hash()
    assert cfg.schema_hash() != ModelConfig(n_heads=2).schema_hash()
    assert cfg.schema_hash() != ModelConfig(seed=cfg.seed + 1).schema_hash()


def test_weight_fingerprint_detects_a_single_perturbed_element(cfg: ModelConfig) -> None:
    model = CleanRoomTransformer(cfg)
    before = model.weight_fingerprint()
    model.layers[2].ffn.w_up[0, 0] += 1e-7
    assert model.weight_fingerprint() != before


def test_profile_pin_tracks_bound_changes(profile) -> None:
    from membrane import CalibrationProfile, MagnitudeBounds

    pinned = profile.pin_hash()
    widened = CalibrationProfile(
        bounds={**profile.bounds, "logits": MagnitudeBounds(1e9, 1e9)},
        margin=profile.margin,
        samples=profile.samples,
        schema_hash=profile.schema_hash,
        weight_fingerprint=profile.weight_fingerprint,
    )
    assert widened.pin_hash() != pinned


def test_profile_round_trips_through_json(profile) -> None:
    import json

    from membrane import CalibrationProfile

    restored = CalibrationProfile.from_record(json.loads(profile.to_json()))
    assert restored.pin_hash() == profile.pin_hash()
    assert restored.bounds.keys() == profile.bounds.keys()
