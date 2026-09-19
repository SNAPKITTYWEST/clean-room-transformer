#!/usr/bin/env python3
"""Produce the verification report: what is detected, by which invariant.

Run from the repository root:

    python3 scripts/verification_report.py

Prints a table mapping each injected fault to the rule that intercepted it, the
detection rate over the fault corpus, the false-positive count over the clean
corpus, and the result of verifying a real run's audit chain.
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import FAULT_CORPUS, PROMPT, TEST_SEED  # noqa: E402
from membrane import (  # noqa: E402
    AuditLog,
    IntegrityMembrane,
    Mode,
    TokenPolicy,
    TraceLevel,
    calibrate,
    default_corpus,
    signing_key_from_seed,
    verify_log,
)
from runtime import CleanRoomEngine, Status  # noqa: E402
from transformer import CleanRoomTransformer, ModelConfig  # noqa: E402

CLEAN_CORPUS = [
    [72, 101, 108, 108, 111],
    [1, 2, 3],
    [200, 201, 202, 203, 204, 205],
    list(range(10, 30)),
    [7, 7, 7, 7],
]


def build(cfg, model, profile, policy=None, audit=None, mode=Mode.ENFORCE):
    membrane = IntegrityMembrane(
        cfg,
        policy or TokenPolicy(cfg.vocab_size),
        profile,
        audit,
        mode,
        TraceLevel.SUMMARY,
        weight_fingerprint=model.weight_fingerprint(),
    )
    return CleanRoomEngine(model=model, membrane=membrane, audit=audit)


def main() -> int:
    cfg = ModelConfig()
    reference = CleanRoomTransformer(cfg)
    profile = calibrate(reference, cfg, default_corpus(cfg), margin=1.5)

    print("=" * 78)
    print("CLEAN-ROOM TRANSFORMER / INTEGRITY MEMBRANE -- VERIFICATION REPORT")
    print("=" * 78)
    print(f"config schema hash   {cfg.schema_hash()}")
    print(f"weight fingerprint   {reference.weight_fingerprint()}")
    print(f"calibration pin      {profile.pin_hash()}")
    print(f"calibrated boundaries {len(profile.bounds)}  margin {profile.margin}")
    print()

    print("-" * 78)
    print("FAULT INJECTION (ENFORCE mode)")
    print("-" * 78)
    print(f"{'fault':<38} {'caught':<7} {'tier':<13} rule(s)")
    caught_count = 0
    for fault_id, inject in FAULT_CORPUS:
        model = CleanRoomTransformer(cfg)
        inject(model)
        result = build(cfg, model, profile).generate(PROMPT, max_new_tokens=6)
        caught = result.status is Status.QUARANTINED
        caught_count += int(caught)
        violations = result.quarantine["violations"] if result.quarantine else []
        tiers = sorted({v["tier"] for v in violations})
        rules = sorted({v["rule"] for v in violations})
        print(
            f"{fault_id:<38} {'YES' if caught else 'NO':<7} "
            f"{','.join(tiers):<13} {','.join(rules)}"
        )
    print()
    print(f"detection rate: {caught_count}/{len(FAULT_CORPUS)} "
          f"({100.0 * caught_count / len(FAULT_CORPUS):.1f}%) over this corpus")
    print()

    print("-" * 78)
    print("CLEAN TRAFFIC (MONITOR mode -- counts every violation, blocks nothing)")
    print("-" * 78)
    false_positives = 0
    inspections = 0
    for prompt in CLEAN_CORPUS:
        result = build(cfg, CleanRoomTransformer(cfg), profile, mode=Mode.MONITOR).generate(
            prompt, max_new_tokens=6
        )
        false_positives += result.report.violations
        inspections += result.report.inspections
        print(f"prompt len {len(prompt):<3} inspections {result.report.inspections:<5} "
              f"violations {result.report.violations}")
    print()
    print(f"false positives: {false_positives} over {inspections} boundary inspections")
    print()

    print("-" * 78)
    print("AUDIT CHAIN")
    print("-" * 78)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "audit.jsonl"
        log = AuditLog(path, signing_key_from_seed(TEST_SEED))
        result = build(cfg, CleanRoomTransformer(cfg), profile, audit=log).generate(
            PROMPT, max_new_tokens=6
        )
        log.close()
        report = verify_log(path, log.public_key, expected_head=result.audit_head)
        print(f"entries              {report.entries}")
        print(f"head                 {report.head}")
        print(f"chain verified       {report.ok}")
        print(f"log size             {path.stat().st_size} bytes")

        # Tamper with one byte of one record and re-verify.
        lines = path.read_text().splitlines()
        target = len(lines) // 2
        lines[target] = lines[target].replace('"max_abs":', '"max_abs_":', 1)
        path.write_text("\n".join(lines) + "\n")
        tampered = verify_log(path, log.public_key, expected_head=result.audit_head)
        print(f"after 1-byte edit    verified={tampered.ok}  {tampered.failures[:1]}")

    print()
    print("Scope note: the detection rate above is measured over the enumerated")
    print("fault corpus. It is not a proof that every possible corruption is")
    print("detectable. See README section 'Honest scoping'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
