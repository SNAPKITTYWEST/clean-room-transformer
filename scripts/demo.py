#!/usr/bin/env python3
"""Minimal end-to-end run, written to be read rather than reused.

    python3 scripts/demo.py [--audit-dir DIR]

Shows the wiring: calibrate, open a signed log, build the membrane, generate,
then verify the log. Then injects one corruption and shows the quarantine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from membrane import (  # noqa: E402
    AuditLog,
    IntegrityMembrane,
    Mode,
    TokenPolicy,
    TraceLevel,
    calibrate,
    default_corpus,
    generate_signing_key,
    verify_log,
)
from runtime import CleanRoomEngine  # noqa: E402
from transformer import CleanRoomTransformer, ModelConfig  # noqa: E402

PROMPT = [ord(c) for c in "Hello"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", default=None)
    args = parser.parse_args()
    audit_dir = Path(args.audit_dir or mkdtemp(prefix="crt-audit-"))
    audit_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig()
    model = CleanRoomTransformer(cfg)

    print(f"config      d_model={cfg.d_model} layers={cfg.n_layers} "
          f"heads={cfg.n_heads} d_ff={cfg.d_ff} vocab={cfg.vocab_size}")
    print(f"schema      {cfg.schema_hash()[:32]}")
    print(f"weights     {model.weight_fingerprint()[:32]}")

    profile = calibrate(model, cfg, default_corpus(cfg), margin=1.5)
    print(f"calibrated  {len(profile.bounds)} boundaries, pin {profile.pin_hash()[:32]}")

    # In production this key lives outside the runtime's reach; here it is
    # generated per run, and the public key is printed so the log can be checked.
    key = generate_signing_key()
    log_path = audit_dir / "session.jsonl"
    audit = AuditLog(log_path, key)

    membrane = IntegrityMembrane(
        cfg,
        TokenPolicy(cfg.vocab_size, max_consecutive_repeats=6),
        profile,
        audit,
        Mode.ENFORCE,
        TraceLevel.SUMMARY,
        weight_fingerprint=model.weight_fingerprint(),
    )
    engine = CleanRoomEngine(model=model, membrane=membrane, audit=audit)

    print("\n--- clean run ---")
    result = engine.generate(PROMPT, max_new_tokens=12)
    print(f"status      {result.status.value}")
    print(f"emitted     {result.emitted}")
    print(f"inspections {result.report.inspections}  violations {result.report.violations}")
    audit.close()

    report = verify_log(log_path, key.public_key(), expected_head=result.audit_head)
    print(f"audit       {report.entries} entries, verified={report.ok}")
    print(f"public key  {membrane.audit.public_key_hex()}")
    print(f"log         {log_path}")

    print("\n--- same run, one hidden state corrupted at layer 2 ---")
    corrupted = CleanRoomTransformer(cfg)
    original_forward = corrupted.layers[2].forward

    def corrupt(x, cache=None):
        out, weights = original_forward(x, cache)
        out = out.copy()
        out[0, 0] = np.float32(1e6)
        return out, weights

    corrupted.layers[2].forward = corrupt

    audit2 = AuditLog(audit_dir / "quarantined.jsonl", key)
    membrane2 = IntegrityMembrane(
        cfg,
        TokenPolicy(cfg.vocab_size),
        profile,
        audit2,
        Mode.ENFORCE,
        TraceLevel.SUMMARY,
        weight_fingerprint=corrupted.weight_fingerprint(),
    )
    bad_result = CleanRoomEngine(model=corrupted, membrane=membrane2, audit=audit2).generate(
        PROMPT, max_new_tokens=12
    )
    audit2.close()

    print(f"status      {bad_result.status.value}")
    print(f"emitted     {bad_result.emitted}")
    print(f"signature   {bad_result.quarantine['signature'][:32]}")
    for violation in bad_result.quarantine["violations"]:
        print(f"violation   [{violation['tier']}] {violation['path']}: {violation['rule']} "
              f"measured={violation.get('measured')} bound={violation.get('bound')}")
    print(f"inspections {bad_result.report.inspections} "
          f"(a clean step inspects {1 + 2 * cfg.n_layers + 2}; "
          "the pass stopped early)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
