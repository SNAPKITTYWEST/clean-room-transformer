"""
load_checkpoint.py — deserialise a checkpoint, verify the weight fingerprint,
configure the engine, and run a generation.

Checkpoint format (JSON + optional .npz weight file):

    model.json  {
        "config": { ...ModelConfig fields... },
        "weight_fingerprint": "<sha256-hex>",
        "schema_hash": "<sha256-hex>"
    }

    model.npz   optional numpy archive; if absent, weights are re-derived
                deterministically from config.seed (same result every time).

Usage:
    python scripts/load_checkpoint.py --checkpoint model.json
    python scripts/load_checkpoint.py --checkpoint model.json --prompt "Hello"
    python scripts/load_checkpoint.py --checkpoint model.json --weights model.npz
    python scripts/load_checkpoint.py --save model.json   # save a fresh checkpoint
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from membrane.audit import AuditLog
from membrane.calibration import calibrate, default_corpus
from membrane.interceptor import IntegrityMembrane, Mode, TraceLevel
from membrane.policy import TokenPolicy
from runtime.engine import CleanRoomEngine
from transformer.config import ModelConfig
from transformer.model import CleanRoomTransformer


# ── serialisation helpers ────────────────────────────────────────────────────

def save_checkpoint(path: str, model: CleanRoomTransformer, cfg: ModelConfig,
                    weights_path: str | None = None) -> None:
    record = {
        "config": json.loads(cfg.canonical_json()),
        "weight_fingerprint": model.weight_fingerprint(),
        "schema_hash": cfg.schema_hash(),
    }
    Path(path).write_text(json.dumps(record, indent=2))
    print(f"checkpoint saved  → {path}")

    if weights_path:
        arrays = {}
        for name, tensor in model._named_parameters():
            arrays[name] = tensor
        np.savez(weights_path, **arrays)
        print(f"weights saved     → {weights_path}")


def load_checkpoint(path: str, weights_path: str | None = None) -> tuple[CleanRoomTransformer, ModelConfig]:
    data = json.loads(Path(path).read_text())

    cfg_dict = data["config"]
    cfg = ModelConfig(
        vocab_size  = cfg_dict.get("vocab_size",   256),
        d_model     = cfg_dict.get("d_model",       64),
        n_heads     = cfg_dict.get("n_heads",        4),
        n_layers    = cfg_dict.get("n_layers",       4),
        d_ff        = cfg_dict.get("d_ff",          176),
        max_seq_len = cfg_dict.get("max_seq_len",   64),
        norm_eps    = cfg_dict.get("norm_eps",       1e-5),
        seed        = cfg_dict.get("seed",           20260918),
    )

    saved_schema = data.get("schema_hash", "")
    actual_schema = cfg.schema_hash()
    if saved_schema and saved_schema != actual_schema:
        print(f"[WARN] schema hash mismatch — checkpoint may be from a different config")
        print(f"       saved : {saved_schema}")
        print(f"       actual: {actual_schema}")

    model = CleanRoomTransformer(cfg)

    if weights_path and Path(weights_path).exists():
        archive = np.load(weights_path)
        model._load_parameters(dict(archive))
        print(f"weights loaded    ← {weights_path}")

    actual_fp = model.weight_fingerprint()
    saved_fp  = data.get("weight_fingerprint", "")

    if saved_fp:
        if actual_fp == saved_fp:
            print(f"weight fingerprint  ✓  {actual_fp[:16]}…")
        else:
            print(f"[WARN] weight fingerprint mismatch")
            print(f"       saved : {saved_fp}")
            print(f"       actual: {actual_fp}")
    else:
        print(f"weight fingerprint  (no saved value to compare)")
        print(f"       actual: {actual_fp[:16]}…")

    print(f"schema hash         {actual_schema[:16]}…")
    print(f"config              d_model={cfg.d_model}  n_layers={cfg.n_layers}"
          f"  n_heads={cfg.n_heads}  vocab={cfg.vocab_size}  seed={cfg.seed}")

    return model, cfg


# ── engine wiring ────────────────────────────────────────────────────────────

def build_engine(model: CleanRoomTransformer, cfg: ModelConfig,
                 audit_dir: str | None = None) -> CleanRoomEngine:
    profile  = calibrate(model, cfg, default_corpus(cfg), margin=1.5)
    key      = Ed25519PrivateKey.generate()
    log      = AuditLog(audit_dir, key) if audit_dir else None
    policy   = TokenPolicy(cfg.vocab_size)
    membrane = IntegrityMembrane(
        cfg=cfg,
        policy=policy,
        profile=profile,
        audit=log,
        mode=Mode.ENFORCE,
        trace=TraceLevel.SUMMARY,
        weight_fingerprint=model.weight_fingerprint(),
    )
    return CleanRoomEngine(model=model, membrane=membrane, audit=log)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="load checkpoint and generate")
    parser.add_argument("--checkpoint", help="path to model.json checkpoint")
    parser.add_argument("--weights",    help="path to model.npz weight archive (optional)")
    parser.add_argument("--prompt",     default="Hello", help="generation prompt (default: Hello)")
    parser.add_argument("--tokens",     type=int, default=12, help="max new tokens (default: 12)")
    parser.add_argument("--save",       help="save a fresh checkpoint to this path instead of loading")
    parser.add_argument("--save-weights", help="also save weights to this .npz path when --save is used")
    args = parser.parse_args()

    if args.save:
        cfg   = ModelConfig()
        model = CleanRoomTransformer(cfg)
        save_checkpoint(args.save, model, cfg, args.save_weights)
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --save is specified")

    print(f"\n{'─'*55}")
    print(f"  load_checkpoint")
    print(f"{'─'*55}")

    model, cfg = load_checkpoint(args.checkpoint, args.weights)

    prompt_tokens = [ord(c) for c in args.prompt]
    print(f"\nprompt  : {args.prompt!r}  ({len(prompt_tokens)} tokens)")
    print(f"max new : {args.tokens}")

    with tempfile.TemporaryDirectory() as audit_dir:
        engine = build_engine(model, cfg, audit_dir)
        result = engine.generate(prompt_tokens, max_new_tokens=args.tokens)

    print(f"\nstatus      : {result.status.value}")
    if result.emitted:
        try:
            decoded = bytes(result.emitted).decode("utf-8", errors="replace")
        except Exception:
            decoded = str(result.emitted)
        print(f"emitted     : {decoded!r}")
    else:
        print(f"emitted     : (none)")

    print(f"steps       : {result.steps}")
    print(f"inspections : {result.report.inspections}")
    print(f"violations  : {result.report.violations}")

    if result.quarantined:
        print(f"\n[QUARANTINED]")
        for v in result.quarantine or []:
            print(f"  tier={v.get('tier')}  rule={v.get('rule')}"
                  f"  path={v.get('path')}  measured={v.get('measured')}")

    print(f"\naudit head  : {(result.audit_head or 'n/a')[:32]}…")
    print(f"fingerprint : {result.weight_fingerprint[:16]}…")
    print(f"schema hash : {result.schema_hash[:16]}…")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()
