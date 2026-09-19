"""
benchmark.py — throughput, per-step latency, and membrane overhead.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --tokens 32 --runs 10
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from membrane.calibration import calibrate, default_corpus
from membrane.interceptor import IntegrityMembrane, Mode, TraceLevel
from membrane.audit import AuditLog
from membrane.policy import TokenPolicy
from runtime.engine import CleanRoomEngine
from transformer.config import ModelConfig
from transformer.model import CleanRoomTransformer

import tempfile
import os


def _make_engine(cfg: ModelConfig, mode: Mode, audit_dir: str | None = None) -> CleanRoomEngine:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    model   = CleanRoomTransformer(cfg)
    profile = calibrate(model, cfg, default_corpus(cfg), margin=1.5)
    key     = Ed25519PrivateKey.generate()
    log     = AuditLog(Path(audit_dir) / "audit.jsonl", key) if audit_dir else None
    policy  = TokenPolicy(cfg.vocab_size)
    membrane = IntegrityMembrane(
        cfg=cfg,
        policy=policy,
        profile=profile,
        audit=log,
        mode=mode,
        trace=TraceLevel.NONE,
        weight_fingerprint=model.weight_fingerprint(),
    )
    return CleanRoomEngine(model=model, membrane=membrane, audit=log)


def bench_throughput(cfg: ModelConfig, max_tokens: int, runs: int) -> dict:
    """Tokens/sec over `runs` independent generate() calls."""
    prompt = [ord(c) for c in "Benchmark prompt: "]
    times  = []
    tokens_emitted = []
    with tempfile.TemporaryDirectory() as tmp:
        engine = _make_engine(cfg, Mode.ENFORCE, tmp)
        # warm-up
        engine.generate(prompt, max_new_tokens=4)
        for _ in range(runs):
            t0     = time.perf_counter()
            result = engine.generate(prompt, max_new_tokens=max_tokens)
            t1     = time.perf_counter()
            times.append(t1 - t0)
            tokens_emitted.append(len(result.emitted))

    total_tok = sum(tokens_emitted)
    total_sec = sum(times)
    per_run   = [tok / t for tok, t in zip(tokens_emitted, times)]

    return {
        "runs":              runs,
        "max_tokens":        max_tokens,
        "total_tokens":      total_tok,
        "total_seconds":     round(total_sec, 4),
        "mean_tok_per_sec":  round(statistics.mean(per_run), 2),
        "stdev_tok_per_sec": round(statistics.stdev(per_run), 2) if len(per_run) > 1 else 0.0,
        "min_tok_per_sec":   round(min(per_run), 2),
        "max_tok_per_sec":   round(max(per_run), 2),
    }


def bench_step_latency(cfg: ModelConfig, max_tokens: int, runs: int) -> dict:
    """Mean latency per generated token (ms)."""
    prompt = [ord(c) for c in "Hello"]
    step_times: list[float] = []
    with tempfile.TemporaryDirectory() as tmp:
        engine = _make_engine(cfg, Mode.ENFORCE, tmp)
        engine.generate(prompt, max_new_tokens=2)  # warm-up
        for _ in range(runs):
            # patch generate to record per-step times
            _orig = engine.model.forward
            _step_start: list[float] = []
            _step_end:   list[float] = []

            def _timed_forward(tokens, caches, observers, step):
                _step_start.append(time.perf_counter())
                result = _orig(tokens, caches, observers, step)
                _step_end.append(time.perf_counter())
                return result

            engine.model.forward = _timed_forward
            engine.generate(prompt, max_new_tokens=max_tokens)
            engine.model.forward = _orig
            for s, e in zip(_step_start, _step_end):
                step_times.append((e - s) * 1000)  # ms

    return {
        "samples":           len(step_times),
        "mean_ms":           round(statistics.mean(step_times), 3),
        "median_ms":         round(statistics.median(step_times), 3),
        "p95_ms":            round(sorted(step_times)[int(len(step_times) * 0.95)], 3),
        "stdev_ms":          round(statistics.stdev(step_times), 3) if len(step_times) > 1 else 0.0,
    }


def bench_membrane_overhead(cfg: ModelConfig, max_tokens: int, runs: int) -> dict:
    """Compare ENFORCE vs MONITOR (membrane off-path as baseline proxy)."""
    prompt = [ord(c) for c in "Hello world"]
    times: dict[str, list[float]] = {"enforce": [], "monitor": []}
    with tempfile.TemporaryDirectory() as tmp:
        for mode_name, mode in [("enforce", Mode.ENFORCE), ("monitor", Mode.MONITOR)]:
            engine = _make_engine(cfg, mode)
            engine.generate(prompt, max_new_tokens=2)  # warm-up
            for _ in range(runs):
                t0 = time.perf_counter()
                engine.generate(prompt, max_new_tokens=max_tokens)
                times[mode_name].append(time.perf_counter() - t0)

    e_mean = statistics.mean(times["enforce"])
    m_mean = statistics.mean(times["monitor"])
    overhead_pct = round((e_mean - m_mean) / m_mean * 100, 1) if m_mean > 0 else 0.0

    return {
        "enforce_mean_s":  round(e_mean, 4),
        "monitor_mean_s":  round(m_mean, 4),
        "overhead_pct":    overhead_pct,
        "note": "MONITOR skips enforcement; overhead_pct ~ membrane cost as % of total",
    }


def _print_section(title: str, data: dict) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")
    for k, v in data.items():
        if k == "note":
            print(f"  note : {v}")
        else:
            print(f"  {k:<28} {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="clean-room-transformer benchmark")
    parser.add_argument("--tokens", type=int, default=16,
                        help="max_new_tokens per run (default 16)")
    parser.add_argument("--runs",   type=int, default=8,
                        help="number of timed runs per benchmark (default 8)")
    args = parser.parse_args()

    cfg = ModelConfig()
    print(f"\nclean-room-transformer benchmark")
    print(f"config : d_model={cfg.d_model}  n_layers={cfg.n_layers}"
          f"  n_heads={cfg.n_heads}  vocab={cfg.vocab_size}")
    print(f"tokens : {args.tokens}   runs : {args.runs}")

    _print_section("Throughput  (tokens / second)",
                   bench_throughput(cfg, args.tokens, args.runs))

    _print_section("Step latency  (ms per token)",
                   bench_step_latency(cfg, args.tokens, args.runs))

    _print_section("Membrane overhead  (ENFORCE vs MONITOR)",
                   bench_membrane_overhead(cfg, args.tokens, args.runs))

    print(f"\n{'─'*55}")
    print("  done.")


if __name__ == "__main__":
    main()
