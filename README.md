# clean-room-transformer

![Tests](https://img.shields.io/badge/tests-154%20passing-brightgreen?style=flat-square)
![Fault Detection](https://img.shields.io/badge/fault%20detection-14%2F14-brightgreen?style=flat-square)
![False Positives](https://img.shields.io/badge/false%20positives-0%2F330-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)
![Dependencies](https://img.shields.io/badge/deps-numpy%20%7C%20cryptography-informational?style=flat-square)
![License](https://img.shields.io/badge/license-FSL--1.1-lightgrey?style=flat-square)
![Build](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)
![Audit](https://img.shields.io/badge/audit%20chain-Ed25519%20signed-blue?style=flat-square)

A decoder-only transformer where every layer boundary is intercepted by an **integrity membrane** — a runtime invariant enforcer that checks structural, mathematical, and statistical contracts, quarantines violations, rolls back KV cache state, and writes to a hash-chained, Ed25519-signed append-only log.

Two dependencies only: `numpy` and `cryptography`, both hash-pinned.

---

## Architecture

### Execution Flow

```mermaid
flowchart TD
    A([Prompt tokens]) --> B[Token + Position Embedding]
    B --> C{Membrane:\nStructural check}
    C -->|pass| D[Decoder Layer 1]
    C -->|fail| Z1[Quarantine + Rollback]

    D --> D1[MHSA\nattn weights checked]
    D1 --> D2[SwiGLU FFN\nhidden state checked]
    D2 --> D3[Layer Output\nmagnitude checked]
    D3 --> E{More layers?}
    E -->|yes| D
    E -->|no| F[Final RMSNorm]
    F --> G[Logit projection]
    G --> H{gate_emission\nPolicy tier}
    H -->|pass| I[Commit token → history]
    H -->|block| Z2[Quarantine + Rollback]
    I --> J{Budget / EOS?}
    J -->|continue| A
    J -->|done| K([GenerationResult])
    Z1 --> K
    Z2 --> K
```

### Membrane Intercept Pipeline

```mermaid
flowchart LR
    subgraph Model Forward Pass
        E[embedding] --> A[attention ×L]
        A --> LO[layer_output ×L]
        LO --> FH[final_hidden]
        FH --> LG[logits]
    end

    subgraph Membrane
        S1[Tier 1: Structural\nshape · dtype · rank]
        S2[Tier 2: Mathematical\nNaN · Inf · float64]
        S3[Tier 3: Statistical\nmagnitude envelopes]
        P[Policy\ndenylist · repetition · budget]
    end

    E -->|StateEvent| S1
    A -->|StateEvent| S1
    A -->|StateEvent| S2
    LO -->|StateEvent| S2
    LO -->|StateEvent| S3
    FH -->|StateEvent| S3
    LG -->|StateEvent| P
```

### Audit Chain

```mermaid
flowchart LR
    R0[session_open\nentry₀] --> R1
    R1[step_0\nentry₁] --> R2
    R2[step_1\nentry₂] --> R3
    R3[...] --> Rn[session_close\nentryₙ]

    R0 -.->|SHA-256\nchain| R1
    R1 -.->|SHA-256\nchain| R2
    R2 -.->|SHA-256\nchain| R3
    Rn -.->|Ed25519\nsigned head| V([verify_log])
```

### Checkpoint Load & Configure

```mermaid
flowchart TD
    CP([model.json / .npz]) --> LC[load_checkpoint]
    LC --> MC[ModelConfig\ndeserialise + validate]
    LC --> WL[Weight arrays\nfingerprint verify]
    MC --> CE[CleanRoomEngine]
    WL --> CE
    CE --> CAL[calibrate\nprofile]
    CAL --> MEM[IntegrityMembrane\nENFORCE mode]
    MEM --> GEN[generate]
```

---

## Quick Start

```bash
pip install -r requirements.txt
pytest                        # 154 tests
python scripts/demo.py        # clean run + injected fault demo
python scripts/benchmark.py   # throughput, latency, membrane overhead
python scripts/load_checkpoint.py --checkpoint model.json --prompt "Hello"
python scripts/verification_report.py
```

Docker (air-gapped):
```bash
docker build -t clean-room-transformer docker/
docker run --network none clean-room-transformer
```

---

## Measured Results

| Metric | Result |
|--------|--------|
| Fault detection | **14 / 14** (fixed corpus) |
| False positives | **0 / 330** clean inspections |
| Audit entries verified | **146** |
| Single-byte edit detected | ✓ |
| Benchmark throughput | see `scripts/benchmark.py` output |

---

## Spec → Code

| Spec item | File |
|-----------|------|
| Isolation | `src/membrane/interceptor.py` |
| Deterministic init | `src/transformer/model.py` |
| KV cache alloc + rollback | `src/runtime/engine.py` |
| MHSA + SwiGLU | `src/transformer/layers.py` |
| State serialisation | `src/transformer/model.py` |
| Invariant enforcement | `src/membrane/invariants.py` |
| Sovereign audit chain | `src/membrane/audit.py` |
| Policy tier | `src/membrane/policy.py` |
| Calibration | `src/membrane/calibration.py` |
| Verification suite | `tests/` |
| Benchmark | `scripts/benchmark.py` |
| Checkpoint load | `scripts/load_checkpoint.py` |

---

## Invariant Tiers

| Tier | Kind | Tolerance |
|------|------|-----------|
| Structural | exact shape · dtype · rank | none |
| Mathematical | NaN · Inf · float64 promotion | none |
| Statistical | calibrated magnitude envelopes | heuristic (margin × calibrated bound) |

---

## Fault → Tier → Rule

| Fault | Tier | Rule |
|-------|------|------|
| NaN / Inf in activations | Mathematical | `no_nan`, `no_inf` |
| Truncated axes | Structural | `exact_shape` |
| float64 promotion | Mathematical | `dtype_preserved` |
| Rank collapse | Structural | `rank_preserved` |
| Magnitude explosion | Statistical | `magnitude_within_envelope` |
| Causal mask leak | Mathematical | `causal_mask_valid` |
| Attention mass error | Mathematical | `attention_mass_unit` |
| Denylist token | Policy | `denylist` |
| Repetition | Policy | `repetition_budget` |
| Token budget | Policy | `budget` |

---

## Model Configuration

Default (`src/transformer/config.py`):

```python
ModelConfig(
    vocab_size  = 256,   # byte-level, no external tokenizer
    d_model     = 64,
    n_heads     = 4,
    n_layers    = 4,
    d_ff        = 176,   # SwiGLU inner ~= 8/3 × d_model
    max_seq_len = 64,
    norm_eps    = 1e-5,
    seed        = 20260918,
)
```

The `schema_hash()` (SHA-256 of canonical JSON) is recorded in every audit entry.

---

## Directory

```
clean-room-transformer/
├── src/
│   ├── transformer/        config, model, layers, hooks, determinism, shapes
│   ├── membrane/           interceptor, invariants, policy, audit, calibration
│   ├── runtime/            engine
│   ├── metal-transformer/  MetalTransformer.swift + Transformer.metal (GPU)
│   └── haskell-transforms/ CoreToLogic.hs + DenseDex.hs (Liquid Haskell)
├── tests/                  154 tests — determinism, invariants, audit, rollback, e2e
├── scripts/
│   ├── demo.py             clean + fault injection walkthrough
│   ├── benchmark.py        throughput / latency / membrane overhead
│   ├── load_checkpoint.py  checkpoint deserialise + generate
│   ├── mutation_probe.py
│   ├── verification_report.py
│   └── run_verification_suite.sh
└── docker/Dockerfile
```

---

## Honest Scoping

- No formal verification. "14/14" and "0/330" are over a fixed corpus, not a universal guarantee.
- No semantic content inspection. The membrane sees tensor values, not meaning.
- Scale: `d_model=64`, 4 layers, 4 heads, untrained noise weights. Production use requires recalibration.
- Mixed precision deliberately unsupported; single `float32` throughout.

---

## Milestone 1 Additions (Sep 18 2026)

Swift modules in `src/metal-transformer/` and Haskell transforms in `src/haskell-transforms/`
added from devflow-finance-twin. See `SwiftTinyLLM-milestone1.zip` for the full Swift package
(ModelConfig · Tensor · Random · ByteTokenizer · RMSNorm · SwiGLU · RoPE).
