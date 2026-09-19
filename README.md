# Clean-Room Transformer with Runtime Integrity Membrane

A working, dependency-minimal decoder-only transformer whose every layer
boundary is intercepted by a governance layer ("the membrane") that checks
invariants before execution may continue, quarantines violations, rolls back
mutated state, and records what happened to a hash-chained, signed,
append-only log.

Two dependencies (`numpy`, `cryptography`), both hash-pinned. No network calls,
no dynamic imports, no `eval` — enforced by a static test, not by assertion.
154 tests. Run it in about two seconds.

```
./scripts/run_verification_suite.sh
```

---

## 1. Quick start

```bash
pip install --require-hashes --no-deps -r requirements.txt   # runtime
pip install -r requirements-dev.txt                          # pytest, tests only

python3 -m pytest tests -q                    # 154 tests
python3 scripts/demo.py                       # clean run, then a quarantined one
python3 scripts/verification_report.py        # detection / false-positive report
python3 scripts/mutation_probe.py             # proves the invariants are load-bearing
```

Air-gapped run:

```bash
docker build -f docker/Dockerfile -t clean-room-transformer .
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,size=16m \
  -v "$PWD/audit:/audit" clean-room-transformer
```

---

## 2. Execution path

```
tokens
  │
  ▼
embedding ──────────────────────────────────► membrane.on_state
  │                                                │ structural
  ▼                                                │ statistical
decoder layer 0 ─ attention weights ─────────► membrane.on_state
  │              └ hidden state ─────────────► membrane.on_state
  ▼                                                │ + mathematical (attention)
decoder layer 1..N-1  (same two boundaries each)   │
  │                                                ▼
  ▼                                          violation?
final layer norm ───────────────────────────► ├── ENFORCE → raise QuarantineError
  │                                           │       └► engine rolls KV cache back,
  ▼                                           │          logs divergence, halts
logits ─────────────────────────────────────► └── MONITOR → log only, continue
  │
  ▼
membrane.gate_emission  ── policy tier: id range, denylist, repetition, budget
  │
  ▼
token committed to output
```

The transformer core imports nothing from the membrane. It publishes
`StateEvent`s to a list of observers (`src/transformer/hooks.py`); an observer
that raises aborts the pass. That is the whole interception mechanism — there is
no post-hoc scan of a finished generation.

---

## 3. Spec deliverables → code

| Spec item | Where | What it actually does |
|---|---|---|
| **A. Isolation boundary** | `docker/Dockerfile`, `tests/test_isolation.py` | Static AST scan rejects any import of `socket`/`urllib`/`requests`/`subprocess`/`ctypes`/`pickle`/`importlib` and any third-party package outside `{numpy, cryptography}`; also rejects URL literals and `eval`/`exec`. Container runs `--network none --read-only`, non-root, no pip in the image. |
| **A. Deterministic init** | `src/transformer/determinism.py` | Per-tensor generators derived by *name* (`SeedSequence` entropy mixing), so no global RNG and no draw-order coupling. `fingerprint()` content-hashes any tensor. |
| **A. Deterministic KV alloc** | `src/transformer/layers.py::KVCache` | Fixed capacity, zero-filled at construction, never reallocates. `rollback_to()` restores byte-identical prior state. |
| **B. MHSA + SwiGLU** | `src/transformer/layers.py` | Explicit single-stream shapes `(T, d_model)` / `(n_heads, T, T_kv)`, max-subtracted softmax, overflow-safe SiLU, pre-norm residual blocks. Every boundary passes `expect(...)`. |
| **B. State serialization** | `src/transformer/hooks.py`, `src/membrane/audit.py` | Hooks publish embedding, per-layer attention weights, per-layer hidden states, final hidden, logits. Writes go to a background thread; the inference thread only enqueues. |
| **C. State interception** | `src/membrane/interceptor.py` | `IntegrityMembrane` is a `StateObserver` sitting between the decoder stack and emission. |
| **C. Constraint checker** | `src/membrane/invariants.py` | Three tiers: structural, mathematical, statistical (§5). |
| **C. Invariant enforcement** | `src/membrane/interceptor.py`, `src/runtime/engine.py` | Violation → divergence signature → audit record → `QuarantineError` → engine rolls the KV cache back and halts. The offending token never enters the output. |
| **C. Sovereign audit chain** | `src/membrane/audit.py` | JSONL, `hash = SHA-256(seq‖prev‖record)`, `sig = Ed25519(hash)`, genesis-linked. `verify_log()` detects modification, deletion, reordering, forgery, and (against a pinned head) truncation. |
| **3. Verification suite** | `tests/`, `scripts/` | 154 tests; 14-fault injection corpus; false-positive measurement; mutation probe. |

---

## 4. Measured results

From `scripts/verification_report.py` and `scripts/mutation_probe.py` on
CPython 3.11 / NumPy 2.4.4:

```
detection rate     14/14 (100.0%) over the enumerated fault corpus
false positives    0 over 330 boundary inspections on clean traffic
audit chain        146 entries verified; single-byte edit detected at entry 37
interception       fault at layer 1 → 5 inspections, not 11: layers 2-3 never ran
```

Which tier catches which fault:

| Fault | Tier | Rule |
|---|---|---|
| NaN / ±Inf in hidden state or attention | structural | `finite` |
| truncated sequence axis, widened feature axis | structural | `extent` |
| float64 promotion | structural | `dtype` |
| rank collapse | structural | `rank` |
| ×10⁴ magnitude explosion, single 10⁶ element | statistical | `max_abs`, `max_row_norm` |
| causal mask leak (uniform attention) | mathematical | `attn_causal_mask` |
| attention mass ×0.5 | mathematical | `attn_rows_sum_to_one` |
| negative / >1 attention weight | mathematical | `attn_nonneg`, `attn_max_one` |
| denylisted token, repetition loop, budget | policy | `token_denylist`, `repetition_loop`, `step_budget` |

The mutation probe disables one tier at a time. With **all** tiers disabled,
0/14 faults are intercepted: 3 crash somewhere downstream in NumPy and 11 flow
through to token emission in silence. That is what the membrane is buying.

---

## 5. The three tiers, and why they are kept apart

**Structural** — the declared contract of the boundary: rank, extents, dtype,
finiteness. Exact, no tolerance. A violation means the tensor is not the object
the next layer expects.

**Mathematical** — properties that follow from the operation itself: softmax
rows sum to 1, attention weights lie in [0, 1], masked positions hold exactly
zero mass. A violation is a defect or a tampered tensor. Checked against an
explicit float tolerance (`1e-4`).

**Statistical** — calibrated magnitude envelopes, fitted by running clean inputs
and multiplying the observed maximum by a margin (default 1.5). These are
heuristics. They catch gross drift and injected corruption; they can miss
anomalies inside the band and can fire on legitimate out-of-distribution input.

They are separate classes in separate functions because they carry different
epistemic weight, and collapsing them into one "safety check" would let the
weakest tier borrow the credibility of the strongest.

---

## 6. Honest scoping

The spec this was built from asks for guarantees stronger than anything here
delivers, and in some cases stronger than anything can deliver. Claim by claim:

**"Formal verification gate / symbolic execution verifier"** — *not
implemented.* What exists is runtime invariant checking. No theorem is proved,
no solver runs, nothing is symbolically executed. The gap is not one of effort:
"this hidden state is safe" is not a formally specified property, so there is
nothing to verify against. Formal methods for neural networks do exist
(interval-bound propagation, Reluplex/Marabou-style SMT, abstract
interpretation), they prove statements of the form "for all inputs in this
ε-ball the output label does not change," and they do not scale to transformer
decoders at useful sizes. If you want to state and check a real property, the
tractable candidates are the *mathematical* tier above — those are genuine
theorems about the operations, and they are checked exactly.

**"Intercepted and neutralized 100% of the time"** — *measured, over a fixed
corpus.* 14/14 of the faults in `tests/conftest.py::FAULT_CORPUS` are
intercepted, reproducibly. That is a detection rate over an enumerated set, not
a universal guarantee, and it should never be quoted without the qualifier.
The false-negative class is structural and known: any perturbation that keeps
the tensor finite, correctly shaped, float32, mathematically consistent, and
inside the calibrated envelope passes by construction. Adding `1e-3` to one
element of a hidden state is undetectable here — and is also enough to change
the emitted token. There is no configuration of this design that closes that
gap, because the gap is the difference between checking integrity and
understanding content.

**"Air-gapped, zero external telemetry"** — *two different claims, one weak and
one strong.* The weak one is enforced here: no module under `src/` imports a
network-capable, process-spawning or dynamic-import module, and a test fails if
that changes. The strong one is a deployment property and this repository cannot
enforce it — a static import scan says nothing about what a linked C extension
does, and NumPy is a large native dependency with BLAS underneath it. Use
`--network none`, a dropped-capability read-only container, and an egress
firewall. Neither claim substitutes for the other.

**"Zero state drift / reproducible execution trails"** — *true with stated
bounds.* Same code path, same input, same weights gives bit-identical output
(`test_same_path_is_bit_exact`). Two limits, both real:
- Prefill and incremental decode agree only to ~2×10⁻⁶, not bitwise. float32
  matmul is not associative and the two paths reduce in different orders. This
  is arithmetic, not a defect; `PATH_EQUIVALENCE_TOL = 1e-4` is the honest
  claim.
- Multi-threaded BLAS reorders reductions nondeterministically. Set
  `OMP_NUM_THREADS=1` (the Dockerfile and suite runner do) or bit-exactness is
  lost.
- The audit head hash is deliberately *not* reproducible: records carry
  `ts_ns`, so two identical runs produce different heads. The reproducible
  artifact is the record sequence with timestamps stripped
  (`test_audit_trail_is_reproducible_except_for_timestamps`).

**"Sovereign audit chain"** — *tamper-evident, not tamper-proof.* An offline
verifier holding the public key can detect modification, deletion, reordering
and forgery. It cannot detect truncation of the tail unless you pin the head
hash out of band — the test for this is explicit about it. And a process that
can rewrite the file *and* holds the signing key can rebuild a shorter history
that verifies cleanly. Real defences are operational: append-only mount or
`chattr +a`, WORM storage, replicating the head hash off-host, and keeping the
signing key somewhere the runtime cannot read it. `signing_key_from_seed()` is
marked test-only for exactly this reason.

**"Safety boundaries" on hidden states** — *magnitude envelopes, nothing
semantic.* Nothing in this system inspects meaning. The policy tier checks token
ids and trajectory shape (range, denylist, repetition run length, step budget),
which are decidable predicates over integers. It is not a content classifier and
must not be described as one; `src/membrane/policy.py` says so in its docstring
so the claim cannot drift as the code is reused.

**Scale and status** — the weights are untrained, deterministically initialized
noise: `d_model=64`, 4 layers, 4 heads, byte vocabulary, single stream, CPU,
NumPy. The deliverable is the execution and governance path, not a language
model. Generated tokens are meaningless and the demo's degenerate output is
expected. Numbers like "132 inspections per 12 tokens" scale linearly with
layers and steps; the membrane's cost is a handful of reductions per boundary,
which is cheap relative to a real model's matmuls but is not free and was not
benchmarked against one.

**Why Python and not Rust** — the spec allowed either. Python plus NumPy was
chosen because the point of this artifact is that a reader can check every
invariant by eye. A Rust port would buy real throughput and memory control and
would move the structural tier into the type system, where shape and dtype
violations become compile-time errors rather than runtime ones. It would not
change anything in this section: the mathematical tier stays a runtime check
with a tolerance, the statistical tier stays a heuristic, and the semantic gap
stays open.

---

## 7. Extending it

**Add an invariant.** Write a function in `src/membrane/invariants.py` returning
`list[Violation]` with the right `Tier`, call it from
`IntegrityMembrane.on_state`, then add a fault to `FAULT_CORPUS` that *only*
your new check can catch and confirm the mutation probe shows it going `SILENT`
when the check is disabled. A check with no fault that isolates it is untested.

**Wire a real model.** Implement the `StateObserver` publication points at your
layer boundaries and keep `begin_step()` honest about the expected extents — the
membrane can only check shapes it was told to expect. Everything else
(calibration, audit chain, rollback, policy) is model-agnostic.

**Tighten or loosen the envelope.** `calibrate(..., margin=)` trades sensitivity
against false positives. Run `verification_report.py` after any change: the
false-positive count on clean traffic is the number that must stay at zero.

**Shadow a new profile.** Run it in `Mode.MONITOR` against production traffic
first; violations are recorded and nothing is blocked. Promote to
`Mode.ENFORCE` only once the clean-traffic violation count is zero.

---

## 8. File map

```
src/transformer/
  config.py        ModelConfig, canonical JSON, schema hash, single dtype
  determinism.py   name-derived generators, tensor fingerprints
  shapes.py        expect(): rank/extent/dtype boundary assertions
  layers.py        LayerNorm, MHSA + KVCache (+rollback), SwiGLU, DecoderLayer
  hooks.py         StateEvent / StateObserver protocol
  model.py         CleanRoomTransformer, weight fingerprint, forward + publish

src/membrane/
  invariants.py    Tier, Violation, StructuralContract, attention math, magnitude
  calibration.py   CalibrationProfile, calibrate(), pin hash, JSON round-trip
  policy.py        TokenPolicy: range, denylist, repetition, step budget
  audit.py         AuditLog (async, chained, signed), verify_log()
  interceptor.py   IntegrityMembrane, Mode, TraceLevel, divergence signatures
  exceptions.py    QuarantineError, CalibrationMismatch

src/runtime/
  engine.py        CleanRoomEngine: snapshot → forward → gate → commit/rollback

tests/
  conftest.py                   fixtures + FAULT_CORPUS (the injection toolkit)
  test_transformer_core.py      shapes, causality, cache, bit-exactness
  test_invariants.py            each invariant in isolation
  test_membrane_interception.py the fault corpus through the real path
  test_rollback.py              byte-identical state restoration
  test_audit_chain.py           every tamper class, async writer semantics
  test_determinism.py           reproducibility of runs and of the trail
  test_monitor_mode.py          shadow mode must not alter the trajectory
  test_isolation.py             static import/URL/eval scan of src/
  test_end_to_end.py            interception ordering, writer thread, reporting
  test_suite_is_not_vacuous.py  disabling a tier must let faults through

scripts/
  demo.py                     clean run, then a quarantined run
  verification_report.py      detection rate, false positives, chain check
  mutation_probe.py           per-tier load-bearing table
  run_verification_suite.sh   all three stages
docker/Dockerfile             sealed, no-egress reference image
```
