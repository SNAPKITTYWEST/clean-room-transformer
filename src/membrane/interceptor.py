"""The Integrity Membrane: state interception between layers and emission.

The membrane is a `StateObserver`. The transformer publishes every instrumented
boundary to it *before* execution continues, so raising from `on_state` stops a
violating tensor from reaching the next layer or the emission head. Nothing in
the transformer core knows the membrane exists.

Modes
-----
ENFORCE  Violations raise `QuarantineError` (the state never propagates).
MONITOR  Violations are recorded only. Used to measure false-positive rates on
         clean traffic and to shadow-test a new profile before enforcing it.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from transformer.config import ModelConfig
from transformer.determinism import fingerprint
from transformer.hooks import StateEvent

from .audit import AuditLog, canonical_json
from .calibration import CalibrationProfile
from .exceptions import CalibrationMismatch, QuarantineError
from .invariants import (
    StructuralContract,
    Tier,
    Violation,
    check_attention_weights,
    check_magnitude,
)
from .policy import TokenPolicy


class Mode(str, Enum):
    ENFORCE = "enforce"
    MONITOR = "monitor"


class TraceLevel(str, Enum):
    NONE = "none"  # divergences only
    SUMMARY = "summary"  # + per-boundary magnitude statistics
    FULL = "full"  # + content fingerprint of every tensor


def divergence_signature(path: str, step: int, violations: list[Violation]) -> str:
    """Stable identifier for a class of divergence.

    Deliberately excludes timestamps and measured magnitudes so that the same
    defect recurring produces the same signature and can be counted.
    """
    payload = canonical_json(
        {
            "path": path,
            "step": step,
            "rules": sorted((v.tier.value, v.rule) for v in violations),
        }
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class MembraneReport:
    inspections: int = 0
    violations: int = 0
    quarantines: int = 0
    by_tier: Counter = field(default_factory=Counter)
    by_rule: Counter = field(default_factory=Counter)
    signatures: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {
            "inspections": self.inspections,
            "violations": self.violations,
            "quarantines": self.quarantines,
            "by_tier": dict(self.by_tier),
            "by_rule": dict(self.by_rule),
            "signatures": list(self.signatures),
        }


class IntegrityMembrane:
    def __init__(
        self,
        cfg: ModelConfig,
        policy: TokenPolicy,
        profile: CalibrationProfile | None = None,
        audit: AuditLog | None = None,
        mode: Mode = Mode.ENFORCE,
        trace: TraceLevel = TraceLevel.SUMMARY,
        weight_fingerprint: str | None = None,
        attention_tolerance: float = 1e-4,
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self.profile = profile
        self.audit = audit
        self.mode = mode
        self.trace = trace
        self.attention_tolerance = attention_tolerance
        self.report = MembraneReport()

        if profile is not None:
            if profile.schema_hash != cfg.schema_hash():
                raise CalibrationMismatch(
                    "calibration profile was fitted against a different model "
                    f"config (profile {profile.schema_hash[:12]}…, "
                    f"runtime {cfg.schema_hash()[:12]}…)"
                )
            if (
                weight_fingerprint is not None
                and profile.weight_fingerprint != weight_fingerprint
            ):
                raise CalibrationMismatch(
                    "calibration profile was fitted against different weights "
                    f"(profile {profile.weight_fingerprint[:12]}…, "
                    f"runtime {weight_fingerprint[:12]}…)"
                )

        d, v, h = cfg.d_model, cfg.vocab_size, cfg.n_heads
        self._contracts: dict[str, StructuralContract] = {
            "embedding": StructuralContract(("t", d)),
            "layer_output": StructuralContract(("t", d)),
            "final_hidden": StructuralContract(("t", d)),
            "logits": StructuralContract(("t", v)),
            "attention": StructuralContract((h, "t", "t_kv")),
        }
        self._step = 0
        self._offset = 0
        self._bindings: dict[str, int] = {}

    # -- step framing ---------------------------------------------------

    def begin_step(self, step: int, n_query: int, offset: int) -> None:
        """Declare the shape contract for the pass about to run.

        Without this the membrane could only check ranks and dtypes: the
        expected sequence extent comes from the caller's intent, not from the
        tensor being inspected.
        """
        self._step = step
        self._offset = offset
        self._bindings = {"t": n_query, "t_kv": offset + n_query}

    # -- observer -------------------------------------------------------

    def on_state(self, event: StateEvent) -> None:
        self.report.inspections += 1
        path = event.path
        tensor = event.tensor

        violations: list[Violation] = []
        contract = self._contracts.get(event.kind)
        if contract is not None:
            violations += contract.check(path, tensor, self._bindings)

        structurally_sound = not violations and isinstance(tensor, np.ndarray)

        if structurally_sound and event.kind == "attention":
            violations += check_attention_weights(
                path, tensor, self._offset, self.attention_tolerance
            )

        if structurally_sound and self.profile is not None:
            bounds = self.profile.for_path(path)
            if bounds is not None:
                violations += check_magnitude(path, tensor, bounds)

        if violations:
            self._handle(path, violations)
            return

        if self.trace is not TraceLevel.NONE and self.audit is not None:
            self.audit.append(self._trace_record(event))

    # -- emission gate --------------------------------------------------

    def gate_emission(
        self, logits: np.ndarray, history: list[int], step: int
    ) -> int:
        """Decode greedily and let the token through only if policy permits.

        Greedy argmax keeps emission a deterministic function of the logits;
        sampling would make the audit trail unreproducible without also
        recording the sampler's RNG state.
        """
        path = f"emission.step{step}"

        if not isinstance(logits, np.ndarray) or logits.ndim != 2:
            # Unrecoverable: no token can be decoded from this, so there is
            # nothing to let through. This raises in MONITOR mode too --
            # "observe, do not block" only makes sense when execution can
            # continue.
            self._handle(
                path,
                [
                    Violation(
                        path,
                        "logits_shape",
                        Tier.STRUCTURAL,
                        f"expected rank-2 logits, got {getattr(logits, 'shape', None)}",
                    )
                ],
                force=True,
            )

        token = int(np.argmax(logits[-1]))
        violations = self.policy.check(token, history, step)

        if violations:
            # ENFORCE: raises, and the engine discards the step.
            # MONITOR: records and returns the token unchanged, so a shadow run
            # follows the same trajectory the unguarded runtime would take.
            self._handle(path, violations)
            return token

        if self.trace is not TraceLevel.NONE and self.audit is not None:
            self.audit.append(
                {
                    "event": "emission",
                    "step": step,
                    "token": token,
                    "logit": float(logits[-1, token]),
                    "margin": float(
                        np.partition(logits[-1], -2)[-1] - np.partition(logits[-1], -2)[-2]
                    ),
                }
            )
        return token

    # -- internals ------------------------------------------------------

    def _handle(
        self, path: str, violations: list[Violation], force: bool = False
    ) -> None:
        """Record a divergence and, unless in MONITOR mode, quarantine.

        `force=True` quarantines regardless of mode, for faults that leave no
        sound way to continue.
        """
        signature = divergence_signature(path, self._step, violations)
        self.report.violations += len(violations)
        self.report.signatures.append(signature)
        for violation in violations:
            self.report.by_tier[violation.tier.value] += 1
            self.report.by_rule[violation.rule] += 1

        if self.audit is not None:
            self.audit.append(
                {
                    "event": "divergence",
                    "step": self._step,
                    "path": path,
                    "signature": signature,
                    "mode": self.mode.value,
                    "violations": [v.as_record() for v in violations],
                }
            )

        if self.mode is Mode.ENFORCE or force:
            self.report.quarantines += 1
            raise QuarantineError(signature, violations, self._step)

    def _trace_record(self, event: StateEvent) -> dict[str, Any]:
        tensor = event.tensor
        record: dict[str, Any] = {
            "event": "state",
            "step": event.step,
            "path": event.path,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        if self.trace in (TraceLevel.SUMMARY, TraceLevel.FULL):
            record["max_abs"] = float(np.abs(tensor).max())
            record["mean"] = float(tensor.mean())
            record["std"] = float(tensor.std())
        if self.trace is TraceLevel.FULL:
            record["content_sha256"] = fingerprint(tensor)
        return record
