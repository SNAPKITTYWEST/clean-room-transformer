"""Inference engine: model + membrane + audit chain, with step rollback.

Execution order for each decode step:

    snapshot KV lengths
      -> membrane.begin_step(...)        declare the shape contract
      -> model.forward(...)             publishes boundaries to the membrane
      -> membrane.gate_emission(...)    policy check on the decoded token
      -> commit (append token to history)

If the membrane raises anywhere in that sequence, the engine rolls the KV cache
back to the snapshot, records the divergence, and stops generating. The step
leaves no residue in model state: `tests/test_rollback.py` asserts that the
cache is byte-identical to its pre-step contents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np

from membrane.audit import AuditLog
from membrane.exceptions import QuarantineError
from membrane.interceptor import IntegrityMembrane, MembraneReport
from transformer.config import ModelConfig
from transformer.layers import KVCache
from transformer.model import CleanRoomTransformer


class Status(str, Enum):
    COMPLETED = "completed"
    QUARANTINED = "quarantined"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CONTEXT_FULL = "context_full"


@dataclass
class GenerationResult:
    prompt: list[int]
    emitted: list[int]
    status: Status
    steps: int
    report: MembraneReport
    quarantine: dict[str, Any] | None = None
    audit_head: str | None = None
    weight_fingerprint: str | None = None
    schema_hash: str | None = None

    @property
    def quarantined(self) -> bool:
        return self.status is Status.QUARANTINED

    def as_record(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "emitted": self.emitted,
            "status": self.status.value,
            "steps": self.steps,
            "report": self.report.as_record(),
            "quarantine": self.quarantine,
            "audit_head": self.audit_head,
            "weight_fingerprint": self.weight_fingerprint,
            "schema_hash": self.schema_hash,
        }


@dataclass
class CleanRoomEngine:
    model: CleanRoomTransformer
    membrane: IntegrityMembrane
    audit: AuditLog | None = None
    caches: list[KVCache] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.caches = self.model.new_caches()

    @property
    def cfg(self) -> ModelConfig:
        return self.model.cfg

    # -- state management -----------------------------------------------

    def _snapshot(self) -> list[int]:
        return [cache.length for cache in self.caches]

    def _rollback(self, snapshot: Sequence[int]) -> None:
        for cache, length in zip(self.caches, snapshot):
            cache.rollback_to(length)

    def reset(self) -> None:
        for cache in self.caches:
            cache.reset()

    # -- generation -----------------------------------------------------

    def generate(self, prompt: Sequence[int], max_new_tokens: int = 16) -> GenerationResult:
        self.reset()
        prompt_list = [int(t) for t in prompt]
        emitted: list[int] = []
        quarantine: dict[str, Any] | None = None
        status = Status.COMPLETED

        if self.audit is not None:
            self.audit.append(
                {
                    "event": "session_open",
                    "schema_hash": self.cfg.schema_hash(),
                    "weight_fingerprint": self.model.weight_fingerprint(),
                    "profile_pin": (
                        self.membrane.profile.pin_hash()
                        if self.membrane.profile is not None
                        else None
                    ),
                    "policy": self.membrane.policy.as_record(),
                    "mode": self.membrane.mode.value,
                    "trace": self.membrane.trace.value,
                    "prompt_len": len(prompt_list),
                }
            )

        step = 0
        pending = np.asarray(prompt_list, dtype=np.int64)

        while True:
            if step >= max_new_tokens:
                status = Status.BUDGET_EXHAUSTED
                break

            offset = self.caches[0].length
            if offset + pending.shape[0] > self.cfg.max_seq_len:
                status = Status.CONTEXT_FULL
                break

            snapshot = self._snapshot()
            self.membrane.begin_step(step, int(pending.shape[0]), offset)

            try:
                logits = self.model.forward(
                    pending, caches=self.caches, observers=[self.membrane], step=step
                )
                token = self.membrane.gate_emission(logits, prompt_list + emitted, step)
            except QuarantineError as exc:
                self._rollback(snapshot)
                quarantine = {
                    "signature": exc.signature,
                    "step": exc.step,
                    "violations": [v.as_record() for v in exc.violations],
                }
                status = Status.QUARANTINED
                break

            emitted.append(token)
            step += 1
            pending = np.asarray([token], dtype=np.int64)

        if self.audit is not None:
            self.audit.append(
                {
                    "event": "session_close",
                    "status": status.value,
                    "emitted": emitted,
                    "steps": step,
                    "membrane": self.membrane.report.as_record(),
                }
            )
            self.audit.flush()

        return GenerationResult(
            prompt=prompt_list,
            emitted=emitted,
            status=status,
            steps=step,
            report=self.membrane.report,
            quarantine=quarantine,
            audit_head=self.audit.head if self.audit is not None else None,
            weight_fingerprint=self.model.weight_fingerprint(),
            schema_hash=self.cfg.schema_hash(),
        )
