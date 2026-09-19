"""Membrane control-flow exceptions."""

from __future__ import annotations

from .invariants import Violation


class MembraneError(Exception):
    """Base class for membrane faults."""


class QuarantineError(MembraneError):
    """Raised when a boundary violates its invariants under ENFORCE mode.

    Raising is the interception mechanism: it unwinds the forward pass before
    the offending tensor can reach the next layer or the emission head. The
    engine catches it, rolls back mutated state and records the divergence.
    """

    def __init__(self, signature: str, violations: list[Violation], step: int) -> None:
        self.signature = signature
        self.violations = violations
        self.step = step
        summary = "; ".join(str(v) for v in violations)
        super().__init__(f"quarantined at step {step} [{signature[:12]}]: {summary}")


class CalibrationMismatch(MembraneError):
    """Raised when a profile does not belong to the loaded weights or config."""
