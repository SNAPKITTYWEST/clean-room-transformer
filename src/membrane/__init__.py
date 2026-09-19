"""The Integrity Membrane: invariant enforcement, policy gating, audit chain."""

from .audit import (
    GENESIS_HASH,
    AuditLog,
    VerificationReport,
    canonical_json,
    entry_hash,
    generate_signing_key,
    read_entries,
    signing_key_from_seed,
    verify_log,
)
from .calibration import CalibrationProfile, calibrate, default_corpus
from .exceptions import CalibrationMismatch, MembraneError, QuarantineError
from .interceptor import (
    IntegrityMembrane,
    MembraneReport,
    Mode,
    TraceLevel,
    divergence_signature,
)
from .invariants import (
    MagnitudeBounds,
    StructuralContract,
    Tier,
    Violation,
    check_attention_weights,
    check_magnitude,
)
from .policy import TokenPolicy

__all__ = [
    "GENESIS_HASH",
    "AuditLog",
    "CalibrationMismatch",
    "CalibrationProfile",
    "IntegrityMembrane",
    "MagnitudeBounds",
    "MembraneError",
    "MembraneReport",
    "Mode",
    "QuarantineError",
    "StructuralContract",
    "Tier",
    "TokenPolicy",
    "TraceLevel",
    "VerificationReport",
    "Violation",
    "calibrate",
    "canonical_json",
    "check_attention_weights",
    "check_magnitude",
    "default_corpus",
    "divergence_signature",
    "entry_hash",
    "generate_signing_key",
    "read_entries",
    "signing_key_from_seed",
    "verify_log",
]
