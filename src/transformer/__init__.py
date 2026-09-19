"""Transformer core: deterministic, instrumented, dependency-minimal."""

from .config import DTYPE, ModelConfig
from .determinism import fingerprint, generator_for
from .hooks import StateEvent, StateObserver
from .layers import DecoderLayer, KVCache, MultiHeadSelfAttention, SwiGLUFeedForward
from .model import CleanRoomTransformer
from .shapes import ShapeViolation, expect

__all__ = [
    "DTYPE",
    "ModelConfig",
    "CleanRoomTransformer",
    "DecoderLayer",
    "KVCache",
    "MultiHeadSelfAttention",
    "SwiGLUFeedForward",
    "StateEvent",
    "StateObserver",
    "ShapeViolation",
    "expect",
    "fingerprint",
    "generator_for",
]
