"""State export hooks.

The transformer core knows nothing about the membrane or the audit chain. It
emits `StateEvent`s to a list of observers; subscribers decide what to do with
them. An observer that raises aborts the forward pass -- that is the mechanism
the membrane uses to stop a violating state from reaching token emission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class StateEvent:
    """One tensor crossing one instrumented boundary."""

    kind: str  # "embedding" | "layer_output" | "attention" | "final_hidden" | "logits"
    tensor: np.ndarray
    step: int  # generation step index (0 for a prefill pass)
    layer: int | None = None  # None for non-layer-scoped events
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def path(self) -> str:
        """Stable identifier for this boundary, e.g. 'layer3.layer_output'."""
        return self.kind if self.layer is None else f"layer{self.layer}.{self.kind}"


@runtime_checkable
class StateObserver(Protocol):
    def on_state(self, event: StateEvent) -> None:  # pragma: no cover - protocol
        ...
