"""Emission policy: the gate between a decoded token and the action engine.

This tier is about *trajectories*, not tensors. It is deliberately simple and
mechanical -- id range, a denylist, degenerate-loop detection, a step budget.
Every rule here is a decidable predicate over the emitted token sequence, which
is why it can be enforced exactly.

It is not a content-safety classifier and makes no claim to be one. Deciding
whether a *meaning* is acceptable is not a property of a token id, and nothing
in this file should be read as doing that job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .invariants import Tier, Violation


@dataclass(frozen=True)
class TokenPolicy:
    vocab_size: int
    forbidden_tokens: frozenset[int] = field(default_factory=frozenset)
    max_consecutive_repeats: int = 8
    max_new_tokens: int = 64

    def check(self, token: int, history: Sequence[int], step: int) -> list[Violation]:
        path = f"emission.step{step}"
        violations: list[Violation] = []

        if not 0 <= token < self.vocab_size:
            violations.append(
                Violation(path, "token_in_range", Tier.POLICY,
                          f"token id {token} outside [0, {self.vocab_size})",
                          measured=float(token), bound=float(self.vocab_size))
            )

        if token in self.forbidden_tokens:
            violations.append(
                Violation(path, "token_denylist", Tier.POLICY,
                          f"token id {token} is on the denylist",
                          measured=float(token))
            )

        run = 1
        for previous in reversed(history):
            if previous != token:
                break
            run += 1
        if run > self.max_consecutive_repeats:
            violations.append(
                Violation(path, "repetition_loop", Tier.POLICY,
                          f"token id {token} repeated {run} times consecutively",
                          measured=float(run), bound=float(self.max_consecutive_repeats))
            )

        if step >= self.max_new_tokens:
            violations.append(
                Violation(path, "step_budget", Tier.POLICY,
                          f"step {step} exceeds budget {self.max_new_tokens}",
                          measured=float(step), bound=float(self.max_new_tokens))
            )

        return violations

    def as_record(self) -> dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "forbidden_tokens": sorted(self.forbidden_tokens),
            "max_consecutive_repeats": self.max_consecutive_repeats,
            "max_new_tokens": self.max_new_tokens,
        }
