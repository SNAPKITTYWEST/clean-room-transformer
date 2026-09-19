#!/usr/bin/env python3
"""Mutation probe: prove each invariant tier is load-bearing.

A verification suite that passes is not evidence of anything unless removing a
check makes it fail. This script disables one tier at a time and reports, for
every injected fault, whether it is still quarantined, whether it merely crashes
the runtime somewhere downstream, or whether it is accepted in silence.

    python3 scripts/mutation_probe.py

"SILENT" cells are the interesting ones: they are exactly what that tier buys.
"CRASH" cells mean NumPy's own shape checks would have refused the tensor
anyway -- honest to know, since it means the membrane is not the only thing
standing there.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import membrane.interceptor as interceptor  # noqa: E402
from conftest import FAULT_CORPUS, PROMPT  # noqa: E402
from membrane import (  # noqa: E402
    IntegrityMembrane,
    Mode,
    TokenPolicy,
    TraceLevel,
    calibrate,
    default_corpus,
)
from membrane.invariants import StructuralContract  # noqa: E402
from runtime import CleanRoomEngine, Status  # noqa: E402
from transformer import CleanRoomTransformer, ModelConfig  # noqa: E402

QUARANTINED, CRASH, SILENT = "CAUGHT", "CRASH", "SILENT"


def disable_structural_rules(rules: set[str]) -> Callable[[], None]:
    original = StructuralContract.check

    def patched(self, path, tensor, bindings):
        return [v for v in original(self, path, tensor, bindings) if v.rule not in rules]

    StructuralContract.check = patched  # type: ignore[method-assign]
    return lambda: setattr(StructuralContract, "check", original)


def disable_function(name: str) -> Callable[[], None]:
    original = getattr(interceptor, name)
    setattr(interceptor, name, lambda *args, **kwargs: [])
    return lambda: setattr(interceptor, name, original)


MUTATIONS: dict[str, Callable[[], Callable[[], None]]] = {
    "baseline (nothing disabled)": lambda: (lambda: None),
    "no finite check": lambda: disable_structural_rules({"finite"}),
    "no shape/dtype check": lambda: disable_structural_rules({"rank", "extent", "dtype"}),
    "no attention math check": lambda: disable_function("check_attention_weights"),
    "no magnitude envelope": lambda: disable_function("check_magnitude"),
    "all tiers disabled": lambda: _disable_everything(),
}


def _disable_everything() -> Callable[[], None]:
    restores = [
        disable_structural_rules({"finite", "rank", "extent", "dtype"}),
        disable_function("check_attention_weights"),
        disable_function("check_magnitude"),
    ]

    def restore() -> None:
        for undo in restores:
            undo()

    return restore


def classify(cfg, profile, inject) -> str:
    # Disabling the finiteness tier deliberately lets NaN propagate through
    # layer_norm, which NumPy warns about. The warning is the expected
    # consequence of the mutation, not a finding; silence it so the table stays
    # readable.
    import warnings

    import numpy as np

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.seterr(all="ignore")

    model = CleanRoomTransformer(cfg)
    inject(model)
    membrane = IntegrityMembrane(
        cfg,
        TokenPolicy(cfg.vocab_size),
        profile,
        None,
        Mode.ENFORCE,
        TraceLevel.NONE,
        weight_fingerprint=model.weight_fingerprint(),
    )
    engine = CleanRoomEngine(model=model, membrane=membrane)
    try:
        result = engine.generate(PROMPT, max_new_tokens=4)
    except Exception:
        return CRASH
    return QUARANTINED if result.status is Status.QUARANTINED else SILENT


def main() -> int:
    cfg = ModelConfig()
    profile = calibrate(CleanRoomTransformer(cfg), cfg, default_corpus(cfg), margin=1.5)

    fault_names = [name for name, _ in FAULT_CORPUS]
    width = max(len(n) for n in fault_names) + 2

    print("=" * 100)
    print("MUTATION PROBE -- each column disables one invariant tier")
    print("=" * 100)

    results: dict[str, dict[str, str]] = {}
    for label, make_mutation in MUTATIONS.items():
        restore = make_mutation()
        try:
            results[label] = {
                name: classify(cfg, profile, inject) for name, inject in FAULT_CORPUS
            }
        finally:
            restore()

    labels = list(MUTATIONS)
    header = "fault".ljust(width) + "".join(f"{l[:22]:<24}" for l in labels)
    print(header)
    print("-" * len(header))
    for name in fault_names:
        row = name.ljust(width)
        for label in labels:
            row += f"{results[label][name]:<24}"
        print(row)

    print()
    for label in labels:
        counts = {
            state: sum(1 for v in results[label].values() if v == state)
            for state in (QUARANTINED, CRASH, SILENT)
        }
        print(
            f"{label:<32} caught {counts[QUARANTINED]:>2}  "
            f"crash {counts[CRASH]:>2}  silent {counts[SILENT]:>2}"
        )

    baseline = results[labels[0]]
    if any(state != QUARANTINED for state in baseline.values()):
        print("\nFAIL: baseline did not catch every fault")
        return 1

    fully_disabled = results[labels[-1]]
    if all(state == QUARANTINED for state in fully_disabled.values()):
        print("\nFAIL: faults still 'caught' with every tier disabled -- the suite "
              "is not measuring the membrane")
        return 1

    print("\nThe membrane is load-bearing: disabling tiers lets faults through.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
