"""Rollback: a quarantined step must leave no residue in model state."""

from __future__ import annotations

import numpy as np

from conftest import PROMPT, inject_into_layer_output
from runtime import Status
from transformer import CleanRoomTransformer, ModelConfig
from transformer.determinism import fingerprint
from transformer.layers import KVCache


def cache_fingerprints(engine) -> list[tuple[int, str, str]]:
    return [
        (cache.length, fingerprint(cache.keys), fingerprint(cache.values))
        for cache in engine.caches
    ]


def test_cache_rollback_restores_byte_identical_state(cfg: ModelConfig) -> None:
    cache = KVCache(cfg.n_heads, cfg.max_seq_len, cfg.d_head)
    k = np.ones((cfg.n_heads, 3, cfg.d_head), dtype=np.float32)
    cache.append(k, k * 2)
    before = (cache.length, fingerprint(cache.keys), fingerprint(cache.values))

    cache.append(k * 7, k * 9)
    assert cache.length == 6
    cache.rollback_to(3)

    assert (cache.length, fingerprint(cache.keys), fingerprint(cache.values)) == before


def test_rollback_to_invalid_length_is_refused(cfg: ModelConfig) -> None:
    cache = KVCache(cfg.n_heads, 8, cfg.d_head)
    k = np.zeros((cfg.n_heads, 2, cfg.d_head), dtype=np.float32)
    cache.append(k, k)
    for bad in (-1, 3):
        try:
            cache.rollback_to(bad)
        except ValueError:
            continue
        raise AssertionError(f"rollback_to({bad}) should have been refused")


def test_quarantined_step_leaves_caches_untouched(cfg: ModelConfig, build_engine) -> None:
    """Run two clean steps, then corrupt step 2 and confirm the cache state is
    exactly what it was before the failed step."""
    model = CleanRoomTransformer(cfg)
    engine = build_engine(model)

    result = engine.generate(PROMPT, max_new_tokens=2)
    assert result.status is Status.BUDGET_EXHAUSTED
    before = cache_fingerprints(engine)
    committed_length = engine.caches[0].length

    # Corrupt the next step only, without resetting the engine.
    inject_into_layer_output(model, 2, lambda x: (x * 1e5).astype(np.float32))
    snapshot = [cache.length for cache in engine.caches]
    engine.membrane.begin_step(2, 1, committed_length)

    from membrane import QuarantineError

    try:
        engine.model.forward(
            np.array([result.emitted[-1]]),
            caches=engine.caches,
            observers=[engine.membrane],
            step=2,
        )
    except QuarantineError:
        engine._rollback(snapshot)
    else:
        raise AssertionError("corrupted step was not quarantined")

    assert cache_fingerprints(engine) == before
    assert engine.caches[0].length == committed_length


def test_quarantine_discards_the_offending_token(cfg: ModelConfig, build_engine) -> None:
    """The token from a quarantined step must never appear in the output."""
    model = CleanRoomTransformer(cfg)
    clean = build_engine(model).generate(PROMPT, max_new_tokens=4)

    corrupted_model = CleanRoomTransformer(cfg)
    inject_into_layer_output(
        corrupted_model, 0, lambda x: (x * 1e5).astype(np.float32)
    )
    quarantined = build_engine(corrupted_model).generate(PROMPT, max_new_tokens=4)

    assert quarantined.status is Status.QUARANTINED
    assert quarantined.emitted == []
    assert len(clean.emitted) == 4
