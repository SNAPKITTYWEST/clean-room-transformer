"""Transformer core: shape contracts, causality, cache behaviour, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from transformer import CleanRoomTransformer, ModelConfig, ShapeViolation
from transformer.layers import KVCache, softmax_last_axis

PROMPT = np.array([72, 101, 108, 108, 111], dtype=np.int64)

# float32 matmul is not associative, so the prefill path (one (T,d)@(d,d) call)
# and the incremental path (T separate (1,d)@(d,d) calls) reduce in different
# orders. The two agree to roughly single-precision epsilon times the number of
# accumulations, not bit-for-bit. This tolerance is the honest claim.
PATH_EQUIVALENCE_TOL = 1e-4


def test_logits_shape_and_dtype(model: CleanRoomTransformer, cfg: ModelConfig) -> None:
    logits = model.forward(PROMPT)
    assert logits.shape == (PROMPT.size, cfg.vocab_size)
    assert logits.dtype == np.float32
    assert np.isfinite(logits).all()


def test_config_rejects_indivisible_head_split() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=65, n_heads=4)


def test_token_id_out_of_range_is_rejected(model: CleanRoomTransformer, cfg: ModelConfig) -> None:
    with pytest.raises(ValueError, match="out of range"):
        model.forward(np.array([cfg.vocab_size], dtype=np.int64))
    with pytest.raises(ValueError, match="out of range"):
        model.forward(np.array([-1], dtype=np.int64))


def test_empty_and_wrong_rank_inputs_are_rejected(model: CleanRoomTransformer) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        model.forward(np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match="rank 1"):
        model.forward(np.zeros((2, 2), dtype=np.int64))


def test_context_overflow_is_refused(cfg: ModelConfig, model: CleanRoomTransformer) -> None:
    too_long = np.zeros(cfg.max_seq_len + 1, dtype=np.int64)
    with pytest.raises(ValueError, match="context overflow"):
        model.forward(too_long)


def test_future_tokens_cannot_affect_earlier_positions(model: CleanRoomTransformer) -> None:
    """The defining property of a causal decoder, checked end to end."""
    a = np.array([10, 20, 30, 40], dtype=np.int64)
    b = np.array([10, 20, 30, 99], dtype=np.int64)
    logits_a = model.forward(a)
    logits_b = model.forward(b)
    assert np.array_equal(logits_a[:3], logits_b[:3])
    assert not np.array_equal(logits_a[3], logits_b[3])


def test_incremental_decode_matches_prefill(model: CleanRoomTransformer) -> None:
    prefill = model.forward(PROMPT)
    caches = model.new_caches()
    incremental = np.stack(
        [
            model.forward(np.array([token]), caches=caches, step=i)[0]
            for i, token in enumerate(PROMPT)
        ]
    )
    assert np.abs(incremental - prefill).max() < PATH_EQUIVALENCE_TOL


def test_same_path_is_bit_exact(model: CleanRoomTransformer) -> None:
    """Reproducibility claim: identical code path plus identical input plus
    identical weights gives bit-identical output, not merely close output."""
    first = model.forward(PROMPT)
    second = model.forward(PROMPT)
    assert np.array_equal(first, second)
    assert first.tobytes() == second.tobytes()


def test_rebuilt_model_is_bit_identical(cfg: ModelConfig) -> None:
    a = CleanRoomTransformer(cfg)
    b = CleanRoomTransformer(cfg)
    assert a.weight_fingerprint() == b.weight_fingerprint()
    assert np.array_equal(a.forward(PROMPT), b.forward(PROMPT))


def test_different_seed_changes_weights(cfg: ModelConfig) -> None:
    other = CleanRoomTransformer(ModelConfig(seed=cfg.seed + 1))
    assert other.weight_fingerprint() != CleanRoomTransformer(cfg).weight_fingerprint()


def test_named_rng_derivation_is_order_independent(cfg: ModelConfig) -> None:
    """Weights are keyed by name, so drawing them in another order is identical."""
    from transformer.determinism import generator_for

    first = generator_for(cfg.seed, "layer0.attn.w_q").normal(size=8)
    _ = generator_for(cfg.seed, "layer3.ffn.w_up").normal(size=99)
    again = generator_for(cfg.seed, "layer0.attn.w_q").normal(size=8)
    assert np.array_equal(first, again)


def test_kv_cache_overflow_is_refused(cfg: ModelConfig) -> None:
    cache = KVCache(cfg.n_heads, 4, cfg.d_head)
    k = np.zeros((cfg.n_heads, 5, cfg.d_head), dtype=np.float32)
    with pytest.raises(ValueError, match="overflow"):
        cache.append(k, k)


def test_shape_helper_reports_axis_and_dtype() -> None:
    from transformer.shapes import expect

    with pytest.raises(ShapeViolation, match="axis 1 expected 4"):
        expect(np.zeros((2, 3), dtype=np.float32), (2, 4), "t")
    with pytest.raises(ShapeViolation, match="dtype"):
        expect(np.zeros((2, 4), dtype=np.float64), (2, 4), "t")
    with pytest.raises(ShapeViolation, match="rank"):
        expect(np.zeros((2,), dtype=np.float32), (2, 4), "t")


def test_softmax_is_stable_at_extreme_inputs() -> None:
    x = np.array([[1000.0, 1000.0, -1000.0]], dtype=np.float32)
    out = softmax_last_axis(x)
    assert np.isfinite(out).all()
    assert out.sum() == pytest.approx(1.0, abs=1e-6)


def test_masked_rows_stay_normalized(model: CleanRoomTransformer) -> None:
    """Row 0 attends to exactly one position; -inf masking must not make NaN."""
    captured = []

    class Capture:
        def on_state(self, event):
            if event.kind == "attention":
                captured.append(event.tensor)

    model.forward(PROMPT, observers=[Capture()])
    assert captured
    for weights in captured:
        assert np.isfinite(weights).all()
        assert np.abs(weights.sum(axis=-1) - 1.0).max() < 1e-5
