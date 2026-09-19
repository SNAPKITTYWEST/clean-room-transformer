"""Unit-level tests of each invariant, independent of the model."""

from __future__ import annotations

import numpy as np
import pytest

from membrane import (
    MagnitudeBounds,
    StructuralContract,
    Tier,
    TokenPolicy,
    check_attention_weights,
    check_magnitude,
)

CONTRACT = StructuralContract(("t", 8))
BINDINGS = {"t": 4, "t_kv": 4}


def rules(violations) -> set[str]:
    return {v.rule for v in violations}


# -- structural -------------------------------------------------------------


def test_clean_tensor_passes_structural_contract() -> None:
    tensor = np.zeros((4, 8), dtype=np.float32)
    assert CONTRACT.check("p", tensor, BINDINGS) == []


def test_non_array_is_rejected() -> None:
    assert rules(CONTRACT.check("p", [1, 2, 3], BINDINGS)) == {"is_ndarray"}


def test_rank_mismatch_is_reported_alone() -> None:
    violations = CONTRACT.check("p", np.zeros((4,), dtype=np.float32), BINDINGS)
    assert rules(violations) == {"rank"}


def test_extent_mismatch_on_bound_dimension() -> None:
    violations = CONTRACT.check("p", np.zeros((3, 8), dtype=np.float32), BINDINGS)
    assert rules(violations) == {"extent"}
    assert violations[0].measured == 3.0 and violations[0].bound == 4.0


def test_dtype_mismatch() -> None:
    violations = CONTRACT.check("p", np.zeros((4, 8), dtype=np.float64), BINDINGS)
    assert rules(violations) == {"dtype"}


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_caught(bad: float) -> None:
    tensor = np.zeros((4, 8), dtype=np.float32)
    tensor[2, 5] = bad
    violations = CONTRACT.check("p", tensor, BINDINGS)
    assert rules(violations) == {"finite"}
    assert violations[0].tier is Tier.STRUCTURAL


def test_unbound_dimension_accepts_any_extent() -> None:
    contract = StructuralContract((None, 8))
    for t in (1, 4, 99):
        assert contract.check("p", np.zeros((t, 8), dtype=np.float32), {}) == []


# -- mathematical -----------------------------------------------------------


def causal_weights(n_heads: int = 2, t: int = 4) -> np.ndarray:
    scores = np.tril(np.ones((t, t), dtype=np.float32))
    weights = scores / scores.sum(axis=-1, keepdims=True)
    return np.broadcast_to(weights, (n_heads, t, t)).copy()


def test_valid_causal_attention_passes() -> None:
    assert check_attention_weights("a", causal_weights(), q_offset=0) == []


def test_uniform_attention_is_caught_by_causality_only() -> None:
    """The case no aggregate statistic can see: rows sum to 1, range is [0,1],
    but mass sits on future positions."""
    weights = np.full((2, 4, 4), 0.25, dtype=np.float32)
    violations = check_attention_weights("a", weights, q_offset=0)
    assert rules(violations) == {"attn_causal_mask"}


def test_unnormalized_rows_are_caught() -> None:
    violations = check_attention_weights("a", causal_weights() * 0.5, q_offset=0)
    assert "attn_rows_sum_to_one" in rules(violations)


def test_negative_and_over_unit_weights_are_caught() -> None:
    weights = causal_weights()
    weights[0, 1, 0] = -0.5
    assert "attn_nonneg" in rules(check_attention_weights("a", weights, 0))

    weights = causal_weights()
    weights[0, 1, 0] = 2.0
    assert "attn_max_one" in rules(check_attention_weights("a", weights, 0))


def test_decode_step_has_no_future_positions_to_check() -> None:
    """At T_q=1 with offset k, every cached position is legitimately visible."""
    weights = np.full((2, 1, 5), 0.2, dtype=np.float32)
    assert check_attention_weights("a", weights, q_offset=4) == []


# -- statistical ------------------------------------------------------------

BOUNDS = MagnitudeBounds(max_abs=10.0, max_row_norm=20.0)


def test_in_envelope_tensor_passes() -> None:
    assert check_magnitude("p", np.full((4, 8), 1.0, dtype=np.float32), BOUNDS) == []


def test_element_magnitude_breach() -> None:
    tensor = np.zeros((4, 8), dtype=np.float32)
    tensor[0, 0] = 50.0
    violations = check_magnitude("p", tensor, BOUNDS)
    assert "max_abs" in rules(violations)
    assert violations[0].tier is Tier.STATISTICAL


def test_row_norm_breach_without_element_breach() -> None:
    """Many moderate values can breach the norm bound while each element is fine."""
    tensor = np.full((2, 8), 9.0, dtype=np.float32)  # row norm ~25.5, max abs 9
    violations = check_magnitude("p", tensor, BOUNDS)
    assert rules(violations) == {"max_row_norm"}


def test_magnitude_tier_defers_on_non_finite_input() -> None:
    tensor = np.zeros((4, 8), dtype=np.float32)
    tensor[0, 0] = np.nan
    assert check_magnitude("p", tensor, BOUNDS) == []


# -- policy -----------------------------------------------------------------

POLICY = TokenPolicy(
    vocab_size=256, forbidden_tokens=frozenset({13, 200}), max_consecutive_repeats=3,
    max_new_tokens=10,
)


def test_ordinary_token_passes_policy() -> None:
    assert POLICY.check(42, [1, 2, 3], step=0) == []


def test_denylisted_token_is_caught() -> None:
    assert rules(POLICY.check(13, [], step=0)) == {"token_denylist"}


def test_out_of_range_token_is_caught() -> None:
    assert rules(POLICY.check(999, [], step=0)) == {"token_in_range"}


def test_repetition_loop_is_caught_at_the_threshold() -> None:
    assert POLICY.check(7, [7, 7], step=0) == []  # run of 3, at the limit
    assert rules(POLICY.check(7, [7, 7, 7], step=0)) == {"repetition_loop"}


def test_repetition_counts_only_the_trailing_run() -> None:
    assert POLICY.check(7, [7, 7, 7, 1], step=0) == []


def test_step_budget_is_enforced() -> None:
    assert rules(POLICY.check(42, [], step=10)) == {"step_budget"}
