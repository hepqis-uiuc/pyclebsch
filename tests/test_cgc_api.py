"""Tests for the public calc_cgcs / print_cgcs / check_cgcs API.

Covers:
- calc_cgcs return-shape filtering (0..4 optional args)
- Singlet (1/sqrt(3)) values and phase convention for SU(3) 3⊗3̄
- Lower-weight algorithm path (non-highest-weight sum_state)
- check_cgcs for SU(2) 2⊗2 (the SU(3) cases are covered in test_cgc_cache.py)
- print_cgcs smoke tests (captures stdout, ensures non-empty output)
- Error path on invalid sum_state

The autouse cache_dir fixture redirects the on-disk CGC cache to tmp_path so
the repo's CGC_Data/ is never written to. The cache survives across tests in
the module thanks to scope='module'.
"""

import math

import pytest

import pyclebsch.cgc as cgc


@pytest.fixture(autouse=True)
def restore_cache_dir(tmp_path):
    """Redirect the CGC cache to a tmp dir and restore the original after."""
    original = cgc.get_cache_dir()
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    yield
    cgc.set_cache_dir(original)


# ---------- calc_cgcs return-shape filtering ----------
# SU(3) 3⊗3̄ = 8 ⊕ 1: sum irreps are (2,1,0) and (0,0,0), each with multiplicity 1.

PRODUCT_3X3BAR = [(1, 0, 0), (1, 1, 0)]


def test_calc_cgcs_no_optional_args_returns_nested_irrep_dict():
    """No optional args → outer dict keyed by sum irrep tuples."""
    result = cgc.calc_cgcs(PRODUCT_3X3BAR)
    assert isinstance(result, dict)
    assert set(result.keys()) == {(2, 1, 0), (0, 0, 0)}
    # Inner level: keyed by mult_idx (int).
    for sum_irrep, by_mult in result.items():
        assert isinstance(by_mult, dict)
        assert set(by_mult.keys()) == {1}  # multiplicity 1 for both irreps


def test_calc_cgcs_sum_iweight_returns_mult_dict():
    """sum_iweight given → dict keyed by mult_idx."""
    result = cgc.calc_cgcs(PRODUCT_3X3BAR, sum_iweight=(0, 0, 0))
    assert isinstance(result, dict)
    assert set(result.keys()) == {1}
    # Inner level: keyed by sum_state (int).
    assert isinstance(result[1], dict)
    for sum_state in result[1]:
        assert isinstance(sum_state, int)


def test_calc_cgcs_sum_iweight_and_mult_idx_returns_state_dict():
    """sum_iweight, mult_idx given → dict keyed by sum_state."""
    result = cgc.calc_cgcs(PRODUCT_3X3BAR, sum_iweight=(0, 0, 0), mult_idx=1)
    assert isinstance(result, dict)
    # Singlet has dimension 1, so exactly one sum_state (index 0).
    assert set(result.keys()) == {0}
    # Inner level: keyed by product_state (tuple).
    inner = result[0]
    assert isinstance(inner, dict)
    for product_state, value in inner.items():
        assert isinstance(product_state, tuple)
        assert len(product_state) == 2  # two factors in the product
        assert isinstance(value, float)


def test_calc_cgcs_three_args_returns_product_state_dict():
    """sum_iweight, mult_idx, sum_state given → dict keyed by product_state."""
    result = cgc.calc_cgcs(
        PRODUCT_3X3BAR, sum_iweight=(0, 0, 0), mult_idx=1, sum_state=0
    )
    assert isinstance(result, dict)
    for product_state, value in result.items():
        assert isinstance(product_state, tuple)
        assert isinstance(value, float)


def test_calc_cgcs_all_four_args_returns_float():
    """All four optional args given → returns a single float (or 0)."""
    # Get any product state with nonzero CGC.
    state_dict = cgc.calc_cgcs(
        PRODUCT_3X3BAR, sum_iweight=(0, 0, 0), mult_idx=1, sum_state=0
    )
    any_product_state = next(iter(state_dict))
    expected_value = state_dict[any_product_state]

    result = cgc.calc_cgcs(
        PRODUCT_3X3BAR,
        sum_iweight=(0, 0, 0),
        mult_idx=1,
        sum_state=0,
        product_state=any_product_state,
    )
    assert isinstance(result, float)
    assert math.isclose(result, expected_value, abs_tol=1e-12)


def test_calc_cgcs_invalid_sum_state_raises():
    """An out-of-range sum_state raises ValueError."""
    with pytest.raises(ValueError):
        cgc.calc_cgcs(PRODUCT_3X3BAR, sum_iweight=(0, 0, 0), mult_idx=1, sum_state=99)


def test_calc_cgcs_zero_for_product_state_with_no_cgc():
    """A product_state not in the dict returns 0 (not KeyError)."""
    state_dict = cgc.calc_cgcs(
        PRODUCT_3X3BAR, sum_iweight=(0, 0, 0), mult_idx=1, sum_state=0
    )
    all_pairs = {(i, j) for i in range(3) for j in range(3)}
    missing_pair = next(iter(all_pairs - set(state_dict.keys())))
    result = cgc.calc_cgcs(
        PRODUCT_3X3BAR,
        sum_iweight=(0, 0, 0),
        mult_idx=1,
        sum_state=0,
        product_state=missing_pair,
    )
    assert result == 0


# ---------- Singlet (1/sqrt(3)) values for SU(3) 3⊗3̄ ----------

def test_singlet_has_three_nonzero_cgcs():
    """The 3⊗3̄ → singlet has exactly 3 nonzero (state, conj-state) pairs."""
    singlet = cgc.calc_cgcs(
        PRODUCT_3X3BAR, sum_iweight=(0, 0, 0), mult_idx=1, sum_state=0
    )
    nonzero = {ps for ps, v in singlet.items() if abs(v) > 1e-10}
    assert len(nonzero) == 3


def test_singlet_magnitudes_are_one_over_sqrt_three():
    """Every nonzero singlet CGC has magnitude 1/sqrt(3)."""
    singlet = cgc.calc_cgcs(
        PRODUCT_3X3BAR, sum_iweight=(0, 0, 0), mult_idx=1, sum_state=0
    )
    for value in singlet.values():
        if abs(value) > 1e-10:
            assert math.isclose(abs(value), 1 / math.sqrt(3), abs_tol=1e-10)


def test_singlet_sum_of_squares_is_one():
    """Σ |CGC|² = 1 (state normalization)."""
    singlet = cgc.calc_cgcs(
        PRODUCT_3X3BAR, sum_iweight=(0, 0, 0), mult_idx=1, sum_state=0
    )
    sum_sq = sum(v * v for v in singlet.values())
    assert math.isclose(sum_sq, 1.0, abs_tol=1e-10)


def test_singlet_phase_convention_smallest_product_state_is_positive():
    """Phase convention: the smallest (tuple-sorted) product_state with
    nonzero CGC has a positive coefficient.

    The source's phase-fix step (in calc_highest_weight_cgcs) sorts product
    states tuple-wise and flips signs so that the smallest has positive CGC.
    """
    singlet = cgc.calc_cgcs(
        PRODUCT_3X3BAR, sum_iweight=(0, 0, 0), mult_idx=1, sum_state=0
    )
    nonzero = {ps: v for ps, v in singlet.items() if abs(v) > 1e-10}
    smallest = min(nonzero.keys())
    assert nonzero[smallest] > 0


# ---------- Lower-weight algorithm path ----------
# SU(3) 3⊗3 = 6 ⊕ 3̄. The 6 has dimension 6 (sum_states 0..5).
# sum_state=0 is HW (uses calc_highest_weight_cgcs). sum_state>0 exercises
# the lower-weight descent (calc_lower_weight_cgcs).

PRODUCT_3X3 = [(1, 0, 0), (1, 0, 0)]


def test_lower_weight_cgcs_orthonormal_for_each_sum_state_of_six():
    """For each basis state of 6=(2,0,0), Σ |CGC|² = 1."""
    six = cgc.calc_cgcs(PRODUCT_3X3, sum_iweight=(2, 0, 0), mult_idx=1)
    for sum_state, state_dict in six.items():
        sum_sq = sum(v * v for v in state_dict.values())
        assert math.isclose(sum_sq, 1.0, abs_tol=1e-10), (
            f"sum_state {sum_state}: Σ|CGC|² = {sum_sq}"
        )


def test_lower_weight_cgcs_six_has_six_sum_states():
    """6=(2,0,0) has dimension 6, so 6 sum_states."""
    six = cgc.calc_cgcs(PRODUCT_3X3, sum_iweight=(2, 0, 0), mult_idx=1)
    assert set(six.keys()) == set(range(6))


def test_lower_weight_cgcs_three_bar_has_three_sum_states():
    """3̄=(1,1,0) has dimension 3, so 3 sum_states."""
    three_bar = cgc.calc_cgcs(PRODUCT_3X3, sum_iweight=(1, 1, 0), mult_idx=1)
    assert set(three_bar.keys()) == set(range(3))


def test_lower_weight_cgcs_lowest_weight_state_orthonormal():
    """Lowest-weight state (largest sum_state index) is fully derived via
    repeated J(k)- application; orthonormality of its CGC row is a strong
    end-to-end check of the lower-weight algorithm."""
    six = cgc.calc_cgcs(PRODUCT_3X3, sum_iweight=(2, 0, 0), mult_idx=1)
    lowest_state = max(six.keys())
    sum_sq = sum(v * v for v in six[lowest_state].values())
    assert math.isclose(sum_sq, 1.0, abs_tol=1e-10)


# ---------- check_cgcs ----------

def test_check_cgcs_su2_2x2():
    """SU(2) 2⊗2 CGCs satisfy orthogonality. (SU(3) 3x3bar and 8x8 are
    covered in test_cgc_cache.py.)"""
    assert cgc.check_cgcs([(1, 0), (1, 0)]) is True


# ---------- print_cgcs smoke tests ----------

def test_print_cgcs_no_optional_args(capsys):
    """print_cgcs runs and produces non-empty stdout when called with no filtering."""
    cgc.print_cgcs(PRODUCT_3X3BAR)
    captured = capsys.readouterr()
    assert len(captured.out) > 0
    # Header marker (the '#' banner) appears at least once.
    assert "#" in captured.out


def test_print_cgcs_with_sum_iweight(capsys):
    """print_cgcs with sum_iweight only."""
    cgc.print_cgcs(PRODUCT_3X3BAR, sum_iweight=(0, 0, 0))
    captured = capsys.readouterr()
    assert "(0, 0, 0)" in captured.out


def test_print_cgcs_with_mult_idx(capsys):
    """print_cgcs with sum_iweight and mult_idx."""
    cgc.print_cgcs(PRODUCT_3X3BAR, sum_iweight=(0, 0, 0), mult_idx=1)
    captured = capsys.readouterr()
    assert "decomposition state:" in captured.out


def test_print_cgcs_with_sum_state(capsys):
    """print_cgcs with sum_iweight, mult_idx, sum_state."""
    cgc.print_cgcs(PRODUCT_3X3BAR, sum_iweight=(0, 0, 0), mult_idx=1, sum_state=0)
    captured = capsys.readouterr()
    assert "decomposition state:" in captured.out


def test_print_cgcs_all_args(capsys):
    """print_cgcs with all args — prints a single CGC value."""
    state_dict = cgc.calc_cgcs(
        PRODUCT_3X3BAR, sum_iweight=(0, 0, 0), mult_idx=1, sum_state=0
    )
    any_product_state = next(iter(state_dict))
    cgc.print_cgcs(
        PRODUCT_3X3BAR,
        sum_iweight=(0, 0, 0),
        mult_idx=1,
        sum_state=0,
        product_state=any_product_state,
    )
    captured = capsys.readouterr()
    assert len(captured.out) > 0
