"""Tests for pyclebsch.su_n_operators.

Covers normalize_iweight, calc_dimension, calc_casimir, calc_dynkin_index,
calc_weight, find_gt_patterns, ladder_op, find_suN_basis, find_direct_sum,
find_symmetry_direct_sum, find_plethysms.

Expected values for casimirs/Dynkin indices are derived in module-level
comments below so failures can be checked against the formula.
"""

import math

import numpy as np
import pytest

from pyclebsch.su_n_operators import (
    calc_casimir,
    calc_dimension,
    calc_dynkin_index,
    calc_weight,
    find_direct_sum,
    find_gt_patterns,
    find_plethysms,
    find_suN_basis,
    find_symmetry_direct_sum,
    ladder_op,
    normalize_iweight,
)


# ---------- normalize_iweight ----------

def test_normalize_iweight_subtracts_last_entry():
    assert normalize_iweight((3, 2, 1)) == (2, 1, 0)
    assert normalize_iweight((5, 5, 5)) == (0, 0, 0)


def test_normalize_iweight_already_normalized_passthrough():
    assert normalize_iweight((2, 1, 0)) == (2, 1, 0)
    assert normalize_iweight((0, 0, 0)) == (0, 0, 0)
    assert normalize_iweight((1, 0)) == (1, 0)


# ---------- calc_dimension ----------

@pytest.mark.parametrize(
    "iweight,expected_dim",
    [
        # SU(2)
        ((0, 0), 1),
        ((1, 0), 2),
        ((2, 0), 3),
        ((3, 0), 4),
        # SU(3)
        ((0, 0, 0), 1),
        ((1, 0, 0), 3),
        ((1, 1, 0), 3),
        ((2, 0, 0), 6),
        ((2, 1, 0), 8),
        ((3, 0, 0), 10),
    ],
)
def test_calc_dimension(iweight, expected_dim):
    assert calc_dimension(iweight) == expected_dim


# ---------- calc_casimir ----------
# SU(3) fundamental (1,0,0):
#   N=3, R=(1,0,0); cas = N*sum(R_i*(R_i+N-1-2i)) - sum(R)^2
#                       = 3*(1*(1+2)) - 1   = 9 - 1 = 8.
#   Casimir = cas/(2N) = 8/6 = 4/3.
# SU(3) adjoint (2,1,0):
#   cas = 3*(2*(2+2) + 1*(1+0)) - 3^2 = 3*9 - 9 = 18.
#   Casimir = 18/6 = 3.

def test_calc_casimir_trivial():
    assert calc_casimir((0, 0, 0)) == 0


def test_calc_casimir_su3_fundamental():
    assert math.isclose(calc_casimir((1, 0, 0)), 4 / 3)


def test_calc_casimir_su3_antifundamental():
    assert math.isclose(calc_casimir((1, 1, 0)), 4 / 3)


def test_calc_casimir_su3_adjoint():
    assert math.isclose(calc_casimir((2, 1, 0)), 3)


def test_calc_casimir_normalization_invariant():
    """Casimir depends only on the normalized i-weight."""
    assert math.isclose(calc_casimir((2, 1, 0)), calc_casimir((5, 4, 3)))


# ---------- calc_dynkin_index ----------
# SU(3) fundamental: cas=8, dimR=3, dimG=8 → T(R) = 8*3/(2*3*8) = 1/2.
# SU(3) adjoint:     cas=18, dimR=8, dimG=8 → T(R) = 18*8/(2*3*8) = 3.

def test_calc_dynkin_index_su3_fundamental():
    assert math.isclose(calc_dynkin_index((1, 0, 0)), 0.5)


def test_calc_dynkin_index_su3_adjoint():
    assert math.isclose(calc_dynkin_index((2, 1, 0)), 3)


# ---------- calc_weight ----------

def test_calc_weight_z_su2_fundamental():
    """SU(2) fund HW [[1,0],[1]] has z-weight 1/2; LW [[1,0],[0]] has -1/2."""
    hw_pattern = [[1, 0], [1]]
    lw_pattern = [[1, 0], [0]]
    assert calc_weight(hw_pattern, "z") == [0.5]
    assert calc_weight(lw_pattern, "z") == [-0.5]


def test_calc_weight_p_su2_fundamental():
    """SU(2) fund HW [[1,0],[1]] has p-weight [1, 0]; LW [[1,0],[0]] has [0, 1]."""
    hw_pattern = [[1, 0], [1]]
    lw_pattern = [[1, 0], [0]]
    assert calc_weight(hw_pattern, "p") == [1, 0]
    assert calc_weight(lw_pattern, "p") == [0, 1]


def test_calc_weight_invalid_kind_raises():
    with pytest.raises(ValueError):
        calc_weight([[1, 0], [1]], "x")


# ---------- find_gt_patterns ----------

@pytest.mark.parametrize(
    "iweight",
    [(1, 0), (2, 0), (3, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 0, 0)],
)
def test_find_gt_patterns_count_matches_dimension(iweight):
    patterns = find_gt_patterns(iweight)
    assert len(patterns) == calc_dimension(iweight)


def test_find_gt_patterns_highest_weight_first():
    """Index 0 of find_gt_patterns is the highest-weight state.

    Its top row equals the i-weight, and every subsequent row equals the
    leading entries of the row above (the unique p-weight-maximizing pattern).
    """
    iweight = (2, 1, 0)
    patterns = find_gt_patterns(iweight)
    hw = patterns[0]
    assert hw[0] == list(iweight)
    assert hw[1] == [iweight[0], iweight[1]]
    assert hw[2] == [iweight[0]]


def test_find_gt_patterns_sorted_by_pweight_descending():
    iweight = (2, 1, 0)
    patterns = find_gt_patterns(iweight)
    pweights = [calc_weight(p, "p") for p in patterns]
    for earlier, later in zip(pweights, pweights[1:]):
        assert earlier >= later


# ---------- ladder_op ----------

def test_ladder_op_raise_annihilates_highest_weight_su2():
    """J(1)+ |HW> = 0 for SU(2) fundamental."""
    hw_pattern = [[1, 0], [1]]
    result = ladder_op([hw_pattern], k=1, kind="+")
    assert result == []


def test_ladder_op_lower_highest_weight_su2_fundamental():
    """J(1)- |HW> = |LW> with coefficient 1 for SU(2) fundamental."""
    hw_pattern = [[1, 0], [1]]
    result = ladder_op([hw_pattern], k=1, kind="-")
    assert len(result) == 1
    coeff, new_state_list = result[0]
    assert math.isclose(coeff, 1.0)
    assert new_state_list == [[[1, 0], [0]]]


def test_ladder_op_lower_annihilates_lowest_weight_su2():
    """J(1)- |LW> = 0 for SU(2) fundamental."""
    lw_pattern = [[1, 0], [0]]
    result = ladder_op([lw_pattern], k=1, kind="-")
    assert result == []


def test_ladder_op_raise_lowest_weight_su2_fundamental():
    """J(1)+ |LW> = |HW> with coefficient 1 for SU(2) fundamental."""
    lw_pattern = [[1, 0], [0]]
    result = ladder_op([lw_pattern], k=1, kind="+")
    assert len(result) == 1
    coeff, new_state_list = result[0]
    assert math.isclose(coeff, 1.0)
    assert new_state_list == [[[1, 0], [1]]]


# ---------- find_suN_basis ----------

def test_find_suN_basis_su2_count_and_shape():
    """su(2) has 3 generators, each 2x2 in the fundamental."""
    basis = find_suN_basis((1, 0))
    assert len(basis) == 3
    for T in basis:
        assert T.shape == (2, 2)


def test_find_suN_basis_su3_count():
    """su(3) has 8 generators in any irrep."""
    basis_fund = find_suN_basis((1, 0, 0))
    assert len(basis_fund) == 8
    for T in basis_fund:
        assert T.shape == (3, 3)


def test_find_suN_basis_su2_fundamental_is_pauli_over_two():
    """In the SU(2) fundamental, basis = (T_z, T_x, T_y) = (σ_z/2, σ_x/2, σ_y/2)."""
    basis = [T.toarray() for T in find_suN_basis((1, 0))]
    Tz, Tx, Ty = basis
    expected_Tz = np.array([[0.5, 0.0], [0.0, -0.5]])
    expected_Tx = np.array([[0.0, 0.5], [0.5, 0.0]])
    expected_Ty = np.array([[0.0, -0.5j], [0.5j, 0.0]])
    np.testing.assert_allclose(Tz, expected_Tz, atol=1e-12)
    np.testing.assert_allclose(Tx, expected_Tx, atol=1e-12)
    np.testing.assert_allclose(Ty, expected_Ty, atol=1e-12)


def test_find_suN_basis_su2_fundamental_commutator():
    """[T_x, T_y] = i T_z in the SU(2) fundamental basis ordering (Tz, Tx, Ty)."""
    basis = [T.toarray() for T in find_suN_basis((1, 0))]
    Tz, Tx, Ty = basis
    np.testing.assert_allclose(Tx @ Ty - Ty @ Tx, 1j * Tz, atol=1e-12)


def test_find_suN_basis_su2_fundamental_normalization():
    """Tr(T_a^2) = 1/2 for each fundamental SU(2) generator."""
    basis = [T.toarray() for T in find_suN_basis((1, 0))]
    for T in basis:
        assert math.isclose(np.trace(T @ T).real, 0.5, abs_tol=1e-12)


# ---------- find_direct_sum ----------

def test_find_direct_sum_su2_2_x_2():
    """SU(2): 2 ⊗ 2 = 3 ⊕ 1."""
    result = find_direct_sum([(1, 0), (1, 0)])
    assert result == {(2, 0): 1, (0, 0): 1}


def test_find_direct_sum_su3_fund_x_afund():
    """SU(3): 3 ⊗ 3̄ = 8 ⊕ 1."""
    result = find_direct_sum([(1, 0, 0), (1, 1, 0)])
    assert result == {(2, 1, 0): 1, (0, 0, 0): 1}


def test_find_direct_sum_su3_fund_x_fund():
    """SU(3): 3 ⊗ 3 = 6 ⊕ 3̄."""
    result = find_direct_sum([(1, 0, 0), (1, 0, 0)])
    assert result == {(2, 0, 0): 1, (1, 1, 0): 1}


def test_find_direct_sum_su3_adjoint_x_adjoint():
    """SU(3): 8 ⊗ 8 = 27 ⊕ 10 ⊕ 10̄ ⊕ 8 ⊕ 8 ⊕ 1.

    i-weight labels: 27=(4,2,0), 10=(3,0,0), 10̄=(3,3,0), 8=(2,1,0), 1=(0,0,0).
    """
    result = find_direct_sum([(2, 1, 0), (2, 1, 0)])
    assert result == {
        (4, 2, 0): 1,
        (3, 3, 0): 1,
        (3, 0, 0): 1,
        (2, 1, 0): 2,
        (0, 0, 0): 1,
    }


def test_find_direct_sum_with_sum_iweight_returns_multiplicity():
    """Passing sum_iweight returns an int multiplicity (0 if absent)."""
    assert find_direct_sum([(2, 1, 0), (2, 1, 0)], sum_iweight=(2, 1, 0)) == 2
    assert find_direct_sum([(2, 1, 0), (2, 1, 0)], sum_iweight=(0, 0, 0)) == 1
    assert find_direct_sum([(1, 0, 0), (1, 0, 0)], sum_iweight=(2, 1, 0)) == 0


# ---------- find_symmetry_direct_sum ----------

def test_find_symmetry_direct_sum_su3_fund_squared():
    """3 ⊗ 3 = 6 (symmetric) ⊕ 3̄ (antisymmetric).

    The 6=(2,0,0) sits under partition (2,) of S_2; the 3̄=(1,1,0) under (1,1).
    idx_list is [[0, 1]] — both factors are the same irrep.
    """
    decomp, idx_list = find_symmetry_direct_sum([(1, 0, 0), (1, 0, 0)])
    assert idx_list == [[0, 1]]
    assert decomp == {
        (2, 0, 0): {((2,),): 1},
        (1, 1, 0): {((1, 1),): 1},
    }


def test_find_symmetry_direct_sum_su2_fund_squared():
    """SU(2) 2 ⊗ 2: triplet (2,0) symmetric, singlet (0,0) antisymmetric."""
    decomp, idx_list = find_symmetry_direct_sum([(1, 0), (1, 0)])
    assert idx_list == [[0, 1]]
    assert decomp == {
        (2, 0): {((2,),): 1},
        (0, 0): {((1, 1),): 1},
    }


# ---------- find_plethysms ----------

def test_find_plethysms_su2_fund_n2():
    """SU(2) fund n=2: triplet (2,0) symmetric, singlet (0,0) antisymmetric."""
    result = find_plethysms((1, 0), 2)
    assert result == {
        (2, 0): {(2,): 1},
        (0, 0): {(1, 1): 1},
    }


def test_find_plethysms_su3_fund_n2():
    """SU(3) fund n=2: 6=(2,0,0) under (2,); 3̄=(1,1,0) under (1,1)."""
    result = find_plethysms((1, 0, 0), 2)
    assert result == {
        (2, 0, 0): {(2,): 1},
        (1, 1, 0): {(1, 1): 1},
    }


def test_find_plethysms_su3_fund_n3():
    """SU(3) fund n=3: 10 under (3,), 8 under (2,1), singlet under (1,1,1).

    Total dim: 10 + 2*8 + 1 = 27 = 3^3. The factor 2 on the adjoint comes
    from the dim-2 (2,1) representation of S_3, not from the multiplicity
    reported here (which is 1).
    """
    result = find_plethysms((1, 0, 0), 3)
    assert result == {
        (3, 0, 0): {(3,): 1},
        (2, 1, 0): {(2, 1): 1},
        (0, 0, 0): {(1, 1, 1): 1},
    }
