"""Tests for pyclebsch.matrix_elements.helpers."""

import math

import pytest

from pyclebsch.matrix_elements.helpers import conjugate_iweight, get_irreps
from pyclebsch.su_n_operators import calc_casimir, calc_dimension


# ---------- conjugate_iweight ----------

@pytest.mark.parametrize(
    "iweight,expected_conjugate",
    [
        # SU(3) basic cases.
        ((0, 0, 0), (0, 0, 0)),    # trivial
        ((1, 0, 0), (1, 1, 0)),    # 3 → 3̄
        ((1, 1, 0), (1, 0, 0)),    # 3̄ → 3
        ((2, 1, 0), (2, 1, 0)),    # 8 (self-conjugate)
        ((2, 0, 0), (2, 2, 0)),    # 6 → 6̄
        ((2, 2, 0), (2, 0, 0)),    # 6̄ → 6
        ((3, 0, 0), (3, 3, 0)),    # 10 → 10̄
        # SU(2): every irrep is self-conjugate.
        ((0, 0), (0, 0)),
        ((1, 0), (1, 0)),
        ((2, 0), (2, 0)),
    ],
)
def test_conjugate_iweight_explicit(iweight, expected_conjugate):
    assert conjugate_iweight(iweight) == expected_conjugate


@pytest.mark.parametrize(
    "iweight",
    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 0, 0), (3, 1, 0)],
)
def test_conjugate_iweight_involution(iweight):
    """Conjugation is an involution: conj(conj(R)) == R."""
    assert conjugate_iweight(conjugate_iweight(iweight)) == iweight


@pytest.mark.parametrize(
    "iweight",
    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 0, 0), (3, 1, 0)],
)
def test_conjugate_iweight_preserves_dimension(iweight):
    """The conjugate irrep has the same dimension."""
    assert calc_dimension(iweight) == calc_dimension(conjugate_iweight(iweight))


# ---------- get_irreps ----------

def test_get_irreps_su3_at_fundamental_casimir():
    """SU(3) with max_casimir=4/3: trivial + fund + antifund.

    The fundamental and antifundamental both have Casimir 4/3 (which is
    accepted via np.isclose). The next-smallest SU(3) Casimir is the 6's
    (=10/3), which exceeds 4/3.
    """
    result = get_irreps(max_casimir=4 / 3, N=3)
    assert set(result.keys()) == {(0, 0, 0), (1, 0, 0), (1, 1, 0)}
    assert result[(0, 0, 0)] == 0
    assert math.isclose(result[(1, 0, 0)], 4 / 3)
    assert math.isclose(result[(1, 1, 0)], 4 / 3)


def test_get_irreps_su3_at_adjoint_casimir():
    """SU(3) with max_casimir=3 (the adjoint Casimir): trivial + fund + antifund + 8.

    Casimirs through 3: trivial=0, fund=antifund=4/3, 8=3.
    The 6=(2,0,0) and 6̄=(2,2,0) have Casimir 10/3 ≈ 3.33, so they are
    excluded. The adjoint sits exactly at the cutoff and is admitted via
    np.isclose.
    """
    result = get_irreps(max_casimir=3, N=3)
    expected = {(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)}
    assert set(result.keys()) == expected
    assert math.isclose(result[(2, 1, 0)], 3)


def test_get_irreps_su3_above_sextet_casimir():
    """SU(3) with max_casimir=10/3: now admits 6 and 6̄ in addition to the adjoint."""
    result = get_irreps(max_casimir=10 / 3, N=3)
    expected = {(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 2, 0), (2, 1, 0)}
    assert set(result.keys()) == expected
    assert math.isclose(result[(2, 0, 0)], 10 / 3)
    assert math.isclose(result[(2, 2, 0)], 10 / 3)


def test_get_irreps_monotonicity():
    """Increasing max_casimir produces a superset of the lower-cutoff result."""
    low = set(get_irreps(max_casimir=4 / 3, N=3).keys())
    high = set(get_irreps(max_casimir=3, N=3).keys())
    assert low.issubset(high)


def test_get_irreps_only_trivial_for_zero_cutoff():
    """max_casimir=0 admits only the trivial irrep."""
    result = get_irreps(max_casimir=0, N=3)
    assert set(result.keys()) == {(0, 0, 0)}


def test_get_irreps_casimir_values_match_calc_casimir():
    """Each returned Casimir equals what calc_casimir computes directly."""
    result = get_irreps(max_casimir=3, N=3)
    for iweight, casimir in result.items():
        assert math.isclose(casimir, calc_casimir(iweight))


def test_get_irreps_su2_at_fundamental_casimir():
    """SU(2) with max_casimir=3/4 (the spin-1/2 Casimir): trivial + 2.

    SU(2) Casimirs: j=0→0, j=1/2→3/4, j=1→2. So 3/4 admits j=0 and j=1/2 only.
    """
    result = get_irreps(max_casimir=3 / 4, N=2)
    assert set(result.keys()) == {(0, 0), (1, 0)}
    assert result[(0, 0)] == 0
    assert math.isclose(result[(1, 0)], 3 / 4)
