"""Tests for pyclebsch.symmetric_group.

Covers:
- tableaux.py: find_partitions, find_tableaux, partition_to_sequence,
               sequence_to_partition, _product_of_hook_lengths.
- young_symmetrizer.py: young_symmetrizer (trivial sym/antisym +
               projector idempotency).

Expected values are derived from textbook formulas:
- Number of partitions of n (OEIS A000041): p(0..7) = 1,1,2,3,5,7,11,15.
- Hook length formula: dim(S_n irrep for partition λ) = n! / prod(hook lengths).
"""

from math import factorial

import pytest

from pyclebsch.symmetric_group.tableaux import (
    _product_of_hook_lengths,
    find_partitions,
    find_tableaux,
    partition_to_sequence,
    sequence_to_partition,
)
from pyclebsch.symmetric_group.young_symmetrizer import young_symmetrizer


# ---------- find_partitions ----------

@pytest.mark.parametrize(
    "n,expected_count",
    [(0, 1), (1, 1), (2, 2), (3, 3), (4, 5), (5, 7), (6, 11), (7, 15)],
)
def test_find_partitions_count(n, expected_count):
    """Number of partitions of n matches the partition function (OEIS A000041)."""
    assert len(list(find_partitions(n))) == expected_count


def test_find_partitions_n4_content():
    """All partitions of 4: {4}, {3,1}, {2,2}, {2,1,1}, {1,1,1,1}."""
    partitions = {tuple(p) for p in find_partitions(4)}
    assert partitions == {(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)}


def test_find_partitions_descending():
    """Every emitted partition has weakly-descending entries."""
    for p in find_partitions(6):
        assert all(p[i] >= p[i + 1] for i in range(len(p) - 1))


def test_find_partitions_sum_to_n():
    """Every emitted partition sums to n (with one quirk: n=0 yields [[0]])."""
    for n in (1, 2, 3, 4, 5, 6):
        for p in find_partitions(n):
            assert sum(p) == n


def test_find_partitions_n0_yields_singleton_zero():
    """n=0 yields a single partition [0] (algorithm quirk; documenting actual behavior)."""
    assert list(find_partitions(0)) == [[0]]


# ---------- partition_to_sequence / sequence_to_partition ----------

@pytest.mark.parametrize(
    "partition",
    [[3, 2, 1], [4], [1, 1, 1, 1], [2, 1], [5, 2, 2, 1], [1]],
)
def test_partition_sequence_roundtrip(partition):
    """sequence_to_partition(partition_to_sequence(p)) == p."""
    assert sequence_to_partition(partition_to_sequence(partition)) == partition


def test_partition_to_sequence_explicit_cases():
    """Hand-derived expected sequences."""
    # [3,2,1]: three rows differing by 1 each → three (1,0) blocks
    assert partition_to_sequence([3, 2, 1]) == [1, 0, 1, 0, 1, 0]
    # [4]: single row of 4 → four 1s then a 0
    assert partition_to_sequence([4]) == [1, 1, 1, 1, 0]
    # [1,1,1,1]: bottom row 1, then 0/0/0 separators
    assert partition_to_sequence([1, 1, 1, 1]) == [1, 0, 0, 0, 0]


# ---------- _product_of_hook_lengths ----------

@pytest.mark.parametrize(
    "partition,expected_product",
    [
        ([1], 1),
        ([2], 2),       # hooks: 2, 1
        ([1, 1], 2),    # hooks: 2, 1
        ([2, 1], 3),    # hooks: 3, 1, 1
        ([3, 1], 8),    # hooks: 4, 2, 1, 1
        ([2, 2], 12),   # hooks: 3, 2, 2, 1
        ([4], 24),      # hooks: 4, 3, 2, 1
        ([1, 1, 1, 1], 24),  # hooks: 4, 3, 2, 1
    ],
)
def test_product_of_hook_lengths(partition, expected_product):
    assert _product_of_hook_lengths(partition) == expected_product


@pytest.mark.parametrize(
    "partition",
    [[2, 1], [3, 1], [2, 2], [3, 2, 1], [4, 2]],
)
def test_hook_length_formula_gives_tableau_count(partition):
    """n! / product(hook lengths) == number of standard Young tableaux."""
    n = sum(partition)
    expected_count = factorial(n) // _product_of_hook_lengths(partition)
    assert len(find_tableaux(partition)) == expected_count


# ---------- find_tableaux ----------

@pytest.mark.parametrize(
    "partition,expected_count",
    [
        ([1], 1),
        ([2], 1),
        ([1, 1], 1),
        ([3], 1),
        ([2, 1], 2),
        ([1, 1, 1], 1),
        ([2, 2], 2),
        ([3, 1], 3),
        ([2, 1, 1], 3),
        ([4], 1),
        ([1, 1, 1, 1], 1),
    ],
)
def test_find_tableaux_count(partition, expected_count):
    assert len(find_tableaux(partition)) == expected_count


def test_find_tableaux_cell_values_are_zero_through_n_minus_one():
    """Every SYT contains exactly the integers 0..n-1."""
    for partition in [[2, 1], [3, 1], [2, 2], [3, 2, 1]]:
        n = sum(partition)
        for tableau in find_tableaux(partition):
            cells = [v for row in tableau for v in row]
            assert sorted(cells) == list(range(n))


def test_find_tableaux_shape_matches_partition():
    """Row lengths of each SYT match the partition."""
    for partition in [[2, 1], [3, 1], [2, 2], [3, 2, 1]]:
        for tableau in find_tableaux(partition):
            assert [len(row) for row in tableau] == partition


def test_find_tableaux_rows_columns_strictly_increasing():
    """Standard Young tableaux: entries strictly increase along rows and columns."""
    for partition in [[2, 1], [3, 1], [2, 2], [3, 2, 1]]:
        for tableau in find_tableaux(partition):
            for row in tableau:
                assert all(row[j] < row[j + 1] for j in range(len(row) - 1))
            num_cols = len(tableau[0])
            for c in range(num_cols):
                col = [tableau[r][c] for r in range(len(tableau)) if c < len(tableau[r])]
                assert all(col[i] < col[i + 1] for i in range(len(col) - 1))


def test_find_tableaux_partition_explicit_2_1():
    """[2,1] has exactly two SYTs: [[0,1],[2]] and [[0,2],[1]]."""
    tableaux = find_tableaux([2, 1])
    assert sorted(tableaux) == [[[0, 1], [2]], [[0, 2], [1]]]


def test_find_tableaux_unsorted_partition_raises():
    """Partition not in weakly-descending order raises ValueError."""
    with pytest.raises(ValueError):
        find_tableaux([1, 2])


# ---------- young_symmetrizer ----------

def _collect(generator):
    """Materialize a Young symmetrizer generator into {perm: coeff} dict."""
    result = {}
    for coeff, perm in generator:
        result[perm] = result.get(perm, 0.0) + coeff
    return result


def _compose_symmetrizers(Y1, Y2):
    """Group-algebra product: (Σ a_p [P]) * (Σ b_q [Q]) = Σ a_p b_q [P∘Q].

    Permutation composition follows the docstring convention `A[i] -> A[P[i]]`:
    applying P then Q gives the combined permutation R with R[i] = P[Q[i]].
    """
    result = {}
    for P, aP in Y1.items():
        for Q, bQ in Y2.items():
            R = tuple(P[Q[i]] for i in range(len(P)))
            result[R] = result.get(R, 0.0) + aP * bQ
    return result


def _dicts_close(d1, d2, atol=1e-10):
    keys = set(d1) | set(d2)
    return all(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) < atol for k in keys)


def test_young_symmetrizer_trivial_n1():
    """n=1 symmetrizer: identity permutation with coefficient 1."""
    Y = _collect(young_symmetrizer([[[0]]], [[0]]))
    assert Y == {(0,): 1.0}


def test_young_symmetrizer_total_symmetric_n2():
    """[[0,1]]: (1/2)(e + (01))."""
    Y = _collect(young_symmetrizer([[[0, 1]]], [[0, 1]]))
    assert set(Y.keys()) == {(0, 1), (1, 0)}
    assert _dicts_close(Y, {(0, 1): 0.5, (1, 0): 0.5})


def test_young_symmetrizer_total_antisymmetric_n2():
    """[[0],[1]]: (1/2)(e - (01))."""
    Y = _collect(young_symmetrizer([[[0], [1]]], [[0, 1]]))
    assert set(Y.keys()) == {(0, 1), (1, 0)}
    assert _dicts_close(Y, {(0, 1): 0.5, (1, 0): -0.5})


def test_young_symmetrizer_total_symmetric_n3():
    """[[0,1,2]]: (1/6) sum over all 6 permutations of S_3."""
    Y = _collect(young_symmetrizer([[[0, 1, 2]]], [[0, 1, 2]]))
    assert len(Y) == 6  # all 6 perms of S_3
    for coeff in Y.values():
        assert abs(coeff - 1.0 / 6) < 1e-12


def test_young_symmetrizer_total_antisymmetric_n3():
    """[[0],[1],[2]]: (1/6) sum over perms with sgn(p) sign."""
    Y = _collect(young_symmetrizer([[[0], [1], [2]]], [[0, 1, 2]]))
    assert len(Y) == 6
    # Identity coefficient is +1/6; the three transpositions are -1/6.
    assert abs(Y[(0, 1, 2)] - 1.0 / 6) < 1e-12
    assert abs(Y[(1, 0, 2)] + 1.0 / 6) < 1e-12
    assert abs(Y[(0, 2, 1)] + 1.0 / 6) < 1e-12
    assert abs(Y[(2, 1, 0)] + 1.0 / 6) < 1e-12
    # Even-parity 3-cycles are +1/6.
    assert abs(Y[(1, 2, 0)] - 1.0 / 6) < 1e-12
    assert abs(Y[(2, 0, 1)] - 1.0 / 6) < 1e-12


@pytest.mark.parametrize(
    "tableau",
    [
        [[0, 1]],          # n=2 symmetric
        [[0], [1]],        # n=2 antisymmetric
        [[0, 1, 2]],       # n=3 symmetric
        [[0], [1], [2]],   # n=3 antisymmetric
        [[0, 1], [2]],     # n=3 mixed (row-ordered)
        [[0, 2], [1]],     # n=3 mixed (not row- or column-ordered)
        [[0, 1, 2], [3]],  # n=4 mixed
        [[0, 1], [2, 3]],  # n=4 mixed
    ],
)
def test_young_symmetrizer_is_a_projector(tableau):
    """Y^2 == Y for any standard-Young-tableau symmetrizer (normalized).

    This is the defining property of the Young projector and verifies the
    `_reduce_perms` algorithm, the MOLD descent, and the final normalization.
    """
    n = sum(len(row) for row in tableau)
    idx_list = [list(range(n))]
    Y = _collect(young_symmetrizer([tableau], idx_list))
    Y_squared = _compose_symmetrizers(Y, Y)
    assert _dicts_close(Y, Y_squared, atol=1e-10), (
        f"tableau={tableau}: Y^2 != Y. Y={Y}, Y^2={Y_squared}"
    )
