"""
Functions on SU(N) irreps that the lattice code uses.

- conjugate_iweight gives the i-weight of the conjugate irrep, for example
  (1, 1, 0) for (1, 0, 0) in SU(3).
- n_ality gives an irrep's charge under the center of SU(N).
- get_irreps gives the irreps whose quadratic Casimir is at most a given
  value, with their Casimirs; irreps_and_singlets (in lattice_data) uses it
  for the "C" and "B" truncations.
Irreps are labeled by i-weights (IrrepWeight, from su_n_operators).
"""

import numpy as np
from itertools import combinations_with_replacement
from ..su_n_operators import calc_casimir, IrrepWeight

def conjugate_iweight(iweight: IrrepWeight) -> IrrepWeight:
    """Returns i-weight corresponding to the conjugate irrep.
    """

    first,last = iweight[0],iweight[-1]
    return tuple(first - j + last for j in iweight[::-1])

def n_ality(iweight: IrrepWeight) -> int:
    """
    The N-ality of an SU(N) irrep: its charge under the center of SU(N),
    the group of the N-th roots of unity times the identity.

    It is the sum of the entries of the i-weight normalized to last entry 0
    (the number of boxes in its Young diagram), mod N, with N the length of
    the i-weight. The N-alities of an irrep and its conjugate add to 0 mod N.
    """
    N = len(iweight)
    last = iweight[-1]
    return sum(entry - last for entry in iweight) % N

def get_irreps(max_casimir: int | float, N: int) -> dict[tuple, float]:
    """
    Identifies all SU(N) irreps whose quadratic Casimirs are at most
    max_casimir. Irreps are returned as keys of a dictionary whose values
    are the corresponding quadratic Casimir.
    """

    # This is a brute-force search for all irreps whose quadratic Casimirs are
    # at most max_casimir. The search stops when all irreps of a 'T' truncation
    # have quadratic Casimirs higher than max_casimir.

    # T stands for the first value of an i-weight.
    T = 0
    searching = True
    irreps = {}

    while searching:

        # Search will stop if stop_search is not changed during for-loop.
        stop_search = True

        # Generate irreps for the current T, and check if their Casimirs
        # are less than max_casimir.
        for trial_rep in combinations_with_replacement(range(T+1), N-2):
            R = (T,) + trial_rep[::-1] + (0,)
            R_casimir = calc_casimir(R)
            if R_casimir < max_casimir or np.isclose(R_casimir, max_casimir):
                stop_search = False
                irreps[R] = R_casimir
            else:
                continue

        # If an acceptable irrep was found during the for-loop, then T will
        # be incremented. Otherwise, searching will cease.
        if not stop_search:
            T += 1
        else:
            searching = False
    
    # Irreps are returned in lexicographic order for neatness.
    return dict(sorted(irreps.items()))
