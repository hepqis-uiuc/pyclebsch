"""
HELPER FUNCTIONS
"""

import numpy as np
from itertools import combinations_with_replacement
from ..su_n_operators import calc_casimir, IrrepWeight

def conjugate_iweight(iweight: IrrepWeight) -> IrrepWeight:
    """Returns i-weight corresponding to the conjugate irrep.
    """

    first,last = iweight[0],iweight[-1]
    return tuple(first - j + last for j in iweight[::-1])

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
