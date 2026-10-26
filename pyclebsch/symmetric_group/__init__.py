"""
Irreps of the symmetric group S_n (the permutations of n objects), which the
SU(N) code needs for tensor products with repeated factors.

Permuting identical factors of a tensor product of SU(N) irreps commutes
with the action of SU(N). So the copies of an SU(N) irrep in such a
product's decomposition can be chosen to transform under irreps of S_n. cgc
chooses them this way (with the Young projection operators of
young_symmetrizer), and su_n_operators.find_plethysms decomposes the n-th
tensor power of an irrep this way (its docstring defines plethysms).

An irrep of S_n is labeled by a partition of n: the row lengths of its Young
diagram, as a list of weakly decreasing integers, for example [2, 1].
calc_Sn_dimension gives its dimension by the hook length formula (J. S.
Frame et al., Can. J. Math. 6, 316 (1954)): n! divided by the product of the
hook lengths of the diagram's boxes, where a box's hook length is 1 plus the
number of boxes to its right plus the number below it. For [2, 1] the hook
lengths are 3, 1 and 1, so the dimension is 3!/3 = 2. The modules:
- tableaux: partitions and standard Young tableaux;
- young_symmetrizer: Young projection operators, which cgc uses;
- plethysm_utils: internal helpers for su_n_operators.find_plethysms.
"""

from math import factorial
from pyclebsch.symmetric_group.tableaux import _product_of_hook_lengths


def calc_Sn_dimension(partition: list) -> int:
    """Returns dimension of a symmetric group irrep,
    given by an integer partition of n via the hook length formula.
    """

    if not all(partition[i] >= partition[i+1] for i in range(len(partition)-1)):
        raise ValueError('Integers in partition must be sorted greatest to least.')

    # Gather initial data.
    n = sum(partition)
    hook_length_prod = _product_of_hook_lengths(partition)

    # Compute the hook length formula and round to ensure integral dimension.
    dim = factorial(n)/hook_length_prod
    return round(dim)
