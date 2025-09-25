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
