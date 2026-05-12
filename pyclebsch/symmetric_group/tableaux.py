from itertools import product, permutations
from collections.abc import Generator


YoungTableau = list[list[int]]

def _conjugate_partition(partition):
    """Computes a conjugate partition (the partition corresponding to the
    "transpose" of a Young tableau for the given partition).
    """

    # Whereas the partition gives the number of cells per row of a tableau,
    # some calculations additionally need the number of cells per column,
    # which is contained in the conjugate partition.

    conjugate_partition = []
    for col in range(1, partition[0]+1):
        num_col_cells = 0
        for num_row_cells in partition:
            if num_row_cells >= col:
                num_col_cells += 1
            else:
                continue
        conjugate_partition.append(num_col_cells)

    return conjugate_partition


def _product_of_hook_lengths(partition):
    """This nested for-loop finds the hook length for each cell of a tableau
    for the partition and multiplies them all.
    """

    conjugate_partition = _conjugate_partition(partition)
    hook_length_prod = 1
    for i in range(len(partition)):
        for j in range(partition[i]):
            hook_length = partition[i] + conjugate_partition[j] - i - j - 1
            hook_length_prod *= hook_length

    return hook_length_prod


def _tableau_data(tableau, extra_data=False):
    """Finds the permutations that preserve the rows and columns of a tableau
    as well as whether the tableau is row and/or column ordered.
    With extra_data, a product of hook lengths are also returned.
    """

    # Gather initial data. Out of these variables, n is returned.
    # tableau_T is the transpose of the tableau.

    partition = [len(row) for row in tableau]
    conjugate_partition = _conjugate_partition(partition)
    tableau_T = [[tableau[j][i] for j in range(conjugate_partition[i])] for i in range(partition[0])]
    n = sum(partition)

    # Extra data includes the product of hook lengths of the tableau
    # (necessary for normalization).

    if extra_data:
        hook_length_prod = _product_of_hook_lengths(partition)
    else:
        pass

    # row_word is the tableau, flattened row-wise.
    # col_word is the tableau, flattened column-wise.
    # is_row_ordered and is_col_ordered check if the lists
    # are ordered from least to greatest.
    # ~Definition 2

    row_word = [i for row in tableau for i in row]
    col_word = [i for col in tableau_T for i in col]
    is_row_ordered = all(row_word[i] < row_word[i+1] for i in range(n-1))
    is_col_ordered = all(col_word[i] < col_word[i+1] for i in range(n-1))

    # row_permutations is a list of permutations that preserve the
    # content of each row. col_permutations is a list of permutations
    # that preserve the content of each column. They are built by joining
    # all combinations X of disjoint permutations on each row/column.
    # sorted_rows/cols sorts each row/column, for reference, to build
    # the mapping that each X corresponds to. perm turns the mapping
    # to the usual tuple version of a permutation.

    row_permutations = []
    sorted_rows = [sorted(row) for row in tableau if len(row)>1]
    for X in product(*(permutations(row) for row in tableau if len(row)>1)):
        mapping = {sorted_rows[i][j]: X[i][j] for i in range(len(sorted_rows)) for j in range(partition[i]) if X[i][j] != sorted_rows[i][j]}
        perm = tuple(mapping[i] if i in mapping else i for i in range(n))
        row_permutations.append(perm)

    col_permutations = []
    sorted_cols = [sorted(col) for col in tableau_T if len(col)>1]
    for X in product(*(permutations(col) for col in tableau_T if len(col)>1)):
        mapping = {sorted_cols[i][j]: X[i][j] for i in range(len(sorted_cols)) for j in range(conjugate_partition[i]) if X[i][j] != sorted_cols[i][j]}
        perm = tuple(mapping[i] if i in mapping else i for i in range(n))
        col_permutations.append(perm)

    if extra_data:
        return n, row_permutations, col_permutations, is_row_ordered, is_col_ordered, hook_length_prod
    else:
        return n, row_permutations, col_permutations, is_row_ordered, is_col_ordered


def partition_to_sequence(p):
    s,num_rows = [],len(p)
    for i in range(num_rows):
        if i==0:
            s += [1]*p[num_rows-1] + [0]
        else:
            s += [1]*(p[num_rows-i-1]- p[num_rows-i]) + [0]
    return s


def sequence_to_partition(s):
    p,num_ones = [],0
    for i in s:
        if i==0:
            p = [num_ones] + p
        else:
            num_ones += 1
    return [x for x in p if x!=0]


def _rim_hooks(partition, l):
    """Finds the rim hooks of a given length l
    for a given partition.
    """

    # Reference for the Murnaghan–Nakayama rule, as suggested by GroupMath.
    # https://www.sciencedirect.com/science/article/pii/S0747717104000112

    sequence = partition_to_sequence(partition)
    rhooks = []
    for i in range(len(sequence)-l):
        if sequence[i]==1 and sequence[i+l]==0:
            rhseq = sequence.copy()
            rhseq[i],rhseq[i+l] = 0,1
            length = sequence[i:i+l+1].count(0)-1
            rhooks.append([sequence_to_partition(rhseq), length])
        else:
            continue

    return rhooks


def find_partitions(n: int) -> Generator[list]:
    """Creates all integer partitions of n using the accel_asc algorithm.
    Partitions are generated as lists of weakly-descending integers.
    """

    # Adapted from
    # https://jeromekelleher.net/generating-integer-partitions.html

    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2][::-1]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1][::-1]


def find_tableaux(partition: list) -> list[YoungTableau]:
    """Generates standard Young tableaux given a partition.
    Returns a list of the tableaux.
    """

    # This code is largely adapted from the PermutationGroup.m file
    # of GroupMath, https://renatofonseca.net/groupmath

    if not all(partition[i] >= partition[i+1] for i in range(len(partition)-1)):
        raise ValueError('Integers in partition must be sorted greatest to least.')

    # Gather initial data.
    n = sum(partition)
    num_rows = len(partition)
    zeros = [0 for i in range(n)]
    conjugate_partition = _conjugate_partition(partition)

    # Compute canonical Young tableau. This tableau has all integers
    # from 0 to n-1 incrementally placed along the rows, top to bottom.
    # For instance, [[0,1,2],[3,4],[5]].
    canonical, count = [], 0
    for i in range(num_rows):
        row = []
        for j in range(partition[i]):
            row.append(count)
            count += 1
        canonical.append(row)

    # idxs_to_check is a list of lists, where each list is a list of
    # at most two indices that a cell of the Young tableau must be
    # immediately less than to remain standard. The index of the cell
    # in the flattened tableau is the index of that cell in idxs_to_check.

    idxs_to_check = [[] for i in range(n)]

    for i in range(num_rows):
        for j in range(partition[i]):
            if j < partition[i]-1:
                # Get index to the right of (i,j).
                idxs_to_check[canonical[i][j]].append(canonical[i][j+1])
            if i < num_rows-1 and j < partition[i+1]:
                # Get index below (i,j).
                idxs_to_check[canonical[i][j]].append(canonical[i+1][j])

    # max_cell_vals gives the maximum value a cell can have on a
    # standard Young tableau. This list is also flattened, so each
    # entry's index corresponds to the cell with that index value
    # on the canonical tableau.

    max_cell_vals = []
    for i in range(num_rows):
        for j in range(partition[i]):
            num = sum(partition[:i]) + sum(conjugate_partition[:j]) - i*j
            max_cell_vals.append(num)

    # Recursively generate the flattened standard Young tableaux.
    # idx is the current index of the tableau that being filled in.
    # tab_to_fill is the flattened tableau that needs to be filled;
    # as idx increases it gradually becomes a standard tableau.
    # min_cell_vals is a dynamical list of minimum values all
    # cells in the tableau can have.

    def generate(idx=0, tab_to_fill=zeros, min_cell_vals=zeros):
            
        # possible_cell_values gives the possible values the current
        # cell can possibly have given the current min_cell_vals.
        # Care is taken to not allow values that are already in tab_to_fill.
        possible_cell_values = [val for val in range(min_cell_vals[idx], max_cell_vals[idx]+1) if val not in tab_to_fill[:idx]]

        # For each value in possible_cell_values, create a new tab_to_fill
        # and min_cell_vals. These go into new_tabs_to_fill and new_min_cell_vals.
        # For new_min_cell_vals, care is taken to update the minimums
        # of the idxs_to_check to the possible value plus one if
        # their current minimum is below it.
        new_tabs_to_fill = []
        new_min_cell_vals = []
        for val in possible_cell_values:
            tab_aux = tab_to_fill.copy()
            tab_aux[idx] = val
            new_tabs_to_fill.append(tab_aux)

            min_aux = min_cell_vals.copy()
            for i in idxs_to_check[idx]:
                min_aux[i] = max(min_aux[i], val+1)
            new_min_cell_vals.append(min_aux)

        # idx cannot be greater than n-1.
        if idx==n-1:
            return new_tabs_to_fill
        else:
            flattened_tableaux = []
            for i in range(len(possible_cell_values)):
                flattened_tableaux += generate(idx+1, new_tabs_to_fill[i], new_min_cell_vals[i])
            return flattened_tableaux

    flat_tabs = generate()

    # Unflatten generated flattened tableaux.

    standard_tableaux = []
    for t in range(len(flat_tabs)):
        new_tab,count = [],0
        for i in range(num_rows):
            row = []
            for j in range(partition[i]):
                row.append(flat_tabs[t][count])
                count += 1
            new_tab.append(row)
        standard_tableaux.append(new_tab)

    return standard_tableaux
