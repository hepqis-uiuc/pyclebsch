"""
IMPORTS
"""

import numpy as np
from math import factorial
from scipy.sparse import csr_array
from scipy.sparse.linalg import eigsh
from itertools import product, combinations, permutations
from collections import Counter, defaultdict
from collections.abc import Generator
from functools import reduce
from more_itertools import locate, product_index
from pathlib import Path, PurePath
from pickle import load, dump

# REFERENCE https://homepages.physik.uni-muenchen.de/~vondelft/PapersVonDelft/Alex2011.pdf
EPS = 1e-10

# Creates CGC_Data directory in directory of this script.
script_directory = Path(__file__).resolve().parent
data_directory = PurePath(script_directory, 'CGC_Data')
Path(data_directory).mkdir(exist_ok=True)

"""
PROPERTIES AND OPERATIONS
"""

def calc_dimension(iweight: tuple) -> int:
    """Returns dimension of an irrep.
    ~Eq. (22)
    """

    dim = 1
    for j in range(1, len(iweight)):
        for jp in range(j):
            dim *= 1 + (iweight[jp] - iweight[j])/(j - jp)

    return round(dim)

def calc_weight(gt_pattern: list[list], kind: str) -> list:
    """Returns the weight of a basis state of an irrep.
    The basis state is given as a gt_pattern.
    The weight can be either a z-weight (kind='z') or a p-weight (kind='p').
    ~Eqs. (24)-(25)
    """

    weight = []
    N = len(gt_pattern)

    # Weights are returned as if the basis state were acted on by
    # J(1)z, ..., J(N-1)z in that order. However, J1z corresponds to
    # the last row of a GT-pattern. Hence, the loops begin at N-1.

    if kind == 'z':
        for k in range(N-1, 0, -1):
            if k==N-1:
                weight.append(sum(gt_pattern[k]) - sum(gt_pattern[k-1])/2)
            else:
                weight.append(sum(gt_pattern[k]) - (sum(gt_pattern[k-1]) + sum(gt_pattern[k+1]))/2)
        return weight
    
    elif kind == 'p':
        for l in range(N-1, -1, -1):
            if l==N-1:
                weight.append(sum(gt_pattern[l]))
            else:
                weight.append(sum(gt_pattern[l]) - sum(gt_pattern[l+1]))
        return weight
    
    else:
        raise ValueError('Invalid weight kind.')

def find_gt_patterns(iweight: tuple) -> list[list[list]]:
    """Creates all GT-patterns for an irrep.
    ~Eqs. (20)-(21)
    """

    gt_patterns = []

    # find_next_rows recursively appends valid GT-patterns to gt_patterns.
    # It starts with the i-weight of the irrep. It then uses the
    # betweenness condition to find valid next rows in the GT-patterns.
    # Once the last row is found, the GT-pattern is appended.

    def find_next_rows(row, prev_rows=[]):
        if len(row)==1:
            gt_patterns.append(prev_rows + [row])
        else:
            for next_row in product(*[range(row[j], row[j+1]-1, -1) for j in range(len(row)-1)]):
                find_next_rows(list(next_row), prev_rows + [row])
    
    find_next_rows(list(iweight))

    # GT-patterns are sorted based on their p-weight (lexicographically)
    # such that the highest-weight state is the first in the list.
    # For states with equal p-weight, their GT-patterns are sorted lexicographically.
    
    gt_patterns.sort(key=lambda pat: calc_weight(pat, 'p'), reverse=True)

    return gt_patterns

def ladder_op(gt_patterns: list[list[list]], k: int, kind: str) -> list:
    """Applies J(k)+- to a direct-product basis state.
    gt_patterns is assumed to be a list of GT-patterns, even if
    the basis state is that of a single irrep. The allowed values
    of k are 1, ..., N-1. kind is '+' for raising operators or
    '-' for lowering operators. Returns a list of [coefficient,
    new direct-product basis state].
    ~Eqs. (28)-(29)
    """

    # J(k)+- acts on the kth row of the GT-pattern.
    # On a direct-product basis state, A x B x C,
    # J(k)+- acts as (J@A) x B x C + A x (J@B) x C + A x B x (J@C).

    linear_combination = []
    for i in range(len(gt_patterns)):
        pattern = gt_patterns[i]
        N = len(pattern[0])

        # row is the kth row of the GT-pattern.
        # prow (previous row) is the (k-1)th row of the GT-pattern.
        # nrow (next row) is the (k+1)th row of the GT-pattern.
        # If k=N-1, then nrow does not exist, so make nrow the empty list.

        row,prow = pattern[k],pattern[k-1]
        nrow = pattern[k+1] if k<N-1 else []
        new_rows,coeffs = [],[]

        for j in range(len(row)):
            if kind=='+' and prow[j] >= row[j]+1 >= prow[j+1]:
                coeff = np.prod([prow[jp] - row[j] + j - jp for jp in range(len(prow))])
                coeff *= np.prod([nrow[jp] - row[j] + j - jp - 1 for jp in range(len(nrow))])
                coeff /= np.prod([(row[jp] - row[j] + j - jp)*(row[jp] - row[j] + j - jp - 1) for jp in range(len(row)) if jp != j])
                if coeff == 0:
                    continue
                else:
                    new_rows.append([row[jp]+1 if jp==j else row[jp] for jp in range(len(row))])
                    coeffs.append(np.sqrt(-coeff))
            elif kind=='-' and prow[j] >= row[j]-1 >= prow[j+1]:
                coeff = np.prod([prow[jp] - row[j] + j - jp + 1 for jp in range(len(prow))])
                coeff *= np.prod([nrow[jp] - row[j] + j - jp for jp in range(len(nrow))])
                coeff /= np.prod([(row[jp] - row[j] + j - jp + 1)*(row[jp] - row[j] + j - jp) for jp in range(len(row)) if jp != j])
                if coeff == 0:
                    continue
                else:
                    new_rows.append([row[jp]-1 if jp==j else row[jp] for jp in range(len(row))])
                    coeffs.append(np.sqrt(-coeff))
        
        for row,coeff in zip(new_rows,coeffs):
            new_pat = pattern[0:k] + [row] + pattern[k+1:]
            patterns = gt_patterns[0:i] + [new_pat] + gt_patterns[i+1:]
            linear_combination.append([coeff,patterns])

    return linear_combination

def find_suN_basis(iweight: tuple) -> list[csr_array]:
    """Returns a basis for an irrep of su(N).
    The basis matrices are returned as orthogonal sparse arrays,
    normalized to 0.5 in the fundamental representation.
    ~Eqs. (16)-(18)
    """

    # A matrix, T, has elements Tij. The indices correspond to
    # basis states of the irrep. i=0 corresponds to the highest-weight
    # state, and i=N-1 corresponds to the lowest-weight state.
    # The overall order of the basis states comes from
    # the output of find_gt_patterns.

    N = len(iweight)
    D = calc_dimension(iweight)
    irrep_basis = find_gt_patterns(iweight)

    # Find N-1 diagonal basis matrices. Their elements are given by
    # the z-weights of each basis state, which are eigenvalues
    # of the N-1 J(k)z operators.

    Jz_list = []
    for k in range(N-1):
        rows,cols,vals = [],[],[]
        for idx in range(D):
            state = irrep_basis[idx]
            num = calc_weight(state, 'z')[k]
            rows.append(idx), cols.append(idx), vals.append(num)
        Jz_list.append(csr_array((vals, (rows,cols))))

    # Find N(N-1) off-diagonal basis matrices. N-1 of them are sums of the
    # J(k)+- operators, whose elements <out| J(k)+- |in> can be
    # read off the output of ladder_op. Because su(N) matrices are Hermitian,
    # only the J(k)+ elements are necessary.

    Jp_list = []
    for k in range(N-1,0,-1):
        rows,cols,vals = [],[],[]
        for state in irrep_basis:
            col_idx = irrep_basis.index(state)
            for num, new_state_in_list in ladder_op([state],k,'+'):
                row_idx = irrep_basis.index(new_state_in_list[0])
                rows.append(row_idx)
                cols.append(col_idx)
                vals.append(num)
        Jp_list.append(csr_array((vals, (rows,cols)), shape=(D,D)))

    basis = []
    in_prod = lambda A,B: (A@B).trace()
    com = lambda A,B: A@B - B@A

    # To orthogonalize the N-1 Jz matrices, use Gram-Schmidt.
    # The normalization of the Jz matrices is kept, however.

    if D==1:
        basis += Jz_list
    else:
        for Jz in Jz_list:
            M = Jz - sum(in_prod(Jz,V)/in_prod(V,V)*V for V in basis)
            M *= np.sqrt(in_prod(Jz,Jz)/in_prod(M,M))
            basis.append(M)
    
    # Each J(k)+ matrix gives two basis matrices, coming from
    # [J(k)+ + J(k)-]/2 and -i[J(k)+ - J(k)-]/2. Note that
    # J(k)- = [J(k)+]^T because the output of ladder_op is real.
    # In total, this gives 2(N-1) more basis matrices.

    for Jp in Jp_list:
        basis.append(0.5*(Jp + Jp.T))
        basis.append(-0.5j*(Jp - Jp.T))

    # Unique nested commutations of the J(k)+ matrices lead to
    # the rest of the basis matrices (for N>2). k1 and k2 index
    # two J(k)+ matrices. V is the nested commutator
    # [J(k1)+, [J(k1+1)+, [J(k1+2)+, ..., [J(k2-1)+, J(k2)+]]]]
    # Finally, two basis matrices are derived from V, giving
    # the remaining (N-2)(N-1) basis matrices.

    for k1,k2 in combinations(range(N-1),2):
        V = reduce(com, [Jp_list[k] for k in range(k1,k2+1)])
        basis.append(0.5*(V + V.T))
        basis.append(-0.5j*(V - V.T))

    return basis

def find_direct_sum(product_iweights: list[tuple], sum_iweight: tuple=None) -> dict[tuple, int]:
    """Decomposes a direct product of irreps (product_iweights) into a
    direct sum of irreps. Returns a dictionary whose keys are the irreps
    appearing in the direct sum, and whose values are the multiplicities of
    the irreps. If sum_iweight is given, then the multiplicity (possibly 0)
    of that irrep is returned.
    ~Pg. 11
    """

    # For efficiency, sort the irreps from lowest weight to highest weight.
    # decomp_memo records decompositions done throughout the algorithm, so
    # repititions are avoided. gt_memo records GT-patterns of irreps
    # encountered in the algorithm to avoid duplicate calculations.

    iweights = sorted(product_iweights, key=calc_dimension)
    decomp_memo,gt_memo = {},{}

    def decompose_two_irreps(R,Rp):

        if R in gt_memo:
            R_basis = gt_memo[R]
        else:
            R_basis = find_gt_patterns(R)
            gt_memo[R] = R_basis

        N = len(R)
        decomposition = []

        # Because this loop depends on the dimension of the irrep R,
        # it is more efficient for dim(R) <= dim(Rp).
        for M in R_basis:
            t_list = list(Rp)
            for i in range(len(M)):
                for j in range(len(M[i])):
                    b = M[j][i] if j==(N-1)-i else M[j][i]-M[j+1][i]
                    t_list[(N-1)-j] += b
                    if j != N-1 and t_list[(N-1)-j-1] < t_list[(N-1)-j]:
                        break
                    else:
                        continue
                else:
                    continue
                break
            else:

                # The i-weights in the decomposition are normalized so that
                # iweight[-1] = 0. However, this algorithm works even if
                # the iweights R and Rp are not normalized in this manner.
                decomposition.append(tuple(t-t_list[-1] for t in t_list))
        
        return decomposition

    # decompose is a recursive function. The routine takes R x K x Rp to
    # K x (A + B) = K x A + K x B. The recursive step is to then decompose
    # K x A and K x B separately. Direct-sum i-weights are returned as a list.

    def decompose(irreps):
        
        R,Rp = irreps[0],irreps[-1]
        label = tuple(sorted((R,Rp)))
        if label in decomp_memo:
            decomp_from_two = decomp_memo[label]
        else:
            decomp_from_two = decompose_two_irreps(R,Rp)
            decomp_memo[label] = decomp_from_two
        irreps.remove(R), irreps.remove(Rp)

        if len(irreps)==0:
            return decomp_from_two
        else:
            res = []
            for irrep in decomp_from_two:
                new_irreps = sorted([irrep] + irreps, key=calc_dimension)
                ref_label = tuple(sorted(new_irreps))

                if ref_label in decomp_memo:
                    decomp = decomp_memo[ref_label]
                else:
                    decomp = decompose(new_irreps)
                    decomp_memo[ref_label] = decomp
                
                res += decomp

            return res
    
    direct_sum = decompose(iweights)

    # count is used if sum_irrep is given. Otherwise,
    # Counter is able to find the multiplicity of each irrep
    # appearing in the direct-sum decomposition. The i-weights
    # are then sorted lexicographically for neatness.

    if sum_iweight is not None:
        return direct_sum.count(sum_iweight)
    else:
        direct_sum = Counter(direct_sum)
        direct_sum = dict(sorted(direct_sum.items(), reverse=True))
        return direct_sum

"""
SYMMETRIC GROUP
"""

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

def calc_Sn_dimension(partition: list) -> int:
    """Returns dimension of a symmetric group irrep,
    given by an integer partition of n via the hook length formula.
    """

    # Gather initial data.
    n = sum(partition)
    num_cols = partition[0]
    num_rows = len(partition)

    # Whereas the partition gives the number of cells per row,
    # the formula additionally needs the number of cells per column,
    # which is contained in the conjugate partition.
    conjugate_partition = []
    for col in range(1, num_cols+1):
        num_col_cells = 0
        for num_row_cells in partition:
            if num_row_cells >= col:
                num_col_cells += 1
            else:
                continue
        conjugate_partition.append(num_col_cells)

    # This nested for-loop finds the hook length for each cell
    # and multiplies them all.
    hook_length_prod = 1
    for i in range(num_rows):
        for j in range(partition[i]):
            hook_length = partition[i] + conjugate_partition[j] - i - j - 1
            hook_length_prod *= hook_length

    # Compute the hook length formula and round to ensure integral dimension.
    dim = factorial(n)/hook_length_prod
    return round(dim)

def find_tableaux(partition: list) -> list:
    """Generates standard Young tableaux given a partition.
    Returns a list of the tableaux.
    """

    # This code is largely adapted from the PermutationGroup.m file
    # of GroupMath, https://renatofonseca.net/groupmath

    # Gather initial data.
    n = sum(partition)
    num_cols = partition[0]
    num_rows = len(partition)
    zeros = [0 for i in range(n)]

    # Compute conjugate partition.
    conjugate_partition = []
    for col in range(1, num_cols+1):
        num_col_cells = 0
        for num_row_cells in partition:
            if num_row_cells >= col:
                num_col_cells += 1
            else:
                continue
        conjugate_partition.append(num_col_cells)

    # Compute canonical Young tableau.
    canonical,count = [],0
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

def young_symmetrizer(tableaux: list[list[list]], idx_list: list[list]) -> Generator[tuple[float, tuple]]:
    """Generates the Young symmetrizer corresponding to a list of standard
    Young tableaux. This is a generator function; it yields the symmetrizer
    as a sum of (coefficient, permutation), and an example of its use
    is as follows.

    Suppose A=(a,b,c,d,e) is to be symmetrized on indices [0,3] according to
    the tableau [[0,1]] and symmetrized on indices [1,2,4] according to the
    tableau [[0,2],[1]]. Then tableaux := [[[0,1]], [[0,2],[1]]] and
    idx_list := [[0,3], [1,2,4]]. The permutations are returned as tuples
    such as P=(3,1,2,0,4) so that A[i] -> A[P[i]] for i=0,...,4 -- or in
    this case, A -> (d,b,c,a,e).

    Each tableau in tableaux has an associated list of indices in idx_list.
    Both arguments are assumed to be a list of lists, even if there is only
    one tableau. The symmetrizer is a projector and is properly normalized.
    """

    # The algorithm presented in https://arxiv.org/pdf/1307.6147
    # is implemented and referenced throughout this code.

    # Finds the permutations that preserve the rows and columns of a tableau
    # as well as whether the tableau is row and/or column ordered. With extra_data,
    # inverses of the permutations and a product of hook lengths are also returned.
    def tableau_data(tableau, extra_data=False):

        # Gather initial data. Out of these variables, n is returned.
        # tableau_T is the transpose of the tableau.

        partition, conjugate_partition = [len(row) for row in tableau], []
        n, num_cols, num_rows = sum(partition), partition[0], len(partition)
        for col in range(1, num_cols+1):
            num_col_cells = 0
            for num_row_cells in partition:
                if num_row_cells >= col:
                    num_col_cells += 1
                else:
                    continue
            conjugate_partition.append(num_col_cells)
        tableau_T = [[tableau[j][i] for j in range(conjugate_partition[i])] for i in range(num_cols)]

        # Extra data includes the product of hook lengths of the tableau (necessary
        # for normalization) and the inverse row and column permutations.

        if extra_data:
            hook_length_prod = 1
            for i in range(num_rows):
                for j in range(partition[i]):
                    hook_length = partition[i] + conjugate_partition[j] - i - j - 1
                    hook_length_prod *= hook_length
            inverse_row_permutations, inverse_col_permutations = [],[]
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
        # to the usual tuple version of a permutation. An inverse permutation
        # is found by using the inverse mapping.

        row_permutations = []
        sorted_rows = [sorted(row) for row in tableau if len(row)>1]
        for X in product(*(permutations(row) for row in tableau if len(row)>1)):
            mapping = {sorted_rows[i][j]: X[i][j] for i in range(len(sorted_rows)) for j in range(partition[i]) if X[i][j] != sorted_rows[i][j]}
            perm = tuple(mapping[i] if i in mapping else i for i in range(n))
            row_permutations.append(perm)
            if extra_data:
                inverse_mapping = {v:k for k,v in mapping.items()}
                inverse_row_permutations.append(tuple(inverse_mapping[i] if i in inverse_mapping else i for i in range(n)))
            else:
                continue

        col_permutations = []
        sorted_cols = [sorted(col) for col in tableau_T if len(col)>1]
        for X in product(*(permutations(col) for col in tableau_T if len(col)>1)):
            mapping = {sorted_cols[i][j]: X[i][j] for i in range(len(sorted_cols)) for j in range(conjugate_partition[i]) if X[i][j] != sorted_cols[i][j]}
            perm = tuple(mapping[i] if i in mapping else i for i in range(n))
            col_permutations.append(perm)
            if extra_data:
                inverse_mapping = {v:k for k,v in mapping.items()}
                inverse_col_permutations.append(tuple(inverse_mapping[i] if i in inverse_mapping else i for i in range(n)))
            else:
                continue

        if extra_data:
            return n, row_permutations, col_permutations, is_row_ordered, is_col_ordered, inverse_row_permutations, inverse_col_permutations, hook_length_prod
        else:
            return n, row_permutations, col_permutations, is_row_ordered, is_col_ordered

    # Calculates the sign of a permutation.
    def sgn(permutation):
        n = len(permutation)
        num_inversions = 0
        for i in range(n):
            for j in range(i+1, n):
                if permutation[i] > permutation[j]:
                    num_inversions += 1
                else:
                    continue
        if num_inversions % 2 == 0:
            return 1
        else:
            return -1

    # Computes the equivalent permutation got by first applying p1 and then p2.
    # p1 and p2 need not be of equal lengths; n ensures that the final permutation
    # is of length n. Nevertheless, p1 and p2 should each contain integers 0,1,...,x.
    def compose_two(p1, p2, n):
        if len(p1) == len(p2):
            perm = tuple(p1[p2[i]] for i in range(len(p1)))
        elif len(p1) > len(p2):
            perm = tuple(p1[p2[i]] if i in p2 else p1[i] for i in range(len(p1)))
        else:
            perm = tuple(p1[p2[i]] if p2[i] in p1 else p2[i] for i in range(len(p2)))
        if len(perm) == n:
            return perm
        else:
            return perm + tuple(range(len(perm), n))
        
    # Simplifies a product of linear combinations of permutations into a single
    # linear combination of distinct permutations. perms_lists is a list of
    # row- or column-preserving permutations. anti_idxs is a list of indices
    # of lists in perms_lists that contain column-preserving permutations;
    # these permutations act with a factor of their sign. As an expression,
    # (1 + p1 + p2 + ...) x (1 + p1 + p2 + ...) x ... is the input and the output
    # is a0*1 + a1*p1 + a2*p2 + ... returned as dictionary whose keys are the
    # distinct permutations pk and whose values are their coefficients ak.
    def reduce_perms(perms_lists, anti_idxs, n):
        # perms_lists is given such that the first list of permutations actually
        # acts first. ordered reverses perms_lists to correct this. res is the
        # final output; because the symmetrizer is so far unnormalized, its
        # permutations' coefficients will be integers.
        ordered = perms_lists[::-1]  
        res = defaultdict(int)
        for i in range(1,len(perms_lists)):
            if i==1:
                for p1,p2 in product(ordered[0], ordered[i]):
                    new = compose_two(p1,p2,n)
                    if 0 in anti_idxs:
                        res[new] += sgn(p1)
                    elif i in anti_idxs:
                        res[new] += sgn(p2)
                    else:
                        res[new] += 1
                temp = res.copy()
            else:
                for p1,p2 in product(temp, ordered[i]):
                    new = compose_two(p1,p2,n)
                    if new==p1:
                        continue
                    elif i in anti_idxs:
                        new_val = temp[new] + res[p1]*sgn(p2)
                    else:
                        new_val = temp[new] + res[p1]
                    if new_val == 0:
                        del temp[new]
                    else:
                        temp[new] = new_val
                res = temp
                temp = res.copy()
        return dict(res)

    symmetrizers = []
    norm = 1

    for tableau in tableaux:

        # Gather data for the input tableau.
        # Only the inverse permutations for the input tableau are necessary.
        # The usual Young symmetrizers are built with the convention where
        # column permutations are applied first, followed by row permutations.
        # ~Eq. (26)
        n, row_perms, col_perms, is_row_ordered, is_col_ordered, inv_row_perms, inv_col_perms, hook_length_prod = tableau_data(tableau, extra_data=True)

        # If the tableau is already row-ordered or column-ordered,
        # then the Young symmetrizer can be immediately built.
        # perms_lists is a tuple of permutations. anti_idxs gives which indices/permutations
        # in perms_lists are part of antisymmetrizers, and therefore come with a
        # factor of sgn(permutation). Inverse permutations are included as
        # necessary. Note that permutation^dagger := permutation^(-1)
        # and the sign of a permutation equals the sign of its inverse.
        # ~Theorem 4

        if is_row_ordered:
            anti_idxs = [1,2]
            perms_lists = [row_perms, col_perms, inv_col_perms, inv_row_perms]
        elif is_col_ordered:
            anti_idxs = [0,3]
            perms_lists = [inv_col_perms, inv_row_perms, row_perms, col_perms]

        else:

            # Determine the MOLD (M) of the tableau by applying the parent
            # map on the tableau. For each parent tableau, gather its
            # tableau_data for later use.
            # ~Definitions 1 and 3

            M, ancestor_perms = 0, []
            child_n, child_tab = n, tableau
            while not (is_row_ordered or is_col_ordered):
                parent_tableau = []
                for i in range(len(child_tab)):
                    row = []
                    for j in range(len(child_tab[i])):
                        if child_tab[i][j] != child_n-1:
                            row.append(child_tab[i][j])
                        else:
                            continue
                    if len(row) == 0:
                        continue
                    else:
                        parent_tableau.append(row)
                child_tab = parent_tableau
                child_n, parent_row_perms, parent_col_perms, is_row_ordered, is_col_ordered = tableau_data(parent_tableau)
                ancestor_perms.append([parent_row_perms, parent_col_perms])
                M += 1

            # The Young symmetrizer depends on the parity of M and whether
            # the Mth ancestor tableau is row-ordered or column-ordered.
            # Each configuration is slightly different. ancestor_..._perms
            # describes the order in which the ancestor row or column
            # permutations appear in a Young symmetrizer.
            # ~Theorem 5

            num = 2*M + 4
            if is_row_ordered:
                if M%2==0:
                    ancestor_col_row_perms = [ancestor_perms[i][1] if i%2==0 else ancestor_perms[i][0] for i in range(M)]
                    perms_lists = ancestor_col_row_perms[::-1] + [row_perms, col_perms, inv_col_perms, inv_row_perms] + ancestor_col_row_perms
                    anti_idxs = [i for i in range(num) if (i%2==1 and i<num//2) or (i%2==0 and i>=num//2)]
                else:
                    ancestor_row_col_row_perms = [ancestor_perms[i][0] if i%2==0 else ancestor_perms[i][1] for i in range(M)]
                    perms_lists = ancestor_row_col_row_perms[::-1] + [inv_col_perms, inv_row_perms, row_perms, col_perms] + ancestor_row_col_row_perms
                    anti_idxs = [i for i in range(num) if (i%2==1 and i<num//2) or (i%2==0 and i>num//2)]
            elif is_col_ordered:
                if M%2==0:
                    ancestor_row_col_perms = [ancestor_perms[i][0] if i%2==0 else ancestor_perms[i][1] for i in range(M)]
                    perms_lists = ancestor_row_col_perms[::-1] + [inv_col_perms, inv_row_perms, row_perms, col_perms] + ancestor_row_col_perms
                    anti_idxs = [i for i in range(num) if (i%2==0 and i<num//2) or (i%2==1 and i>num//2)]
                else:
                    ancestor_col_row_col_perms = [ancestor_perms[i][1] if i%2==0 else ancestor_perms[i][0] for i in range(M)]
                    perms_lists = ancestor_col_row_col_perms[::-1] + [row_perms, col_perms, inv_col_perms, inv_row_perms] + ancestor_col_row_col_perms
                    anti_idxs = [i for i in range(num) if (i%2==0 and i<num//2) or (i%2==1 and i>=num//2)]

        # perms_lists is simplified with reduce_perms. combined_perms is the
        # (unnormalized) Young symmetrizer for this tableau and portion of idx_list.
        # The normalization factor norm can be found by taking the coefficient of
        # identity permutation and the product of hook lengths for tableau.

        combined_perms = reduce_perms(perms_lists, anti_idxs, n)
        id_coeff = combined_perms[tuple(i for i in range(n))]
        norm *= 1/(id_coeff*hook_length_prod)
        symmetrizers.append(combined_perms)

    # Each Young symmetrizer in symmetrizers is made of permutations on 0,...,x.
    # These integers need to be translated to appropriate indices in idx_list.
    # Taking advantage of symmetrizers containing disjoint Young symmetrizers,
    # this for-loop generates the complete, normalized Young symmetrizer by
    # expanding (a0*1 + a1*p1 + ...) x (b0*1 + b1*q1 + ...) x ...

    num_idxs = sum(len(x) for x in idx_list)
    num_idx_lists = len(idx_list)
    for X in product(*symmetrizers):
        temp = [0 for i in range(num_idxs)]
        coeff = 1
        for i in range(num_idx_lists):
            coeff *= symmetrizers[i][X[i]]
            for j in range(len(idx_list[i])):
                temp[idx_list[i][j]] = idx_list[i][X[i][j]]
        prefactor = norm*coeff
        perm = tuple(temp)
        yield prefactor, perm

def find_plethysms(iweight: tuple, n: int) -> dict[tuple, dict[tuple, int]]:
    """Decomposes a direct product of n factors of an irrep (iweight)
    into a direct sum of irreps. The decomposition is returned as a dictionary
    whose keys are the direct-sum irreps, and whose values are dictionaries
    giving the symmetric group irrep (given as a partition of n) those
    irreps transform under, with multiplicity.
    """

    # This code is largely adapted from the PermutationGroup.m file
    # of GroupMath, https://renatofonseca.net/groupmath

    # Gather initial data.
    N = len(iweight)
    fundamental_rep = tuple(1 if i==0 else 0 for i in range(N))
    longest_Weyl_word = [j for i in range(N-2,-1,-1) for j in range(i+1)]

    # The dominant weights are z-weights whose entries are all nonnegative.
    # dominant_zweights records these weights as well as their multiplicities.
    dominant_zweights = defaultdict(int)
    for pat in find_gt_patterns(iweight):
        zweight = calc_weight(pat, 'z')
        if all(x>=0 for x in zweight):
            dominant_zweights[tuple(zweight)] += 1
        else:
            continue

    # Simple roots can be extracted from the z-weights of the
    # fundamental representation.
    simple_roots,fund_basis = [],find_gt_patterns(fundamental_rep)
    for i in range(N-1):
        wi = np.array(calc_weight(fund_basis[i], 'z'))
        wi_1 = np.array(calc_weight(fund_basis[i+1], 'z'))
        simple_roots.append(2*(wi - wi_1))

    # Calculates the order of the conjugacy
    # class of the Sn irrep given by partition.
    def class_order(partition):
        prod,cycle_tally = 1,Counter(partition)
        for cycle_len, num in cycle_tally.items():
            prod *= cycle_len**num * factorial(num)
        return round(factorial(n)/prod)
    
    # Finds the rim hooks of a given length l
    # for a given partition.
    def rim_hooks(partition, l):

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

    # Computes the character of an Sn_partition
    # in the irrep given by partition recursively.
    def class_character(partition, Sn_partition):
        if len(partition)==0:
            return 1
        else:
            new_irreps = rim_hooks(partition, Sn_partition[0])
            new_Sn_irrep = Sn_partition[1:]
            character = sum((-1)**l * class_character(p,new_Sn_irrep) for p,l in new_irreps)
            return character

    # Finds the i-weight corresponding to a dominant weight,
    # assumed to be the highest weight of an irrep.
    def find_iweight(zweight):
        pweight,count = [0],-1
        for i in range(len(zweight)-1,-1,-1):
            if i==len(zweight)-1:
                pweight.append(int(2*zweight[i]))
            else:
                pweight.append(int(2*zweight[i] + pweight[count+1]))
            count += 1
        return tuple(pweight[::-1])

    # Finds Weyl orbit of a given z-weight.
    def Weyl_orbit(zweight):
        orbit,weights = [np.array(zweight)],[np.array(zweight)]
        while len(weights) != 0:
            trial_weights,new_weights = [],[]
            for w in weights:
                if not np.any(w):
                    trial_weights.append(None)
                else:
                    reflections = []
                    for i in range(N-1):
                        reflections.append(w - w[i]*simple_roots[i])
                    trial_weights.append(reflections)
            for i in range(N-1):
                for j in range(len(weights)):
                    if weights[j][i] <= 0 or trial_weights[j] is None:
                        continue
                    else:
                        check = trial_weights[j][i][i+1:N-1]
                        if np.all(check==abs(check)):
                            new_weights.append(trial_weights[j][i])
                        else:
                            continue
            weights = new_weights
            if len(weights) != 0:
                orbit += weights
            else:
                continue
        return orbit

    # Applies the kth Adams operator to the irrep (iweight).
    def Adams(k):

        def v_decomp(dweight):
            poly = [[w,1] for w in Weyl_orbit(dweight)]
            for i in range(len(longest_Weyl_word)):
                for j in range(len(poly)):
                    if poly[j][1] != 0:
                        letter = longest_Weyl_word[i]
                        if poly[j][0][letter] >= 0:
                            continue
                        elif poly[j][0][letter] == -0.5:
                            poly[j][1] = 0
                        elif poly[j][0][letter] <= -1:
                            factor = poly[j][0][letter] + 0.5
                            poly[j][1] *= -1
                            poly[j][0] = poly[j][0] - factor*simple_roots[letter]
                        else:
                            continue
                    else:
                        continue
            for monomial,coeff in poly:
                if coeff == 0:
                    continue
                else:
                    yield find_iweight(monomial.tolist()), coeff

        polynomial = defaultdict(int)
        for dweight,mult in dominant_zweights.items():
            scaled_dweight = [k*x for x in dweight]
            for R,coeff in v_decomp(scaled_dweight):
                polynomial[R] += coeff*mult
        return {R:num for R,num in polynomial.items() if num != 0}

    # A plethysm has a polynomial where each term is an SU(N) irrep with
    # a multiplicity coefficient. The formula for the polynomial can be found
    # in page 72 of http://wwwmathlabo.univ-poitiers.fr/~maavl/pdf/LiE-manual.pdf
    # The algorithm is iterative and many computations are repeated; to
    # mitigate this, Adams_dict stores all possible evaluations of Adams.
    # decomp_dict stores all direct-sum decompositions. part_dict effectively
    # evaluates the formula once, storing each term from it, up to a coefficient.

    Adams_dict = {k: Adams(k) for k in range(1,n+1)}
    part_dict,decomp_dict = {},{}

    # The formula may produce a direct product of (direct-sum) polynomials
    # of the form (R1 + R2 + R3 + ...) x (S1 + S2 + S3 + ...) x ...
    # with integer coefficients on each irrep. The direct products are
    # decomposed with find_direct_sum and all direct-sum irreps are combined
    # at the end. Irreps with coefficient zero are discarded.

    for P in find_partitions(n):

        # poly_dict represents the final polynomial as a dictionary
        # whose keys are irreps and whose values are their coefficients.
        # polynomial_factors is a list of the (direct-sum) polynomials,
        # which are also lists of [irrep, prefactor].
        polynomial_factors = [Adams_dict[k] for k in P]

        if len(P) == 1:
             part_dict[tuple(P)] = polynomial_factors[0]
        else:
            poly_dict = defaultdict(int)

            for i in range(1,len(P)):
                if i == 1:
                    for R,Rp in product(polynomial_factors[0], polynomial_factors[i]):
                        label = tuple(sorted([R,Rp]))
                        num = polynomial_factors[0][R]*polynomial_factors[i][Rp]
                        if label in decomp_dict:
                            decomp = decomp_dict[label]
                        else:
                            decomp = find_direct_sum([R,Rp])
                            decomp_dict[label] = decomp
                        for S,mult in decomp.items():
                            poly_dict[S] += num*mult
                    poly_dict = {R:num for R,num in poly_dict.items() if num != 0}
                else:
                    temp = defaultdict(int)
                    for R,Rp in product(poly_dict, polynomial_factors[i]):
                        label = tuple(sorted([R,Rp]))
                        num = poly_dict[R]*polynomial_factors[i][Rp]
                        if label in decomp_dict:
                            decomp = decomp_dict[label]
                        else:
                            decomp = find_direct_sum([R,Rp])
                            decomp_dict[label] = decomp
                        for S,mult in decomp.items():
                            temp[S] += num*mult
                    poly_dict = {R:num for R,num in temp.items() if num != 0}

            part_dict[tuple(P)] = {R: num for R,num in poly_dict.items() if num != 0}

    # plethysms holds the plethysm formula computed for each Sn irrep (partition).
    # The formula contains a coefficient that is generally a float, unlike
    # all previous coefficients. round is used to detect zeros and
    # ensure all multiplicities are integer-valued.

    plethysms = defaultdict(dict)

    for partition in find_partitions(n):
        plethysm = defaultdict(float)
        for P in find_partitions(n):
            coeff = class_order(P)*class_character(partition,P)/factorial(n)
            if coeff == 0:
                continue
            else:
                for R,num in part_dict[tuple(P)].items():
                    plethysm[R] += coeff*num
        for R,num in plethysm.items():
            mult = round(num)
            if mult == 0:
                continue
            else:
                plethysms[R][tuple(partition)] = mult
    
    # Sort SU(N) irreps lexicographically.
    plethysms = {R: plethysms[R] for R in sorted(plethysms.keys(), reverse=True)}
    return plethysms

def find_symmetry_direct_sum(product_iweights: list[tuple], sum_iweight: tuple=None) -> tuple[dict, list]:
    """Decomposes a direct product of irreps (product_iweights) into a
    direct sum of irreps and provides the symmetry group irreps they
    transform under. Returns a dictionary of the direct sum and a list
    giving the lists of indices that transform under symmetric group irreps.
    If sum_iweight is given, then the (possibly empty) dictionary of
    the symmetry group irreps for only that sum_iweight is returned.
    """

    # Find all plethysms for each repeated irrep in product_iweights.
    # keys is nearly list(set(product_iweights)), except the order of
    # the irreps in keys is that of the irreps in plethysms. indices
    # is a list of lists of indices of each irrep in keys as it appears
    # in product_iweights.

    plethysms = {R: find_plethysms(R,num) for R,num in Counter(product_iweights).items()}
    keys = list(plethysms.keys())
    indices = [list(locate(product_iweights, lambda x: x==R)) for R in keys]

    # direct_sum initializes the dictionary for the final result.
    # The plethysm polynomials are multiplied together, leading to
    # further direct-sum decompositions. The individual terms of these
    # decompositions are added together into direct_sum.

    if sum_iweight is None:
        direct_sum = defaultdict(lambda: defaultdict(int))
    else:
        symmetric_group_irreps = defaultdict(int)

    # irreps is a tuple of irreps, representing a term of the distributed
    # direct product of polynomials. For each irrep in irreps, there are
    # potentially many symmetric group irreps attached to it from the
    # plethysms. Each combination of these irreps must therefore be compiled.
    # Sn_irreps gives the direct-sum irreps' Sn irreps with their own
    # multiplicities.

    if sum_iweight is None:
        for irreps in product(*(plethysms[R] for R in plethysms)):
            if len(irreps)==1:
                decomp = {irreps[0]: 1}
            else:
                decomp = find_direct_sum(list(irreps))
            for Sn_irreps in product(*(plethysms[keys[i]][irreps[i]].items() for i in range(len(keys)))):
                partitions,mult = (),1
                for p,m in Sn_irreps:
                    partitions += (p,)
                    mult *= m
                for sum_irrep in decomp:
                    direct_sum[sum_irrep][partitions] += decomp[sum_irrep]*mult
    else:
        for irreps in product(*(plethysms[R] for R in plethysms)):
            if len(irreps)==1:
                if irreps[0]==sum_iweight:
                    multiplicity = 1
                else:
                    multiplicity = 0
            else:
                multiplicity = find_direct_sum(list(irreps), sum_iweight)
            if multiplicity == 0:
                continue
            else:
                for Sn_irreps in product(*(plethysms[keys[i]][irreps[i]].items() for i in range(len(keys)))):
                    partitions,mult = (),1
                    for p,m in Sn_irreps:
                        partitions += (p,)
                        mult *= m
                    symmetric_group_irreps[partitions] += multiplicity*mult
        return dict(symmetric_group_irreps), indices

    # Sort the direct-sum irreps lexicographically.    
    direct_sum = {R: dict(direct_sum[R]) for R in sorted(direct_sum.keys(), reverse=True)}
    return direct_sum, indices

"""
CLEBSCH-GORDAN COEFFICIENTS
"""

def calc_highest_weight_cgcs(product_iweights: list[tuple], sum_iweight: tuple, multiplicity: int) -> dict[int, dict[tuple, float]]:
    """Calculates the Clebsch-Gordan Coefficients for the highest-weight state
    of an irrep (sum_iweight) appearing in the direct-sum decomposition of a
    direct product of irreps (product_iweights). A multiplicity number
    of orthonormal CGC vectors are produced. The CGCs transform under
    irreps of the symmetric group, as given by find_symmetry_direct_sum.
    Returns a dictionary whose keys are the multiplicity indices of
    sum_iweight, and whose values are dictionaries of the form
    {product basis state: CGC}.
    ~Eqs. (33)-(34) and Pg. 13
    """

    # Return CGCs if already computed.

    highest_weight_cgc_data_path = PurePath(data_directory, str(product_iweights), 'highest_weight_CGC_' + str(sum_iweight))
    if Path(highest_weight_cgc_data_path).exists():
        with open(highest_weight_cgc_data_path, 'rb') as fp:
            return load(fp)
    else:
        pass

    # Gather initial data. The GT-pattern for the highest-weight state
    # of sum_iweight can be manually made. gt_patterns is a dictionary
    # of GT-patterns for basis states of irreps in product_iweights.

    N = len(sum_iweight)
    highest_weight_state = [[sum_iweight[i] for i in range(j)] for j in range(N,0,-1)]
    highest_weight = calc_weight(highest_weight_state, 'z')
    gt_patterns = {R: find_gt_patterns(R) for R in set(product_iweights)}

    # Sn_irreps gives the symmetric group irreps the CGCs of sum_iweight
    # should transform under. idx_list are the indices each irrep acts on.
    # tableaux_cache stores necessary standard Young tableaux.

    Sn_irreps, idx_list = find_symmetry_direct_sum(product_iweights, sum_iweight)
    Sn_irrep_set = set(partition for partitions in Sn_irreps for partition in partitions)
    tableaux_cache = {partition: find_tableaux(list(partition)) for partition in Sn_irrep_set}

    # The product basis is an exponentially large set. However, in order
    # for a basis state to have a nonzero CGC with the highest-weight state,
    # that basis state must match the J(k)z eigenvalues of the highest-weight state.
    # The 'selected basis' is the subset of product basis states whose total
    # z-weights match the z-weight (highest_weight) of the highest-weight state.

    # product_zweights is a nested dictionary that ultimately gives the
    # GT-patterns that have a J(k)z eigenvalue at an index of their z-weight.
    # The format is {irrep: {z-weight index: {J(k)z eigenvalue: [GT-patterns (indices)]}}}.

    product_zweights = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for R in set(product_iweights):
        for gt_idx in range(len(gt_patterns[R])):
            zweight = calc_weight(gt_patterns[R][gt_idx], 'z')
            for i in range(N-1):
                product_zweights[R][i][zweight[i]].append(gt_idx)

    # bounds are maximum J(k)z eigenvalues for the direct-product irreps.
    # partition_zval is a recursive function called as an iterator.
    # It partitions one of the J(k)z eigenvalues (a half-integer, zval)
    # into a tuple of half-integers, such that the sum of the tuple equals zval.
    # The entries of the tuple are bounded in magnitude by bounds.
    # idx tracks whether the tuple's length matches that of bounds.
    # The recursive step creates the next element of the tuple by subtracting
    # a half-integer within the bounds of the current element irrep from the input.
    # When idx=len(bounds)-1, the input may be the last element of the tuple
    # if it is within bounds; otherwise, the function terminates that tuple.

    bounds = [0.5*R[0] for R in product_iweights]
    def partition_zval(zval, idx=0):
        if idx==len(bounds)-1:
            if abs(zval) <= bounds[idx]:
                yield (zval,)
            return
        for i in np.arange(-bounds[idx], bounds[idx]+0.5, 0.5):
            for p in partition_zval(zval-i, idx+1):
                yield (i,) + p

    # The selected basis is now found by partitioning the J(1)z eigenvalue
    # of the highest-weight state. Using product_zweights, product basis states
    # are grouped into candidates, whose J(1)z eigenvalue decomposition matches
    # that of the partition p. The candidates are looped over, checking if their
    # total J(k>1)z eigenvalues match those of the highest-weight state.
    # In addition, selected basis states are grouped by permutations per
    # indices in idx_list. Each group is recorded in perm_dict, whose key
    # is the lexicographic version of each state, and whose value is a
    # dictionary of the form {selected basis state index: selected basis state}.

    selected_basis = []
    perm_dict,state_idx = defaultdict(dict),0
    for p in partition_zval(highest_weight[0]):
        candidates = [product_zweights[product_iweights[i]][0][p[i]] for i in range(len(product_iweights))]
        for state in product(*candidates):
            for k in range(1,N-1):
                # == should be fine for half-integers.
                if sum(calc_weight(gt_patterns[product_iweights[i]][state[i]], 'z')[k] for i in range(len(product_iweights))) == highest_weight[k]:
                    continue
                else:
                    break
            else:
                selected_basis.append(state)
                # This relies on product_iweights being a sorted list of i-weights.
                sorted_state = tuple(x for idxs in idx_list for x in sorted(state[i] for i in idxs))
                perm_dict[sorted_state][state_idx] = state
                state_idx += 1

    # The highest-weight (hw) state can be expanded in the selected basis (sb);
    # the coefficients of the expansion are the CGCs: |hw state> = sum_{sb} C_sb |sb state>.
    # Acting on the highest-weight state with the J(k)+ operators annihilates it.
    # When some selected basis states raise to the same product basis state,
    # their CGCs satisfy an equation of the form j1 C1 + ... + jn Cn = 0.
    
    # row elements index a separate equation.
    # col elements index a CGC.
    # val elements are coefficients of the equation.
    # Act on all selected basis states with J(k)+. Gather the raised states
    # and the corresponding (coefficient, selected basis state that was raised)
    # into raised_dict. Each raised state corresponds to a separate equation,
    # assigned to a row by row_count. The coefficients are values in that row,
    # and the nonzero columns correspond to the selected basis states that
    # were raised, and therefore, their CGCs. This results in an equation of the form
    # [sparse matrix]@[CGC vector] = [zero vector].

    row,col,val=[],[],[]
    row_count = 0
    for k in range(N-1,0,-1):
        raised_dict = defaultdict(list)
        for i in range(len(selected_basis)):
            product_gts = [gt_patterns[product_iweights[j]][selected_basis[i][j]] for j in range(len(product_iweights))]
            for coeff,raised_gts in ladder_op(product_gts, k, '+'):
                raised_state = tuple(gt_patterns[product_iweights[j]].index(raised_gts[j]) for j in range(len(product_iweights)))
                raised_dict[raised_state].append((coeff,i))
        for raised_state in raised_dict:
            for num,col_count in raised_dict[raised_state]:
                row.append(row_count), col.append(col_count), val.append(num)
            row_count += 1

    # Let system=S be the sparse matrix. Then it is true that S^T@S @ [CGC vector] = [zero vector].
    # Therefore, the CGCs are eigenvectors of S^T@S with eigenvalue zero.
    # The eigsh algorithm can be finicky; v0 and ncv were chosen here by trial-and-error.

    if len(val) == 0:
        vecs = np.array([[1.0]])
    else:
        system = csr_array((val, (row,col)))
        v0 = np.ones(len(selected_basis))/np.sqrt(len(selected_basis))
        ncv = len(selected_basis)
        vals,vecs = eigsh(system.T@system, k=multiplicity, which='SM', v0=v0, ncv=ncv)
        
        # This is a check that the eigenvectors found have eigenvalue zero.
        if not all(abs(v)<EPS for v in vals): raise ArithmeticError('Some CGC vectors have nonzero eigenvalues:', [v for v in vals if abs(v)>=EPS])

    # Generally, the highest-weight state may transform under some nontrivial
    # irrep of the symmetric group. In these cases, the highest-weight CGCs
    # (the columns of vecs above) must be symmetrized. This can be done with
    # Young symmetrizers. For a given set of standard Young tableaux, the Young
    # symmetrizer Y will be a linear combination of permutations on the selected
    # basis states (Y = sum_m a_m*p_m). The corresponding matrix is a projector
    # with matrix elements <i|Y|j>, where |i> is a selected basis state. This is
    # also sum_m a_m*<i|p_m|j>. Therefore, the only permutations that matter are
    # those that take |j> to |i>. Using the fact that the projector is symmetric,
    # ij_perms is a dictionary made to identify these permutations.

    ij_perms = defaultdict(list)
    for i in range(len(selected_basis)):
        istate = selected_basis[i]
        ikey = tuple(x for idxs in idx_list for x in sorted(istate[i] for i in idxs))
        
        # ival_idxs and jval_idxs give the states of indices of |i> and |j>
        # per set of indices in idx_list. For example, in the case of RxRxSxS
        # and |i>=(0,0,1,2), ival_idxs could be {0: {0: [0,1]}, 1: {1: [0], 2: [1]}}.
        # The first set of keys refer to indices of idx_list (0 refers to [0,1] indices
        # and 1 refers to [2,3] indices). This splits |i> into 'substates' -- namely,
        # (0,0) and (1,2). The second set of keys are states of irreps. The final
        # lists give indices of the 'substate' that have the state value.

        ival_idxs = {}
        for x in range(len(idx_list)):
            substate = [istate[y] for y in idx_list[x]]
            ival_idxs[x] = {n: list(locate(substate, lambda z: z==n)) for n in set(substate)}

        # Only the upper triangular pairs of indices are necessary. Furthermore,
        # nonzero matrix elements for (i,j) will come from states |j> that are
        # related to |i> by permutation. Hence, the calls to perm_dict.

        for j in perm_dict[ikey]:
            if j < i:
                continue
            else:
                jstate = perm_dict[ikey][j]

                jval_idxs = {}
                for x in range(len(idx_list)):
                    substate = [jstate[y] for y in idx_list[x]]
                    jval_idxs[x] = {n: list(locate(substate, lambda z: z==n)) for n in set(substate)}

                # initial_perm is one of the most basic permutations that take
                # |j> to |i>. All other relevant permutations can be built out of
                # appropriate permutations of initial_perm.

                initial_perm = {}
                for x in range(len(idx_list)):
                    for n in ival_idxs[x]:
                        iidxs,jidxs = ival_idxs[x][n],jval_idxs[x][n]
                        for y in range(len(iidxs)):
                            if jidxs[y] != iidxs[y]:
                                initial_perm[idx_list[x][iidxs[y]]] = idx_list[x][jidxs[y]]
                            else:
                                continue
                initial_perm = tuple(initial_perm[k] if k in initial_perm else k for k in range(len(product_iweights)))
                
                # Remaining permutations can be made by permuting the indices found
                # in ival_idxs in initial_perm.

                for perms in product(*(permutations([idx_list[x][y] for y in ival_idxs[x][n]]) for x in range(len(idx_list)) for n in ival_idxs[x])):
                    temp,pidx = [0 for k in range(len(product_iweights))],0
                    for x in range(len(idx_list)):
                        for n in ival_idxs[x]:
                            for k in range(len(ival_idxs[x][n])):
                                idx = ival_idxs[x][n][k]
                                temp[idx_list[x][idx]] = initial_perm[perms[pidx][k]]
                            pidx += 1
                    ij_perms[(i,j)].append(tuple(temp))
                
    # Irreps of the symmetric group of degree n are given by partitions of n.
    # The dimension of the Sn irrep is the number of standard Young tableaux
    # that are possible to make with that partition. Different Young symmetrizers
    # can be constructed with different Young tableaux. These combinations
    # are accounted for by the first two for-loops. The next for-loop
    # builds the upper-triangular of the projector matrix. These matrices
    # are stored in highest_weight_state_symmetrizers.

    highest_weight_state_symmetrizers = defaultdict(list)
    for partitions in Sn_irreps:
        for tableaux in product(*(tableaux_cache[irrep] for irrep in partitions)):

            # The fact that the matrix is a projector comes from young_symmetrizer
            # being a properly normalized Young symmetrizer.
            Y = {perm: coeff for coeff,perm in young_symmetrizer(tableaux, idx_list)}
            
            row,col,val = [],[],[]
            for i in range(len(selected_basis)):
                ikey = tuple(x for idxs in idx_list for x in sorted(selected_basis[i][k] for k in idxs))
                for j in perm_dict[ikey]:
                    if j < i:
                        continue
                    else:
                        num = sum(Y[p] for p in ij_perms[(i,j)] if p in Y)
                        if abs(num) > EPS:
                            row.append(i), col.append(j)
                            # The other half of the diagonal is added at the end.
                            if i==j:
                                val.append(num/2)
                            else:
                                val.append(num)
                        else:
                            continue
            symmetrizer = csr_array((val, (row,col)), shape=((len(selected_basis), len(selected_basis))), dtype=float)
            
            # Add the lower-triangular and store the matrix.
            symmetrizer += symmetrizer.T
            highest_weight_state_symmetrizers[partitions].append(symmetrizer)
    
    # RREF returns the reduced row echelon form of a matrix. This is only used for
    # outer multiplicities greater than one, within the same symmetric group irrep.
    def RREF(A):
        # i gives index of current row to be put into normal form.
        n_rows, n_cols = A.shape
        i = 0
        # Iterate over columns of A to find pivots (row index with leading term).
        # Pivots are chosen based on maximum value found in column.
        for j in range(n_cols):
            pivot = np.argmax(abs(A[i:n_rows,j])) + i
            max_at_pivot = abs(A[pivot,j])
            # If max_at_pivot~0 then column must be zero vector.
            if max_at_pivot < EPS:
                A[i:n_rows,j] = np.zeros(n_rows-i)
            else:
                # If pivot is not current row then swap pivot and i
                # so that pivot row is ith row.
                if pivot != i:
                    A[[pivot,i], j:n_cols] = A[[i,pivot], j:n_cols]
                else:
                    pass
                # Make sure ith row has a leading 1.
                A[i, j:n_cols] = A[i, j:n_cols]/A[i,j]
                # Subtract multiples of ith reduced row from other rows
                # to make all entries above/below leading 1 zero.
                reduced_row = A[i, j:n_cols]
                if i > 0:
                    row_inds_above = range(i)
                    A[row_inds_above, j:n_cols] = A[row_inds_above, j:n_cols] - np.outer(reduced_row, A[row_inds_above, j]).T
                if i < n_rows-1:
                    row_inds_below = range(i+1,n_rows)
                    A[row_inds_below, j:n_cols] = A[row_inds_below, j:n_cols] - np.outer(reduced_row, A[row_inds_below, j]).T
                else:
                    pass
                # Subsequent iterations will now
                # find pivots in sub-matrix A[i:n_rows, j:n_cols].
                i += 1
            # Conditional for non-square matrices.
            if i == n_rows:
                break
            else:
                continue
        return A

    # Let P be a projection matrix. P can be written in the basis of vecs as Ptilde.
    # Ptilde is still a projector with eigenvalues (tvals) zero and one. The
    # eigenvectors (tvecs) with eigenvalue one give the coefficients of the linear
    # combinations of vecs that are properly symmetrized according to P. The rest of
    # the eigenvectors give the coefficients of the linear combinations of vecs
    # that are orthogonal to the symmetrized vectors. These remaining orthogonal,
    # unsymmetrized vectors can be fed to another projection matrix, and the
    # algorithm proceeds as before.

    # num_remaining tracks how many CGC vectors still need to be symmetrized.
    # remaining_vecs are the remaining CGC vectors (as rows of an array).
    # sym_vecs will be a list of individual symmetrized CGC vectors.

    num_remaining = multiplicity
    remaining_vecs = vecs.T
    sym_vecs = []
    for partitions in Sn_irreps:
        mult = Sn_irreps[partitions]
        for symmetrizer in highest_weight_state_symmetrizers[partitions]:

            # Build Ptilde out of the Young symmetrizer and the remaining
            # vectors. The eigenvalues tvals will be ones and zeros.

            Ptilde = []
            for i in range(num_remaining):
                row = []
                for j in range(num_remaining):
                    row.append(remaining_vecs[i]@symmetrizer@remaining_vecs[j])
                Ptilde.append(row)
            tvals,tvecs = np.linalg.eigh(Ptilde)

            # ones contains the indices of tvals where eigenvalues equal one.
            # coeffs are the corresponding eigenvectors. symmetrized is an array
            # whose rows are properly symmetrized CGCs. If the multiplicity (mult)
            # of the symmetric group irrep is greater than one then the CGCs are
            # further refined by passing them through RREF and Gram-Schmidt.
            # Otherwise, ensure that the CGCs are normalized.

            ones = np.where(np.isclose(tvals, 1.0))[0]
            coeffs = tvecs[:,ones].T
            symmetrized = coeffs@remaining_vecs
            if mult > 1:
                symmetrized = RREF(symmetrized)
                symmetrized = np.linalg.qr(symmetrized.T)[0].T
            else:
                symmetrized /= np.linalg.norm(symmetrized)

            # The phase convention is such that the CGC of the highest-weight
            # product basis state is positive.

            for k in range(mult):
                max_state_idx = selected_basis.index(min(selected_basis[n] for n in range(len(selected_basis)) if abs(symmetrized[k][n]) > EPS))
                if symmetrized[k][max_state_idx] < 0:
                    symmetrized[k] *= -1
                else:
                    pass
                sym_vecs.append(symmetrized[k])

            # zeros contains the indices of tvals where eigenvalues equal zero.
            # coeffs are the corresponding eigenvectors. remaining_vecs are turned
            # into vectors that are orthogonal to the current symmetrized CGCs.

            if num_remaining-mult == 0:
                continue
            else:
                zeros = [k for k in range(len(tvals)) if k not in ones]
                coeffs = tvecs[:,zeros].T
                remaining_vecs = coeffs@remaining_vecs
                num_remaining -= mult
    vecs = np.array(sym_vecs)

    # Gather all nonzero CGCs for each multiplicity index.
    # A CGC counts as zero if abs(CGC) < EPS.
    
    cgc_dict = {}
    for a in range(multiplicity):
        cgc_dict[a+1] = {selected_basis[i]: vecs[a][i] for i in range(len(selected_basis)) if abs(vecs[a][i]) > EPS}

    # Save CGCs.

    with open(highest_weight_cgc_data_path, 'wb') as fp:
        dump(cgc_dict, fp)

    return cgc_dict

def calc_lower_weight_cgcs(product_iweights: list[tuple], sum_iweight_mult_idx: tuple[tuple, int], highest_dict: dict[tuple, float]) -> dict[int, dict[tuple, float]]:
    """Calculates the Clebsch-Gordan Coefficients for all lower-weight states
    of an irrep (sum_iweight, mult_idx) appearing in the direct-sum
    decomposition of a direct product of irreps (product_iweights). The CGCs
    for the highest-weight state of the irrep (highest_dict) is required.
    Returns a dictionary whose keys enumerate the states of sum_iweight
    (highest to lowest weight), and whose values are dictionaries of the
    form {product basis state: CGC}.
    ~Pg. 14
    """

    sum_iweight, _ = sum_iweight_mult_idx

    # Return CGCs if already computed.

    lower_weight_cgc_data_path = PurePath(data_directory, str(product_iweights), f'lower_weight_CGC_{sum_iweight_mult_idx}')
    if Path(lower_weight_cgc_data_path).exists():
        with open(lower_weight_cgc_data_path, 'rb') as fp:
            return load(fp)
    else:
        pass

    # Gather initial data. gt_patterns is a dictionary of GT-patterns for 
    # basis states of irreps in product_iweights as well as sum_iweight.
    # pweights contains the p-weights for the basis states of sum_iweight.
    # inner_mults gives the inner multiplicities of sum_iweight p-weights. 

    N = len(sum_iweight)
    gt_patterns = {R: find_gt_patterns(R) for R in set(product_iweights + [sum_iweight])}
    pweights = {gt_patterns[sum_iweight].index(gt): tuple(calc_weight(gt,'p')) for gt in gt_patterns[sum_iweight]}
    inner_mults = Counter(pweights.values())

    # When J(k)- is applied on a state S of sum_iweight, a linear combination
    # of states generally appears. Let S' be one of those states. Call S the
    # parent of the child S'. parent_dict is a nested dictionary of parents and
    # their children, arranged for later convenience. It has the form
    # {p-weight of S': {index for S: {J(k)- index k: [(coeff, index for S')]}}}
    # where coeff is coefficient of S' when applying J(k)- on S. There can be
    # multiple children with the same weight, hence the final list of tuples.

    parent_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for pat in gt_patterns[sum_iweight]:
        parent = gt_patterns[sum_iweight].index(pat)
        for k in range(N-1,0,-1):
            for coeff,child_in_list in ladder_op([pat], k, '-'):
                child = gt_patterns[sum_iweight].index(child_in_list[0])
                child_weight = pweights[child]
                parent_dict[child_weight][parent][k].append((coeff, child))
    
    cgc_dict = {}

    # There are potentially multiple sum_iweight states that have the same
    # p-weight. These states have a set of parents from different J(k)- operators.
    # Over the course of this for-loop, the CGCs of those parents will be found.
    # Then for a given p-weight, gather all the parent states P. Apply the
    # relevant J(k)- operators on each P. This creates a system of equations.
    # The LHS has rows J(k)-@P, which have at most inner_mults[p-weight]
    # nonzero entries. The RHS has J(k)- acting on product basis states
    # found in the CGCs of P. The system is generally overdetermined.

    for pweight in inner_mults:

        # For the highest-weight state, simply use highest_dict.
        if pweight==sum_iweight:
            cgc_dict[0] = highest_dict

        else:

            # The relevant sum (lhs) and product (rhs) basis states
            # will be gathered iteratively in lhs_basis and rhs_basis.
            LHS,RHS = [],[]
            lhs_basis,rhs_basis = {},{}
            lhs_basis_idx,rhs_basis_idx = 0,0

            # To create the LHS and RHS of the system, iterate over the parents
            # of pweight. Then act on the parent with J(k)- operators. All this
            # information is stored in parent_dict already at least for the LHS.
            for parent in parent_dict[pweight]:
                for k in parent_dict[pweight][parent]:
                    lhs_row = [0]*inner_mults[pweight]
                    for coeff,child in parent_dict[pweight][parent][k]:
                        # Augment lhs_basis.
                        if child in lhs_basis:
                            pass
                        else:
                            lhs_basis[child] = lhs_basis_idx
                            lhs_basis_idx += 1
                        # Augment LHS.
                        lhs_row[lhs_basis[child]] = coeff
                    LHS.append(lhs_row)

                    # Unlike the LHS, the number of columns in the RHS is unknown
                    # until all CGC product states are acted on by J(k)- operators.
                    # rhs_row tracks what product states actually show up for a row.
                    # The final RHS, will fill in zeros in rows where some product
                    # states did not show up.
                    rhs_row = defaultdict(float)
                    for product_state in cgc_dict[parent]:
                        product_pats = [gt_patterns[product_iweights[i]][product_state[i]] for i in range(len(product_iweights))]
                        for coeff,new_product_pats in ladder_op(product_pats, k, '-'):
                            new_product_state = tuple(gt_patterns[product_iweights[i]].index(new_product_pats[i]) for i in range(len(product_iweights)))
                            # Augment rhs_basis.
                            if new_product_state in rhs_basis:
                                pass
                            else:
                                rhs_basis[new_product_state] = rhs_basis_idx
                                rhs_basis_idx += 1
                            # Augment RHS.
                            rhs_row[rhs_basis[new_product_state]] += coeff*cgc_dict[parent][product_state]
                    RHS.append(rhs_row)

            # Make RHS a dense matrix, filling in zeros within rows.
            # This also aranges the row entries of RHS according to rhs_basis.
            # Use a least squares algorithm to the system of equations.
            # The solution gives the CGCs for all child states with pweight.
            RHS = [[row[i] if i in row else 0 for i in range(len(rhs_basis))] for row in RHS]
            cgcs = np.linalg.lstsq(LHS, RHS, rcond=None)[0]

            # Gather all nonzero CGCs for each child state.
            # A CGC counts as zero if abs(CGC) < EPS.
            for child in lhs_basis:
                child_cgcs = cgcs[lhs_basis[child]]
                nonzero_cgcs = {}
                for product_state in rhs_basis:
                    basis_idx = rhs_basis[product_state]
                    if abs(child_cgcs[basis_idx]) > EPS:
                        nonzero_cgcs[product_state] = child_cgcs[basis_idx]
                    else:
                        continue    
                cgc_dict[child] = nonzero_cgcs

    # Save CGCs.

    with open(lower_weight_cgc_data_path, 'wb') as fp:
        dump(cgc_dict, fp)

    return cgc_dict

def calc_cgcs(product_iweights: list[tuple], sum_iweight: tuple=None, mult_idx: int=None, sum_state: int=None, product_state: tuple=None) -> dict:
    """Manages the calculation of desired Clebsch-Gordan Coefficients in the
    direct-sum decomposition of a direct product of irreps (product_iweights).

    sum_iweight is an irrep appearing in the direct-sum decomposition.
    mult_idx is the multiplicity index of sum_iweight.
    sum_state is an integer indexing the basis state of sum_iweight.
    product_state is a product basis state of product_iweights.

    CGCs are saved in CGC_Data. To reduce redundant computations,
    product_iweights are sorted; computed CGCs are then returned with
    product basis states unsorted according to the input product_iweights.
    """

    # Get direct-sum decomposition and sort product_iweights.
    decomposition = find_direct_sum(product_iweights)
    product_irreps = sorted(product_iweights)

    # Ensure directory for product_iweights exists to save CGCs.
    cgc_data_directory = PurePath(data_directory, str(product_irreps))
    Path(cgc_data_directory).mkdir(exist_ok=True)
    
    # reorder is created to unsort the product basis states in computed CGCs.
    # The ith index of mapping gives the index the ith unsorted irrep becomes
    # in the sorted product_irreps. When the same irreps appear in
    # product_iweights, reorder simply slides these irreps together to the right.
    mapping = [0]*len(product_iweights)
    srt_idx = 0
    for irrep in sorted(set(product_iweights)):
        for unsrt_idx in locate(product_iweights, lambda x: x==irrep):
            mapping[unsrt_idx] = srt_idx
            srt_idx += 1
    reorder = lambda X: tuple(X[idx] for idx in mapping)

    # Returns specific CGC of a product basis state.
    if None not in {sum_iweight,mult_idx,sum_state,product_state}:
        multiplicity = decomposition[sum_iweight]
        highest_dict = calc_highest_weight_cgcs(product_irreps, sum_iweight, multiplicity)
        if sum_state==0:
            highest_dict = {reorder(P): highest_dict[mult_idx][P] for P in highest_dict[mult_idx]}
            if product_state in highest_dict:
                return highest_dict[product_state]
            else:
                return 0
        else:
            lower_dict = calc_lower_weight_cgcs(product_irreps, (sum_iweight,mult_idx), highest_dict[mult_idx])
            lower_dict = {reorder(P): lower_dict[sum_state][P] for P in lower_dict[sum_state]}
            if product_state in lower_dict:
                return lower_dict[product_state]
            else:
                return 0
    
    # Returns CGCs of a sum basis state.
    elif None not in {sum_iweight,mult_idx,sum_state}:
        multiplicity = decomposition[sum_iweight]
        highest_dict = calc_highest_weight_cgcs(product_irreps, sum_iweight, multiplicity)
        if sum_state==0:
            highest_dict = {reorder(P): highest_dict[mult_idx][P] for P in highest_dict[mult_idx]}
            return highest_dict
        else:
            lower_dict = calc_lower_weight_cgcs(product_irreps, (sum_iweight,mult_idx), highest_dict[mult_idx])
            lower_dict = {reorder(P): lower_dict[sum_state][P] for P in lower_dict[sum_state]}
            return lower_dict
    
    # Returns CGCs of a direct-sum irrep.
    elif None not in {sum_iweight,mult_idx}:
        multiplicity = decomposition[sum_iweight]
        highest_dict = calc_highest_weight_cgcs(product_irreps, sum_iweight, multiplicity)
        lower_dict = calc_lower_weight_cgcs(product_irreps, (sum_iweight,mult_idx), highest_dict[mult_idx])
        lower_dict = {S: {reorder(P): lower_dict[S][P] for P in lower_dict[S]} for S in lower_dict}
        return lower_dict
    
    # Returns CGCs of a direct-sum i-weight.
    elif sum_iweight is not None:
        cgc_dict = {}
        multiplicity = decomposition[sum_iweight]
        highest_dict = calc_highest_weight_cgcs(product_irreps, sum_iweight, multiplicity)
        for a in range(1,multiplicity+1):
            lower_dict = calc_lower_weight_cgcs(product_irreps, (sum_iweight,a), highest_dict[a])
            lower_dict = {S: {reorder(P): lower_dict[S][P] for P in lower_dict[S]} for S in lower_dict}
            cgc_dict[a] = lower_dict
        return cgc_dict

    # Returns all CGCs.
    else:
        cgc_dict = defaultdict(dict)
        for sum_irrep in decomposition:
            multiplicity = decomposition[sum_irrep]
            highest_dict = calc_highest_weight_cgcs(product_irreps, sum_irrep, multiplicity)
            for a in range(1,multiplicity+1):
                lower_dict = calc_lower_weight_cgcs(product_irreps, (sum_irrep,a), highest_dict[a])
                lower_dict = {S: {reorder(P): lower_dict[S][P] for P in lower_dict[S]} for S in lower_dict}
                cgc_dict[sum_irrep][a] = lower_dict
        return dict(cgc_dict)

def print_cgcs(product_iweights: list[tuple], sum_iweight: tuple=None, mult_idx: int=None, sum_state: int=None, product_state: tuple=None) -> None:
    """Prints all desired Clebsch-Gordan Coefficients in the
    direct-sum decomposition of a direct product of irreps (product_iweights).

    sum_iweight is an irrep appearing in the direct-sum decomposition.
    mult_idx is the multiplicity index of sum_iweight.
    sum_state is an integer indexing the basis state of sum_iweight.
    product_state is a product basis state of product_iweights.
    """

    cgc_res = calc_cgcs(product_iweights, sum_iweight, mult_idx, sum_state, product_state)
    gt_patterns = {R: dict(enumerate(find_gt_patterns(R))) for R in set(product_iweights)}

    if None not in {sum_iweight,mult_idx,sum_state,product_state}:
        sum_gts = find_gt_patterns(sum_iweight)
        print('#'*50)
        print('(i-weight, multiplicity index):', sum_iweight, mult_idx)
        print('#'*50)
        print()
        print('decomposition state:', sum_gts[sum_state])
        print('='*50)
        print(cgc_res, product_state)
        print()

    elif None not in {sum_iweight,mult_idx,sum_state}:
        sum_gts = find_gt_patterns(sum_iweight)
        print('#'*50)
        print('(i-weight, multiplicity index):', sum_iweight, mult_idx)
        print('#'*50)
        print()
        print('decomposition state:', sum_gts[sum_state])
        print('='*50)
        for product_state in cgc_res:
            cgc = cgc_res[product_state]
            print(cgc, product_state)
        print()

    elif None not in {sum_iweight,mult_idx}:
        sum_gts = find_gt_patterns(sum_iweight)
        print('#'*50)
        print('(i-weight, multiplicity index):', sum_iweight, mult_idx)
        print('#'*50)
        print()
        for sum_state in range(len(sum_gts)):
            print('decomposition state:', sum_gts[sum_state])
            print('='*50)
            for product_state in cgc_res[sum_state]:
                cgc = cgc_res[sum_state][product_state]
                print(cgc, product_state)
            print()

    elif sum_iweight is not None:
        sum_gts = find_gt_patterns(sum_iweight)
        for mult_idx in cgc_res:
            print('#'*50)
            print('(i-weight, multiplicity index):', sum_iweight, mult_idx)
            print('#'*50)
            print()
            for sum_state in range(len(sum_gts)):
                print('decomposition state:', sum_gts[sum_state])
                print('='*50)
                for product_state in cgc_res[mult_idx][sum_state]:
                    cgc = cgc_res[mult_idx][sum_state][product_state]
                    print(cgc, product_state)
                print()

    else:
        for sum_irrep in cgc_res:
            sum_gts = find_gt_patterns(sum_irrep)
            for mult_idx in cgc_res[sum_irrep]:
                print('#'*50)
                print('(i-weight, multiplicity index):', sum_irrep, mult_idx)
                print('#'*50)
                print()
                for sum_state in range(len(sum_gts)):
                    print('decomposition state:', sum_gts[sum_state])
                    print('='*50)
                    for product_state in cgc_res[sum_irrep][mult_idx][sum_state]:
                        cgc = cgc_res[sum_irrep][mult_idx][sum_state][product_state]
                        print(cgc, product_state)
                    print()
    
    print('states of', product_iweights)
    for R in gt_patterns:
        print(R)
        print(gt_patterns[R])
        print()

def check_cgcs(product_iweights: list[tuple]) -> bool:
    """Checks if the Clebsch-Gordan Coefficient matrix satisfies
    the expected orthogonality conditions. Returns True if orthogonal.
    """

    cgc_dict = calc_cgcs(product_iweights)
    ranges = [range(calc_dimension(R)) for R in product_iweights]
    dim = 1
    for R in product_iweights:
        dim *= calc_dimension(R)

    # Create the CGC matrix as a sparse array. product_index is called
    # to avoid iterating over every product basis state.

    row,col,val = [],[],[]
    row_idx = 0
    for sum_irrep in cgc_dict:
        for mult_idx in cgc_dict[sum_irrep]:
            for sum_state in cgc_dict[sum_irrep][mult_idx]:
                for product_state in cgc_dict[sum_irrep][mult_idx][sum_state]:
                    cgc = cgc_dict[sum_irrep][mult_idx][sum_state][product_state]
                    col_idx = product_index(product_state, *ranges)
                    row.append(row_idx), col.append(col_idx), val.append(cgc)
                row_idx += 1
    cgc_matrix = csr_array((val, (row,col)))

    # Multiply the CGC matrix with its transpose and eliminate
    # entries that are practically zero.

    CtC = cgc_matrix.T@cgc_matrix
    CCt = cgc_matrix@cgc_matrix.T
    CtC.data[abs(CtC.data) < EPS],CCt.data[abs(CCt.data) < EPS] = 0,0
    CtC.eliminate_zeros(),CCt.eliminate_zeros()
    Id = np.ones(dim)

    # Check orthogonality.

    orthogonal = np.allclose(CtC.data, Id)
    orthogonal = orthogonal and np.allclose(CCt.data, Id)
    orthogonal = orthogonal and np.allclose(CtC.data, CCt.data)

    return orthogonal
