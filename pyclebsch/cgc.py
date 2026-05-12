import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import eigsh
from itertools import product, permutations
from collections import Counter, defaultdict
from more_itertools import locate, product_index
from pathlib import Path
from pickle import load, dump

from pyclebsch.su_n_operators import (
    calc_dimension,
    calc_weight,
    find_gt_patterns,
    find_direct_sum,
    find_symmetry_direct_sum,
    ladder_op,
)
from pyclebsch.symmetric_group import calc_Sn_dimension
from pyclebsch.symmetric_group.tableaux import find_tableaux
from pyclebsch.symmetric_group.young_symmetrizer import young_symmetrizer

EPS = 1e-10

_cgc_cache_dir: Path | None = Path("./CGC_Data")

def set_cache_dir(path: str | Path | None) -> None:
    """Set the directory where computed CGCs are cached.
    Pass None to disable caching entirely.
    """
    global _cgc_cache_dir
    _cgc_cache_dir = Path(path) if path is not None else None


def get_cache_dir() -> Path | None:
    """Return the current CGC cache directory, or None if caching is disabled."""
    return _cgc_cache_dir


def _resolve_cache_dir() -> Path | None:
    """Return the effective cache directory, creating it if needed.
    Returns None when caching is disabled.
    """
    if _cgc_cache_dir is None:
        return None
    _cgc_cache_dir.mkdir(parents=True, exist_ok=True)
    return _cgc_cache_dir


def calc_highest_weight_cgcs(
    product_iweights: list[tuple], sum_iweight: tuple
) -> dict[int, dict[tuple, float]]:
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

    sum_irrep = tuple(j-sum_iweight[-1] for j in sum_iweight)
    cache_dir = _resolve_cache_dir()
    if cache_dir is not None:
        highest_weight_cgc_data_path = cache_dir / str(product_iweights) / ('highest_weight_CGC_' + str(sum_irrep))
        if highest_weight_cgc_data_path.exists():
            with open(highest_weight_cgc_data_path, 'rb') as fp:
                return load(fp)

    # Gather initial data. The GT-pattern for the highest-weight state
    # of sum_iweight can be manually made. gt_patterns is a dictionary
    # of GT-patterns for basis states of irreps in product_iweights.

    N = len(sum_irrep)
    highest_weight_state = [[sum_irrep[i] for i in range(j)] for j in range(N,0,-1)]
    highest_weight = calc_weight(highest_weight_state, 'z')
    gt_patterns = {R: find_gt_patterns(R) for R in set(product_iweights)}

    # Sn_irreps gives the symmetric group irreps the CGCs of sum_iweight
    # should transform under. idx_list are the indices each irrep acts on.
    # multiplicity is the number of sum_iweight copies in product_iweights.
    # tableaux_cache stores necessary standard Young tableaux.

    Sn_irreps, idx_list = find_symmetry_direct_sum(product_iweights, sum_irrep)
    multiplicity = sum(np.prod([calc_Sn_dimension(irrep) for irrep in partitions])*mult for partitions, mult in Sn_irreps.items())
    if multiplicity==0: raise ValueError('sum_iweight not in product_iweights decomposition.')
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
                    temp,pidx = [0 for _ in range(len(product_iweights))],0
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

    if cache_dir is not None:
        highest_weight_cgc_data_path = cache_dir / str(product_iweights) / ('highest_weight_CGC_' + str(sum_irrep))
        highest_weight_cgc_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(highest_weight_cgc_data_path, 'wb') as fp:
            dump(cgc_dict, fp)

    return cgc_dict


def calc_lower_weight_cgcs(product_iweights: list[tuple], sum_iweight: tuple, mult_idx: int) -> dict[int, dict[tuple, float]]:
    """Calculates the Clebsch-Gordan Coefficients for all lower-weight states
    of an irrep (sum_iweight, mult_idx) appearing in the direct-sum
    decomposition of a direct product of irreps (product_iweights).
    Returns a dictionary whose keys enumerate the states of sum_iweight
    (highest to lowest weight), and whose values are dictionaries of the
    form {product basis state: CGC}.
    ~Pg. 14
    """

    # Return CGCs if already computed.

    sum_irrep = tuple(j-sum_iweight[-1] for j in sum_iweight)
    cache_dir = _resolve_cache_dir()
    if cache_dir is not None:
        lower_weight_cgc_data_path = cache_dir / str(product_iweights) / f'lower_weight_CGC_{(sum_irrep, mult_idx)}'
        if lower_weight_cgc_data_path.exists():
            with open(lower_weight_cgc_data_path, 'rb') as fp:
                return load(fp)

    # The CGCs for the highest-weight state of sum_iweight are required.
    try:
        highest_dict = calc_highest_weight_cgcs(product_iweights, sum_irrep)[mult_idx]
    except KeyError:
        raise ValueError('Invalid mult_idx.')

    # Gather initial data. gt_patterns is a dictionary of GT-patterns for 
    # basis states of irreps in product_iweights as well as sum_iweight.
    # pweights contains the p-weights for the basis states of sum_iweight.
    # inner_mults gives the inner multiplicities of sum_iweight p-weights. 

    N = len(sum_irrep)
    gt_patterns = {R: find_gt_patterns(R) for R in set(product_iweights + [sum_irrep])}
    pweights = {gt_patterns[sum_irrep].index(gt): tuple(calc_weight(gt,'p')) for gt in gt_patterns[sum_irrep]}
    inner_mults = Counter(pweights.values())

    # When J(k)- is applied on a state S of sum_iweight, a linear combination
    # of states generally appears. Let S' be one of those states. Call S the
    # parent of the child S'. parent_dict is a nested dictionary of parents and
    # their children, arranged for later convenience. It has the form
    # {p-weight of S': {index for S: {J(k)- index k: [(coeff, index for S')]}}}
    # where coeff is coefficient of S' when applying J(k)- on S. There can be
    # multiple children with the same weight, hence the final list of tuples.

    parent_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for pat in gt_patterns[sum_irrep]:
        parent = gt_patterns[sum_irrep].index(pat)
        for k in range(N-1,0,-1):
            for coeff,child_in_list in ladder_op([pat], k, '-'):
                child = gt_patterns[sum_irrep].index(child_in_list[0])
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
        if pweight==sum_irrep:
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

    if cache_dir is not None:
        lower_weight_cgc_data_path = cache_dir / str(product_iweights) / f'lower_weight_CGC_{(sum_irrep, mult_idx)}'
        lower_weight_cgc_data_path.parent.mkdir(parents=True, exist_ok=True)
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
    # Normalize product_iweights.
    N = len(product_iweights[0])
    normalized_iweights = [tuple(j-iweight[-1] for j in iweight) for iweight in product_iweights]
    
    # Sort nontrivial iweights. Trivial irreps can be appended later.
    # product_irreps contains the direct product CGCs will be calculated for.
    # (1) If at least two irreps are nontrivial, then product_irreps = nontrivial_iweights.
    # (2) If only one irrep is nontrivial, then CGCs are calculated for trivial x irrep.
    # (3) If all irreps are trivial, then CGCs are calculated for trivial x trivial.
    # excess gives the number of trivial irrep states to be eventually appended.
    trivial = (0,)*N
    nontrivial_iweights = sorted(iweight for iweight in normalized_iweights if iweight != trivial)
    if len(nontrivial_iweights) >= 2:
        product_irreps = nontrivial_iweights
        excess = normalized_iweights.count(trivial)
    elif len(nontrivial_iweights) == 1:
        product_irreps = [trivial] + nontrivial_iweights
        excess = normalized_iweights.count(trivial)-1
    else:
        product_irreps = [trivial, trivial]
        excess = normalized_iweights.count(trivial)-2

    # Ensure directory for product_iweights exists to save CGCs.
    cache_dir = _resolve_cache_dir()
    if cache_dir is not None:
        cgc_data_directory = cache_dir / str(product_irreps)
        cgc_data_directory.mkdir(parents=True, exist_ok=True)
    
    # reorder is created to unsort the product basis states in computed CGCs,
    # and to append trivial states (to the front, as they would be for a sorted state).
    # The ith index of mapping gives the index the ith unsorted irrep becomes
    # in the sorted product_irreps. When the same irreps appear in
    # normalized_iweights, reorder simply slides these irreps together to the right.
    mapping = [0]*len(normalized_iweights)
    srt_idx = 0
    for irrep in sorted(set(normalized_iweights)):
        for unsrt_idx in locate(normalized_iweights, lambda x: x==irrep):
            mapping[unsrt_idx] = srt_idx
            srt_idx += 1
    def reorder(X):
        Xp = (0,)*excess + X
        return tuple(Xp[idx] for idx in mapping)

    # Returns specific CGC of a product basis state.
    if None not in {sum_iweight,mult_idx,sum_state,product_state}:
        try:
            lower_dict = calc_lower_weight_cgcs(product_irreps, sum_iweight, mult_idx)
            lower_dict = {reorder(P): lower_dict[sum_state][P] for P in lower_dict[sum_state]}
            if product_state in lower_dict:
                return lower_dict[product_state]
            else:
                return 0
        except KeyError:
            raise ValueError('Invalid sum_state.')
    
    # Returns CGCs of a sum basis state.
    elif None not in {sum_iweight,mult_idx,sum_state}:
        try:
            lower_dict = calc_lower_weight_cgcs(product_irreps, sum_iweight, mult_idx)
            lower_dict = {reorder(P): lower_dict[sum_state][P] for P in lower_dict[sum_state]}
            return lower_dict
        except KeyError:
            raise ValueError('Invalid sum_state.')
    
    # Returns CGCs of a direct-sum irrep.
    elif None not in {sum_iweight,mult_idx}:
        lower_dict = calc_lower_weight_cgcs(product_irreps, sum_iweight, mult_idx)
        lower_dict = {S: {reorder(P): lower_dict[S][P] for P in lower_dict[S]} for S in lower_dict}
        return lower_dict
    
    # Returns CGCs of a direct-sum i-weight.
    elif sum_iweight is not None:
        cgc_dict = {}
        multiplicity = find_direct_sum(product_iweights, sum_iweight)
        if multiplicity==0: raise ValueError('sum_iweight not in product_iweights decomposition.')
        for a in range(1,multiplicity+1):
            lower_dict = calc_lower_weight_cgcs(product_irreps, sum_iweight, a)
            lower_dict = {S: {reorder(P): lower_dict[S][P] for P in lower_dict[S]} for S in lower_dict}
            cgc_dict[a] = lower_dict
        return cgc_dict

    # Returns all CGCs.
    else:
        cgc_dict = defaultdict(dict)
        decomposition = find_direct_sum(product_iweights)
        for sum_irrep in decomposition:
            multiplicity = decomposition[sum_irrep]
            for a in range(1,multiplicity+1):
                lower_dict = calc_lower_weight_cgcs(product_irreps, sum_irrep, a)
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
    dim,ranges = 1,[]
    for R in product_iweights:
        Rdim = calc_dimension(R)
        dim *= Rdim
        ranges.append(range(Rdim))

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
