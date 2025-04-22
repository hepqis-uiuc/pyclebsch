"""
IMPORTS
"""

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import eigsh
from itertools import product, combinations
from collections import Counter, defaultdict
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
CLEBSCH-GORDAN COEFFICIENTS
"""

def calc_highest_weight_cgcs(product_iweights: list[tuple], sum_iweight: tuple, multiplicity: int) -> dict[int, dict[tuple, float]]:
    """Calculates the Clebsch-Gordan Coefficients for the highest-weight state
    of an irrep (sum_iweight) appearing in the direct-sum decomposition of a
    direct product of irreps (product_iweights). A multiplicity number
    of orthonormal CGC vectors are produced. Returns a dictionary whose
    keys are the multiplicity indices of sum_iweight, and whose values
    are dictionaries of the form {product basis state: CGC}.
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

    selected_basis = []
    for p in partition_zval(highest_weight[0]):
        candidates = [product_zweights[product_iweights[i]][0][p[i]] for i in range(len(product_iweights))]
        for state in product(*candidates):
            for k in range(1,N-1):
                if sum(calc_weight(gt_patterns[product_iweights[i]][state[i]], 'z')[k] for i in range(len(product_iweights))) == highest_weight[k]:
                    continue
                else:
                    break
            else:
                selected_basis.append(state)

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
    # The eigsh algorithm can be finicky; v0 and ncv are chosen here by trial-and-error.

    if len(val) == 0:
        vecs = np.array([[1.0]])
    else:
        system = csr_array((val, (row,col)))
        v0 = np.ones(len(selected_basis))/np.sqrt(len(selected_basis))
        ncv = len(selected_basis)
        vals,vecs = eigsh(system.T@system, k=multiplicity, which='SM', v0=v0, ncv=ncv)
        
        # This is a check that the eigenvectors found have eigenvalue zero.
        if not all(abs(v)<EPS for v in vals): raise ArithmeticError('Some CGC vectors have nonzero eigenvalues:', [v for v in vals if abs(v)>=EPS])

    # The phase convention is such that the CGC of the highest-weight
    # product basis state is positive.

    for a in range(multiplicity):
        max_state_idx = selected_basis.index(min(selected_basis[i] for i in range(len(selected_basis)) if abs(vecs[:,a][i]) > EPS))
        if vecs[:,a][max_state_idx] < 0:
            vecs[:,a] *= -1
        else:
            continue

    # Gather all nonzero CGCs for each multiplicity index.
    # A CGC counts as zero if abs(CGC) < EPS.
    
    cgc_dict = {}
    for a in range(multiplicity):
        cgc_dict[a+1] = {selected_basis[i]: vecs[:,a][i] for i in range(len(selected_basis)) if abs(vecs[:,a][i]) > EPS}

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
