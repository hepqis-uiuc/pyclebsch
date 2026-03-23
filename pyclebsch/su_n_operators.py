from math import factorial
import numpy as np
from scipy.sparse import csr_array
from itertools import product, combinations
from collections import Counter, defaultdict
from functools import reduce
from more_itertools import locate
from typing import Optional

from pyclebsch.symmetric_group.plethysm_utils import _Adams, _class_character, _class_order
from pyclebsch.symmetric_group.tableaux import find_partitions


type IrrepWeight = tuple[int, int, ...] # Length N corresponds to SU(N) iweight.

def calc_dimension(iweight: IrrepWeight) -> int:
    """Returns dimension of an irrep.
    ~Eq. (22)
    """

    dim = 1
    for j in range(1, len(iweight)):
        for jp in range(j):
            dim *= 1 + (iweight[jp] - iweight[j])/(j - jp)

    return round(dim)


def calc_casimir(iweight: IrrepWeight) -> float:
    """Returns the quadratic Casimir eigenvalue of an irrep.
    """

    N = len(iweight)
    R = [j-iweight[-1] for j in iweight]

    res = N*sum(R[i]*(R[i] + N - 1 - 2*i) for i in range(N))
    res -= sum(R)**2

    return res/(2*N)


def calc_dynkin_index(iweight: IrrepWeight) -> float:
    """Returns the Dynkin index of an irrep.
    """

    N = len(iweight)
    R = [j-iweight[-1] for j in iweight]

    cas = N*sum(R[i]*(R[i] + N - 1 - 2*i) for i in range(N))
    cas -= sum(R)**2

    dimR = calc_dimension(R)
    dimG = N**2 - 1

    return cas*dimR/(2*N*dimG)


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


def find_gt_patterns(iweight: IrrepWeight) -> list[list[list]]:
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


def find_suN_basis(iweight: IrrepWeight) -> list[csr_array]:
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


def find_direct_sum(product_iweights: list[IrrepWeight], sum_iweight: Optional[IrrepWeight]=None) -> dict[tuple, int]:
    """Decomposes a direct product of irreps (product_iweights) into a
    direct sum of irreps. Returns a dictionary whose keys are the irreps
    appearing in the direct sum, and whose values are the multiplicities of
    the irreps. If sum_iweight is given, then the multiplicity (possibly 0)
    of that irrep is returned.
    ~Pg. 11
    """

    # For efficiency, sort the irreps from lowest dimension to highest dimension.
    # Moreover, normalize the i-weights to avoid redundancies.
    # decomp_memo records decompositions done throughout the algorithm, so
    # repititions are avoided. gt_memo records GT-patterns of irreps
    # encountered in the algorithm to avoid duplicate calculations.

    iweights = sorted((tuple(j-iweight[-1] for j in iweight) for iweight in product_iweights), key=calc_dimension)
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


def find_symmetry_direct_sum(product_iweights: list[IrrepWeight], sum_iweight: Optional[IrrepWeight]=None) -> tuple[dict, list]:
    """Decomposes a direct product of irreps (product_iweights) into a
    direct sum of irreps and provides the symmetry group irreps they
    transform under. Returns a dictionary of the direct sum and a list
    giving the lists of indices that transform under symmetric group irreps.
    If sum_iweight is given, then the (possibly empty) dictionary of
    the symmetry group irreps for only that sum_iweight is returned.
    """

    # Find all plethysms for each repeated irrep in product_iweights.
    # i-weights are normalized in normalized_iweights to identify repeated irreps.
    # keys is nearly list(set(normalized_iweights)), except the order of
    # the irreps in keys is that of the irreps in plethysms. indices
    # is a list of lists of indices of each irrep in keys as it appears
    # in normalized_iweights.

    normalized_iweights = [tuple(j-iweight[-1] for j in iweight) for iweight in product_iweights]
    plethysms = {R: find_plethysms(R,num) for R,num in Counter(normalized_iweights).items()}
    keys = list(plethysms.keys())
    indices = [list(locate(normalized_iweights, lambda x: x==R)) for R in keys]

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


def find_plethysms(iweight: IrrepWeight, n: int) -> dict[tuple, dict[tuple, int]]:
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
    simple_roots, fund_basis = [], find_gt_patterns(fundamental_rep)
    for i in range(N-1):
        wi = np.array(calc_weight(fund_basis[i], 'z'))
        wi_1 = np.array(calc_weight(fund_basis[i+1], 'z'))
        simple_roots.append(2*(wi - wi_1))

    # A plethysm has a polynomial where each term is an SU(N) irrep with
    # a multiplicity coefficient. The formula for the polynomial can be found
    # in page 72 of http://wwwmathlabo.univ-poitiers.fr/~maavl/pdf/LiE-manual.pdf
    # The algorithm is iterative and many computations are repeated; to
    # mitigate this, Adams_dict stores all possible evaluations of Adams.
    # decomp_dict stores all direct-sum decompositions. part_dict effectively
    # evaluates the formula once, storing each term from it, up to a coefficient.

    Adams_dict = {k: _Adams(k, N, dominant_zweights, simple_roots) for k in range(1,n+1)}
    part_dict, decomp_dict = {}, {}

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
            coeff = _class_order(P,n)*_class_character(partition,P)/factorial(n)
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
