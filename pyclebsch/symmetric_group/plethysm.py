import numpy as np
from math import factorial
from itertools import product
from collections import Counter, defaultdict


def _class_order(partition, n):
    """Calculates the order of the conjugacy
    class of the Sn irrep given by partition.
    """

    prod,cycle_tally = 1,Counter(partition)
    for cycle_len, num in cycle_tally.items():
        prod *= cycle_len**num * factorial(num)

    return round(factorial(n)/prod)

def _class_character(partition, Sn_partition):
    """Computes the character of an Sn_partition
    in the irrep given by partition recursively.
    """

    if len(partition)==0:
        return 1
    else:
        new_irreps = _rim_hooks(partition, Sn_partition[0])
        new_Sn_irrep = Sn_partition[1:]
        character = sum((-1)**l * _class_character(p,new_Sn_irrep) for p,l in new_irreps)
        return character

def _find_iweight(zweight):
    """Finds the i-weight corresponding to a dominant weight,
    assumed to be the highest weight of an irrep.
    """

    pweight,count = [0],-1
    for i in range(len(zweight)-1,-1,-1):
        if i==len(zweight)-1:
            pweight.append(int(2*zweight[i]))
        else:
            pweight.append(int(2*zweight[i] + pweight[count+1]))
        count += 1

    return tuple(pweight[::-1])

def _Weyl_orbit(zweight, N, simple_roots):
    """Finds Weyl orbit of a given SU(N) z-weight.
    """

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

def _Adams(k, N, dominant_zweights, simple_roots):
    """Applies the kth Adams operator to the irrep (iweight).
    """

    longest_Weyl_word = [j for i in range(N-2,-1,-1) for j in range(i+1)]

    def v_decomp(dweight):
        poly = [[w,1] for w in _Weyl_orbit(dweight, N, simple_roots)]
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
                yield _find_iweight(monomial.tolist()), coeff

    polynomial = defaultdict(int)
    for dweight,mult in dominant_zweights.items():
        scaled_dweight = [k*x for x in dweight]
        for R,coeff in v_decomp(scaled_dweight):
            polynomial[R] += coeff*mult

    return {R:num for R,num in polynomial.items() if num != 0}

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
