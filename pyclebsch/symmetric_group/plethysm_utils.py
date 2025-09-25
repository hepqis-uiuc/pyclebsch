import numpy as np
from math import factorial
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
    pweight, count = [0],-1
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
