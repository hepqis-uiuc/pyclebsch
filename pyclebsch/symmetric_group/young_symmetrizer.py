from collections import defaultdict
from collections.abc import Generator


def _sgn(permutation):
    """Calculates the sign of a permutation.
    """

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

def _compose_two(p1, p2, n):
    """Computes the equivalent permutation got by first applying p1 and then p2.
    p1 and p2 need not be of equal lengths; n ensures that the final permutation
    is of length n. Nevertheless, p1 and p2 should each contain integers 0,1,...,x.
    """

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

def _reduce_perms(perms_lists, anti_idxs, n):
    """Simplifies a product of linear combinations of permutations into a single
    linear combination of distinct permutations. perms_lists is a list of
    row- or column-preserving permutations. anti_idxs is a list of indices
    of lists in perms_lists that contain column-preserving permutations;
    these permutations act with a factor of their sign. As an expression,
    (1 + p1 + p2 + ...) x (1 + p1 + p2 + ...) x ... is the input and the output
    is a0*1 + a1*p1 + a2*p2 + ... returned as dictionary whose keys are the
    distinct permutations pk and whose values are their coefficients ak.
    """

    # perms_lists is given such that the first list of permutations actually
    # acts first. ordered reverses perms_lists to correct this. res is the
    # final output; because the symmetrizer is so far unnormalized, its
    # permutations' coefficients will be integers.
    ordered = perms_lists[::-1]  
    res = defaultdict(int)
    for i in range(1,len(perms_lists)):
        if i==1:
            for p1,p2 in product(ordered[0], ordered[i]):
                new = _compose_two(p1,p2,n)
                if 0 in anti_idxs:
                    res[new] += _sgn(p1)
                elif i in anti_idxs:
                    res[new] += _sgn(p2)
                else:
                    res[new] += 1
            temp = res.copy()
        else:
            for p1,p2 in product(temp, ordered[i]):
                new = _compose_two(p1,p2,n)
                if new==p1:
                    continue
                elif i in anti_idxs:
                    new_val = temp[new] + res[p1]*_sgn(p2)
                else:
                    new_val = temp[new] + res[p1]
                if new_val == 0:
                    del temp[new]
                else:
                    temp[new] = new_val
            res = temp
            temp = res.copy()
    return dict(res)

def young_symmetrizer(tableaux: list[YoungTableau], idx_list: list[list]) -> Generator[tuple[float, tuple]]:
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

    # The algorithm presented in https://arxiv.org/pdf/1610.10088
    # is implemented and referenced throughout this code.

    symmetrizers = []
    norm = 1

    for tableau in tableaux:

        # Gather data for the input tableau.
        # Only the inverse permutations for the input tableau are necessary.
        # The usual Young symmetrizers are built with the convention where
        # column permutations are applied first, followed by row permutations.
        # ~Eq. (26)
        n, row_perms, col_perms, is_row_ordered, is_col_ordered, inv_row_perms, inv_col_perms, hook_length_prod = _tableau_data(tableau, extra_data=True)

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
                child_n, parent_row_perms, parent_col_perms, is_row_ordered, is_col_ordered = _tableau_data(parent_tableau)
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

        combined_perms = _reduce_perms(perms_lists, anti_idxs, n)
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
