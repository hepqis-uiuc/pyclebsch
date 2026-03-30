"""
PLAQUETTE MATRIX ELEMENTS
"""

import numpy as np
from tqdm import tqdm
from ..cgc import calc_cgcs
from itertools import product
from multiprocessing import Pool
from collections import defaultdict
from .helpers import conjugate_irrep
from ..su_n_operators import calc_dimension, find_direct_sum

def plaquette_site_factor(N, s, initial, final, URij, ij_idxs, ctrl_irreps, conj_dict, EPS):
    """
    Calculates a site factor. The inputs are as follows.
    s -> The plaquette site number. Different sites require different conjugations.
    initial/final -> initial/final singlet data given as
    (ith plaquette link irrep, jth plaquette link irrep, FORDERd irreps, multiplicity index).
    URij -> Irreps that change the ith and jth plaquette links given as (ith UR irrep, jth UR irrep);
    these irreps are the fundamental or antifundamental representations.
    ij_idxs -> Indices of the ith and jth plaquette link within the FORDERd singlets,
    given as (ith index, jth index).
    ctrl_irreps -> Tuple of FORDERd control link irreps.
    conj_dict -> Dictionary of irreps and their conjugate basis states plus phases.
    The output is a tuple (site factor value, plaquette site number, site factor data),
    with the last entry being (initial ith irrep, initial jth irrep, final ith irrep,
    final jth irrep, initial multiplicity index, final multiplicity index, control link irreps).
    """

    # For clarity, the ith and jth plaquette links are, respectively:
    # l1 and l4 for s1
    # l1 and l2 for s2
    # l3 and l2 for s3
    # l3 and l4 for s4.

    # RES will be the site factor value.
    # Rii is the initial ith plaquette link irrep.
    # Rji is the initial jth plaquette link irrep.
    # Rif is the final ith link irrep.
    # Rjf is the final jth link irrep.
    # singleti and singletf are the initial and final singlets, FORDERd.
    # Gi and Gf are the initial and final singlet multiplicity indices.
    # UR_i is the irrep that changes the ith plaquette link irrep.
    # UR_j is the irrep that changes the jth plaquette link irrep.
    # fund is the fundamental representation i-weight (1, N-1 0's).
    RES = 0
    Rii,Rji,singleti,Gi = initial
    Rif,Rjf,singletf,Gf = final
    UR_i,UR_j = URij
    i_idx,j_idx = ij_idxs
    trivial_rep = (0,)*N
    fund = tuple(1 if i==0 else 0 for i in range(N))

    # Note, for example, that Rii may be conjugate to the ith plaquette link
    # irrep in singleti due to Gauss' law. This is accounted for below.

    # The rows and columns refer to the following notation:
    #  Rii Gi Rji
    # UR_i C UR_j
    #  Rif Gf Rjf
    # where C is the tuple of control link irreps.
    # top_row and bottom_row have the Clebsch-Gordon coefficients of the direct
    # product all irreps in singleti and singletf into the Gi'th/Gf'th trivial representation.
    # left_col has the coefficients for Rii x UR_i -> Rif.
    # right_col has the coefficients for Rji x UR_j -> Rjf.

    top_row = calc_cgcs(singleti, trivial_rep, Gi+1, 0)
    bottom_row = calc_cgcs(singletf, trivial_rep, Gf+1, 0)
    left_col = calc_cgcs([Rii, UR_i], Rif, 1)
    right_col = calc_cgcs([Rji, UR_j], Rjf, 1)
    top_dict,bottom_dict,left_dict,right_dict = defaultdict(dict),defaultdict(dict),defaultdict(dict),defaultdict(dict)
    
    # The dict versions have the coefficients organized for later use.
    for state in top_row:
        RiRj_state = (state[i_idx],state[j_idx])
        C_state = tuple(state[i] for i in range(len(singleti)) if i not in [i_idx,j_idx])
        top_dict[RiRj_state][C_state] = top_row[state]
    for state in bottom_row:
        RiRj_state = (state[i_idx],state[j_idx])
        C_state = tuple(state[i] for i in range(len(singletf)) if i not in [i_idx,j_idx])
        bottom_dict[RiRj_state][C_state] = bottom_row[state]
    for rif in left_col:
        for rii,ui in left_col[rif]:
            left_dict[ui][(rii,rif)] = left_col[rif][(rii,ui)]
    for rjf in right_col:
        for rji,uj in right_col[rjf]:
            right_dict[uj][(rji,rjf)] = right_col[rjf][(rji,uj)]

    # "scan" refers to the following for-loop. The u variable is chosen to be
    # a state of the fundamental representation (except for when s=4, where
    # the only relevant states are those of the antifundamental, which both
    # left_dict and right_dict have). left_dict will have these states when
    # s=1 and s=2. When s=3, right_dict has these states.
    scan_dict = left_dict if s in [1,2,4] else right_dict
    nonscan_dict = right_dict if s in [1,2,4] else left_dict

    for u in scan_dict:

        # The UR_i/j states are from the same irrep when s=2 or s=4.
        # For s=1 or s=3, a conjugate state must be found. This conjugate
        # state is named ut and the associated phase is phiu. Also, a check is
        # made if ut is in the nonscan_dict; if it is not, then proceeding with
        # this iteration would lead to a contribution of 0 to RES.
        if s in [1,3]:
            if conj_dict[fund][u][0] in nonscan_dict:
                ut,phiu = conj_dict[fund][u]
            else:
                continue
        else:
            if u in nonscan_dict:
                ut,phiu = u,1.0
            else:
                continue
        
        # More phases are gathered here. When s=1,2,4, Ri and Rj states must
        # get conjugated to match what will appear in top_dict and bottom_dict.
        # These conjugations have associated phases that are all multiplied together
        # with phiu into phi. Also, scan_states has ith plaquette link irrep
        # data unless s=3, hence the conditionals when extracting the ri's and rj's.
        for scan_states, nonscan_states in product(scan_dict[u], nonscan_dict[ut]):
            rii,rif = scan_states if s in [1,2,4] else nonscan_states
            rji,rjf = nonscan_states if s in [1,2,4] else scan_states
            if s==1:
                rii,phi1 = conj_dict[Rii][rii]
                rif,phi2 = conj_dict[Rif][rif]
                rji,phi3 = conj_dict[Rji][rji]
                rjf,phi4 = conj_dict[Rjf][rjf]
                phi = phiu*phi1*phi2*phi3*phi4
            elif s==2:
                rji,phi3 = conj_dict[Rji][rji]
                rjf,phi4 = conj_dict[Rjf][rjf]
                phi = phiu*phi3*phi4
            elif s==4:
                rii,phi1 = conj_dict[Rii][rii]
                rif,phi2 = conj_dict[Rif][rif]
                phi = phiu*phi1*phi2
            else:
                phi = phiu
            
            # Check if the plaquette link irrep states are in the singlet coefficients;
            # if they are then a sum over all control link basis state configurations is
            # made in row_cgs. A minor efficiency conditional is made as well.
            if (rii,rji) not in top_dict or (rif,rjf) not in bottom_dict:
                continue
            elif len(top_dict[(rii, rji)]) <= len(bottom_dict[(rif, rjf)]):
                row_cgs = sum(top_dict[(rii, rji)][c]*bottom_dict[(rif, rjf)][c] for c in top_dict[(rii, rji)] if c in bottom_dict[(rif, rjf)])
            else:
                row_cgs = sum(top_dict[(rii, rji)][c]*bottom_dict[(rif, rjf)][c] for c in bottom_dict[(rif, rjf)] if c in top_dict[(rii, rji)])

            # If row_cgs is small, then ignore this contribution. Otherwise, add to RES.
            if abs(row_cgs) < EPS:
                continue
            else:
                RES += phi*scan_dict[u][scan_states]*nonscan_dict[ut][nonscan_states]*row_cgs
    
    # Return RES if not a rounding error.
    if abs(RES) < EPS:
        return 0
    else:
        return (RES,s,(Rii,Rji,Rif,Rjf,Gi,Gf,ctrl_irreps))

def calc_plaquette_site_factors(N, P, sites, plaquettes, truncation_irreps, singlets, conj_dict, FORDER, EPS, parallelize=True):
    """
    Calculates all plaquette site factors for a plaquette P. P is a
    tuple (site coordinate, lattice plane), such as ((0,0,0), (1,2)).
    Site factors are returned as a nested dictionary of the form
    {site number: {site factor: site factor value}}. Site factors are nested tuples
    of the form (initial ith link irrep, initial jth link irrep, final ith link irrep,
    final jth link irrep, initial multiplicity index, final multiplicity index,
    control link irreps). Here, i and j refer to lattice directions, given by
    the lattice plane (i,j) of P. The control link irreps are ordered according to FORDER.
    """

    # Gather information for P.
    signature = (P[1], tuple(tuple(sorted(sites[s])) for s in plaquettes[P][2]))
    plane, half_link_dirs_per_site = signature
    site_factors, site_factor_args = defaultdict(dict), []
    fund = tuple(1 if i==0 else 0 for i in range(N))
    afund = conjugate_irrep(fund)

    # Pre-compute the allowed direct-sum irreps when tensoring an irrep from
    # truncation_irreps with either the fundamental or antifundamental representations.
    decomp_dict = defaultdict(dict)
    possible_num_links_per_site = set(len(half_link_dirs) for half_link_dirs in half_link_dirs_per_site)
    for n in possible_num_links_per_site:
        for UR in [fund, afund]:
            decomp_dict[n][UR] = {R: [S for S in find_direct_sum([R,UR]) if S in truncation_irreps[n]] for R in truncation_irreps[n]}

    def add_args(s, n, UR_i, UR_j, Ri_conj, Rj_conj, ij_idxs, ctrl_idxs, singlet_irreps):
        """
        Adds arguments to site_factor_args to make calls to plaquette_site_factor.
        """

        # Unpack some data.
        URij = [UR_i,UR_j]
        i_idx,j_idx = ij_idxs
        i_decomp = decomp_dict[n][UR_i]
        j_decomp = decomp_dict[n][UR_j]

        # ctrl_irreps conjugates irreps as necessary.
        # A "re" prefix means "real"; the real lattice irrep is conjugated
        # relative to how it appears in a singlet.
        for C in singlet_irreps:
            ctrl_irreps = tuple(C[i] if ctrl_idxs[i][1]<0 else conjugate_irrep(C[i]) for i in range(len(C)))
            for plinks in singlet_irreps[C]:
                Rii,Rji = plinks
                reRii = conjugate_irrep(Rii) if Ri_conj else Rii
                reRji = conjugate_irrep(Rji) if Rj_conj else Rji
                for reRif,reRjf in product(i_decomp[reRii], j_decomp[reRji]):
                    Rif = conjugate_irrep(reRif) if Ri_conj else reRif
                    Rjf = conjugate_irrep(reRjf) if Rj_conj else reRjf
                    if (Rif,Rjf) not in singlet_irreps[C]: continue
                    for Gi,Gf in product(range(singlet_irreps[C][(Rii,Rji)]), range(singlet_irreps[C][(Rif,Rjf)])):
                        singleti, singletf, c_count = [], [], 0
                        for i in range(n):
                            if i==i_idx:
                                singleti.append(Rii)
                                singletf.append(Rif)
                            elif i==j_idx:
                                singleti.append(Rji)
                                singletf.append(Rjf)
                            else:
                                singleti.append(C[c_count])
                                singletf.append(C[c_count])
                                c_count += 1
                        initial = [reRii,reRji,singleti,Gi]
                        final = [reRif,reRjf,singletf,Gf]
                        site_factor_args.append((N,s,initial,final,URij,ij_idxs,ctrl_irreps,conj_dict,EPS))
    
    # Set up site factor calculations. site_factor_args contains a list of all
    # necessary inputs for plaquette_site_factor. These inputs are prepared with
    # this for-loop and add_args.

    for i in range(4):

        # Gather information on plaquette and control link directions for this site.
        s = i+1
        half_link_dirs = half_link_dirs_per_site[i]
        plaq_link_dirs = [plane[0],plane[1]] if s==1 else [-plane[0],plane[1]] if s==2 else [-plane[0],-plane[1]] if s==3 else [plane[0],-plane[1]]
        ctrl_link_dirs = list(set(half_link_dirs).symmetric_difference(plaq_link_dirs))

        # Gather information on plaquette and control link FORDER indices for this site.
        ctrl_idxs_and_dirs,count = [],0
        for direction in FORDER:
            if direction==plaq_link_dirs[0]:
                i_link_idx = count
                count += 1
            elif direction==plaq_link_dirs[1]:
                j_link_idx = count
                count += 1
            elif direction in ctrl_link_dirs:
                ctrl_idxs_and_dirs.append((count,direction))
                count += 1
            else:
                continue
        ij_idxs = [i_link_idx,j_link_idx]
        num_links_at_site = len(half_link_dirs)

        # Call add_args. The booleans tell whether Ri and Rj must get conjugated
        # relative to how they appear in the singlet. (If they are outgoing then
        # they must get conjugated.)
        match s:
            case 1:
                add_args(s, num_links_at_site, fund, afund, True, True, ij_idxs, ctrl_idxs_and_dirs, singlets[num_links_at_site])
            case 2:
                add_args(s, num_links_at_site, fund, fund, False, True, ij_idxs, ctrl_idxs_and_dirs, singlets[num_links_at_site])
            case 3:
                add_args(s, num_links_at_site, afund, fund, False, False, ij_idxs, ctrl_idxs_and_dirs, singlets[num_links_at_site])
            case 4:
                add_args(s, num_links_at_site, afund, afund, True, False, ij_idxs, ctrl_idxs_and_dirs, singlets[num_links_at_site])
    
    # If Clebsch-Gordan coefficients have not been computed before, a parallelized
    # computation may result in an EOFError.
    if parallelize:
        with Pool(5) as pool:
            sf_res = pool.starmap(func=plaquette_site_factor, iterable=tqdm(site_factor_args, desc='Site Factors', total=len(site_factor_args), leave=False), chunksize=1)
    else:
        sf_res = []
        for args in tqdm(site_factor_args, desc='Site Factors', leave=False):
            sf_res.append(plaquette_site_factor(*args))

    # Collect site factors from sf_res.
    for res in sf_res:
        if res==0:
            continue
        else:
            sf,s,info = res
            site_factors[s][info] = sf

    return dict(site_factors)

def glue_plaquette_site_factors(s1, info, site_factors, ctrl_idxs, BCs, dims, EPS, PRES):
    """
    Glues together site factors given an s1 site factor seed. info is nested
    dictionary of site factors organized in a way to streamline the glueing.
    site_factors is a dictionary of each site's site factors and their values.
    ctrl_idxs and BCs aid in checking boundary condition restrictions. dims
    is a dictionary of pre-computed irrep dimensions to avoid repeated calculations.
    matrix elements are returned as a dictionary of the form:
    {(final plaquette state, initial plaquette state): matrix element value}.
    """

    # Unpack data
    i_ctrl_idxs,j_ctrl_idxs = ctrl_idxs
    s2_has_BCs,s3_has_BCs,s4_has_BCs = BCs
    matrix_elements = {}

    # REFERENCE: (Rii, Rji, Rif, Rjf, Gi, Gf, ctrl_irreps)
    #              0    1    2    3   4   5       6

    # Calculate first irrep dimension coefficient, and record s1
    # i/j control link irrep for later boundary condition (BC) restrictions
    # with s2 and s4.
    dim1 = dims[s1[0]]*dims[s1[1]]/dims[s1[2]]/dims[s1[3]]
    if 1 in i_ctrl_idxs: s1_i_ctrl = s1[6][i_ctrl_idxs[1]]
    if 1 in j_ctrl_idxs: s1_j_ctrl = s1[6][j_ctrl_idxs[1]]

    # Gather s2 site factors that match the s1 ith plaquette link irrep,
    # and the control link irrep if BCs apply.
    # Guard: on non-periodic lattices, sites may have different link counts
    # and thus different allowed irreps; skip if s2 has no matching irreps.
    if (s1[0],s1[2]) not in info[2]: return matrix_elements
    if s2_has_BCs:
        S2 = info[2][(s1[0],s1[2])][s1_i_ctrl]
    else:
        S2 = info[2][(s1[0],s1[2])]

    # Loop through s2 site factors.
    for s2 in S2:

        # Record s2 j control link irrep for BCs with s3.
        if 2 in j_ctrl_idxs: s2_j_ctrl = s2[6][j_ctrl_idxs[2]]

        # Gather s3 site factors that match s2 jth plaquette link irreps,
        # accounting for BCs.
        if (s2[1],s2[3]) not in info[3]: continue
        if s3_has_BCs:
            S3 = info[3][(s2[1],s2[3])][s2_j_ctrl]
        else:
            S3 = info[3][(s2[1],s2[3])]

        # Loop through s3 site factors.
        for s3 in S3:

            # For some lattices, all BC checks may still be insufficient
            # to identify glueable site factors. In these cases, just
            # manually check for which site factors can be glued.
            if (s3[0],s1[1],s3[2],s1[3]) not in info[4]: continue

            # Calculate second irrep dimension coefficient, and record
            # s3 i control link irrep for BCs with s4.
            dim3 = dims[s3[0]]*dims[s3[1]]/dims[s3[2]]/dims[s3[3]]
            if 3 in i_ctrl_idxs: s3_i_ctrl = s3[6][i_ctrl_idxs[3]]

            # Gather acceptable s4 site factors.
            if s4_has_BCs==(True,True):
                S4 = info[4][(s3[0],s1[1],s3[2],s1[3])][(s3_i_ctrl, s1_j_ctrl)]
            elif s4_has_BCs==(True,False):
                S4 = info[4][(s3[0],s1[1],s3[2],s1[3])][s3_i_ctrl]
            elif s4_has_BCs==(False,True):
                S4 = info[4][(s3[0],s1[1],s3[2],s1[3])][s1_j_ctrl]
            else:
                S4 = info[4][(s3[0],s1[1],s3[2],s1[3])]

            # Glue site factors together along with dimension coefficients.
            for s4 in S4:
                val = site_factors[1][s1]*site_factors[2][s2]*site_factors[3][s3]*site_factors[4][s4]
                if abs(val) < EPS:
                    continue
                else:
                    coeff = np.sqrt(dim1*dim3)
                    val = round(coeff*val, PRES)
                    Pf = (s1[2], s3[3], s3[2], s1[3], s1[6], s2[6], s3[6], s4[6], s1[5], s2[5], s3[5], s4[5])
                    Pi = (s1[0], s3[1], s3[0], s1[1], s1[6], s2[6], s3[6], s4[6], s1[4], s2[4], s3[4], s4[4])
                    matrix_elements[(Pf,Pi)] = val

    return matrix_elements

def calc_plaquette_elements(N, P, sites, plaquettes, truncation_irreps, singlets, conj_dict, FORDER, EPS, PRES, parallelize=True):
    """
    Calculates all plaquette matrix elements for a given plaquette P. P is a
    tuple (site coordinate, lattice plane), such as ((0,0,0), (1,2)). Matrix
    elements are returned as a dictionary of the form:
    {(final plaquette state, initial plaquette state): matrix element value}.
    Plaquette states are nested tuples of irreps and multiplicity indices of the form:
    (l1, l2, l3, l4, (s1 ctrls), (s2 ctrls), (s3 ctrls), (s4 ctrls), s1, s2, s3, s4).
    Matrix elements are calculated using FORDER; this may change the values
    of matrix elements for plaquettes on different planes, and it is required
    to perform consistent site factor sums over irrep indices in products of CGCs.
    """

    # Gather information for P.
    signature = (P[1], tuple(tuple(sorted(sites[s])) for s in plaquettes[P][2]))
    plane, half_link_dirs_per_site = signature
    unique_ctrls = plaquettes[P][-1]

    # Find control link FORDER indices.
    i_ctrl_idxs,j_ctrl_idxs = {},{}
    for i in range(4):
        ctrl_idx,s = 0,i+1
        plaq_link_dirs = [plane[0],plane[1]] if s==1 else [-plane[0],plane[1]] if s==2 else [-plane[0],-plane[1]] if s==3 else [plane[0],-plane[1]]
        for direction in FORDER:
            if direction in half_link_dirs_per_site[i]:
                if direction in plaq_link_dirs:
                    continue
                elif abs(direction)==plane[0]:
                    i_ctrl_idxs[s] = ctrl_idx
                    ctrl_idx += 1
                elif abs(direction)==plane[1]:
                    j_ctrl_idxs[s] = ctrl_idx
                    ctrl_idx += 1
                else:
                    ctrl_idx += 1
            else:
                continue
    
    # Calculate site factors.
    site_factors = calc_plaquette_site_factors(N, P, sites, plaquettes, truncation_irreps, singlets, conj_dict, FORDER, EPS, parallelize)
    if len(site_factors)==0:
        return {}
    else:
        info = {}
    
    # Check boundary conditions.
    s2_has_BCs = (1 in i_ctrl_idxs) and (2 in i_ctrl_idxs) and (i_ctrl_idxs[2] not in unique_ctrls[1])
    s3_has_BCs = (2 in j_ctrl_idxs) and (3 in j_ctrl_idxs) and (j_ctrl_idxs[3] not in unique_ctrls[2])
    enforce_i_ctrls = (3 in i_ctrl_idxs) and (4 in i_ctrl_idxs) and (i_ctrl_idxs[4] not in unique_ctrls[3])
    enforce_j_ctrls = (1 in j_ctrl_idxs) and (4 in j_ctrl_idxs) and (j_ctrl_idxs[4] not in unique_ctrls[3])
    s4_has_BCs = (enforce_i_ctrls, enforce_j_ctrls)

    # Organize site factors for glueing. info is a dictionary of site factors for
    # s2, s3, and s4. info has the form {s_k: {site factor identifier: site factor}}.
    # The site factor identifier helps the glueing process. Without boundary condition
    # constraints, the identifiers are simply initial and final plaquette link
    # irreps. For example, s2 site factors are glued to s1 site factors if both
    # site factors have the same Rii and Rif. With boundary condition constraints,
    # a more selective identifier is used involving the shared control link irreps.
    # For example, if s2 and s1 have the same Rii and Rif, then only site factors
    # that have the same i control link irrep will be glued.
    for i in range(4):
        s = i+1
        match s:
            # REFERENCE: (Rii, Rji, Rif, Rjf, Gi, Gf, ctrl_irreps)
            #              0    1    2    3   4   5       6
            case 1:
                pass
            case 2:
                if s2_has_BCs:
                    s2_info = defaultdict(lambda: defaultdict(list))
                    for sf in site_factors[s]:
                        s2_info[(sf[0],sf[2])][sf[6][i_ctrl_idxs[2]]].append(sf)
                else:
                    s2_info = defaultdict(list)
                    for sf in site_factors[s]:
                        s2_info[(sf[0],sf[2])].append(sf)
                info[s] = dict(s2_info)
            case 3:
                if s3_has_BCs:
                    s3_info = defaultdict(lambda: defaultdict(list))
                    for sf in site_factors[s]:
                        s3_info[(sf[1],sf[3])][sf[6][j_ctrl_idxs[3]]].append(sf)
                else:
                    s3_info = defaultdict(list)
                    for sf in site_factors[s]:
                        s3_info[(sf[1],sf[3])].append(sf)
                info[s] = dict(s3_info)
            case 4:
                if enforce_i_ctrls and enforce_j_ctrls:
                    s4_info = defaultdict(lambda: defaultdict(list))
                    for sf in site_factors[s]:
                        s4_info[sf[0:4]][(sf[6][i_ctrl_idxs[4]], sf[6][j_ctrl_idxs[4]])].append(sf)
                elif enforce_i_ctrls:
                    s4_info = defaultdict(lambda: defaultdict(list))
                    for sf in site_factors[s]:
                        s4_info[sf[0:4]][sf[6][i_ctrl_idxs[4]]].append(sf)
                elif enforce_j_ctrls:
                    s4_info = defaultdict(lambda: defaultdict(list))
                    for sf in site_factors[s]:
                        s4_info[sf[0:4]][sf[6][j_ctrl_idxs[4]]].append(sf)
                else:
                    s4_info = defaultdict(list)
                    for sf in site_factors[s]:
                        s4_info[sf[0:4]].append(sf)
                info[s] = dict(s4_info)

    # Condense data for site factor glueing.
    ctrl_idxs = [i_ctrl_idxs,j_ctrl_idxs]
    BCs = [s2_has_BCs,s3_has_BCs,s4_has_BCs]
    dims = {R: calc_dimension(R) for R in set().union(*truncation_irreps.values())}

    # Because s1 site factors have no restriction in the glueing process,
    # they can be glued separately (in parallel), or consecutively.
    # glue_plaquette_site_factors is called on a given s1 site factor, generating
    # all possible matrix elements from that s1 seed.
    if parallelize:
        with Pool(5) as pool:
            res = pool.starmap(func=glue_plaquette_site_factors, iterable=tqdm(((s1,info,site_factors,ctrl_idxs,BCs,dims,EPS,PRES) for s1 in site_factors[1]), desc='Matrix Elements', total=len(site_factors[1]), leave=False), chunksize=1)
    else:
        res = []
        for s1 in tqdm(site_factors[1], desc='Matrix Elements', leave=False):
            res.append(glue_plaquette_site_factors(s1,info,site_factors,ctrl_idxs,BCs,dims,EPS,PRES))

    # Combine all matrix element dictionaries in res.
    matrix_elements = {k:v for d in res for k,v in d.items()}
    return matrix_elements
