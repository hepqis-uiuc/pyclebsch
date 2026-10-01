"""
LATTICE DATA
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from tqdm import tqdm
from .helpers import *
from ..cgc import calc_cgcs
from collections import defaultdict
from ..su_n_operators import calc_casimir, find_direct_sum, IrrepWeight
from more_itertools import distinct_permutations
from itertools import product, combinations_with_replacement

type SiteMultiplicityIndex = int
type SiteControlLinks = tuple[IrrepWeight, ...]
type ActiveLink = IrrepWeight
type PlaquetteState = tuple[
    ActiveLink,
    ActiveLink,
    ActiveLink,
    ActiveLink,
    SiteControlLinks,
    SiteControlLinks,
    SiteControlLinks,
    SiteControlLinks,
    SiteMultiplicityIndex,
    SiteMultiplicityIndex,
    SiteMultiplicityIndex,
    SiteMultiplicityIndex,
]
type SiteCoordinate = tuple[int, int] | tuple[int, int, int]
type LinkDirection = Literal[1, 2, 3, -1, -2, -3]
type Plane = tuple[LinkDirection, LinkDirection]
type SiteHalfLinks = list[LinkDirection]
type LinkAddress = tuple[SiteCoordinate, LinkDirection]
type PlaquetteActiveLinkAddresses = tuple[LinkAddress, LinkAddress, LinkAddress, LinkAddress]
type PlaquetteControlLinkAddresses = tuple[list[LinkAddress], list[LinkAddress], list[LinkAddress], list[LinkAddress]]
type PlaquetteSiteCoordinates = tuple[SiteCoordinate, SiteCoordinate, SiteCoordinate, SiteCoordinate]
type PlaquetteUniqueControls = tuple[list[int], list[int], list[int], list[int]]
type PlaquetteAddress = tuple[SiteCoordinate, Plane]
type PlaquetteSignature = tuple[
    Plane, tuple[SiteHalfLinks, SiteHalfLinks, SiteHalfLinks, SiteHalfLinks] | tuple[()]
] # TODO make a class to enforce forder?


@dataclass
class LatticeDef:
    """Class for defining the geometry of a lattice."""

    num_sites: tuple[int, int, int]
    PBCs: tuple[bool, bool, bool]
    FORDER: tuple[LinkDirection, LinkDirection, LinkDirection, LinkDirection, LinkDirection, LinkDirection] | list[LinkDirection]

    @property
    def planes(self) -> tuple[Plane] | tuple[Plane, Plane, Plane]:
        if hasattr(self, "_lattice_planes"):
            return self._lattice_planes
        if any(self.num_sites) < 1:
            raise ValueError(
                f"Malformed lattice encountered while attempting to compute plane. num_sites should consist of positive integers but encountered the tuple '{self.num_sites}'."
            )
        possible_planes = [(1, 2), (1, 3), (2, 3)]
        self._lattice_planes = tuple(filter(self._plane_exists, possible_planes))

        return self._lattice_planes

    @property
    def sites(self) -> dict[SiteCoordinate, SiteHalfLinks]:
        """
        The F-ordered half-links connected to a given lattice site.
        """
        if not hasattr(self, "_sites"):
            self._sites, self._links, self._plaquettes = sites_links_and_plaquettes(self.num_sites, self.PBCs, self.FORDER)
        return self._sites

    @property
    def links(self) -> dict[LinkAddress, tuple[SiteCoordinate, SiteCoordinate]]:
        """
        The start and stop site coordinates associated with links in the lattice.
        """
        if not hasattr(self, "_links"):
            self._sites, self._links, self._plaquettes = sites_links_and_plaquettes(self.num_sites, self.PBCs, self.FORDER)
        return self._links

    @property
    def plaquettes(self) -> dict[
            PlaquetteAddress, tuple[
                PlaquetteActiveLinkAddresses,
                PlaquetteControlLinkAddresses,
                PlaquetteSiteCoordinates,
                PlaquetteUniqueControls]]:
        """
        Four-tuples of active links, control links, site coordinates, and unique controls for plaquettes.

        - Active links go CCW in the plaquette plane starting from the "bottom-left" site.
        - Control links are presented as a length-4 list of F-ordered non-active links
          attached to each site, following the "CCW from bottom-left site" ordering convention.
        - Sites are the lattice coordinates of each site in the plaquette, also CCW-ordered.
        - Unique controls is a list of indices for the controls in each corresponding site,
          with later duplicates filtered out. This allows for determination of the set of
          actual physical links which are controls without overcounting. For example, if
          the control link at index zero on the first site is also a control for the second
          site, then the first element of the unique controls list will contain the index '0',
          and the index in the second element of unique controls which corresponds to that
          physical link will be omitted.
        """
        if not hasattr(self, "_plaquettes"):
            self._sites, self._links, self._plaquettes = sites_links_and_plaquettes(self.num_sites, self.PBCs, self.FORDER)
        return self._plaquettes

    def site_exists(self, site: SiteCoordinate, periodic_ok: bool = True) -> bool:
        """
        Check whether site exists on the lattice.

        If periodic_ok is True, then (for any periodic directions) the lattice coordinate
        is wrapped around before checking for site existence (i.e. any int is allowed).
        For nonperiodic directions, this is NOT done (the allowed range is between zero
        and the corresponding entry in num_sites minus 1).
        """
        if len(site) != 3:
            raise ValueError(f"Lattice site '{site}' should be a 3-tuple of ints.")
        for idx, current_dir_max in enumerate(self.num_sites):
            if not isinstance(site[idx], int):
                raise ValueError(f"Lattice site '{site}' should be a 3-tuple of ints.")
            current_dir_is_periodic_and_periodic_ok = self.PBCs[idx] and periodic_ok
            site_is_between_zero_and_current_dir_max = (site[idx] >= 0) and (site[idx] < current_dir_max)
            current_dir_in_range = True if current_dir_is_periodic_and_periodic_ok is True else site_is_between_zero_and_current_dir_max
            if current_dir_in_range is False:
                return False

        return True
            

    def _plane_exists(self, plane: Plane) -> bool:
        """Check if plane can be formed on the lattice."""
        # Switch to using zero-indexed plane info.
        dir_1, dir_2 = sorted(plane)
        dir_1_zero_indexed = abs(dir_1) - 1
        dir_2_zero_indexed = abs(dir_2) - 1

        # Boolean tests to establish existence.
        lattice_plane_large_enough_despite_boundary_conds = (
            self.num_sites[dir_1_zero_indexed] >= 2
            and self.num_sites[dir_2_zero_indexed] >= 2
        )
        lattice_plane_has_one_large_direction_and_one_small_periodic_direction = (
            self.num_sites[dir_1_zero_indexed] >= 2
            and self.num_sites[dir_2_zero_indexed] == 1
            and (self.PBCs[dir_2_zero_indexed] is True)
        ) or (
            self.num_sites[dir_1_zero_indexed] == 1
            and self.num_sites[dir_2_zero_indexed] >= 2
            and (self.PBCs[dir_1_zero_indexed] is True)
        )
        lattice_plane_has_two_small_but_periodic_directions = (
            self.num_sites[dir_1_zero_indexed] == 1
            and self.num_sites[dir_2_zero_indexed] == 1
            and (self.PBCs[dir_1_zero_indexed] is True)
            and (self.PBCs[dir_2_zero_indexed] is True)
        )
        if (
            lattice_plane_large_enough_despite_boundary_conds
            or lattice_plane_has_one_large_direction_and_one_small_periodic_direction
            or lattice_plane_has_two_small_but_periodic_directions
        ):
            return True
        else:
            return False
        

def sites_links_and_plaquettes(num_sites: tuple[int, int, int] | list[int], PBCs: tuple[bool, bool, bool] | list[bool], FORDER) -> tuple[
        dict[SiteCoordinate, SiteHalfLinks],
        dict[LinkAddress, tuple[SiteCoordinate, SiteCoordinate]],
        dict[PlaquetteAddress, tuple[PlaquetteActiveLinkAddresses, PlaquetteControlLinkAddresses, PlaquetteSiteCoordinates, PlaquetteUniqueControls]]
]:
    """
    Sets up sites, links, and plaquettes of the cubic lattice.
    Sites are returned as a dictionary whose keys are site coordinates and whose
    values are lists of directions along the axes that a site's half-links extend to.
    Links are returned as keys of a dictionary whose values are
    [initial site, final site].
    Plaquettes are returned as keys of a dictionary with values of the form
    [ [plaquette links], [lists of control links], [plaquette sites],
    [unique control links] ]. Unique control links are identified by whether
    they appear multiple times due to periodic boundary conditions (i.e. the
    list of unique control links is a list counting how many times each
    link in the list of controls appears; if an element is zero, then
    that means the corresponding control link is unique).
    """

    # Check inputs.
    if any(n<1 for n in num_sites):
        raise ValueError('Number of sites along an axis must be at least 1.')
    elif any(n==1==c for n,c in zip(num_sites,PBCs)):
        raise ValueError('Axes with periodic boundary conditions must at least 2 sites.')
    else:
        pass

    # The maximum number of spatial lattice dimensions. This variable is
    # currently placed here for possible future development.
    MAX_LATTICE_DIM = 3

    if len(FORDER) != 2*MAX_LATTICE_DIM:
        raise NotImplementedError('>3d lattices are not implemented yet.')
    elif sorted(FORDER) != [-3,-2,-1,1,2,3]:
        raise ValueError('FORDER should be a permutation of [1,2,3,-1,-2,-3].')
    else:
        pass
    
    # Initialize sites as a dictionary of site coordinates with empty lists.
    # lengths gives the lengths along each lattice axis (lattice spacing = 1).
    sites = {s: [] for s in product(*(range(i) for i in num_sites))}
    lengths = [num_sites[i] if PBCs[i] else num_sites[i]-1 for i in range(3)]
    
    # Links are labeled by a tuple (lattice coordinate, positive lattice direction).
    # Using lengths and PBCs, each lattice link is found, and, in the process,
    # the directions of half-links connected to each site are appended. Values
    # of the dictionary links are (starting site, ending site).
    links = {}
    for s in sites:
        for i in range(3):
            if s[i]==lengths[i] and not PBCs[i]:
                continue
            else:
                s_i = list(s)
                if s[i]+1==lengths[i] and PBCs[i]:
                    s_i[i] = 0
                else:
                    s_i[i] += 1
                s_i = tuple(s_i)
                sites[s].append(i+1), sites[s_i].append(-(i+1))
                links[(s, i+1)] = tuple([s,s_i])
    
    # Plaquette links and sites are labeled as
    # s4 -- l3 -- s3
    #  |          |
    # l4          l2
    #  |          |
    # s1 -- l1 -- s2
    # By convention, l1 is a link extending in the +1 or +2 direction.
    # Each plaquette is defined with (s1 coordinate, plaquette plane),
    # with possible planes being (1,2), (1,3), or (2,3). Additionally,
    # multiple control links may extend from each plaquette site. However,
    # due to boundary conditions, some sites may share the same control link(s).
    # s1 will always have unique control links; the control links s2 or s4 share
    # with it will be counted as non-unique. Likewise, control links s3 (s4) shares
    # with s2 (s3) will be counted as non-unique.
    plaquettes = {}
    for s1 in sites:
        for j in range(1,3):
            for i in range(j):
                if (s1[i]==lengths[i] and not PBCs[i]) or (s1[j]==lengths[j] and not PBCs[j]):
                    continue
                else:
                    s2,s3,s4 = list(s1),list(s1),list(s1)
                    if s1[i]+1==lengths[i] and PBCs[i]:
                        s2[i] = 0
                        s3[i] = 0
                    else:
                        s2[i] += 1
                        s3[i] += 1
                    if s1[j]+1==lengths[j] and PBCs[j]:
                        s3[j] = 0
                        s4[j] = 0
                    else:
                        s3[j] += 1
                        s4[j] += 1
                    s2,s3,s4 = tuple(s2),tuple(s3),tuple(s4)
                    l1,l2,l3,l4 = (s1,i+1),(s2,j+1),(s4,i+1),(s1,j+1)

                    # plaq_link_directions gives the directions each l_k extends to at each site s_k.
                    ctrl_links, unique_ctrls = [], []
                    plaq_link_directions = [[i+1,j+1], [-(i+1),j+1], [-(i+1),-(j+1)], [i+1,-(j+1)]]
                    for n in range(4):
                        s = [s1,s2,s3,s4][n]
                        count, clinks, uniques = 0, [], []
                        # clinks is a list of control links for a site. Links are appended
                        # to clinks according to FORDER. If a control link is added to clinks and
                        # it has not been part of any clinks list, then it is counted as unique,
                        # and its index within clinks is appended to the uniques list. All clinks
                        # and uniques lists are gathered in ctrl_links and unique_ctrls, respectively.
                        for d in sorted(sites[s], key=lambda x: FORDER.index(x)):
                            if d in plaq_link_directions[n]:
                                continue
                            else:
                                if s[abs(d)-1] + np.sign(d, dtype=int) < 0:
                                    C = (tuple(lengths[k]-1 if k==abs(d)-1 else s[k] for k in range(3)), abs(d))
                                elif d < 0:
                                    C = (tuple(s[k]-1 if k==abs(d)-1 else s[k] for k in range(3)), abs(d))
                                else:
                                    C = (s, d)
                                clinks.append(C)
                                if all(C not in clist for clist in ctrl_links): uniques.append(count)
                                count += 1
                        ctrl_links.append(clinks), unique_ctrls.append(uniques)
                    plaquettes[(s1, (i+1,j+1))] = [tuple([l1,l2,l3,l4])] + [tuple(ctrl_links)] + [tuple([s1,s2,s3,s4])] + [tuple(unique_ctrls)]

    return sites, links, plaquettes

def irreps_and_singlets(N, sites, truncation_mode, cutoff):
    """
    Sets up irreps and singlets of the truncated Hilbert space.
    Irreps and singlets are returned as a dictionaries. Due to boundary
    conditions, different sites' links and singlets may have different possible
    irreps. Keys of these nested dictionaries are the number n of links that meet
    at a site, which can range from 2 to 6.
    Values of link_irreps are lists of irreps for a given n.
    Values of site_singlets are nested dictionaries representing all possible
    permutations of a singlet; they have the form
    {control link irreps: {plaquette link irreps: multiplicity of singlet}}.
    Additionally, a dictionary of irrep basis states and their conjugate states
    along with associated phases are returned as conj_dict. It has the form
    {irrep (R): {irrep basis state: (conjugate basis state from Rbar, phase)}}.
    """

    # Check inputs
    if isinstance(N, int) and N>1:
        pass
    else:
        raise ValueError('N must be an integer greater than 1.')
    if truncation_mode=='T' and isinstance(cutoff, int) and cutoff >= 0:
        trial_irreps = {R: calc_casimir(R) for R in sorted(r[::-1]+(0,) for r in combinations_with_replacement(range(cutoff+1), N-1))}
    elif truncation_mode in ['C','B'] and cutoff >= 0:
        trial_irreps = get_irreps(cutoff, N)
    else:
        raise ValueError('Invalid truncation mode and/or cutoff.')
    
    # trial_irreps is a dictionary {irrep: quadratic Casimir of irrep}.
    # Not all irreps in trial_irreps may be used, particularly in 'B'-type
    # truncations. The used irreps are gathered in used_irreps lists below.

    # Check each site and the number n of links meeting at each site. The set of
    # these distinct values is in possible_num_links_per_site.
    possible_num_links_per_site = sorted(set(len(half_link_dirs) for half_link_dirs in sites.values()))
    link_irreps, site_singlets, conj_dict = {}, {}, {}
    trivial_rep = (0,)*N

    # trial_singlet attempts to form distinct singlets out of irreps in trial_irreps.
    # find_direct_sum is called to calculate multiplicities of singlets.
    # If multiplicity=0 then no singlet is present. Otherwise, as long as the singlet
    # is accepted by a possible B truncation, all permutations of the singlet
    # are found and added in singlets in a form that streamlines building physical
    # plaquette states later.
    for n in possible_num_links_per_site:
        used_irreps, singlets = set(), defaultdict(dict)
        for trial_singlet in combinations_with_replacement(trial_irreps, n):
            sum_of_casimirs = sum(trial_irreps[R] for R in trial_singlet)
            B_valid = sum_of_casimirs < cutoff or np.isclose(sum_of_casimirs, cutoff)
            if truncation_mode in ['T','C'] or (truncation_mode=='B' and B_valid):
                multiplicity = find_direct_sum(trial_singlet, trivial_rep)
                if multiplicity > 0:
                    used_irreps.update(trial_singlet)
                    for permutation in distinct_permutations(trial_singlet):
                        singlets[permutation[2:]][permutation[0:2]] = multiplicity
                else:
                    continue
            else:
                continue
        link_irreps[n] = sorted(used_irreps)
        site_singlets[n] = dict(singlets)
    
    # conj_dict is built with all possible irreps (all_irreps) found in the
    # lattice Hilbert space. inv_cgcs [inv=invariant] are the Clebsch-Gordan
    # coefficients of R x Rbar -> trivial_rep. This provides a map from states
    # of R to those of Rbar (which may be R). Associated phases are signs of
    # these coefficients. (st=state, cst=conjugate state)
    all_irreps = sorted(set().union(*link_irreps.values()))
    for R in all_irreps:
        inv_cgcs = calc_cgcs([R, conjugate_iweight(R)], trivial_rep, 1, 0)
        conj_dict[R] = {st: (cst, np.sign(inv_cgcs[(st,cst)])) for st,cst in inv_cgcs}

    return link_irreps, site_singlets, conj_dict

def physical_plaquette_states(P, sites, plaquettes, singlets, FORDER) -> list[PlaquetteState]:
    """
    Generates a list of physical plaquette states for a plaquette P. P is a
    tuple (site coordinate, lattice plane), such as ((0,0,0), (1,2)). The states
    are returned as a list of nested tuples of irreps and multiplicity indices of the form:
    (l1, l2, l3, l4, (s1 ctrls), (s2 ctrls), (s3 ctrls), (s4 ctrls), s1, s2, s3, s4).
    """

    # Gather information for P.
    signature = (P[1], tuple(tuple(sorted(sites[s])) for s in plaquettes[P][2]))
    plane, half_link_dirs_per_site = signature
    unique_ctrls = plaquettes[P][-1]

    # Each plaquette site will support a singlet, which involves all links meeting
    # at that site. Those links are conventionally ordered according to FORDER,
    # so that when a singlet tensor product such as 2 x 2 x 2 x 2 in SU(2) is
    # written down, a specific link is inferred for each tensor product factor.
    # i/j_idxs gives the FORDER indices of a plaquette link for each site.
    # The i/j directions are known from the plane (i,j) the plaquette is on.
    # So i_idxs=[0,1,0,0] would mean the 0th link irrep appearing in the s1 singlet
    # belongs to the link (s1, i), the 1st link irrep appearing in the s2 singlet
    # belongs to the link (s1, i), the 0th link irrep appearing in the s3 singlet
    # belongs to the link (s4, i), and the 0th link irrep appearing in the s4 singlet
    # belongs to the link (s4, i). (Similarly for j_idxs.)
    # i/j_ctrl_idxs gives the FORDER indices of control links that are specifically
    # in the i or j directions for each site. In addition, the index of the control
    # link within the list of control links for that site is also recorded.
    # i/j_ctrl_idxs is of the form {s_k: (FORDER index, control link list index)}.
    i_idxs,j_idxs,i_ctrl_idxs,j_ctrl_idxs = [],[],{},{}
    for i in range(4):
        idx,ctrl_idx,s = 0,0,i+1
        plaq_link_dirs = [plane[0],plane[1]] if s==1 else [-plane[0],plane[1]] if s==2 else [-plane[0],-plane[1]] if s==3 else [plane[0],-plane[1]]
        for direction in FORDER:
            if direction in half_link_dirs_per_site[i]:
                if direction==plaq_link_dirs[0]:
                    i_idxs.append(idx)
                elif direction==plaq_link_dirs[1]:
                    j_idxs.append(idx)
                elif abs(direction)==plane[0]:
                    i_ctrl_idxs[s] = (idx,ctrl_idx)
                    ctrl_idx += 1
                elif abs(direction)==plane[1]:
                    j_ctrl_idxs[s] = (idx,ctrl_idx)
                    ctrl_idx += 1
                else:
                    ctrl_idx += 1
                idx += 1
            else:
                continue

    # i/j_ctrl_idxs is useful for figuring out if boundary conditions require
    # two sites' control links to have the same irrep. This can happen for
    # control links in s2 and s3, which is given by the booleans s2/3_has_BCs.
    # It can also happen in s4 in either i or j directions which are separately
    # given by booleans enforce_i/j_ctrls.
    s2_has_BCs = (1 in i_ctrl_idxs) and (2 in i_ctrl_idxs) and (i_ctrl_idxs[2][1] not in unique_ctrls[1])
    s3_has_BCs = (2 in j_ctrl_idxs) and (3 in j_ctrl_idxs) and (j_ctrl_idxs[3][1] not in unique_ctrls[2])
    enforce_i_ctrls = (3 in i_ctrl_idxs) and (4 in i_ctrl_idxs) and (i_ctrl_idxs[4][1] not in unique_ctrls[3])
    enforce_j_ctrls = (1 in j_ctrl_idxs) and (4 in j_ctrl_idxs) and (j_ctrl_idxs[4][1] not in unique_ctrls[3])

    # The rest of this function pieces together site singlets to form physical
    # plaquette states. At each site, num_links is the number of links meeting
    # at a site and hlinks is a list of directions each half-link at the site
    # extends to. X and Y are tuples of irreps, where X+Y gives some permutation
    # of a distinct singlet S. G is the multiplicity of S. By Gauss' law, irreps
    # appearing in singlets are not necessarily the true irreps on the lattice.
    # Outgoing/negative/left (half-)links are conjugated in singlets relative
    # to the actual irrep on the link. This conjugation is accounted for and,
    # as a result, "modified" singlets are pieced together through this function.
    # i/j_idxs is used to ensure plaquette link irreps of the modified singlets
    # match, and i/j_ctrl_idxs as well as the BC booleans are used to check if
    # certain control link irreps match.

    # s1
    s1 = []
    num_links = len(half_link_dirs_per_site[0])
    hlinks = sorted(half_link_dirs_per_site[0], key=lambda x: FORDER.index(x))
    for X in tqdm(singlets[num_links], desc='s1', leave=False):
        for Y in singlets[num_links][X]:
            S = X+Y
            G = singlets[num_links][X][Y]

            site = []
            for i in range(len(S)):
                if hlinks[i]<0:
                    site.append(S[i])
                else:
                    site.append(conjugate_iweight(S[i]))
            s1.append((tuple(site), G))

    # s2
    s2 = []
    num_links = len(half_link_dirs_per_site[1])
    hlinks = sorted(half_link_dirs_per_site[1], key=lambda x: FORDER.index(x))
    for X in tqdm(singlets[num_links], desc='s2', leave=False):
        for Y in singlets[num_links][X]:
            S = X+Y
            G = singlets[num_links][X][Y]
        
            site = []
            for i in range(len(S)):
                if hlinks[i]<0:
                    site.append(S[i])
                else:
                    site.append(conjugate_iweight(S[i]))

            for s,g in s1:
                if s2_has_BCs and site[i_ctrl_idxs[2][0]]!=s[i_ctrl_idxs[1][0]]:
                    continue
                elif site[i_idxs[1]]==s[i_idxs[0]]:
                    s2.append(((s,g), (tuple(site),G)))
                else:
                    continue
    del s1

    # s3
    s3 = []
    num_links = len(half_link_dirs_per_site[2])
    hlinks = sorted(half_link_dirs_per_site[2], key=lambda x: FORDER.index(x))
    for X in tqdm(singlets[num_links], desc='s3', leave=False):
        for Y in singlets[num_links][X]:
            S = X+Y
            G = singlets[num_links][X][Y]
        
            site = []
            for i in range(len(S)):
                if hlinks[i]<0:
                    site.append(S[i])
                else:
                    site.append(conjugate_iweight(S[i]))

            for site1,site2 in s2:
                s,g = site2
                if s3_has_BCs and site[j_ctrl_idxs[3][0]]!=s[j_ctrl_idxs[2][0]]:
                    continue
                elif site[j_idxs[2]]==s[j_idxs[1]]:
                    s3.append((site1, site2, (tuple(site),G)))
                else:
                    continue
    del s2

    # s4
    s4 = []
    num_links = len(half_link_dirs_per_site[3])
    hlinks = sorted(half_link_dirs_per_site[3], key=lambda x: FORDER.index(x))
    for X in tqdm(singlets[num_links], desc='s4', leave=False):
        for Y in singlets[num_links][X]:
            S = X+Y
            G = singlets[num_links][X][Y]
        
            site = []
            for i in range(len(S)):
                if hlinks[i]<0:
                    site.append(S[i])
                else:
                    site.append(conjugate_iweight(S[i]))

            for site1,site2,site3 in s3:
                s,g1 = site1
                r,g3 = site3
                if (enforce_i_ctrls and site[i_ctrl_idxs[4][0]]!=r[i_ctrl_idxs[3][0]]) or (enforce_j_ctrls and site[j_ctrl_idxs[4][0]]!=s[j_ctrl_idxs[1][0]]):
                    continue
                elif site[i_idxs[3]]==r[i_idxs[2]] and site[j_idxs[3]]==s[j_idxs[0]]:
                    s4.append((site1, site2, site3, (tuple(site),G)))
                else:
                    continue
    del s3

    # Finally, all possible physical plaquette states are found by looping
    # over all multiplicity indices, which are 0-indexed.
    states = []
    for site1,site2,site3,site4 in tqdm(s4, desc='Unpack', leave=False):
        S1,G1 = site1
        S2,G2 = site2
        S3,G3 = site3
        S4,G4 = site4
        site_data = [S1,S2,S3,S4]
        ctrls = [tuple(site_data[i][j] for j in range(len(half_link_dirs_per_site[i])) if j not in [i_idxs[i], j_idxs[i]]) for i in range(4)]
        for g1,g2,g3,g4 in product(*(range(G) for G in [G1,G2,G3,G4])):
            states.append((S1[i_idxs[0]], S2[j_idxs[1]], S3[i_idxs[2]], S4[j_idxs[3]], ctrls[0], ctrls[1], ctrls[2], ctrls[3], g1, g2, g3, g4))
    del s4

    return states

def compute_plaquette_signature(
        plaquette_address: PlaquetteAddress, lattice: LatticeDef
) -> PlaquetteSignature:
    """
    Obtain the 'signature' associated with the plaquette defined by plaquette_address.

    The argument plaquette_address is a 2-tuple whose first element is a site coordinate,
    and whose second element is a plane (defined as a 2-tuple of sorted, positive link directions).
    In the given plane, the site coordinate is the "bottom-left" vertex of the plaquette.

    A plaquette's signature captures the notion of whether a plaquette is on the edge, interior, or corner
    of a lattice. This is relevant because that information (along with plane and FORDER) are necessary to
    unambiguously compute matrix elements of Wilson loops. As a convenience, the half links at each site
    appearing in the signature are sorted according to the value of the FORDER property on lattice.

    If it is not possible to form a plaquette in the requested plane, then the returned PlaquetteSignature
    will have an empty tuple as its second element (which otherwise gives half-link data per site). This can
    occur when requesting a plaquette signature on the boundaries of a non-periodic lattice direction.
    """
    site_coordinate, plane = plaquette_address
    if (site_coordinate not in lattice.sites) or (plane not in lattice.planes):
        raise KeyError(f"Plaquette address {plaquette_address} not found in lattice {lattice}.")
    
    if plaquette_address in lattice.plaquettes.keys():
        _, _, plaquette_site_coordinates, _ = lattice.plaquettes[plaquette_address]
        sites_with_half_links = tuple(
            tuple(sorted(lattice.sites[site_coordinate], key=lambda x: lattice.FORDER.index(x)))
            for site_coordinate in plaquette_site_coordinates)
    else:
        sites_with_half_links = () # Plaquette doesn't exist, return empty tuple.

    signature = (plane, sites_with_half_links)
    return signature
