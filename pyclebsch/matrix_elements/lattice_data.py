"""
The geometry of a cubic lattice, and the physical states of its plaquettes.

Geometry. LatticeDef(num_sites, PBCs, FORDER) describes a lattice with three
axes, numbered 1, 2 and 3, each open or periodic. A lattice with fewer than
three dimensions has a single site along each axis it does not use, and that
axis must be open: num_sites=(3, 2, 1) with PBCs=(False, False, False) is a
3x2 lattice along axes 1 and 2. A LatticeDef cannot be modified. It provides
the lattice's sites, links, plaquettes and cubes, and, with orientation signs,
the boundary of each plaquette and cube and the coboundary of each link and
plaquette. sites_links_and_plaquettes computes the sites, links and
plaquettes without a LatticeDef. The conventions:
- A site is a tuple of three coordinates, each counted from 0. A direction is
  1, 2 or 3 for the positive direction along that axis, and -1, -2 or -3 for
  the negative one.
- A link's address (s, d) names the link from site s to its neighbor in the
  positive direction d. The link has a half-link at each end: an outgoing
  one at s, in direction d, and an incoming one at the neighbor, in
  direction -d.
- FORDER, a permutation of (1, 2, 3, -1, -2, -3), is a conventional order
  of the half-links at every site (P. Balaji et al., arXiv:2509.25865). It
  assigns each irrep in a site's singlet, and so in its Clebsch-Gordan
  coefficients (calc_cgcs), to one of the site's links, which is otherwise
  ambiguous when two links carry the same irrep. It also orders the control
  links in a plaquette state.
- A plaquette's address (s1, (i, j)), with i < j, names the plaquette in
  plane (i, j) whose other corners are reached from s1 by steps in the
  positive directions i and j, wrapping around on periodic axes.
  PlaquetteGeometry labels its sites s1-s4 and its links l1-l4. The other
  links at its sites are its control links.

Physical plaquette states. Irreps of SU(N) are labeled by i-weights
(IrrepWeight, from su_n_operators), and the basis states of an irrep are
numbered as in calc_cgcs. irreps_and_singlets gives, for a truncation, the
irreps allowed on links and the singlets that Gauss's law allows at a site
where a given number of links meet. The truncation is a mode and a cutoff:
mode "T" keeps the irreps whose i-weight has first entry at most the cutoff;
mode "C" keeps the irreps whose quadratic Casimir is at most the cutoff;
mode "B" keeps the same irreps as "C", and keeps only the singlets whose
irreps have quadratic Casimirs that sum to at most the cutoff.
physical_plaquette_states lists a plaquette's physical states: the irreps
on its links and control links, and the multiplicity index of the singlet
at each of its sites (numbered from 0; calc_cgcs's mult_idx, from 1, is one
more), that together satisfy Gauss's law at all four sites
(PlaquetteState).

Within one lattice, plaquettes with the same signature
(compute_plaquette_signature: the plane, and the half-links at each site)
have the same physical states.
plaquettes_by_signature groups a lattice's plaquettes by signature, and
physical_plaquette_states_by_signature computes the states once per
signature and caches them in memory (see pyclebsch.cache). For example,

    lattice = LatticeDef(
        num_sites=(3, 2, 1),
        PBCs=(False, False, False),
        FORDER=(1, 2, 3, -1, -2, -3),
    )
    states = physical_plaquette_states_by_signature(
        lattice, N=3, truncation_mode="T", cutoff=1
    )

maps each of the two signatures of this lattice, one per plaquette, to its
9 physical states.
"""

import functools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
from tqdm import tqdm
from .helpers import *
from ..cache import _plaquette_cache
from ..cgc import calc_cgcs
from collections import defaultdict
from ..su_n_operators import calc_casimir, find_direct_sum, IrrepWeight
from more_itertools import distinct_permutations
from itertools import product, combinations_with_replacement

type SiteMultiplicityIndex = int
type SiteControlLinks = tuple[IrrepWeight, ...]
type ActiveLink = IrrepWeight
# The SiteSinglets returned by irreps_and_singlets take the form
# of a dictionary whose keys are the total number of links
# meeting at a site. The values of SiteSinglets are themselves
# dictionaries which break down the possible singlets according
# to control sector: the key at the first nesting-level down
# is the tuple SiteControlLinks consisting of i-weights on
# the control links (with the caveat that outgoing links
# have had their i-weight conjugated). The value is a second
# nested level of dictionary whose keys are all the active
# link configurations that are compatible with the current
# control sector (again with conjugation of outgoing link irreps),
# and whose value is the number of physically distinct singlets
# corresponding to the current control sector + active link
# combination.
type SiteSinglets = dict[int, dict[SiteControlLinks, dict[tuple[ActiveLink, ActiveLink], int]]]
# For each irrep R returned by irreps_and_singlets, {R: {m: (m', sign)}}:
# the singlet in R x conj(R) is a sum of products |m>|m'>, with m a basis
# state of R and m' a basis state of conj(R), in which each m appears in
# exactly one product. Each m maps to its partner m' and to the sign (1.0 or
# -1.0) of that product's Clebsch-Gordan coefficient. Basis states are
# numbered as in calc_cgcs: Gelfand-Tsetlin patterns, highest weight first,
# starting from 0.
type ConjugateStates = dict[IrrepWeight, dict[int, tuple[int, float]]]
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
type SiteHalfLinks = tuple[LinkDirection, ...]
type LinkAddress = tuple[SiteCoordinate, LinkDirection]
type PlaquetteActiveLinkAddresses = tuple[LinkAddress, LinkAddress, LinkAddress, LinkAddress]
type PlaquetteControlLinkAddresses = tuple[
    tuple[LinkAddress, ...], tuple[LinkAddress, ...], tuple[LinkAddress, ...], tuple[LinkAddress, ...]
]
type PlaquetteSiteCoordinates = tuple[SiteCoordinate, SiteCoordinate, SiteCoordinate, SiteCoordinate]
type PlaquetteUniqueControls = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
type PlaquetteAddress = tuple[SiteCoordinate, Plane]
type PlaquetteSignature = tuple[
    Plane, tuple[SiteHalfLinks, SiteHalfLinks, SiteHalfLinks, SiteHalfLinks] | tuple[()]
] # TODO make a class to enforce forder?
type CubeAddress = SiteCoordinate
type OrientationSign = Literal[1, -1]

# For each plane (i, j), the axis k normal to it and the sign of the
# plaquette's normal e_i x e_j along e_k: e_1 x e_2 = +e_3,
# e_1 x e_3 = -e_2, e_2 x e_3 = +e_1.
_PLANE_NORMALS: dict[Plane, tuple[int, OrientationSign]] = {(1, 2): (3, 1), (1, 3): (2, -1), (2, 3): (1, 1)}


class PlaquetteGeometry(NamedTuple):
    """
    The links and sites of one plaquette, as stored in LatticeDef.plaquettes.

    The plaquette's links and sites are labeled
    s4 -- l3 -- s3
     |          |
    l4          l2
     |          |
    s1 -- l1 -- s2
    where, for a plaquette in plane (i, j), l1 points along +i from s1 and
    l4 along +j from s1.
    """

    active_links: PlaquetteActiveLinkAddresses
    """The links l1, l2, l3, l4."""
    control_links: PlaquetteControlLinkAddresses
    """For each site s1, s2, s3, s4, its other links, in the F-order of their half-links at the site."""
    site_coordinates: PlaquetteSiteCoordinates
    """The coordinates of s1, s2, s3, s4."""
    unique_controls: PlaquetteUniqueControls
    """
    For each site, the indices into its control links of the links that are
    not a control link of an earlier site, so that the plaquette's distinct
    control links can be found without counting one twice. For example, if
    the control link at index 0 of s1 is also a control link of s2, then
    the indices for s1 include 0, and the index of that link among s2's
    control links is omitted from the indices for s2. A control link is
    shared by two sites only along an edge of the plaquette whose axis is
    periodic with length 2.
    """


@dataclass(frozen=True)
class LatticeDef:
    """
    Class for defining the geometry of a lattice.

    A LatticeDef cannot be modified: its fields are stored as tuples and
    cannot be assigned, and everything it computes from them is built from
    immutable values: frozendicts, tuples, integers and PlaquetteGeometry. So
    the data cannot become inconsistent with the lattice it describes. A
    mutable copy of a property is the caller's to make: dict(lattice.sites)
    is a plain dict, and for the properties whose values are frozendicts,
    {key: dict(value) for key, value in lattice.plaquette_boundaries.items()}
    is a plain dict of plain dicts. (copy.copy and copy.deepcopy return
    frozendicts.)
    """

    num_sites: tuple[int, int, int]
    PBCs: tuple[bool, bool, bool]
    FORDER: tuple[LinkDirection, LinkDirection, LinkDirection, LinkDirection, LinkDirection, LinkDirection] | list[LinkDirection]

    def __post_init__(self):
        # Store lists given for the fields as tuples. object.__setattr__ is
        # how a frozen dataclass sets its own fields after __init__.
        object.__setattr__(self, "num_sites", tuple(self.num_sites))
        object.__setattr__(self, "PBCs", tuple(self.PBCs))
        object.__setattr__(self, "FORDER", tuple(self.FORDER))

    @functools.cached_property
    def planes(self) -> tuple[Plane] | tuple[Plane, Plane, Plane]:
        if any(n < 1 for n in self.num_sites):
            raise ValueError(
                f"Malformed lattice encountered while attempting to compute plane. num_sites should consist of positive integers but encountered the tuple '{self.num_sites}'."
            )
        possible_planes = [(1, 2), (1, 3), (2, 3)]
        return tuple(filter(self._plane_exists, possible_planes))

    # sites_links_and_plaquettes computes all three together, so they are
    # cached together here, and sites, links and plaquettes are plain
    # properties that read from this cache.
    @functools.cached_property
    def _sites_links_and_plaquettes(self) -> tuple[
            frozendict[SiteCoordinate, SiteHalfLinks],
            frozendict[LinkAddress, tuple[SiteCoordinate, SiteCoordinate]],
            frozendict[PlaquetteAddress, PlaquetteGeometry]]:
        sites, links, plaquettes = sites_links_and_plaquettes(self.num_sites, self.PBCs, self.FORDER)
        return frozendict(sites), frozendict(links), frozendict(plaquettes)

    @property
    def sites(self) -> frozendict[SiteCoordinate, SiteHalfLinks]:
        """
        The half-links connected to each lattice site, as the directions they
        extend in.
        """
        return self._sites_links_and_plaquettes[0]

    @property
    def links(self) -> frozendict[LinkAddress, tuple[SiteCoordinate, SiteCoordinate]]:
        """
        The start and stop site coordinates associated with links in the lattice.
        """
        return self._sites_links_and_plaquettes[1]

    @property
    def plaquettes(self) -> frozendict[PlaquetteAddress, PlaquetteGeometry]:
        """
        The plaquettes of the lattice: for each plaquette address (the
        coordinates of its site s1 and its plane), a PlaquetteGeometry with its
        active_links, control_links, site_coordinates and unique_controls. See
        PlaquetteGeometry for the labels s1-s4 and l1-l4 and what each field
        holds.
        """
        return self._sites_links_and_plaquettes[2]

    @functools.cached_property
    def cubes(self) -> tuple[CubeAddress, ...]:
        """
        The cubes of the lattice, as a tuple of cube addresses in lexicographic
        order.

        A cube's address is its corner c from which the other seven corners are
        reached by steps in the positive axis directions (c + e_1, c + e_2, ...,
        c + e_1 + e_2 + e_3), with coordinates wrapping around on periodic axes.
        A plaquette's address uses its site s1 in the same way. Along a periodic
        axis, c can be any coordinate; along an open axis, any but the last. So
        a lattice with an open axis of a single site has no cubes.
        """
        x_corners, y_corners, z_corners = (
            range(n) if periodic else range(n - 1) for n, periodic in zip(self.num_sites, self.PBCs)
        )
        return tuple(product(x_corners, y_corners, z_corners))

    @functools.cached_property
    def plaquette_boundaries(self) -> frozendict[PlaquetteAddress, frozendict[LinkAddress, OrientationSign]]:
        """
        The boundary of each plaquette: its four links, each with its
        orientation sign in the plaquette, +1 for l1 and l2 and -1 for l3 and
        l4 (see PlaquetteGeometry for the labels). The loop s1 -> s2 -> s3 -> s4
        -> s1 traverses l1 and l2 along their directions and l3 and l4
        against them, so the boundary is l1 + l2 - l3 - l4. This is the
        direction of the Wilson loop whose matrix elements
        calc_plaquette_elements computes.
        """
        return frozendict(
            (plaquette_address, frozendict(zip(plaquette_geometry.active_links, (1, 1, -1, -1))))
            for plaquette_address, plaquette_geometry in self.plaquettes.items()
        )

    @functools.cached_property
    def link_coboundaries(self) -> frozendict[LinkAddress, frozendict[PlaquetteAddress, OrientationSign]]:
        """
        The coboundary of each link: the plaquettes that contain it, each with
        the link's sign in that plaquette's boundary (see plaquette_boundaries).
        Every link is a key; a link in no plaquette has an empty frozendict.
        """
        return self._coboundary(self.links, self.plaquette_boundaries)

    @functools.cached_property
    def cube_boundaries(self) -> frozendict[CubeAddress, frozendict[PlaquetteAddress, OrientationSign]]:
        """
        The boundary of each cube: its six faces, each with the sign of the
        face's normal e_i x e_j (for a face in plane (i, j)) along the cube's
        outward normal there.

        A cube with address c has, for each plane with normal axis k,
        a lower face at c, where the outward normal is -e_k, and an upper face
        at c + e_k, where it is +e_k. On a periodic axis the upper face's
        address wraps around. With these signs, the boundary of each cube's
        boundary is zero.
        """
        boundaries = {}
        for cube_address in self.cubes:
            face_signs = {}
            for plane, (normal_axis, normal_sign) in _PLANE_NORMALS.items():
                # The upper face's site s1 is c + e_k. On an open axis, c's
                # coordinate is at most L - 2, so c + e_k is inside the lattice
                # and the modulus changes only a periodic axis's coordinate.
                upper_face_s1 = list(cube_address)
                upper_face_s1[normal_axis - 1] = (cube_address[normal_axis - 1] + 1) % self.num_sites[normal_axis - 1]
                face_signs[(cube_address, plane)] = -normal_sign
                face_signs[(tuple(upper_face_s1), plane)] = normal_sign
            boundaries[cube_address] = frozendict(face_signs)
        return frozendict(boundaries)

    @functools.cached_property
    def plaquette_coboundaries(self) -> frozendict[PlaquetteAddress, frozendict[CubeAddress, OrientationSign]]:
        """
        The coboundary of each plaquette: the cubes that contain it, each with
        the plaquette's sign in that cube's boundary (see cube_boundaries).
        Every plaquette is a key; a plaquette in no cube has an empty
        frozendict. A plaquette in two cubes has opposite signs in them.
        """
        return self._coboundary(self.plaquettes, self.cube_boundaries)

    @staticmethod
    def _coboundary(cells: Iterable, higher_cell_boundaries: Mapping[Any, Mapping]) -> frozendict:
        """
        Invert higher_cell_boundaries to get the coboundary of each of cells.

        cells holds the addresses of every k-cell of the lattice.
        higher_cell_boundaries is a mapping whose keys are the addresses of
        every (k+1)-cell, and whose value for each is that cell's boundary: a
        mapping from the addresses of the k-cells on it to their orientation
        signs. The result maps each k-cell's address to its coboundary: a
        frozendict from the addresses of the (k+1)-cells that contain it to its
        sign in each of their boundaries. A k-cell that no (k+1)-cell contains
        maps to an empty frozendict.

        Example: _coboundary(links, plaquette_boundaries) maps each link to the
        plaquettes that contain it, with signs. That is link_coboundaries.
        """
        coboundary = {cell: {} for cell in cells}
        for higher_cell, signs in higher_cell_boundaries.items():
            for cell, sign in signs.items():
                coboundary[cell][higher_cell] = sign
        return frozendict((cell, frozendict(signs)) for cell, signs in coboundary.items())

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
        dict[PlaquetteAddress, PlaquetteGeometry]
]:
    """
    Sets up sites, links, and plaquettes of the cubic lattice.
    Sites are returned as a dictionary whose keys are site coordinates and whose
    values are tuples of directions along the axes that a site's half-links extend to.
    Links are returned as keys of a dictionary whose values are
    (initial site, final site).
    Plaquettes are returned as keys of a dictionary whose values are
    PlaquetteGeometry: the plaquette's links, the control links at each of its
    sites, its sites, and the unique controls at each site (the indices of
    the control links that are not a control link of an earlier site).
    Everything in the values is a tuple; the three dictionaries are new on
    every call.
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
                        ctrl_links.append(tuple(clinks)), unique_ctrls.append(tuple(uniques))
                    plaquettes[(s1, (i+1,j+1))] = PlaquetteGeometry(tuple([l1,l2,l3,l4]), tuple(ctrl_links), tuple([s1,s2,s3,s4]), tuple(unique_ctrls))

    # Cast the values of sites to tuples so that a user cannot modify
    # the data inside a LatticeDef.
    sites = {site: tuple(half_link_dirs) for site, half_link_dirs in sites.items()}
    return sites, links, plaquettes


def irreps_and_singlets(
        N: int,
        sites: Mapping[SiteCoordinate, SiteHalfLinks],
        truncation_mode: str,
        cutoff: float,
) -> tuple[dict[int, list[IrrepWeight]], SiteSinglets, ConjugateStates]:
    """
    Sets up irreps and singlets of the truncated Hilbert space.

    Returns (link_irreps, site_singlets, conj_dict). Due to boundary
    conditions, different sites' links and singlets may have different possible
    irreps, so link_irreps and site_singlets are keyed by the number n of links
    that meet at a site (2 to 6 on a lattice with plaquettes).
    link_irreps[n] lists the irreps that appear in at least one singlet at a
    site where n links meet. site_singlets is a SiteSinglets and conj_dict a
    ConjugateStates; the comments on those type aliases describe them.
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

def physical_plaquette_states(
        P: PlaquetteAddress,
        sites: Mapping[SiteCoordinate, SiteHalfLinks],
        plaquettes: Mapping[PlaquetteAddress, PlaquetteGeometry],
        singlets: SiteSinglets,
        FORDER: Sequence[LinkDirection]
) -> list[PlaquetteState]:
    """
    Generates a list of physical plaquette states for a plaquette P. P is a
    tuple (site coordinate, lattice plane), such as ((0,0,0), (1,2)). The states
    are returned as a list of nested tuples of irreps and multiplicity indices of the form:
    (l1, l2, l3, l4, (s1 ctrls), (s2 ctrls), (s3 ctrls), (s4 ctrls), s1, s2, s3, s4).
    """

    # Gather information for P.
    signature = (P[1], tuple(tuple(sorted(sites[s])) for s in plaquettes[P].site_coordinates))
    plane, half_link_dirs_per_site = signature
    unique_ctrls = plaquettes[P].unique_controls

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
    # Outgoing (positive-direction) half-links are conjugated in singlets
    # relative to the actual irrep on the link. This conjugation is accounted
    # for and, as a result, "modified" singlets are pieced together through
    # this function.
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
        sites_with_half_links = tuple(
            tuple(sorted(lattice.sites[site_coordinate], key=lambda x: lattice.FORDER.index(x)))
            for site_coordinate in lattice.plaquettes[plaquette_address].site_coordinates)
    else:
        sites_with_half_links = () # Plaquette doesn't exist, return empty tuple.

    signature = (plane, sites_with_half_links)
    return signature

def plaquettes_by_signature(lattice: LatticeDef) -> dict[PlaquetteSignature, list[PlaquetteAddress]]:
    """
    Every plaquette of the lattice, grouped by signature (see compute_plaquette_signature).

    Each list of plaquette addresses is sorted, and the groups are ordered by
    their smallest addresses.

    Within one lattice, plaquettes with the same signature have the same
    physical plaquette states and the same matrix elements. Those depend on a
    plaquette's plane, the half-links at its corners, its unique controls,
    and inputs shared by the whole lattice (N, the truncation, FORDER, EPS,
    PRES), but not on its position. The signature holds the plane and the
    half-links, and it also determines the unique controls, for this reason.
    Controls at two corners of a plaquette are the same link only along an
    edge of the plaquette whose axis is periodic with length 2: in plane
    (1, 2), the -1 control of corner 1 starts at s1 - e1 and the +1 control
    of corner 2 at s1 + e1, the same site exactly when axis 1 is periodic
    with length 2. Which corners have such controls is in the signature, and
    the length and periodicity of an axis are the same for every plaquette.
    The two controls of one corner along an axis normal to the plane start at
    s and s - e, which coincide only on a periodic axis of length 1. The
    grouping relies on sites_links_and_plaquettes refusing such axes.
    """
    groups: dict[PlaquetteSignature, list[PlaquetteAddress]] = defaultdict(list)
    for plaquette in sorted(lattice.plaquettes):
        groups[compute_plaquette_signature(plaquette, lattice)].append(plaquette)
    return dict(groups)

def _physical_plaquette_states_cache_key(
        lattice: LatticeDef, plaquette: PlaquetteAddress, N: int, truncation_mode: str, cutoff: float
) -> tuple:
    """
    The key under which the physical states of plaquette are cached (in
    pyclebsch.cache._PlaquetteCache.physical_states): (signature, unique
    controls, N, truncation mode, cutoff, FORDER), all of them hashable.

    The key leaves out the plaquette's position and the lattice's size and
    boundary conditions: those affect the states only through the signature
    (an open boundary removes half-links) and the unique controls (a
    periodic axis of length 2 makes controls coincide). The unique controls
    are needed across lattices: every plaquette of the periodic 2x2x2 and
    3x3x3 lattices has the same signature, but on 2x2x2 some of its controls
    coincide.
    """
    return (
        compute_plaquette_signature(plaquette, lattice),
        lattice.plaquettes[plaquette].unique_controls,
        N,
        truncation_mode,
        cutoff,
        lattice.FORDER,
    )

def physical_plaquette_states_by_signature(
        lattice: LatticeDef, N: int, truncation_mode: str, cutoff: float
) -> dict[PlaquetteSignature, list[PlaquetteState]]:
    """
    physical_plaquette_states for each plaquette signature of the lattice,
    computed once per signature (see plaquettes_by_signature for why that
    suffices).

    The singlets are those of irreps_and_singlets(N, lattice.sites,
    truncation_mode, cutoff), and FORDER is the lattice's. Each signature's
    states are computed on the smallest plaquette address with that
    signature, and hold for every plaquette with it.

    The states are cached in memory across calls, keyed by (signature,
    unique controls, N, truncation mode, cutoff, FORDER);
    _physical_plaquette_states_cache_key says why. clear_memory_cache()
    clears the cache and set_physical_plaquette_states_cache_limit() caps
    it. Each returned list is a new copy of the cached one, so the caller
    may modify it. The copy is shallow; the states in it are tuples and
    cannot be modified.
    """
    # irreps_and_singlets is only needed, and only computed, on a cache miss.
    truncation_data = functools.cache(lambda: irreps_and_singlets(N, lattice.sites, truncation_mode, cutoff))

    def compute(plaquette: PlaquetteAddress) -> list[PlaquetteState]:
        _, singlets, _ = truncation_data()
        return physical_plaquette_states(plaquette, lattice.sites, lattice.plaquettes, singlets, lattice.FORDER)

    states = {}
    for signature, plaquettes in plaquettes_by_signature(lattice).items():
        states[signature] = list(_plaquette_cache.physical_states.get_or_compute(
            _physical_plaquette_states_cache_key(lattice, plaquettes[0], N, truncation_mode, cutoff),
            functools.partial(compute, plaquettes[0]),
        ))
    return states
