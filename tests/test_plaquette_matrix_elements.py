"""Tests for plaquette matrix element computation on a small PBC d=2 lattice.

Lattice: SU(3), num_sites=[2,2,1], PBCs=[True,True,False], T-truncation cutoff 1.
This is the smallest lattice with non-empty matrix elements for every plaquette
and exercises the full glueing pipeline (4 plaquettes, all interior under PBCs).

A second group of tests at the bottom of this module covers open boundary
conditions with B truncation, where the four vertices of a plaquette can carry
different numbers of half-links and hence different sets of allowed irreps.
"""

import math
from itertools import product

import numpy as np
import pytest

import pyclebsch.cgc as cgc
from pyclebsch.matrix_elements.lattice_data import (
    irreps_and_singlets,
    physical_plaquette_states,
    sites_links_and_plaquettes,
)
from pyclebsch.matrix_elements.plaquette_matrix_elements import (
    calc_plaquette_elements,
    calc_plaquette_site_factors,
)
from pyclebsch.su_n_operators import calc_dimension

FORDER = [1, 2, 3, -1, -2, -3]
N_COLORS = 3
EPS = 1e-10
PRES = 10
NUM_SITES = [2, 2, 1]
PBCS = [True, True, False]
TRUNCATION_MODE = "T"
CUTOFF = 1


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    """Redirect the on-disk CGC cache to a tmp dir for this module."""
    original = cgc.get_cache_dir()
    cgc.set_cache_dir(tmp_path_factory.mktemp("CGC_Data"))
    yield
    cgc.set_cache_dir(original)


@pytest.fixture(scope="module")
def lattice_objects(cache_dir):
    """Build sites/plaquettes and the truncation/singlet/conjugation data."""
    sites, _links, plaquettes = sites_links_and_plaquettes(NUM_SITES, PBCS, FORDER)
    truncation_irreps, singlets, conj_dict = irreps_and_singlets(
        N_COLORS, sites, TRUNCATION_MODE, CUTOFF
    )
    return sites, plaquettes, truncation_irreps, singlets, conj_dict


@pytest.fixture(scope="module")
def matrix_elements_sequential(lattice_objects):
    """Compute matrix elements once (parallelize=False) for every plaquette."""
    sites, plaquettes, truncation_irreps, singlets, conj_dict = lattice_objects
    results = {}
    for P in sorted(plaquettes.keys()):
        results[P] = calc_plaquette_elements(
            N_COLORS, P, sites, plaquettes,
            truncation_irreps, singlets, conj_dict,
            FORDER, EPS, PRES, parallelize=False,
        )
    return results


def test_calc_plaquette_elements_shape(matrix_elements_sequential):
    """Every plaquette yields a non-empty dict of (Pf, Pi) -> float.

    Each plaquette state is a 12-tuple: 4 active-link i-weights (3-tuples),
    4 control-link tuples (each containing i-weights), and 4 integer
    multiplicity indices.
    """
    assert len(matrix_elements_sequential) > 0
    for P, mat_elems in matrix_elements_sequential.items():
        assert len(mat_elems) > 0, f"Plaquette {P} produced no matrix elements"
        for (Pf, Pi), value in mat_elems.items():
            assert len(Pf) == 12, f"Pf has length {len(Pf)}, expected 12"
            assert len(Pi) == 12, f"Pi has length {len(Pi)}, expected 12"

            for active_link in tuple(Pf[:4]) + tuple(Pi[:4]):
                assert isinstance(active_link, tuple)
                assert len(active_link) == N_COLORS

            for site_ctrls in tuple(Pf[4:8]) + tuple(Pi[4:8]):
                assert isinstance(site_ctrls, tuple)
                for ctrl in site_ctrls:
                    assert isinstance(ctrl, tuple)
                    assert len(ctrl) == N_COLORS

            for g in tuple(Pf[8:]) + tuple(Pi[8:]):
                assert isinstance(g, int)

            assert isinstance(value, float)


def test_calc_plaquette_elements_parallelize_consistency(
    lattice_objects, matrix_elements_sequential
):
    """parallelize=True must produce the same dict as parallelize=False.

    The sequential fixture has already populated the on-disk CGC cache, so the
    multiprocessing workers will read from cache rather than recompute (avoiding
    the parallel-pickle race noted in SKILL.md).
    """
    sites, plaquettes, truncation_irreps, singlets, conj_dict = lattice_objects
    for P in sorted(plaquettes.keys()):
        result_par = calc_plaquette_elements(
            N_COLORS, P, sites, plaquettes,
            truncation_irreps, singlets, conj_dict,
            FORDER, EPS, PRES, parallelize=True,
        )
        result_seq = matrix_elements_sequential[P]
        assert set(result_par.keys()) == set(result_seq.keys()), (
            f"Plaquette {P}: parallel/sequential key sets differ."
        )
        for key, v_seq in result_seq.items():
            v_par = result_par[key]
            assert math.isclose(v_seq, v_par, abs_tol=1e-10), (
                f"Plaquette {P} key {key}: sequential={v_seq} parallel={v_par}"
            )


# ---------------------------------------------------------------------------
# Open boundary conditions with B truncation.
#
# Under open BCs the four vertices of a plaquette need not carry the same
# number of half-links: a corner site of a d=2 lattice has 2, an edge site 3,
# an interior site 4. B truncation bounds the sum of Casimirs of the irreps
# meeting at a site, so the set of allowed irreps -- and therefore the set of
# active-link transitions that can be dressed into a singlet -- differs from
# vertex to vertex. A transition of the shared link l1 that exists at s1 can
# then be impossible at s2, and likewise for l2 between s2 and s3. The glueing
# code must treat such a mismatch as a vanishing contribution rather than
# failing to look the site factor up.
# ---------------------------------------------------------------------------

OPEN_BC_CASES = [
    # (num_sites, PBCs, cutoff, plaquette address). Both plaquettes below sit
    # against a lattice corner; the first has a 2-half-link vertex at s2, the
    # second at s3, which exercises both link-matching lookups in the glueing.
    pytest.param(
        [3, 2, 1], [False, False, False], 6, ((1, 0, 0), (1, 2)),
        id="3x2x1_OBC_B6_corner_at_s2",
    ),
    pytest.param(
        [2, 3, 1], [False, False, False], 6, ((0, 1, 0), (1, 2)),
        id="2x3x1_OBC_B6_corner_at_s3",
    ),
]


def _lattice_objects_for(num_sites, PBCs, cutoff):
    """Build sites/plaquettes plus B-truncation singlet data for one case."""
    sites, _links, plaquettes = sites_links_and_plaquettes(num_sites, PBCs, FORDER)
    truncation_irreps, singlets, conj_dict = irreps_and_singlets(
        N_COLORS, sites, "B", cutoff
    )
    return sites, plaquettes, truncation_irreps, singlets, conj_dict


def _brute_force_plaquette_elements(P, sites, plaquettes, truncation_irreps,
                                    singlets, conj_dict):
    """Reference implementation of the site factor glueing.

    Enumerates every quadruple of site factors whose shared active-link irreps
    agree, and keeps the quadruple when the plaquette states it spells out are
    both physical. Boundary conditions are therefore imposed by
    ``physical_plaquette_states`` rather than by the control-link index
    bookkeeping used in ``glue_plaquette_site_factors``, which makes this an
    independent check of that bookkeeping. Only usable on small lattices: the
    enumeration is quartic in the number of site factors.
    """
    site_factors = calc_plaquette_site_factors(
        N_COLORS, P, sites, plaquettes, truncation_irreps, singlets,
        conj_dict, FORDER, EPS, parallelize=False,
    )
    if len(site_factors) == 0:
        return {}
    physical = set(physical_plaquette_states(P, sites, plaquettes, singlets, FORDER))
    dims = {R: calc_dimension(R) for R in set().union(*truncation_irreps.values())}

    # REFERENCE: site factor key is (Rii, Rji, Rif, Rjf, Gi, Gf, ctrl_irreps)
    #                                 0    1    2    3   4   5       6
    matrix_elements = {}
    for s1, s2 in product(site_factors[1], site_factors[2]):
        if (s1[0], s1[2]) != (s2[0], s2[2]):        # shared link l1
            continue
        for s3 in site_factors[3]:
            if (s2[1], s2[3]) != (s3[1], s3[3]):    # shared link l2
                continue
            for s4 in site_factors[4]:
                if (s3[0], s3[2]) != (s4[0], s4[2]):    # shared link l3
                    continue
                if (s1[1], s1[3]) != (s4[1], s4[3]):    # shared link l4
                    continue
                Pf = (s1[2], s3[3], s3[2], s1[3],
                      s1[6], s2[6], s3[6], s4[6],
                      s1[5], s2[5], s3[5], s4[5])
                Pi = (s1[0], s3[1], s3[0], s1[1],
                      s1[6], s2[6], s3[6], s4[6],
                      s1[4], s2[4], s3[4], s4[4])
                if Pf not in physical or Pi not in physical:
                    continue
                val = (site_factors[1][s1] * site_factors[2][s2]
                       * site_factors[3][s3] * site_factors[4][s4])
                if abs(val) < EPS:
                    continue
                dim1 = dims[s1[0]] * dims[s1[1]] / dims[s1[2]] / dims[s1[3]]
                dim3 = dims[s3[0]] * dims[s3[1]] / dims[s3[2]] / dims[s3[3]]
                matrix_elements[(Pf, Pi)] = round(np.sqrt(dim1 * dim3) * val, PRES)

    return matrix_elements


@pytest.mark.parametrize("num_sites,PBCs,cutoff,P", OPEN_BC_CASES)
def test_open_bc_b_truncation_glueing_succeeds(cache_dir, num_sites, PBCs, cutoff, P):
    """Vertices with unequal half-link counts must not break the glueing.

    Regression test: the glueing used to raise KeyError when an active-link
    transition present at one vertex had no counterpart at the next one.
    """
    sites, plaquettes, truncation_irreps, singlets, conj_dict = _lattice_objects_for(
        num_sites, PBCs, cutoff
    )
    half_link_counts = [len(sites[s]) for s in plaquettes[P][2]]
    assert len(set(half_link_counts)) > 1, (
        f"Case {num_sites} {P} is meant to have vertices with differing "
        f"half-link counts, got {half_link_counts}."
    )

    mat_elems = calc_plaquette_elements(
        N_COLORS, P, sites, plaquettes, truncation_irreps, singlets,
        conj_dict, FORDER, EPS, PRES, parallelize=False,
    )

    assert len(mat_elems) > 0, f"Plaquette {P} produced no matrix elements"

    # Every state label must be a physical plaquette state (subset rather than
    # equality because vanishing matrix elements are not stored).
    physical = set(physical_plaquette_states(P, sites, plaquettes, singlets, FORDER))
    labels = {state for key in mat_elems for state in key}
    assert labels <= physical, (
        f"Plaquette {P}: {len(labels - physical)} unphysical state labels."
    )


@pytest.mark.parametrize("num_sites,PBCs,cutoff,P", OPEN_BC_CASES)
def test_open_bc_b_truncation_matches_brute_force(cache_dir, num_sites, PBCs,
                                                  cutoff, P):
    """Skipping absent site factors must not drop or invent matrix elements."""
    sites, plaquettes, truncation_irreps, singlets, conj_dict = _lattice_objects_for(
        num_sites, PBCs, cutoff
    )
    mat_elems = calc_plaquette_elements(
        N_COLORS, P, sites, plaquettes, truncation_irreps, singlets,
        conj_dict, FORDER, EPS, PRES, parallelize=False,
    )
    expected = _brute_force_plaquette_elements(
        P, sites, plaquettes, truncation_irreps, singlets, conj_dict
    )

    assert len(expected) > 0, f"Brute force reference is empty for plaquette {P}"
    assert set(mat_elems) == set(expected), (
        f"Plaquette {P}: {len(set(expected) - set(mat_elems))} missing and "
        f"{len(set(mat_elems) - set(expected))} spurious matrix elements."
    )
    for key, expected_value in expected.items():
        assert math.isclose(mat_elems[key], expected_value, abs_tol=1e-10), (
            f"Plaquette {P} key {key}: got {mat_elems[key]}, "
            f"expected {expected_value}"
        )
