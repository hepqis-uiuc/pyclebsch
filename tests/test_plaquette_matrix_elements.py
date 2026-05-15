"""Tests for plaquette matrix element computation on a small PBC d=2 lattice.

Lattice: SU(3), num_sites=[2,2,1], PBCs=[True,True,False], T-truncation cutoff 1.
This is the smallest lattice with non-empty matrix elements for every plaquette
and exercises the full glueing pipeline (4 plaquettes, all interior under PBCs).
"""

import math

import pytest

import pyclebsch.cgc as cgc
from pyclebsch.matrix_elements.lattice_data import (
    irreps_and_singlets,
    sites_links_and_plaquettes,
)
from pyclebsch.matrix_elements.plaquette_matrix_elements import calc_plaquette_elements

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
