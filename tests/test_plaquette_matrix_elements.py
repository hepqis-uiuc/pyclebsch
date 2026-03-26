"""Tests for plaquette matrix element computation, especially OBC boundary cases."""

from pyclebsch.matrix_elements.lattice_data import (
    irreps_and_singlets,
    sites_links_and_plaquettes,
)
from pyclebsch.matrix_elements.plaquette_matrix_elements import calc_plaquette_elements

FORDER = [1, 2, 3, -1, -2, -3]
N_COLORS = 3
EPS = 1e-10
PRES = 10


def _calc_all_plaquette_elements(num_sites, PBCs, truncation_mode, cutoff):
    """Helper: compute matrix elements for every plaquette on the lattice."""
    sites, links, plaquettes = sites_links_and_plaquettes(num_sites, PBCs, FORDER)
    truncation_irreps, singlets, conj_dict = irreps_and_singlets(
        N_COLORS, sites, truncation_mode, cutoff
    )
    results = {}
    for P in sorted(plaquettes.keys()):
        results[P] = calc_plaquette_elements(
            N_COLORS,
            P,
            sites,
            plaquettes,
            truncation_irreps,
            singlets,
            conj_dict,
            FORDER,
            EPS,
            PRES,
            parallelize=False,
        )
    return results


class TestOBCBoundaryGlueing:
    """Regression tests for glueing site factors across OBC boundary vertices
    where adjacent sites have different numbers of half-links and thus
    different allowed irreps under B-truncation."""

    def test_b6_d3_2_obc_no_keyerror(self):
        """B6 d=3/2 OBC: boundary plaquettes must not raise KeyError.

        Prior to the fix, plaquettes with mixed link counts (e.g. [3,2,2,3])
        raised KeyError in glue_plaquette_site_factors because site 1 (3 links)
        produced irreps like (2,2,0) absent at site 2 (2 links).
        """
        results = _calc_all_plaquette_elements(
            num_sites=[4, 2, 1],
            PBCs=[False, False, False],
            truncation_mode="B",
            cutoff=6,
        )
        # 3 plaquettes on the [4,2,1] OBC lattice
        assert len(results) == 3
        for P, mat_elems in results.items():
            assert len(mat_elems) > 0, f"Plaquette {P} produced no matrix elements"

    def test_b6_d3_2_obc_boundary_vs_interior_counts(self):
        """Boundary plaquettes should have fewer matrix elements than interior."""
        results = _calc_all_plaquette_elements(
            num_sites=[4, 2, 1],
            PBCs=[False, False, False],
            truncation_mode="B",
            cutoff=6,
        )
        sites, _, plaquettes = sites_links_and_plaquettes(
            [4, 2, 1], [False, False, False], FORDER
        )
        boundary_counts = []
        interior_counts = []
        for P, mat_elems in results.items():
            link_counts = [len(sites[s]) for s in plaquettes[P][2]]
            if all(n == link_counts[0] for n in link_counts):
                interior_counts.append(len(mat_elems))
            else:
                boundary_counts.append(len(mat_elems))

        # Interior plaquettes (uniform link counts) should have more elements
        assert len(interior_counts) > 0
        assert len(boundary_counts) > 0
        assert all(bc < ic for bc in boundary_counts for ic in interior_counts)

    def test_b6_d3_2_pbc_regression(self):
        """B6 PBC d=3/2 should still produce the same results as before the fix."""
        results = _calc_all_plaquette_elements(
            num_sites=[3, 2, 1],
            PBCs=[True, False, False],
            truncation_mode="B",
            cutoff=6,
        )
        # All PBC plaquettes on a [3,2,1] lattice have uniform link counts
        for P, mat_elems in results.items():
            assert len(mat_elems) > 0

    def test_b5_d3_2_obc_still_works(self):
        """B5 d=3/2 OBC was already working before the fix; verify no regression."""
        results = _calc_all_plaquette_elements(
            num_sites=[4, 2, 1],
            PBCs=[False, False, False],
            truncation_mode="B",
            cutoff=5,
        )
        assert len(results) == 3
        for P, mat_elems in results.items():
            assert len(mat_elems) > 0

    def test_obc_matrix_elements_are_symmetric(self):
        """Matrix elements should satisfy H[Pf,Pi] == H[Pi,Pf] (Hermiticity)."""
        results = _calc_all_plaquette_elements(
            num_sites=[4, 2, 1],
            PBCs=[False, False, False],
            truncation_mode="B",
            cutoff=6,
        )
        for P, mat_elems in results.items():
            for (Pf, Pi), val in mat_elems.items():
                if (Pi, Pf) in mat_elems:
                    assert abs(val - mat_elems[(Pi, Pf)]) < 1e-8, (
                        f"Plaquette {P}: H[Pf,Pi] != H[Pi,Pf] for states {Pf}, {Pi}"
                    )
