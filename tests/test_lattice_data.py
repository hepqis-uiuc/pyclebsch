import pytest

import pyclebsch.cgc as cgc
from pyclebsch.matrix_elements.lattice_data import (
    LatticeDef,
    LinkDirection,
    PlaquetteAddress,
    PlaquetteSignature,
    compute_plaquette_signature,
    irreps_and_singlets,
    physical_plaquette_states,
    sites_links_and_plaquettes,
)

FORDER: list[LinkDirection] = [1, 2, 3, -1, -2, -3]


@pytest.fixture(autouse=True)
def restore_cache_dir():
    """Save/restore the CGC cache dir around each test.

    Tests that compute CGCs (irreps_and_singlets / physical_plaquette_states)
    should call cgc.set_cache_dir(tmp_path / "CGC_Data") at the top to redirect
    writes; other tests are unaffected.
    """
    original = cgc.get_cache_dir()
    yield
    cgc.set_cache_dir(original)


def test_lattice_def_planes():
    """Check that we get the right planes for various lattices."""
    # d=3/2 cases, PBCs
    lattice = LatticeDef(num_sites=(2, 2, 1), PBCs=(True, False, False), FORDER=FORDER)
    assert lattice.planes == ((1, 2),)
    lattice = LatticeDef(num_sites=(10, 2, 1), PBCs=(True, False, False), FORDER=FORDER)
    assert lattice.planes == ((1, 2),)

    # d=3/2 cases, OBCs
    lattice = LatticeDef(num_sites=(2, 2, 1), PBCs=(False, False, False), FORDER=FORDER)
    assert lattice.planes == ((1, 2),)
    lattice = LatticeDef(num_sites=(3, 2, 1), PBCs=(False, False, False), FORDER=FORDER)
    assert lattice.planes == ((1, 2),)

    # d=2 cases, PBCs
    lattice = LatticeDef(num_sites=(2, 2, 1), PBCs=(True, True, False), FORDER=FORDER)
    assert lattice.planes == ((1, 2),)
    lattice = LatticeDef(num_sites=(4, 3, 1), PBCs=(True, True, False), FORDER=FORDER)
    assert lattice.planes == ((1, 2),)

    # d=2 cases, OBCs
    lattice = LatticeDef(num_sites=(3, 3, 1), PBCs=(False, False, False), FORDER=FORDER)
    assert lattice.planes == ((1, 2),)
    lattice = LatticeDef(num_sites=(10, 10, 1), PBCs=(False, False, False), FORDER=FORDER)
    assert lattice.planes == ((1, 2),)

    # d=3 cases, PBCs
    lattice = LatticeDef(num_sites=(2, 2, 2), PBCs=(True, True, True), FORDER=FORDER)
    assert lattice.planes == ((1, 2), (1, 3), (2, 3))
    lattice = LatticeDef(num_sites=(4, 1, 7), PBCs=(False, True, False), FORDER=FORDER) # mixed BCs
    assert lattice.planes == ((1, 2), (1, 3), (2, 3))

    # d=3 cases, OBCs
    lattice = LatticeDef(num_sites=(2, 2, 2), PBCs=(False, False, False), FORDER=FORDER)
    assert lattice.planes == ((1, 2), (1, 3), (2, 3))
    lattice = LatticeDef(num_sites=(4, 3, 2), PBCs=(False, False, False), FORDER=FORDER)
    assert lattice.planes == ((1, 2), (1, 3), (2, 3))


def test_lattice_def_site_exists():
    """Check that the 'site exists' functionality gives the correct results."""

    lattice = LatticeDef(num_sites=(2, 1, 4), PBCs=(False, True, True), FORDER=FORDER)
    nonexistent_sites = [(2, 0, 0), (2, 1, 4), (2, -1, -1), (2, -1, 3)]
    existent_sites_strict = [(0, 0, 0), (0, 0, 1), (1, 0, 3)]
    existent_sites_periodic_ok = [(0, -1, 10), (0, -3, 20), (1, 1, 4)]

    for site in nonexistent_sites:
        assert lattice.site_exists(site) is False

    for site in existent_sites_strict:
        assert lattice.site_exists(site, periodic_ok=False) is True
        assert lattice.site_exists(site, periodic_ok=True) is True

    for site in existent_sites_periodic_ok:
        assert lattice.site_exists(site, periodic_ok=False) is False
        assert lattice.site_exists(site, periodic_ok=True) is True


def test_lattice_def_site_exists_value_error():
    """Should get ValueError for non 3-tuple of ints input."""
    lattice = LatticeDef(num_sites=(2, 1, 4), PBCs=(False, True, True), FORDER=FORDER)
    with pytest.raises(ValueError) as e_info:
        lattice.site_exists(site=(1.0, 1, 1))
    with pytest.raises(ValueError) as e_info:
        lattice.site_exists(site=(1, 1, 1, 1))


def test_compute_plaquette_key_error_on_bad_plaquette_address():
    lattice = LatticeDef(num_sites=(2, 2, 1), PBCs=[True, False, False], FORDER=FORDER)
    plaquette_address_bad_site = ((3, 0, 0), (1, 2))
    plaquette_address_bad_plane = ((0, 0, 0), (1, 3))
    
    with pytest.raises(KeyError) as e_info:
        compute_plaquette_signature(plaquette_address_bad_site, lattice)
    with pytest.raises(KeyError) as e_info:
        compute_plaquette_signature(plaquette_address_bad_plane, lattice)

        
def test_compute_plaquette_signature():

    # Test cases includes a strict option because for d=3 and for larger lattices,
    # it is tedious to write out every single possible plaquette. Suffices
    # To spot check different "kinds" of plaquettes (like corners, edges, interiors).
    test_cases = {
        "d=3/2": {
            "2 plaquettes, PBCs": {
                "strict_test": True,
                "lattice": LatticeDef(
                    num_sites=(2, 2, 1),
                    PBCs=(True, False, False),
                    FORDER=[1, -1, 2, -2, 3, -3],
                ),
                "expected_signatures": {
                    ((0, 0, 0), (1, 2)): (
                        (1, 2),
                        ((1, -1, 2), (1, -1, 2), (1, -1, -2), (1, -1, -2)),
                    ),
                    ((1, 0, 0), (1, 2)): (
                        (1, 2),
                        ((1, -1, 2), (1, -1, 2), (1, -1, -2), (1, -1, -2)),
                    )
                },
                "expected_empty_signatures": {
                    ((0, 1, 0), (1, 2)): ((1, 2), ()),
                    ((1, 1, 0), (1, 2)): ((1, 2), ()),
                }
            },
            "3 plaquettes, OBCs": {  # 4 sites needed along 1 dir since OBCs
                "strict_test": True,
                "lattice": LatticeDef(
                    num_sites=(4, 2, 1),
                    PBCs=(False, False, False),
                    FORDER=[-1, 3, 2, -2, 1, -3],
                ),
                "expected_signatures": {
                    ((0, 0, 0), (1, 2)): (
                        (1, 2),
                        ((2, 1), (-1, 2, 1), (-1, -2, 1), (-2, 1)),
                    ),
                    ((1, 0, 0), (1, 2)): (
                        (1, 2),
                        ((-1, 2, 1), (-1, 2, 1), (-1, -2, 1), (-1, -2, 1)),
                    ),
                    ((2, 0, 0), (1, 2)): (
                        (1, 2),
                        ((-1, 2, 1), (-1, 2), (-1, -2), (-1, -2, 1)),
                    )},
                "expected_empty_signatures": {
                    ((3, 0, 0), (1, 2)): ((1, 2), ()),
                    ((0, 1, 0), (1, 2)): ((1, 2), ()),
                    ((1, 1, 0), (1, 2)): ((1, 2), ()),
                    ((2, 1, 0), (1, 2)): ((1, 2), ()),
                    ((3, 1, 0), (1, 2)): ((1, 2), ()),
                },
            },
        },
        "d=2": {
            "4 plaquettes, PBCs": {
                "strict_test": True,
                "lattice": LatticeDef(
                    num_sites=(2, 2, 1),
                    PBCs=(True, True, False),
                    FORDER=[1, 2, 3, -1, -2, -3],
                ),
                "expected_signatures": {
                    ((0, 0, 0), (1, 2)): (
                        (1, 2),
                        ((1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2)),
                    ),
                    ((1, 0, 0), (1, 2)): (
                        (1, 2),
                        ((1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2))
                    ),
                    ((0, 1, 0), (1, 2)): (
                        (1, 2),
                        ((1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2))
                    ),
                    ((1, 1, 0), (1, 2)): (
                        (1, 2),
                        ((1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2), (1, 2, -1, -2))
                    ),
                },
                "expected_empty_signatures": {
                }
            },
            "9 plaquettes, mixed BCs": {  # 4 sites needed along 1 dir since OBCs; 3 along 2 dir since PBCs
                "strict_test": True,
                "lattice": LatticeDef(
                    num_sites=(4, 3, 1),
                    PBCs=(False, True, False),
                    FORDER=[-1, -2, -3, 1, 2, 3],
                ),
                "expected_signatures": {
                    ((0, 0, 0), (1, 2)): (
                        (1, 2),
                        ((-2, 1, 2), (-1, -2, 1, 2), (-1, -2, 1, 2), (-2, 1, 2)),
                    ),
                    ((1, 0, 0), (1, 2)): (
                        (1, 2),
                        ((-1, -2, 1, 2), (-1, -2, 1, 2), (-1, -2, 1, 2), (-1, -2, 1, 2)),
                    ),
                    ((2, 0, 0), (1, 2)): (
                        (1, 2),
                        ((-1, -2, 1, 2), (-1, -2, 2), (-1, -2, 2), (-1, -2, 1, 2)),
                    ),
                    ((0, 1, 0), (1, 2)): (
                        (1, 2),
                        ((-2, 1, 2), (-1, -2, 1, 2), (-1, -2, 1, 2), (-2, 1, 2))
                    ),
                    ((1, 1, 0), (1, 2)): (
                        (1, 2),
                        ((-1, -2, 1, 2), (-1, -2, 1, 2), (-1, -2, 1, 2), (-1, -2, 1, 2)),
                    ),
                    ((2, 1, 0), (1, 2)): (
                        (1, 2),
                        ((-1, -2, 1, 2), (-1, -2, 2), (-1, -2, 2), (-1, -2, 1, 2)),
                    ),
                    ((0, 2, 0), (1, 2)): (
                        (1, 2),
                        ((-2, 1, 2), (-1, -2, 1, 2), (-1, -2, 1, 2), (-2, 1, 2)),
                    ),
                    ((1, 2, 0), (1, 2)): (
                        (1, 2),
                        ((-1, -2, 1, 2), (-1, -2, 1, 2), (-1, -2, 1, 2), (-1, -2, 1, 2)),
                    ),
                    ((2, 2, 0), (1, 2)): (
                        (1, 2),
                        ((-1, -2, 1, 2), (-1, -2, 2), (-1, -2, 2), (-1, -2, 1, 2)),
                    ),
                },
                "expected_empty_signatures": {
                    ((3, 0, 0), (1, 2)): ((1, 2), ()),
                    ((3, 1, 0), (1, 2)): ((1, 2), ()),
                    ((3, 2, 0), (1, 2)): ((1, 2), ()),
                },
            },
        },
        "d=3": {
            "6 plaquettes, OBCs": { # i.e. the faces of a single cube
                "strict_test": True,
                "lattice": LatticeDef(
                    num_sites=(2, 2, 2),
                    PBCs=(False, False, False),
                    FORDER=[1, 2, 3, -1, -2, -3],
                ),
                "expected_signatures": {
                    ((0, 0, 0), (1, 2)): ( # bottom
                        (1, 2),
                        ((1, 2, 3), (2, 3, -1), (3, -1, -2), (1, 3, -2)),
                    ),
                    ((0, 0, 0), (1, 3)): ( # front
                        (1, 3),
                        ((1, 2, 3), (2, 3, -1), (2, -1, -3), (1, 2, -3))
                    ),
                    ((0, 0, 0), (2, 3)): ( # left side
                        (2, 3),
                        ((1, 2, 3), (1, 3, -2), (1, -2, -3), (1, 2, -3))
                    ),
                    ((1, 0, 0), (2, 3)): ( # right side
                        (2, 3),
                        ((2, 3, -1), (3, -1, -2), (-1, -2, -3), (2, -1, -3))
                    ),
                    ((0, 1, 0), (1, 3)): ( # back
                        (1, 3),
                        ((1, 3, -2), (3, -1, -2), (-1, -2, -3), (1, -2, -3))
                    ),
                    ((0, 0, 1), (1, 2)): ( # top
                        (1, 2),
                        ((1, 2, -3), (2, -1, -3), (-1, -2, -3), (1, -2, -3))
                    )
                },
                "expected_empty_signatures": { # three coordinates have two empties (appeared in 'expected'), three coordinates have three empties (no appearance in 'expected')
                    ((1, 0, 0), (1, 2)): ((1, 2), ()),
                    ((1, 0, 0), (1, 3)): ((1, 3), ()),
                    ((0, 1, 0), (1, 2)): ((1, 2), ()),
                    ((0, 1, 0), (2, 3)): ((2, 3), ()),
                    ((0, 0, 1), (1, 3)): ((1, 3), ()),
                    ((0, 0, 1), (2, 3)): ((2, 3), ()),
                    ((0, 1, 1), (1, 2)): ((1, 2), ()),
                    ((0, 1, 1), (1, 3)): ((1, 3), ()),
                    ((0, 1, 1), (2, 3)): ((2, 3), ()),
                    ((1, 0, 1), (1, 2)): ((1, 2), ()),
                    ((1, 0, 1), (1, 3)): ((1, 3), ()),
                    ((1, 0, 1), (2, 3)): ((2, 3), ()),
                    ((1, 1, 1), (1, 2)): ((1, 2), ()),
                    ((1, 1, 1), (1, 3)): ((1, 3), ()),
                    ((1, 1, 1), (2, 3)): ((2, 3), ()),
                }
            },
            "3x3x3, mixed BCs": {  # Many plaquettes, just spot checking a few
                "strict_test": False,
                "lattice": LatticeDef(
                    num_sites=(4, 3, 3),
                    PBCs=(False, True, True),
                    FORDER=[1, 2, 3, -1, -2, -3],
                ),
                "expected_signatures": {
                    ((0, 0, 0), (1, 2)): ( # 1 dir "edge"
                        (1, 2),
                        ((1, 2, 3, -2, -3), (1, 2, 3, -1, -2, -3), (1, 2, 3, -1, -2, -3), (1, 2, 3, -2, -3))
                    ),
                    ((0, 0, 0), (1, 3)): ( # 1 dir "edge"
                        (1, 3),
                        ((1, 2, 3, -2, -3), (1, 2, 3, -1, -2, -3), (1, 2, 3, -1, -2, -3), (1, 2, 3, -2, -3))
                    ),
                    ((0, 0, 0), (2, 3)): ( # 1 dir "face"
                        (2, 3),
                        ((1, 2, 3, -2, -3), (1, 2, 3, -2, -3), (1, 2, 3, -2, -3), (1, 2, 3, -2, -3))
                    ),
                    ((1, 1, 1), (1, 3)): ( # interior
                        (1, 3),
                        ((1, 2, 3, -1, -2, -3), (1, 2, 3, -1, -2, -3), (1, 2, 3, -1, -2, -3), (1, 2, 3, -1, -2, -3))
                    ),
                },
                "expected_empty_signatures": {
                    ((3, 0, 0), (1, 2)): ((1, 2), ()), # Far 1 dir edge of lattice, can't form (1, 2) plaquette there.
                },
            },
        }
    }
   
    for dim_str, lattices in test_cases.items():
        for lattice_str, current_test_data in lattices.items():
            computed_signatures = {}
            for plaquette_address in list(current_test_data["lattice"].plaquettes.keys()):
                computed_signatures[plaquette_address] = compute_plaquette_signature(
                    plaquette_address=plaquette_address, lattice=current_test_data["lattice"]
                )
            if current_test_data['strict_test'] is True:
                assert computed_signatures == current_test_data["expected_signatures"], f"{dim_str}, {lattice_str} (strict test) yielded unexpected signatures.\nExpected: {current_test_data['expected_signatures']}\nEncountered: {computed_signatures}"
            else:
                for expected_plaquette_address, expected_plaquette_signature in current_test_data["expected_signatures"].items():
                    current_site = expected_plaquette_address[0]
                    assert computed_signatures[expected_plaquette_address] == expected_plaquette_signature, f"{dim_str}, {lattice_str} (non-strict test) yielded unexpected signature at site {current_site}.\nExpected: {expected_plaquette_signature}\nEncountered: {computed_signatures[expected_plaquette_address]}"
            for impossible_plaquette_address, empty_signature_result in current_test_data['expected_empty_signatures'].items():
                assert compute_plaquette_signature(impossible_plaquette_address, current_test_data['lattice']) == empty_signature_result


# ===========================================================================
# sites_links_and_plaquettes
# ===========================================================================

def test_sites_links_and_plaquettes_2x2x1_pbc_x():
    """num_sites=(2,2,1) with PBC only along x: 4 sites, 6 links, 2 plaquettes."""
    sites, links, plaquettes = sites_links_and_plaquettes(
        (2, 2, 1), (True, False, False), FORDER
    )
    assert len(sites) == 4
    assert set(sites.keys()) == {(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)}
    assert len(links) == 6
    assert len(plaquettes) == 2
    expected_plaq_keys = {((0, 0, 0), (1, 2)), ((1, 0, 0), (1, 2))}
    assert set(plaquettes.keys()) == expected_plaq_keys


def test_sites_links_and_plaquettes_3x2x1_obc():
    """num_sites=(3,2,1) with all OBC: 6 sites, 7 links, 2 plaquettes."""
    sites, links, plaquettes = sites_links_and_plaquettes(
        (3, 2, 1), (False, False, False), FORDER
    )
    assert len(sites) == 6
    assert len(links) == 7
    assert len(plaquettes) == 2


def test_sites_links_and_plaquettes_2x2x1_pbc_xy():
    """d=2 PBC 2x2: 4 sites with 4 half-links each, 8 links, 4 plaquettes."""
    sites, links, plaquettes = sites_links_and_plaquettes(
        (2, 2, 1), (True, True, False), FORDER
    )
    assert len(sites) == 4
    for half_links in sites.values():
        assert sorted(half_links) == [-2, -1, 1, 2]
    assert len(links) == 8
    assert len(plaquettes) == 4


def test_sites_links_and_plaquettes_rejects_obc_axis_of_length_one_with_pbc():
    """PBC on an axis with only 1 site must raise ValueError."""
    with pytest.raises(ValueError):
        sites_links_and_plaquettes((2, 2, 1), (False, False, True), FORDER)


def test_sites_links_and_plaquettes_rejects_zero_sites():
    """num_sites entries must be at least 1."""
    with pytest.raises(ValueError):
        sites_links_and_plaquettes((0, 1, 1), (False, False, False), FORDER)


def test_sites_links_and_plaquettes_rejects_bad_forder():
    """FORDER must be a permutation of [1,2,3,-1,-2,-3]."""
    with pytest.raises(ValueError):
        sites_links_and_plaquettes(
            (2, 2, 1), (True, False, False), [1, 2, 3, 4, 5, 6]
        )


# ===========================================================================
# irreps_and_singlets
# ===========================================================================

def _build_d2_pbc_2x2_sites():
    sites, _, _ = sites_links_and_plaquettes((2, 2, 1), (True, True, False), FORDER)
    return sites


def test_irreps_and_singlets_su3_t1_d2_link_irreps(tmp_path):
    """T=1, SU(3): admitted irreps at each 4-link site are {trivial, fund, afund}."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    sites = _build_d2_pbc_2x2_sites()
    link_irreps, _site_singlets, _conj_dict = irreps_and_singlets(
        N=3, sites=sites, truncation_mode="T", cutoff=1
    )
    # All 4 sites have 4 half-links each (d=2 PBC).
    assert set(link_irreps.keys()) == {4}
    assert set(link_irreps[4]) == {(0, 0, 0), (1, 0, 0), (1, 1, 0)}


def test_irreps_and_singlets_su3_t1_d2_trivial_singlet_present(tmp_path):
    """The all-trivial singlet ((0,0,0)^4) has multiplicity 1.

    The source records site_singlets[n][permutation[2:]][permutation[0:2]] = mult.
    For an all-trivial 4-tuple there is exactly one distinct permutation, so
    site_singlets[4][((0,0,0),(0,0,0))][((0,0,0),(0,0,0))] == 1.
    """
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    sites = _build_d2_pbc_2x2_sites()
    _, site_singlets, _ = irreps_and_singlets(3, sites, "T", 1)
    trivial = (0, 0, 0)
    assert site_singlets[4][(trivial, trivial)][(trivial, trivial)] == 1


def test_irreps_and_singlets_su3_t1_d2_conj_dict_keys(tmp_path):
    """conj_dict covers every irrep that appears in link_irreps."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    sites = _build_d2_pbc_2x2_sites()
    link_irreps, _, conj_dict = irreps_and_singlets(3, sites, "T", 1)
    all_irreps = set().union(*link_irreps.values())
    assert set(conj_dict.keys()) == all_irreps


def test_irreps_and_singlets_su3_t1_d2_conj_dict_trivial(tmp_path):
    """conj_dict[(0,0,0)] = {0: (0, +1)} — trivial is self-conjugate, phase +1."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    sites = _build_d2_pbc_2x2_sites()
    _, _, conj_dict = irreps_and_singlets(3, sites, "T", 1)
    trivial_conj = conj_dict[(0, 0, 0)]
    assert set(trivial_conj.keys()) == {0}
    cst, phase = trivial_conj[0]
    assert cst == 0
    assert phase == 1


def test_irreps_and_singlets_su3_t1_d2_conj_dict_fundamental(tmp_path):
    """conj_dict[(1,0,0)] has 3 entries (one per fund state) mapping to (afund_state, ±1).

    Each fund state has |phase| == 1 (sign of a nonzero CGC of magnitude 1/√3).
    """
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    sites = _build_d2_pbc_2x2_sites()
    _, _, conj_dict = irreps_and_singlets(3, sites, "T", 1)
    fund_conj = conj_dict[(1, 0, 0)]
    assert set(fund_conj.keys()) == {0, 1, 2}
    # Conjugate map is a bijection onto the 3 antifund states.
    target_states = {cst for cst, _phase in fund_conj.values()}
    assert target_states == {0, 1, 2}
    for _cst, phase in fund_conj.values():
        assert abs(phase) == 1


def test_irreps_and_singlets_c_mode_runs(tmp_path):
    """C-mode truncation runs to completion on a small lattice."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    sites = _build_d2_pbc_2x2_sites()
    link_irreps, _, _ = irreps_and_singlets(3, sites, "C", 4 / 3)
    assert set(link_irreps[4]) >= {(0, 0, 0), (1, 0, 0), (1, 1, 0)}


def test_irreps_and_singlets_b_mode_runs(tmp_path):
    """B-mode truncation runs to completion on a small lattice."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    sites = _build_d2_pbc_2x2_sites()
    link_irreps, _, _ = irreps_and_singlets(3, sites, "B", 6)
    assert (0, 0, 0) in link_irreps[4]


def test_irreps_and_singlets_rejects_invalid_truncation_mode(tmp_path):
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    sites = _build_d2_pbc_2x2_sites()
    with pytest.raises(ValueError):
        irreps_and_singlets(3, sites, "Z", 1)


def test_irreps_and_singlets_rejects_invalid_N(tmp_path):
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    sites = _build_d2_pbc_2x2_sites()
    with pytest.raises(ValueError):
        irreps_and_singlets(1, sites, "T", 1)


# ===========================================================================
# physical_plaquette_states
# ===========================================================================

def test_physical_plaquette_states_su3_t1_d2_smoke(tmp_path):
    """Smoke test on SU(3), T=1, d=2 PBC 2x2:
    - Non-empty list of states.
    - Every state is a 12-tuple.
    - Active links are 3-tuples; control slots are tuples; multiplicities are ints.
    - The all-trivial state appears.
    """
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    sites, _, plaquettes = sites_links_and_plaquettes(
        (2, 2, 1), (True, True, False), FORDER
    )
    _, site_singlets, _ = irreps_and_singlets(3, sites, "T", 1)
    # Pick any plaquette (all 4 are equivalent under PBC d=2 2x2).
    plaq_address = next(iter(plaquettes))
    states = physical_plaquette_states(
        plaq_address, sites, plaquettes, site_singlets, FORDER
    )
    assert len(states) > 0
    for state in states:
        assert len(state) == 12
        for active_link in state[:4]:
            assert isinstance(active_link, tuple) and len(active_link) == 3
        for site_ctrls in state[4:8]:
            assert isinstance(site_ctrls, tuple)
            for ctrl in site_ctrls:
                assert isinstance(ctrl, tuple) and len(ctrl) == 3
        for g in state[8:]:
            assert isinstance(g, int)

    trivial = (0, 0, 0)
    # Each site has 4 half-links; 2 are active, 2 are controls. So each
    # site_ctrls is a length-2 tuple.
    all_trivial = (
        trivial, trivial, trivial, trivial,
        (trivial, trivial), (trivial, trivial),
        (trivial, trivial), (trivial, trivial),
        0, 0, 0, 0,
    )
    assert all_trivial in states
