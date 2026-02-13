import pytest
from pyclebsch.matrix_elements.lattice_data import (
    LatticeDef,
    LinkDirection,
    PlaquetteAddress,
    PlaquetteSignature,
    compute_plaquette_signature,
)

FORDER: list[LinkDirection] = [1, 2, 3, -1, -2, -3]


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
