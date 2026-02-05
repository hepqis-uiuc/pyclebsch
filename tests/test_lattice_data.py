import pytest
from pyclebsch.matrix_elements.lattice_data import (
    LatticeDef,
    compute_plaquette_signature,
)


def test_lattice_def_planes():
    """Check that we get the right planes for various lattices."""

    # d=3/2 cases, PBCs
    lattice = LatticeDef(num_sites=(2, 2, 1), PBCs=(True, False, False))
    assert lattice.planes == ((1, 2),)
    lattice = LatticeDef(num_sites=(10, 2, 1), PBCs=(True, False, False))
    assert lattice.planes == ((1, 2),)

    # d=3/2 cases, OBCs
    lattice = LatticeDef(num_sites=(2, 2, 1), PBCs=(False, False, False))
    assert lattice.planes == ((1, 2),)
    lattice = LatticeDef(num_sites=(3, 2, 1), PBCs=(False, False, False))
    assert lattice.planes == ((1, 2),)

    # d=2 cases, PBCs
    lattice = LatticeDef(num_sites=(2, 2, 1), PBCs=(True, True, False))
    assert lattice.planes == ((1, 2),)
    lattice = LatticeDef(num_sites=(4, 3, 1), PBCs=(True, True, False))
    assert lattice.planes == ((1, 2),)

    # d=2 cases, OBCs
    lattice = LatticeDef(num_sites=(3, 3, 1), PBCs=(False, False, False))
    assert lattice.planes == ((1, 2),)
    lattice = LatticeDef(num_sites=(10, 10, 1), PBCs=(False, False, False))
    assert lattice.planes == ((1, 2),)

    # d=3 cases, PBCs
    lattice = LatticeDef(num_sites=(2, 2, 2), PBCs=(True, True, True))
    assert lattice.planes == ((1, 2), (1, 3), (2, 3))
    lattice = LatticeDef(num_sites=(4, 1, 7), PBCs=(False, True, False)) # mixed BCs
    assert lattice.planes == ((1, 2), (1, 3), (2, 3))

    # d=3 cases, OBCs
    lattice = LatticeDef(num_sites=(2, 2, 2), PBCs=(False, False, False))
    assert lattice.planes == ((1, 2), (1, 3), (2, 3))
    lattice = LatticeDef(num_sites=(4, 3, 2), PBCs=(False, False, False))
    assert lattice.planes == ((1, 2), (1, 3), (2, 3))


def test_lattice_def_site_exists():
    """Check that the 'site exists' functionality gives the correct results."""

    lattice = LatticeDef(num_sites=(2, 1, 4), PBCs=(False, True, True))
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
    lattice = LatticeDef(num_sites=(2, 1, 4), PBCs=(False, True, True))
    with pytest.raises(ValueError) as e_info:
        lattice.site_exists(site=(1.0, 1, 1))
    with pytest.raises(ValueError) as e_info:
        lattice.site_exists(site=(1, 1, 1, 1))


def test_compute_plaquette_signature():
    # test_cases = {
    #     "d=3/2": [
    #     LatticeDef(num_sites=(2, 2, 1), PBCs=(True, False, False)),
    #     LatticeDef(num_sites=(3, 2, 1), PBCs=(True, False, False)),
    #     LatticeDef(num_sites=(4, 2, 1), PBCs=(True, False, False)),
    #     LatticeDef(num_sites=(5, 2, 1), PBCs=(True, False, False))
    #     ],
    #     "d=2"
    # }
    lattice = LatticeDef(num_sites=(2, 2, 1), PBCs=(True, False, False))
    sites = [(x, y, z) for z in range(lattice.num_sites[2]) for y in range(lattice.num_sites[1]) for x in range(lattice.num_sites[0])]
    # for 
    raise NotImplementedError("Test not yet written.")
