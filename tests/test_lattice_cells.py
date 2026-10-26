"""Tests for LatticeDef's cell data: its cubes, and the boundaries and
coboundaries between neighboring dimensions with their orientation signs.
Also tests that a LatticeDef, and everything it returns, cannot be modified.

Every comparison is exact: the signs are integers, and N-alities are
integers mod N.
"""

import copy
import dataclasses
import pickle
from math import prod

import pytest

from pyclebsch.matrix_elements.helpers import n_ality
from pyclebsch.matrix_elements.lattice_data import (
    LatticeDef,
    LinkDirection,
    irreps_and_singlets,
)
from pyclebsch.matrix_elements.plaquette_matrix_elements import calc_plaquette_elements

FORDER: list[LinkDirection] = [1, 2, 3, -1, -2, -3]
N_COLORS = 3
EPS = 1e-10
PRES = 10

OPEN = (False, False, False)
PERIODIC = (True, True, True)

# Open, periodic, mixed and planar lattices, as (num_sites, PBCs). In
# "3x2x2 periodic in y" a periodic axis of length 2 meets open axes, so one
# cube can have faces that wrap around and faces at an open boundary.
LATTICES: dict[str, tuple[tuple[int, int, int], tuple[bool, bool, bool]]] = {
    "2x2x2 open": ((2, 2, 2), OPEN),
    "3x2x2 open": ((3, 2, 2), OPEN),
    "3x3x2 open": ((3, 3, 2), OPEN),
    "3x3x3 open": ((3, 3, 3), OPEN),
    "2x2x2 periodic": ((2, 2, 2), PERIODIC),
    "3x2x2 periodic": ((3, 2, 2), PERIODIC),
    "3x3x3 periodic": ((3, 3, 3), PERIODIC),
    "3x3x2 periodic in x and y": ((3, 3, 2), (True, True, False)),
    "2x2x3 periodic in z": ((2, 2, 3), (False, False, True)),
    "3x2x2 periodic in y": ((3, 2, 2), (False, True, False)),
    "3x2x1 open": ((3, 2, 1), OPEN),
    "2x2x1 open": ((2, 2, 1), OPEN),
}

# The axis normal to each plane (1-indexed).
NORMAL_AXIS = {(1, 2): 3, (1, 3): 2, (2, 3): 1}

# Everything a LatticeDef computes from its fields.
COMPUTED_PROPERTIES = (
    "planes",
    "sites",
    "links",
    "plaquettes",
    "cubes",
    "plaquette_boundaries",
    "link_coboundaries",
    "cube_boundaries",
    "plaquette_coboundaries",
)


def _lattice(name: str) -> LatticeDef:
    num_sites, pbcs = LATTICES[name]
    return LatticeDef(num_sites=num_sites, PBCs=pbcs, FORDER=FORDER)


def _incidences(table) -> list[tuple]:
    """The (cell, incident cell, sign) triples of a boundary or coboundary."""
    return sorted(
        (cell, incident_cell, sign)
        for cell, signs in table.items()
        for incident_cell, sign in signs.items()
    )


# ---------- boundaries and coboundaries ----------


@pytest.mark.parametrize("name", LATTICES)
def test_boundary_of_a_boundary_vanishes(name):
    """For every cube c and every link l of its faces,
    sum over faces p of cube_boundaries[c][p] * plaquette_boundaries[p][l] is 0.
    (Planar lattices have no cubes, so there is nothing to check on them.)"""
    lattice = _lattice(name)
    for cube_address, face_signs in lattice.cube_boundaries.items():
        links = {
            link
            for face_address in face_signs
            for link in lattice.plaquette_boundaries[face_address]
        }
        for link in links:
            total = sum(
                sign * lattice.plaquette_boundaries[face_address].get(link, 0)
                for face_address, sign in face_signs.items()
            )
            assert total == 0, f"cube {cube_address}, link {link}: {total}"


@pytest.mark.parametrize("name", LATTICES)
def test_signs_are_plus_or_minus_one(name):
    lattice = _lattice(name)
    for table in (
        lattice.plaquette_boundaries,
        lattice.link_coboundaries,
        lattice.cube_boundaries,
        lattice.plaquette_coboundaries,
    ):
        assert {sign for _, _, sign in _incidences(table)} <= {1, -1}


@pytest.mark.parametrize("name", LATTICES)
def test_link_coboundaries_invert_plaquette_boundaries(name):
    """The same (plaquette, link, sign) triples, read from each side; every
    link is a key of link_coboundaries, including links in no plaquette."""
    lattice = _lattice(name)
    assert set(lattice.plaquette_boundaries) == set(lattice.plaquettes)
    assert set(lattice.link_coboundaries) == set(lattice.links)
    from_boundary = _incidences(lattice.plaquette_boundaries)
    from_coboundary = [
        (plaquette_address, link, sign)
        for link, plaquette_address, sign in _incidences(lattice.link_coboundaries)
    ]
    assert sorted(from_coboundary) == from_boundary


@pytest.mark.parametrize("name", LATTICES)
def test_plaquette_coboundaries_invert_cube_boundaries(name):
    """The same (cube, face, sign) triples, read from each side; every
    plaquette is a key of plaquette_coboundaries, including those in no cube."""
    lattice = _lattice(name)
    assert set(lattice.cube_boundaries) == set(lattice.cubes)
    assert set(lattice.plaquette_coboundaries) == set(lattice.plaquettes)
    from_boundary = _incidences(lattice.cube_boundaries)
    from_coboundary = [
        (cube_address, plaquette_address, sign)
        for plaquette_address, cube_address, sign in _incidences(
            lattice.plaquette_coboundaries
        )
    ]
    assert sorted(from_coboundary) == from_boundary


@pytest.mark.parametrize("name", LATTICES)
def test_faces_are_in_the_expected_number_of_cubes(name):
    """A face normal to axis k is in two cubes if k is periodic. If k is
    open with length L >= 2, it is in one cube when its k coordinate is 0
    or L - 1 and in two otherwise; if k is open with length 1, in none. A
    face in two cubes has opposite signs in them."""
    lattice = _lattice(name)
    for face_address, cube_signs in lattice.plaquette_coboundaries.items():
        site, plane = face_address
        k = NORMAL_AXIS[plane] - 1
        length = lattice.num_sites[k]
        if lattice.PBCs[k]:
            expected = 2
        elif length >= 2:
            expected = 1 if site[k] in (0, length - 1) else 2
        else:
            expected = 0
        assert len(cube_signs) == expected, face_address
        if expected == 2:
            assert sorted(cube_signs.values()) == [-1, 1], face_address


@pytest.mark.parametrize("name", LATTICES)
def test_cube_count_and_order(name):
    """prod(n_i) cubes, with n_i = L_i on a periodic axis and L_i - 1 on an
    open one, in lexicographic order of their addresses."""
    lattice = _lattice(name)
    expected = prod(
        n if periodic else n - 1 for n, periodic in zip(lattice.num_sites, lattice.PBCs)
    )
    assert len(lattice.cubes) == expected
    assert list(lattice.cubes) == sorted(lattice.cube_boundaries)
    if lattice.num_sites[2] == 1:
        assert lattice.cubes == ()


# ---------- single-lattice checks of cubes, faces and signs ----------


def test_cubes_of_3x2x2_open():
    lattice = _lattice("3x2x2 open")
    assert lattice.cubes == ((0, 0, 0), (1, 0, 0))
    for face_signs in lattice.cube_boundaries.values():
        assert len(face_signs) == 6


def test_shared_face_signs_on_3x2x2_open():
    """The face ((1,0,0), (2,3)) has normal e_2 x e_3 = +e_1. It is the upper
    x face of cube (0,0,0), where the outward normal is +e_1, and the lower x
    face of cube (1,0,0), where the outward normal is -e_1."""
    lattice = _lattice("3x2x2 open")
    face_address = ((1, 0, 0), (2, 3))
    assert lattice.cube_boundaries[(0, 0, 0)][face_address] == 1
    assert lattice.cube_boundaries[(1, 0, 0)][face_address] == -1
    assert dict(lattice.plaquette_coboundaries[face_address]) == {
        (0, 0, 0): 1,
        (1, 0, 0): -1,
    }


def test_plaquette_link_sign_pattern():
    """The plaquette's boundary is +l1 + l2 - l3 - l4: the loop
    s1 -> s2 -> s3 -> s4 -> s1 traverses l3 and l4 backward."""
    lattice = _lattice("3x2x2 open")
    plaquette_address = ((0, 0, 0), (1, 2))
    l1, l2, l3, l4 = lattice.plaquettes[plaquette_address].active_links
    assert dict(lattice.plaquette_boundaries[plaquette_address]) == {
        l1: 1,
        l2: 1,
        l3: -1,
        l4: -1,
    }


def test_link_coboundary_sizes_in_bulk_and_at_edge():
    """On 3x3x3 open the x-link from the center site lies in two xy and two
    xz plaquettes, and the x-link along an edge of the lattice in one of
    each. (No link of 3x2x2 lies in four plaquettes.)"""
    lattice = _lattice("3x3x3 open")
    assert len(lattice.link_coboundaries[((1, 1, 1), 1)]) == 4
    assert len(lattice.link_coboundaries[((0, 0, 0), 1)]) == 2


def test_link_coboundary_sizes_on_3x2x2_open():
    """On 3x2x2 open, y and z have two sites, so every x-link is in one xy
    and one xz plaquette. A y-link is in one yz plaquette, and in xy
    plaquettes on both sides when x = 1 but on one side when x = 0 or 2; the
    same holds for z-links with xz. So the four y- and z-links at x = 1 lie
    in three plaquettes and the other sixteen links in two."""
    lattice = _lattice("3x2x2 open")
    links_in_three_plaquettes = {
        ((1, 0, 0), 2),
        ((1, 0, 1), 2),
        ((1, 0, 0), 3),
        ((1, 1, 0), 3),
    }
    coboundary_sizes = {
        link_address: len(plaquette_signs)
        for link_address, plaquette_signs in lattice.link_coboundaries.items()
    }
    assert coboundary_sizes == {
        link: 3 if link in links_in_three_plaquettes else 2 for link in lattice.links
    }
    assert len(coboundary_sizes) == 20


def test_link_in_no_plaquette_has_empty_coboundary():
    """A 3x1x1 lattice is a line of links with no plaquette at all."""
    lattice = LatticeDef(num_sites=(3, 1, 1), PBCs=OPEN, FORDER=FORDER)
    assert lattice.plaquettes == {}
    assert len(lattice.link_coboundaries) == 2
    assert all(
        len(plaquette_signs) == 0
        for plaquette_signs in lattice.link_coboundaries.values()
    )


# ---------- orientation against the matrix elements ----------


@pytest.mark.parametrize("name", ["2x2x1 open", "2x2x1 periodic in x and y"])
def test_box_changes_active_link_n_ality_by_orientation(name):
    """Under every matrix element of the plaquette operator, the N-ality of
    active link l changes by plaquette_boundaries[p][l] mod N: +1 on l1 and l2,
    which the Wilson loop traverses forward, -1 on l3 and l4. The other
    tests check that the signs agree with each other, which flipping every
    sign would not break; this one ties them to the Wilson loop. On the
    periodic lattice, the links l3 and l4 of some plaquettes wrap around."""
    num_sites, pbcs = {
        "2x2x1 open": ((2, 2, 1), OPEN),
        "2x2x1 periodic in x and y": ((2, 2, 1), (True, True, False)),
    }[name]
    lattice = LatticeDef(num_sites=num_sites, PBCs=pbcs, FORDER=FORDER)
    truncation_irreps, singlets, conj_dict = irreps_and_singlets(
        N_COLORS, lattice.sites, "B", 3
    )
    matrix_element_count = 0
    mismatches = []
    for plaquette_address, plaquette_data in lattice.plaquettes.items():
        expected = tuple(
            lattice.plaquette_boundaries[plaquette_address][link] % N_COLORS
            for link in plaquette_data.active_links
        )
        matrix_elements = calc_plaquette_elements(
            N_COLORS,
            plaquette_address,
            lattice.sites,
            lattice.plaquettes,
            truncation_irreps,
            singlets,
            conj_dict,
            lattice.FORDER,
            EPS,
            PRES,
            parallelize=False,
        )
        # Keys are (final state, initial state); a state starts with the
        # irreps of l1, l2, l3, l4.
        for final, initial in matrix_elements:
            matrix_element_count += 1
            observed = tuple(
                (n_ality(after) - n_ality(before)) % N_COLORS
                for before, after in zip(initial[:4], final[:4])
            )
            if observed != expected:
                mismatches.append((plaquette_address, initial, final))
    assert matrix_element_count > 0
    assert mismatches == []


# ---------- nothing a LatticeDef holds can be modified ----------


def test_fields_cannot_be_assigned():
    lattice = _lattice("3x2x2 open")
    for field, value in (
        ("num_sites", (4, 2, 2)),
        ("PBCs", PERIODIC),
        ("FORDER", tuple(FORDER)),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(lattice, field, value)


@pytest.mark.parametrize("name", LATTICES)
def test_computed_properties_cannot_be_assigned_or_modified(name):
    """Assigning a property fails, and so does modifying its value at any
    level. hash() succeeds only on a value whose every level is immutable:
    it fails on any list, dict or set inside, at any depth."""
    lattice = _lattice(name)
    for prop in COMPUTED_PROPERTIES:
        value = getattr(lattice, prop)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(lattice, prop, value)
        assert isinstance(value, (frozendict, tuple)), prop
        hash(value)


def test_mapping_values_cannot_be_modified():
    """Item assignment fails on the outer and inner mappings."""
    lattice = _lattice("3x2x2 open")
    plaquette_address = ((0, 0, 0), (1, 2))
    with pytest.raises(TypeError):
        lattice.plaquette_boundaries[plaquette_address] = {}  # ty: ignore[invalid-assignment]  # on purpose
    with pytest.raises(TypeError):
        lattice.plaquette_boundaries[plaquette_address][((0, 0, 0), 1)] = -1  # ty: ignore[invalid-assignment]  # on purpose
    with pytest.raises(TypeError):
        lattice.sites[(0, 0, 0)] = (1,)  # ty: ignore[invalid-assignment]  # on purpose


def test_fields_are_stored_as_tuples():
    """Lists given for the fields are stored as tuples, so the lattice is
    hashable and equal to one built from tuples."""
    # Lists for num_sites and PBCs, on purpose.
    from_lists = LatticeDef(
        num_sites=[3, 2, 2],  # ty: ignore[invalid-argument-type]
        PBCs=[False, False, False],  # ty: ignore[invalid-argument-type]
        FORDER=FORDER,
    )
    from_tuples = LatticeDef(
        num_sites=(3, 2, 2), PBCs=OPEN, FORDER=(1, 2, 3, -1, -2, -3)
    )
    assert from_lists.num_sites == (3, 2, 2)
    assert from_lists.PBCs == OPEN
    assert from_lists.FORDER == tuple(FORDER)
    assert from_lists == from_tuples
    assert hash(from_lists) == hash(from_tuples)


def test_copies_and_pickles():
    """copy.copy, copy.deepcopy and pickling give equal lattices with equal
    cell data, also after the cell data has been computed."""
    lattice = _lattice("3x2x2 periodic in y")
    for prop in COMPUTED_PROPERTIES:
        getattr(lattice, prop)
    for duplicate in (
        copy.copy(lattice),
        copy.deepcopy(lattice),
        pickle.loads(pickle.dumps(lattice)),
    ):
        assert duplicate == lattice
        for prop in COMPUTED_PROPERTIES:
            assert getattr(duplicate, prop) == getattr(lattice, prop), prop


def test_documented_mutable_copy():
    """The mutable copy the docstrings describe is made of plain dicts, and
    changing it leaves the lattice unchanged."""
    lattice = _lattice("3x2x2 open")
    boundary = {
        plaquette_address: dict(link_signs)
        for plaquette_address, link_signs in lattice.plaquette_boundaries.items()
    }
    changed_plaquette_address = ((0, 0, 0), (1, 2))
    link = ((0, 0, 0), 1)
    boundary[changed_plaquette_address][link] = -1
    assert lattice.plaquette_boundaries[changed_plaquette_address][link] == 1
