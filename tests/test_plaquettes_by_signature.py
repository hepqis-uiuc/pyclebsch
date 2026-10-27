"""Tests for the plaquette tables computed once per plaquette signature.

Every comparison here is exact (==): both sides run the same code on the same
inputs, with parallelize=False so that no summation order can differ.
"""

from collections import Counter, defaultdict

import pytest

from pyclebsch import cache
from pyclebsch.matrix_elements import lattice_data, plaquette_matrix_elements
from pyclebsch.matrix_elements.lattice_data import (
    LatticeDef,
    LinkDirection,
    compute_plaquette_signature,
    irreps_and_singlets,
    physical_plaquette_states,
    physical_plaquette_states_by_signature,
    plaquettes_by_signature,
)
from pyclebsch.matrix_elements.plaquette_matrix_elements import (
    calc_plaquette_elements,
    plaquette_matrix_elements_by_signature,
)

FORDER: list[LinkDirection] = [1, 2, 3, -1, -2, -3]
N_COLORS = 3
EPS = 1e-10
PRES = 10

OPEN = (False, False, False)
PERIODIC = (True, True, True)

# The lattices of the brute-force comparison, as (num_sites, PBCs).
LATTICES: dict[str, tuple[tuple[int, int, int], tuple[bool, bool, bool]]] = {
    "2x2x2 open": ((2, 2, 2), OPEN),
    "3x2x2 open": ((3, 2, 2), OPEN),
    "4x3x1 open": ((4, 3, 1), OPEN),
    "3x2x1 periodic in x": ((3, 2, 1), (True, False, False)),
    "2x2x1 periodic in x and y": ((2, 2, 1), (True, True, False)),
    "2x2x2 periodic": ((2, 2, 2), PERIODIC),
    "3x3x2 periodic in x and y": ((3, 3, 2), (True, True, False)),
}

# Truncations as (mode, cutoff). B4 has B3's irreps (1, 3, 3bar) with more
# physical states; B6 adds 6, 6bar and 8 (three-bit links) and site singlets
# of multiplicity greater than 1; B7 has B6's irreps with more states; T1 has
# B3's irreps with no site-level cutoff.
ALL_TRUNCATIONS = (("T", 1), ("B", 3), ("B", 4), ("B", 6), ("B", 7))

# The periodic d=3 lattices at T1, B6 and B7 take minutes or several GB of
# memory per plaquette, so they are compared at B3 and B4 only.
LATTICE_TRUNCATIONS = {
    name: (("B", 3), ("B", 4))
    if name in ("2x2x2 periodic", "3x3x2 periodic in x and y")
    else ALL_TRUNCATIONS
    for name in LATTICES
}

SLOW_CELLS = {
    ("2x2x2 periodic", ("B", 4)),
    ("3x3x2 periodic in x and y", ("B", 4)),
    ("3x2x2 open", ("B", 7)),
}

TABLE = [
    pytest.param(
        name,
        truncation,
        id=f"{name}-{truncation[0]}{truncation[1]}",
        marks=[pytest.mark.slow] if (name, truncation) in SLOW_CELLS else [],
    )
    for name in LATTICES
    for truncation in LATTICE_TRUNCATIONS[name]
]

# The two plaquette caches, each as (label used by call_counts, function
# that sets its limit, name of its attribute on cache._plaquette_cache).
PLAQUETTE_CACHES = [
    pytest.param(
        "states",
        cache.set_physical_plaquette_states_cache_limit,
        "physical_states",
        id="physical states",
    ),
    pytest.param(
        "matrix_elements",
        cache.set_plaquette_matrix_elements_cache_limit,
        "matrix_elements",
        id="matrix elements",
    ),
]


def _lattice(name: str) -> LatticeDef:
    num_sites, pbcs = LATTICES[name]
    return LatticeDef(num_sites=num_sites, PBCs=pbcs, FORDER=FORDER)


@pytest.fixture
def plaquette_caches_cleared():
    """Start with empty plaquette caches; remove any cap a test sets."""
    cache._plaquette_cache.clear()
    yield
    cache.set_physical_plaquette_states_cache_limit(None)
    cache.set_plaquette_matrix_elements_cache_limit(None)
    cache.set_cgc_memory_cache_limit(None)


@pytest.fixture
def call_counts(monkeypatch):
    """Count the calls the by-signature functions make to
    physical_plaquette_states ("states") and calc_plaquette_elements
    ("matrix_elements"), each wrapped so that it still computes its result."""
    counts: Counter[str] = Counter()

    def counting(label, function):
        def wrapper(*args, **kwargs):
            counts[label] += 1
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        lattice_data,
        "physical_plaquette_states",
        counting("states", physical_plaquette_states),
    )
    monkeypatch.setattr(
        plaquette_matrix_elements,
        "calc_plaquette_elements",
        counting("matrix_elements", calc_plaquette_elements),
    )
    return counts


def _states_by_signature(lattice: LatticeDef, truncation: tuple[str, int]) -> dict:
    mode, cutoff = truncation
    return physical_plaquette_states_by_signature(lattice, N_COLORS, mode, cutoff)


def _matrix_elements_by_signature(
    lattice: LatticeDef, truncation: tuple[str, int], eps: float = EPS, pres: int = PRES
) -> dict:
    mode, cutoff = truncation
    return plaquette_matrix_elements_by_signature(
        lattice, N_COLORS, mode, cutoff, eps, pres, parallelize=False
    )


class _PerPlaquette:
    """physical_plaquette_states and calc_plaquette_elements on one lattice,
    called directly on each plaquette."""

    def __init__(self, lattice: LatticeDef, truncation: tuple[str, int]) -> None:
        mode, cutoff = truncation
        self.lattice = lattice
        self.irreps, self.singlets, self.conj_dict = irreps_and_singlets(
            N_COLORS, lattice.sites, mode, cutoff
        )

    def states(self, plaquette):
        lattice = self.lattice
        return physical_plaquette_states(
            plaquette, lattice.sites, lattice.plaquettes, self.singlets, lattice.FORDER
        )

    def matrix_elements(self, plaquette):
        lattice = self.lattice
        return calc_plaquette_elements(
            N_COLORS,
            plaquette,
            lattice.sites,
            lattice.plaquettes,
            self.irreps,
            self.singlets,
            self.conj_dict,
            lattice.FORDER,
            EPS,
            PRES,
            parallelize=False,
        )


@pytest.mark.parametrize(("lattice_name", "truncation"), TABLE)
def test_signature_determines_results_within_lattice(
    plaquette_caches_cleared, lattice_name, truncation
):
    """For every plaquette, the results computed on the plaquette itself equal
    the by-signature results for its signature. The caches start empty, so
    the by-signature results are computed on this lattice."""
    lattice = _lattice(lattice_name)
    states_by_signature = _states_by_signature(lattice, truncation)
    matrix_elements_by_signature = _matrix_elements_by_signature(lattice, truncation)
    per_plaquette = _PerPlaquette(lattice, truncation)
    for plaquette in sorted(lattice.plaquettes):
        signature = compute_plaquette_signature(plaquette, lattice)
        assert per_plaquette.states(plaquette) == states_by_signature[signature], (
            plaquette
        )
        assert (
            per_plaquette.matrix_elements(plaquette)
            == matrix_elements_by_signature[signature]
        ), plaquette


@pytest.mark.parametrize(
    "truncation",
    [
        pytest.param(
            truncation,
            id=f"{truncation[0]}{truncation[1]}",
            marks=[pytest.mark.slow] if truncation == ("B", 7) else [],
        )
        for truncation in ALL_TRUNCATIONS
    ],
)
def test_cache_keys_complete_across_lattices(
    plaquette_caches_cleared, call_counts, truncation
):
    """A cached result whose key occurs on several lattices of the table
    equals the result computed directly on each of them.

    The key of a plaquette's results, besides inputs common to the whole
    table, is its signature and its unique controls. The first lattice with a
    key fills the cache; on every later lattice with that key the
    by-signature functions must serve it from the cache (they compute only
    the keys no earlier lattice had), and it must equal the result computed
    directly on the later lattice."""
    names = [name for name in LATTICES if truncation in LATTICE_TRUNCATIONS[name]]
    lattices = {name: _lattice(name) for name in names}

    # Smallest plaquette address per lattice for each (signature, unique
    # controls), found directly from the plaquettes.
    occurrences: dict[tuple, dict[str, tuple]] = defaultdict(dict)
    for name, lattice in lattices.items():
        for plaquette in sorted(lattice.plaquettes):
            unique_controls = tuple(tuple(c) for c in lattice.plaquettes[plaquette][3])
            key = (compute_plaquette_signature(plaquette, lattice), unique_controls)
            occurrences[key].setdefault(name, plaquette)
    shared = {
        key: by_lattice
        for key, by_lattice in occurrences.items()
        if len(by_lattice) > 1
    }
    assert shared, (
        "no key occurs on more than one lattice; the test would check nothing"
    )

    seen_keys: set[tuple] = set()
    for name, lattice in lattices.items():
        keys = {key for key, by_lattice in occurrences.items() if name in by_lattice}
        num_new_keys = len(keys - seen_keys)
        seen_keys |= keys
        call_counts.clear()
        states_by_signature = _states_by_signature(lattice, truncation)
        matrix_elements_by_signature = _matrix_elements_by_signature(
            lattice, truncation
        )
        assert call_counts == Counter(
            states=num_new_keys, matrix_elements=num_new_keys
        ), name
        per_plaquette = None
        for key, by_lattice in shared.items():
            if name not in by_lattice:
                continue
            signature, _ = key
            states = states_by_signature[signature]
            matrix_elements = matrix_elements_by_signature[signature]
            per_plaquette = per_plaquette or _PerPlaquette(lattice, truncation)
            plaquette = by_lattice[name]
            assert per_plaquette.states(plaquette) == states, (name, plaquette)
            assert per_plaquette.matrix_elements(plaquette) == matrix_elements, (
                name,
                plaquette,
            )


@pytest.mark.parametrize(
    ("num_sites", "pbcs", "expected_groups"),
    [
        ((2, 2, 2), OPEN, 6),
        ((3, 2, 2), OPEN, 11),
        ((3, 3, 3), OPEN, 36),
        ((6, 6, 6), OPEN, 81),
        ((3, 3, 3), PERIODIC, 3),
        ((2, 2, 2), PERIODIC, 3),
    ],
)
def test_signature_counts(num_sites, pbcs, expected_groups):
    lattice = LatticeDef(num_sites=num_sites, PBCs=pbcs, FORDER=FORDER)
    assert len(plaquettes_by_signature(lattice)) == expected_groups


@pytest.mark.parametrize("lattice_name", LATTICES)
def test_groups_partition_the_plaquettes(lattice_name):
    """Every plaquette is in exactly one group, the group of its signature,
    and each group is sorted."""
    lattice = _lattice(lattice_name)
    groups = plaquettes_by_signature(lattice)
    grouped = [plaquette for group in groups.values() for plaquette in group]
    assert sorted(grouped) == sorted(lattice.plaquettes)
    for signature, group in groups.items():
        assert group == sorted(group)
        assert all(
            compute_plaquette_signature(plaquette, lattice) == signature
            for plaquette in group
        )


def test_one_computation_per_signature(plaquette_caches_cleared, call_counts):
    """Each by-signature function computes once per signature, and a repeated
    call with the same arguments computes nothing."""
    cache.clear_memory_cache()
    truncation = ("B", 3)
    for num_sites, pbcs, num_signatures in [
        ((3, 2, 2), OPEN, 11),
        ((3, 3, 3), PERIODIC, 3),
    ]:
        lattice = LatticeDef(num_sites=num_sites, PBCs=pbcs, FORDER=FORDER)
        call_counts.clear()
        _states_by_signature(lattice, truncation)
        _matrix_elements_by_signature(lattice, truncation)
        assert call_counts == Counter(
            states=num_signatures, matrix_elements=num_signatures
        ), num_sites
        call_counts.clear()
        _states_by_signature(lattice, truncation)
        _matrix_elements_by_signature(lattice, truncation)
        assert call_counts == Counter(), num_sites


def test_modifying_returned_tables_leaves_cache_unchanged(
    plaquette_caches_cleared, call_counts
):
    """Emptying the returned tables changes neither the cache nor what a
    later call, served from the cache, returns."""
    lattice = LatticeDef(num_sites=(3, 2, 2), PBCs=OPEN, FORDER=FORDER)
    truncation = ("B", 3)
    states = _states_by_signature(lattice, truncation)
    matrix_elements = _matrix_elements_by_signature(lattice, truncation)
    assert all(states.values()) and any(matrix_elements.values())
    expected_states = {signature: list(t) for signature, t in states.items()}
    expected_matrix_elements = {
        signature: dict(t) for signature, t in matrix_elements.items()
    }
    for table in [*states.values(), *matrix_elements.values()]:
        table.clear()

    call_counts.clear()
    assert _states_by_signature(lattice, truncation) == expected_states
    assert (
        _matrix_elements_by_signature(lattice, truncation) == expected_matrix_elements
    )
    assert call_counts == Counter()


# Values of EPS and PRES other than the defaults, used only to give the
# matrix elements a different cache key; they need not be sensible choices.
OTHER_EPS = 1e-12
OTHER_PRES = 8


def test_eps_and_pres_affect_only_matrix_elements(
    plaquette_caches_cleared, call_counts
):
    """Changing EPS or PRES recomputes the matrix elements of every
    signature, and no physical states."""
    lattice = LatticeDef(num_sites=(3, 2, 2), PBCs=OPEN, FORDER=FORDER)
    truncation = ("B", 3)
    num_signatures = 11
    _states_by_signature(lattice, truncation)
    _matrix_elements_by_signature(lattice, truncation)
    for eps, pres in [(OTHER_EPS, PRES), (EPS, OTHER_PRES)]:
        call_counts.clear()
        _matrix_elements_by_signature(lattice, truncation, eps, pres)
        _states_by_signature(lattice, truncation)
        assert call_counts == Counter(matrix_elements=num_signatures), (eps, pres)


def _one_signature_lattices() -> list[LatticeDef]:
    """Three lattices with one plaquette signature each and three different
    cache keys, so that each by-signature call adds one entry per cache. The
    last two have the same signature and differ only in their unique
    controls."""
    lattices = [
        LatticeDef(num_sites=num_sites, PBCs=pbcs, FORDER=FORDER)
        for num_sites, pbcs in [
            ((3, 2, 1), (True, False, False)),
            ((2, 2, 1), (True, True, False)),
            ((3, 3, 1), (True, True, False)),
        ]
    ]
    signatures = [list(plaquettes_by_signature(lattice)) for lattice in lattices]
    assert [len(s) for s in signatures] == [1, 1, 1]
    assert signatures[1] == signatures[2]
    return lattices


def _compute(label: str, lattice: LatticeDef) -> None:
    """The by-signature table of kind label ("states" or "matrix_elements")
    on lattice, at B3."""
    truncation = ("B", 3)
    if label == "states":
        _states_by_signature(lattice, truncation)
    else:
        _matrix_elements_by_signature(lattice, truncation)


@pytest.mark.parametrize(("label", "set_limit", "_attribute"), PLAQUETTE_CACHES)
def test_plaquette_cache_limit(
    plaquette_caches_cleared, call_counts, label, set_limit, _attribute
):
    """With a cap of 2 entries, the least recently used entry is evicted, and
    the other plaquette cache, uncapped, evicts nothing meanwhile."""
    lattices = _one_signature_lattices()
    other_label = "matrix_elements" if label == "states" else "states"

    set_limit(2)
    for lattice in lattices:
        _compute(label, lattice)
    assert call_counts[label] == 3
    _compute(label, lattices[0])  # evicted as the least recently used
    assert call_counts[label] == 4
    _compute(label, lattices[2])  # still cached
    assert call_counts[label] == 4

    # Had the cap of 2 reached the other cache too, its second pass would
    # recompute the first lattice, evicted by the third.
    for _ in range(2):
        for lattice in lattices:
            _compute(other_label, lattice)
    assert call_counts[other_label] == 3
    assert call_counts[label] == 4

    set_limit(None)
    _compute(label, lattices[1])  # evicted above, so computed once more
    assert call_counts[label] == 5
    for lattice in lattices:
        _compute(label, lattice)
    assert call_counts[label] == 5

    cache.clear_memory_cache()
    assert len(cache._cgc_cache._memory) == 0
    for lattice in lattices:
        _compute(label, lattice)
    assert call_counts[label] == 8


def test_cgc_and_plaquette_limits_do_not_act_on_each_other(
    plaquette_caches_cleared, call_counts
):
    """Caps on the plaquette caches evict no CGC tables, and a cap on the CGC
    cache evicts no plaquette tables. Each half first checks that its cap
    does evict, so that it is not checking an idle cap."""
    lattices = _one_signature_lattices()
    labels = ("states", "matrix_elements")
    cache.clear_memory_cache()

    # Plaquette caps of 1: every new lattice evicts the previous one's tables.
    cache.set_physical_plaquette_states_cache_limit(1)
    cache.set_plaquette_matrix_elements_cache_limit(1)
    before = cache.cgc_cache_stats()
    for lattice in lattices:
        for label in labels:
            _compute(label, lattice)
    for label in labels:
        _compute(label, lattices[0])
    assert call_counts == Counter(states=4, matrix_elements=4)
    assert cache.cgc_cache_stats()["evictions"] == before["evictions"]

    # A CGC cap of 1, with the plaquette caps removed.
    cache.set_physical_plaquette_states_cache_limit(None)
    cache.set_plaquette_matrix_elements_cache_limit(None)
    cache.clear_memory_cache()
    cache.set_cgc_memory_cache_limit(1)
    call_counts.clear()
    before = cache.cgc_cache_stats()
    for _ in range(2):
        for lattice in lattices:
            for label in labels:
                _compute(label, lattice)
    assert cache.cgc_cache_stats()["evictions"] > before["evictions"]
    assert call_counts == Counter(states=3, matrix_elements=3)


@pytest.mark.parametrize(
    ("label", "set_limit", "stats", "other_stats"),
    [
        pytest.param(
            "states",
            cache.set_physical_plaquette_states_cache_limit,
            cache.physical_plaquette_states_cache_stats,
            cache.plaquette_matrix_elements_cache_stats,
            id="physical states",
        ),
        pytest.param(
            "matrix_elements",
            cache.set_plaquette_matrix_elements_cache_limit,
            cache.plaquette_matrix_elements_cache_stats,
            cache.physical_plaquette_states_cache_stats,
            id="matrix elements",
        ),
    ],
)
def test_plaquette_cache_stats(
    plaquette_caches_cleared, label, set_limit, stats, other_stats
):
    """Each plaquette cache counts its own hits, computations and evictions;
    the other plaquette cache's counters do not move. The counters run for
    the whole process, so the test compares differences. The keys are those
    of the CGC cache's statistics that apply to a cache with no disk tier."""
    lattices = _one_signature_lattices()
    before, other_before = stats(), other_stats()
    set_limit(2)
    for lattice in lattices:
        _compute(label, lattice)  # 3 computed; the third evicts the first
    _compute(label, lattices[2])  # 1 hit
    after = stats()
    assert {key: after[key] - before[key] for key in after} == {
        "memory_hits": 1,
        "computed": 3,
        "evictions": 1,
    }
    assert other_stats() == other_before
    assert set(after) < set(cache.cgc_cache_stats())


def test_cache_limits_are_independent(plaquette_caches_cleared):
    """Each of the three limits (CGC tables, physical plaquette states,
    plaquette matrix elements) changes only its own cache's cap."""
    plaquette_cache = cache._plaquette_cache

    def limits():
        return (
            cache._cgc_cache.max_memory_entries,
            plaquette_cache.physical_states.max_entries,
            plaquette_cache.matrix_elements.max_entries,
        )

    cache.set_cgc_memory_cache_limit(7)
    assert limits() == (7, None, None)
    cache.set_physical_plaquette_states_cache_limit(2)
    assert limits() == (7, 2, None)
    cache.set_plaquette_matrix_elements_cache_limit(3)
    assert limits() == (7, 2, 3)
    cache.set_cgc_memory_cache_limit(None)
    assert limits() == (None, 2, 3)
    cache.set_physical_plaquette_states_cache_limit(None)
    assert limits() == (None, None, 3)
    cache.set_plaquette_matrix_elements_cache_limit(None)
    assert limits() == (None, None, None)


@pytest.mark.parametrize(("_label", "set_limit", "attribute"), PLAQUETTE_CACHES)
def test_plaquette_cache_limit_rejects_invalid_values(
    plaquette_caches_cleared, _label, set_limit, attribute
):
    """The cap must be None or an integer of at least 1."""
    set_limit(5)
    for invalid in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            set_limit(invalid)
    assert getattr(cache._plaquette_cache, attribute).max_entries == 5
