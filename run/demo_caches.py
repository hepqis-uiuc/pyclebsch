"""
Demonstrates pyclebsch's three caches: the CGC cache (in memory, and
optionally on disk), and the in-memory caches of physical plaquette states
and of plaquette matrix elements. Each step prints the cache's counters,
which accumulate over the whole process.

Run with: uv run -m run.demo_caches
"""

import tempfile

import pyclebsch
from pyclebsch.cgc import calc_cgcs
from pyclebsch.matrix_elements.lattice_data import (
    LatticeDef,
    LinkDirection,
    physical_plaquette_states_by_signature,
    plaquettes_by_signature,
)
from pyclebsch.matrix_elements.plaquette_matrix_elements import (
    plaquette_matrix_elements_by_signature,
)

FORDER: list[LinkDirection] = [1, 2, 3, -1, -2, -3]
N = 3
EPS = 1e-10
PRES = 10
TRUNCATION_MODE = "B"
CUTOFF = 3

# 3 x 3bar x 8 in SU(3).
PRODUCT = [(1, 0, 0), (1, 1, 0), (2, 1, 0)]


def demo_cgc_cache() -> None:
    print("CGC cache, in memory")
    pyclebsch.set_cgc_cache_dir(None)  # no disk tier, the default
    calc_cgcs(PRODUCT)
    print("  first call (tables computed):", pyclebsch.cgc_cache_stats())
    calc_cgcs(PRODUCT)
    print("  second call (memory hits):   ", pyclebsch.cgc_cache_stats())

    pyclebsch.set_cgc_memory_cache_limit(1)
    pyclebsch.clear_memory_cache()
    calc_cgcs(PRODUCT)
    print("  with a limit of 1 table:     ", pyclebsch.cgc_cache_stats())
    pyclebsch.set_cgc_memory_cache_limit(None)

    print("CGC cache, on disk")
    with tempfile.TemporaryDirectory() as directory:
        pyclebsch.set_cgc_cache_dir(directory)  # also clears the CGC memory tier
        calc_cgcs(PRODUCT)
        print("  computed, written to disk:   ", pyclebsch.cgc_cache_stats())
        pyclebsch.clear_memory_cache()
        calc_cgcs(PRODUCT)
        print("  read back from disk:         ", pyclebsch.cgc_cache_stats())
        pyclebsch.set_cgc_cache_dir(None)


def demo_plaquette_caches() -> None:
    open_4x4x4 = LatticeDef(
        num_sites=(4, 4, 4), PBCs=(False, False, False), FORDER=FORDER
    )
    open_6x6x6 = LatticeDef(
        num_sites=(6, 6, 6), PBCs=(False, False, False), FORDER=FORDER
    )
    for name, lattice in [("4x4x4", open_4x4x4), ("6x6x6", open_6x6x6)]:
        print(
            f"Open {name} lattice: {len(lattice.plaquettes)} plaquettes, "
            f"{len(plaquettes_by_signature(lattice))} signatures"
        )

    print(f"Physical plaquette states, {TRUNCATION_MODE}{CUTOFF} truncation")
    physical_plaquette_states_by_signature(open_4x4x4, N, TRUNCATION_MODE, CUTOFF)
    print("  4x4x4, first call: ", pyclebsch.physical_plaquette_states_cache_stats())
    physical_plaquette_states_by_signature(open_4x4x4, N, TRUNCATION_MODE, CUTOFF)
    print("  4x4x4, second call:", pyclebsch.physical_plaquette_states_cache_stats())
    # Every signature of the open 6x6x6 lattice also occurs on 4x4x4, with the
    # same unique controls (on an open lattice no two controls coincide), so
    # 6x6x6 is served entirely from the cache.
    physical_plaquette_states_by_signature(open_6x6x6, N, TRUNCATION_MODE, CUTOFF)
    print("  6x6x6, first call: ", pyclebsch.physical_plaquette_states_cache_stats())

    print(f"Plaquette matrix elements, {TRUNCATION_MODE}{CUTOFF} truncation")
    pyclebsch.set_plaquette_matrix_elements_cache_limit(40)
    plaquette_matrix_elements_by_signature(
        open_4x4x4, N, TRUNCATION_MODE, CUTOFF, EPS, PRES, parallelize=False
    )
    print(
        "  4x4x4, limit of 40 tables:",
        pyclebsch.plaquette_matrix_elements_cache_stats(),
    )
    pyclebsch.set_plaquette_matrix_elements_cache_limit(None)

    # The returned tables are copies: modifying one leaves the cache intact.
    states = physical_plaquette_states_by_signature(
        open_4x4x4, N, TRUNCATION_MODE, CUTOFF
    )
    signature = next(iter(states))
    states[signature].clear()
    again = physical_plaquette_states_by_signature(
        open_4x4x4, N, TRUNCATION_MODE, CUTOFF
    )
    print(
        f"A returned table, emptied: {len(states[signature])} states; "
        f"the same table from the cache: {len(again[signature])} states"
    )

    pyclebsch.clear_memory_cache()  # drops the CGC and plaquette tables alike


def main() -> None:
    demo_cgc_cache()
    demo_plaquette_caches()


if __name__ == "__main__":
    main()
