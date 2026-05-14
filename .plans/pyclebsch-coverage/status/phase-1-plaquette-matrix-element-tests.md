# Phase 1: Plaquette matrix element tests — COMPLETE

## What was done

Created `tests/test_plaquette_matrix_elements.py` with:

- A module-scoped fixture redirecting the on-disk CGC cache to a `tmp_path_factory` directory (no pollution of repo `CGC_Data/`).
- A module-scoped `lattice_objects` fixture that builds sites/plaquettes/singlets for `num_sites=[2,2,1], PBCs=[T,T,F], SU(3), T=1` once.
- A module-scoped `matrix_elements_sequential` fixture that computes matrix elements with `parallelize=False` once per module run.

Two tests:

1. **`test_calc_plaquette_elements_shape`** — every plaquette yields a non-empty dict; every key is a pair of 12-tuples with 4 active i-weights, 4 control tuples, 4 integer multiplicity indices; every value is a float. **PASSED.**
2. **`test_calc_plaquette_elements_parallelize_consistency`** — `parallelize=True` and `parallelize=False` produce identical dicts within `1e-10`. **PASSED.**

Final test run: 16 passed total (14 pre-existing + 2 new).

## What was dropped

A Hermiticity / orientation-reversal symmetry test was originally in the plan. Two iterations were attempted and failed:

1. Strict Hermiticity `H[Pf,Pi] == H[Pi,Pf]` — fails because `calc_plaquette_elements` returns matrix elements of the unitary Wilson loop `U_□`, not the Hermitian Hamiltonian `U + U†`.
2. Per-position active-link conjugation `<Pf|U|Pi> == <P̄i|U|P̄f>` — holds on vacuum-endpoint entries (where my observational evidence came from) but fails on the first mixed-control entry. The likely missing piece is that reversing Wilson loop direction also implicitly reverses FORDER, which reorders per-site control tuples. Deriving the full symmetry is out of scope.

User decision: **drop the symmetry test.** Recorded in plan.md.

## Coverage gained

`pyclebsch/matrix_elements/plaquette_matrix_elements.py` now has direct test coverage of:
- `calc_plaquette_elements` end-to-end on a 4-plaquette PBC d=2 lattice
- Both `parallelize=True` (multiprocessing path) and `parallelize=False` (sequential path) execute and agree
- The full glue + site-factor pipeline (`plaquette_site_factor`, `calc_plaquette_site_factors`, `glue_plaquette_site_factors`) runs to completion

Also indirectly exercises `lattice_data.sites_links_and_plaquettes` and `lattice_data.irreps_and_singlets` for SU(3), T=1, which Phase 5 will hit directly.

## Status

**COMPLETE.** Ready to proceed to Phase 2 (`test_su_n_operators.py`).
