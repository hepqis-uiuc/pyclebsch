# Overview: pyclebsch Coverage Improvement — COMPLETE

**Status: all 5 phases complete.**

| Phase | File(s) | Tests added | Result |
|---|---|---|---|
| 1 | `tests/test_plaquette_matrix_elements.py` (new) | 2 | Hermiticity/symmetry test dropped after investigation (Wilson loop is unitary, not Hermitian) |
| 2 | `tests/test_su_n_operators.py` (new) | 50 | Caught 2 plan errors during writing |
| 3 | `tests/test_symmetric_group.py` (new) | 61 | Young symmetrizer idempotency verified including MOLD-descent branch |
| 4 | `tests/test_cgc_api.py` (new) | 21 | All 5 return-shape branches + singlet phase convention |
| 5 | `tests/test_helpers.py` (new), `tests/test_lattice_data.py` (extended) | 29 + 16 | Caught 1 plan error during writing |

**Final test count: 193 passing (started at 14).** Net gain: **179 tests** across all of `pyclebsch`'s public API.

## Modules covered

- `pyclebsch/cgc.py` — cache API, 5 return-shape branches, error paths, phase convention, lower-weight algorithm, `print_cgcs` at all 5 levels, `check_cgcs` for SU(2)/SU(3).
- `pyclebsch/su_n_operators.py` — dimension/Casimir/Dynkin, GT patterns, ladder ops, basis generators, direct sum & symmetry-aware direct sum, plethysms.
- `pyclebsch/symmetric_group/tableaux.py` — partitions, tableaux, hook lengths, partition↔sequence round-trips.
- `pyclebsch/symmetric_group/young_symmetrizer.py` — trivial sym/antisym cases plus Y² = Y idempotency on 8 representative tableaux including the MOLD-descent branch.
- `pyclebsch/matrix_elements/helpers.py` — `conjugate_iweight`, `get_irreps`.
- `pyclebsch/matrix_elements/lattice_data.py` — `LatticeDef`, `sites_links_and_plaquettes`, `irreps_and_singlets` (T/C/B modes), `physical_plaquette_states`, `compute_plaquette_signature`.
- `pyclebsch/matrix_elements/plaquette_matrix_elements.py` — `calc_plaquette_elements` shape + parallelize-consistency.

## Plan errors caught before they reached production

1. **Phase 2**: SU(3) fund n=3 plethysm — adjoint multiplicity is **1**, not 2. The factor of 2 is `dim_{S_3}((2,1))`.
2. **Phase 2**: `find_symmetry_direct_sum` returns nested `((partition,),)` keys (one partition per repeated-irrep group), not bare `(partition,)`.
3. **Phase 5**: `get_irreps(max_casimir=3, N=3)` does **not** include 6 = (2,0,0) or 6̄ = (2,2,0) — their Casimir is 10/3, not below 3. Plan said the cutoff would admit them; the 6 and 6̄ require `max_casimir ≥ 10/3`.

## What was deliberately dropped

- **Phase 1 Hermiticity/symmetry test**: `calc_plaquette_elements` returns matrix elements of the unitary Wilson loop `U_□`, not the Hermitian Hamiltonian `U + U†`. The intended orientation-reversal symmetry interacts with FORDER's per-site half-link ordering in non-trivial ways. Deriving and testing the full symmetry was out of scope; coverage of the function still gained via the shape and parallelize-consistency tests.

## Files produced (under `tests/`)

- `tests/test_plaquette_matrix_elements.py` (new, 2 tests)
- `tests/test_su_n_operators.py` (new, 50 tests)
- `tests/test_symmetric_group.py` (new, 61 tests)
- `tests/test_cgc_api.py` (new, 21 tests)
- `tests/test_helpers.py` (new, 29 tests)
- `tests/test_lattice_data.py` (extended, +16 tests)

No package code modified.
