# Phase 5: matrix_elements gap tests — COMPLETE

## What was done

Created **`tests/test_helpers.py`** with 29 tests covering `matrix_elements/helpers.py`:

- **`conjugate_iweight`** (16 tests): 10 parametrized explicit cases (SU(3) trivial/fund/antifund/8/6/6̄/10, plus SU(2) self-conjugates); involution check across 6 inputs; dimension preservation across 6 inputs.
- **`get_irreps`** (7 tests):
  - SU(3) at `max_casimir=4/3` → `{trivial, fund, antifund}`.
  - SU(3) at `max_casimir=3` (adjoint Casimir) → `{trivial, fund, antifund, adjoint}` — **6 and 6̄ are excluded** since their Casimir is 10/3.
  - SU(3) at `max_casimir=10/3` → adds 6 and 6̄.
  - Monotonicity (low cutoff ⊆ high cutoff).
  - `max_casimir=0` → trivial only.
  - SU(2) at j=1/2 Casimir (3/4) → `{trivial, j=1/2}`.
  - All returned Casimirs match `calc_casimir`.

Appended **16 tests to `tests/test_lattice_data.py`** for `lattice_data.py` gaps:

- Added autouse `restore_cache_dir` fixture so tests that need CGCs redirect to `tmp_path`.
- **`sites_links_and_plaquettes`** (6 tests): 3 lattice geometries with hand-verified site/link/plaquette counts; 3 error paths (length-1 PBC axis, zero sites, bad FORDER).
- **`irreps_and_singlets`** (9 tests): `link_irreps` content for T=1 SU(3); all-trivial singlet entry present; `conj_dict` covers exactly the `link_irreps` irreps; trivial conj_dict structure (single state, +1 phase); fundamental conj_dict bijects onto antifundamental with ±1 phases; C-mode and B-mode smoke tests; ValueError on invalid mode or N≤1.
- **`physical_plaquette_states`** (1 test): SU(3) T=1 on d=2 PBC 2×2 — non-empty list of correctly-structured 12-tuples, all-trivial state present.

## Discrepancies from plan expectations

**One plan error caught and corrected:**

The plan said `get_irreps(max_casimir=3, N=3)` should admit the 6 = (2,0,0) and 6̄ = (2,2,0). It doesn't: their Casimir is 10/3 ≈ 3.33, which is above 3. The first run of `test_get_irreps_su3_at_adjoint_casimir` failed and surfaced this; I re-derived the Casimirs (10/3 for the 6, exactly 3 for the adjoint) and corrected the expectation to `{trivial, fund, antifund, adjoint}` (4 irreps, not 6). Added a separate test at `max_casimir=10/3` that exercises the higher-T branch where the 6 and 6̄ are admitted.

## Coverage gained

- `pyclebsch/matrix_elements/helpers.py` — both public functions exercised.
- `pyclebsch/matrix_elements/lattice_data.py`:
  - `sites_links_and_plaquettes` direct counts + 3 error paths.
  - `irreps_and_singlets` — all three truncation modes (T/C/B), invalid-input branches, and the `conj_dict` construction loop that calls `calc_cgcs`.
  - `physical_plaquette_states` end-to-end on the smallest non-trivial PBC d=2 lattice.
  - `LatticeDef`, `compute_plaquette_signature` were already covered before Phase 5.

## Status

**COMPLETE.** Full suite: 193 passing (was 148 pre-Phase 5; started at 14 before the project).
