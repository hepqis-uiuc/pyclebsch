# pyclebsch Coverage Improvement

Complexity: medium
Status: **COMPLETE — all 5 phases done, 193 tests passing (started at 14).**

## Goal

Raise test coverage of `pyclebsch` by adding tests that exercise the public API of each module. Avoid modifying package source unless absolutely necessary; new code goes under `tests/`.

## Current State (audit summary)

Existing test files in `tests/`:
- `test_cgc_cache.py` — `set_cache_dir/get_cache_dir`, caching side effects, one `check_cgcs([(2,1,0),(2,1,0)])` correctness check.
- `test_lattice_data.py` — `LatticeDef.planes`, `LatticeDef.site_exists`, `compute_plaquette_signature`.

Modules with **no direct test file**:
- `pyclebsch/su_n_operators.py` (565 lines, ~10 public functions)
- `pyclebsch/symmetric_group/tableaux.py`
- `pyclebsch/symmetric_group/young_symmetrizer.py`
- `pyclebsch/symmetric_group/plethysm_utils.py` (all helpers private; covered indirectly via `find_plethysms`)
- `pyclebsch/matrix_elements/helpers.py`
- `pyclebsch/matrix_elements/plaquette_matrix_elements.py`

Partially tested:
- `pyclebsch/cgc.py` — only cache + one correctness check; no tests for return-shape filtering, `calc_lower_weight_cgcs`, `print_cgcs`, error paths.
- `pyclebsch/matrix_elements/lattice_data.py` — no tests for `sites_links_and_plaquettes`, `irreps_and_singlets`, `physical_plaquette_states`.

Note: SKILL.md says "Python 3.14+" but `pyproject.toml` says `>=3.12`; SKILL.md describes `find_suN_basis` and `print_cgcs` which exist in the source. The `B` and `C` truncation modes documented are present in `irreps_and_singlets`. SKILL.md appears largely accurate; minor discrepancies noted but **do not affect test design**.

## Conventions for all new tests

- Tests live in `tests/` with `test_*.py` naming.
- Use the autouse fixture pattern (already in `test_cgc_cache.py`) to redirect the CGC cache to `tmp_path` so tests don't pollute the repo `CGC_Data/`.
- Mark anything that triggers full CGC computation for non-trivial irreps with `@pytest.mark.slow`.
- Prefer pure-Python expectations (no recomputation via the same code path being tested).
- Use SU(2) and SU(3) low-dim irreps for hand-checkable expected values.

## Phase 1: Plaquette matrix element tests (new file `tests/test_plaquette_matrix_elements.py`)

**Status: COMPLETE.** 2 tests passing.

- [x] Build the file with module-scoped fixtures for cache redirection (`cgc.set_cache_dir → tmp_path_factory`) and lattice construction (`num_sites=[2,2,1], PBCs=[T,T,F], SU(3), T=1`).
  - Acceptance met: pytest collects without import errors.

- [x] **Shape / non-emptiness smoke test**: every plaquette yields a non-empty dict; every key is a pair of 12-tuples with 4 active i-weights, 4 control tuples, 4 integer multiplicity indices; every value is a float.
  - Acceptance met.

- [x] **`parallelize=True` vs `parallelize=False` consistency**: identical dicts (`math.isclose`, `abs_tol=1e-10`) across both paths for every plaquette.
  - Acceptance met.

- [~] ~~Hermiticity / orientation-reversal symmetry test~~: **dropped per user decision.** Strict Hermiticity does not hold because `calc_plaquette_elements` returns matrix elements of the unitary Wilson loop `U_□`, not the Hermitian Hamiltonian. The naive per-position active-link conjugation symmetry (`<Pf|U|Pi> == <P̄i|U|P̄f>`) holds only on vacuum-endpoint entries; deriving the full symmetry would require accounting for FORDER reversal in the per-site control orderings under direction reversal, which is out of scope for a coverage task.

## Phase 2: `su_n_operators.py` tests (new file `tests/test_su_n_operators.py`)

**Status: COMPLETE.** 50 tests passing.

- [x] `normalize_iweight`: subtracts last entry; idempotent on already-normalized inputs.
- [x] `calc_dimension`: SU(2) and SU(3) cases against Weyl formula expectations (parametrized).
- [x] `calc_casimir`: trivial=0, SU(3) fund=4/3, antifund=4/3, adjoint=3, plus normalization invariance.
- [x] `calc_dynkin_index`: SU(3) fund=1/2, adjoint=3.
- [x] `calc_weight`: SU(2) fund HW/LW z- and p-weight values; invalid kind raises `ValueError`.
- [x] `find_gt_patterns`: count = `calc_dimension` (parametrized over 7 irreps); index 0 is highest-weight pattern; sorted by p-weight descending.
- [x] `ladder_op`: J± annihilate HW/LW correctly; J− HW = LW with coefficient 1; J+ LW = HW with coefficient 1 (SU(2) fundamental).
- [x] `find_suN_basis`: generator count = N²−1 for SU(2) and SU(3); fundamental basis matrices are exactly (σ_z/2, σ_x/2, σ_y/2) for SU(2); `[T_x, T_y] = i T_z`; `Tr(T_a²) = 1/2` (standard fundamental normalization).
- [x] `find_direct_sum`: SU(2) 2⊗2; SU(3) 3⊗3̄, 3⊗3, 8⊗8 (the last includes (2,1,0):2); `sum_iweight` argument returns int multiplicity (including 0).
- [x] `find_symmetry_direct_sum`: SU(2) and SU(3) fund⊗fund — 6/triplet symmetric, 3̄/singlet antisymmetric; `idx_list == [[0,1]]`. **Discovery:** the S_n irrep keys in the inner dict are nested tuples `((partition,),)` (one partition per repeated-irrep group), not bare `(partition,)`. Test expectations adjusted accordingly.
- [x] `find_plethysms`: SU(2) n=2; SU(3) n=2; SU(3) n=3. **Discovery:** SU(3) fund n=3 has the adjoint (2,1,0) under partition (2,1) with multiplicity **1**, not 2 as the plan originally noted. The factor of 2 in `dim(adjoint) × 2 = 16` comes from `dim((2,1))=2` of S_3, not from the plethysm multiplicity.

8⊗8 ran fast enough (≈0.5s) that `@pytest.mark.slow` was unnecessary.

## Phase 3: `symmetric_group/` tests (new file `tests/test_symmetric_group.py`)

**Status: COMPLETE.** 61 tests passing.

- [x] `find_partitions(n)`: count matches OEIS A000041 for n=0..7; exact content for n=4; descending-order invariant; sum-to-n invariant; n=0 yields `[[0]]` (algorithm quirk documented).
- [x] `find_tableaux(partition)`: count vs. hook-length formula `n!/Π(hooks)` (parametrized over [2,1], [3,1], [2,2], [3,2,1], [4,2]); standalone count assertions for 11 partitions including the 1D edge cases; cell-value set check; row/column strict-increase check; exact content for [2,1]; `ValueError` on unsorted partition input.
- [x] `partition_to_sequence` / `sequence_to_partition`: round-trip across 6 partitions plus explicit expected sequences for `[3,2,1]`, `[4]`, `[1,1,1,1]`.
- [x] `_product_of_hook_lengths`: hand-verified hook products for 8 partitions including `[2,1]→3`, `[3,1]→8`, `[2,2]→12`, `[4]→24`.
- [x] `young_symmetrizer`:
  - Trivial n=1 yields `{(0,): 1.0}`.
  - Full symmetric/antisymmetric tableaux for n=2, n=3 — exact coefficients and signs verified against `(1/n!)·Σ_p (sgn?) [P]`.
  - **Idempotency Y² = Y verified on 8 tableaux** covering: pure symmetric, pure antisymmetric, row-ordered mixed `[[0,1],[2]]`, **non-row/column-ordered mixed `[[0,2],[1]]`** (exercises the MOLD descent branch), and n=4 cases `[[0,1,2],[3]]` and `[[0,1],[2,3]]`. Group-algebra composition implemented in test helper `_compose_symmetrizers`.

## Phase 4: `cgc.py` extended tests (new file `tests/test_cgc_api.py`)

**Status: COMPLETE.** 21 tests passing.

- [x] `calc_cgcs` return-shape filtering: 5 levels (0..4 optional args), each verified to return the correct dict-of-dict-of-dict structure or float; sum irreps `{(2,1,0), (0,0,0)}` for 3⊗3̄; multiplicity 1 for both.
- [x] Error paths: invalid `sum_state` raises `ValueError`; product_state with no CGC returns `0` (not `KeyError`).
- [x] Singlet checks for SU(3) 3⊗3̄: exactly 3 nonzero CGCs; each has magnitude 1/√3; Σ|CGC|² = 1; smallest tuple-ordered product_state has positive CGC (phase convention from source's phase-fix step).
- [x] `check_cgcs` for SU(2) 2⊗2 (SU(3) 3⊗3̄ and 8⊗8 already covered in `test_cgc_cache.py`).
- [x] `print_cgcs` smoke tests: captures stdout via `capsys`, asserts non-empty output and the presence of expected formatting markers (the `#` banner, "decomposition state:") at each of the 5 filter levels.
- [x] Lower-weight algorithm path: SU(3) 3⊗3 → 6 has 6 sum_states 0..5; orthonormality (Σ|CGC|² = 1) holds for every sum_state including the lowest-weight one (verifies the lower-weight descent end-to-end); 3⊗3 → 3̄ similarly has 3 sum_states.

## Phase 5: `matrix_elements/helpers.py` and `lattice_data.py` gap tests

**Status: COMPLETE.** 29 tests in new `tests/test_helpers.py`; 16 new tests appended to `tests/test_lattice_data.py`. All passing.

- [x] `conjugate_iweight`: parametrized over 10 SU(3) and SU(2) cases; involution check; dimension preservation.
- [x] `get_irreps`: fundamental Casimir cutoff (3 irreps); adjoint Casimir cutoff (4 irreps — **the 6 is excluded** since its Casimir is 10/3 > 3, plan was wrong here); above-sextet cutoff (6 irreps); monotonicity; zero-cutoff yields trivial only; SU(2) j=1/2 cutoff; values cross-check against `calc_casimir`.
- [x] `sites_links_and_plaquettes`: 2×2×1 PBC-x (4 sites, 6 links, 2 plaquettes); 3×2×1 OBC (6 sites, 7 links, 2 plaquettes); 2×2×1 PBC-xy (4 sites with 4 half-links each, 8 links, 4 plaquettes); error paths for length-1 PBC axis, zero sites, and bad FORDER.
- [x] `irreps_and_singlets`: T=1 SU(3) on d=2 PBC 2×2 — `link_irreps[4] = {trivial, fund, afund}`; all-trivial singlet present with multiplicity 1; `conj_dict` covers exactly the irreps in `link_irreps`; trivial maps to `{0: (0, +1)}`; fundamental maps bijectively onto the 3 antifund states with ±1 phases; C-mode and B-mode runs to completion; error paths for invalid truncation mode and invalid N.
- [x] `physical_plaquette_states`: SU(3) T=1 PBC d=2 2×2 — non-empty list; every element is a 12-tuple with correct slot structure; all-trivial state appears.

**Plan error caught (during writing):** the `(2,0,0)` and `(2,2,0)` 6/6̄ have Casimir 10/3, not below 3, so they are not admitted at `max_casimir=3`. The test that hand-derived this caught the plan's mistake on first execution; corrected expectation locked in.

## Out of scope

- Performance/timing tests.
- Tests requiring `ymcirc` integration (JSON generation script).
- Tests for the demo / Wilson-loop scripts under `run/`.
- Modifying any package code (only new files under `tests/`).
