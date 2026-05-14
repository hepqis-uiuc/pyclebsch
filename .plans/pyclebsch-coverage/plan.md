# pyclebsch Coverage Improvement

Complexity: medium

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

## Phase 4: `cgc.py` extended tests (extend `tests/test_cgc_cache.py` or new `tests/test_cgc_api.py`)

- [ ] `calc_cgcs` return-shape filtering:
  - No optional args → outer dict keyed by sum irreps; for SU(3) 3⊗3̄, contains `(2,1,0)` and `(0,0,0)`.
  - With `sum_iweight=(0,0,0)` → dict keyed by mult indices.
  - With `mult_idx=1` → dict keyed by sum states.
  - With `sum_state=0` → dict keyed by product-state tuples.
  - All four args → float.
- [ ] Singlet check: for 3⊗3̄ in SU(3), the singlet CGCs (sum=(0,0,0), mult=1) over the diagonal of (state, conjugate-state) should equal `1/sqrt(3)` up to phase, all with the same sign (phase convention).
- [ ] `check_cgcs` returns `True` for SU(2) `[(1,0),(1,0)]` and SU(3) `[(1,0,0),(1,1,0)]`. Mark slow if too expensive.
- [ ] `print_cgcs` smoke test: capture stdout (`capsys`), assert non-empty output for a small case. (Avoids regression where formatting crashes.)
- [ ] Lower-weight algorithm path: for SU(3) 3⊗3 → 6, request `calc_cgcs([(1,0,0),(1,0,0)], (2,0,0), mult_idx=1, sum_state=k)` for k=0 and k=last; verify magnitudes (sum of squares of CGCs for a given sum_state == 1).

## Phase 5: `matrix_elements/helpers.py` and `lattice_data.py` gap tests

- [ ] `conjugate_iweight`:
  - `(1,0,0) -> (1,1,0)`, `(1,1,0) -> (1,0,0)`, `(2,1,0) -> (2,1,0)` (self-conjugate), `(0,0,0) -> (0,0,0)`.
  - Generic check: dimension preserved (`calc_dimension(R) == calc_dimension(conjugate_iweight(R))`).
- [ ] `get_irreps(max_casimir, N)`:
  - For SU(3) and `max_casimir=4/3`: returns `{(0,0,0): 0, (1,0,0): 4/3, (1,1,0): 4/3}` (or whatever the code returns — verify and lock in).
  - Monotonicity: increasing `max_casimir` yields a superset.
- [ ] `sites_links_and_plaquettes`:
  - `num_sites=(2,2,1), PBCs=(True,False,False)` yields 4 sites, expected number of links, exactly 2 plaquettes in the (1,2) plane (matches the existing `compute_plaquette_signature` test fixtures).
  - OBC `num_sites=(3,2,1)`: 6 sites, 2 plaquettes.
- [ ] `irreps_and_singlets`:
  - SU(3), `T=1` on a 2-half-link site set: `link_irreps[2]` includes `{(0,0,0), (1,0,0), (1,1,0)}`.
  - `site_singlets` for a 2-half-link site under T=1: contains the trivial-trivial singlet and 3⊗3̄ singlet with multiplicity 1.
  - `conj_dict[(1,0,0)]` maps each fundamental basis state to its antifundamental partner with a definite phase (|value|≈1).
  - Sanity: `irreps_and_singlets` runs without error for `C` and `B` modes on small lattices.
- [ ] `physical_plaquette_states`:
  - Smallest case: SU(3), `num_sites=(2,2,1), PBCs=(True,True,False), T=1`. Verify the returned list of 12-tuples is non-empty, every element has the expected length, and the **all-trivial** state appears.

## Out of scope

- Performance/timing tests.
- Tests requiring `ymcirc` integration (JSON generation script).
- Tests for the demo / Wilson-loop scripts under `run/`.
- Modifying any package code (only new files under `tests/`).
