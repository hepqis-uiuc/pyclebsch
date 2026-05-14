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

For each, give an expected value derivable by hand or from a textbook.

- [ ] `normalize_iweight`: `(3,2,1) -> (2,1,0)`, `(5,5,5) -> (0,0,0)`, already-normalized passes through unchanged.
- [ ] `calc_dimension` (Weyl formula):
  - SU(2): `(0,0) -> 1`, `(1,0) -> 2`, `(2,0) -> 3`, `(3,0) -> 4`.
  - SU(3): `(0,0,0) -> 1`, `(1,0,0) -> 3`, `(1,1,0) -> 3`, `(2,1,0) -> 8`, `(2,0,0) -> 6`, `(3,0,0) -> 10`.
- [ ] `calc_casimir`:
  - SU(3) fundamental `(1,0,0)`: `4/3`.
  - SU(3) adjoint `(2,1,0)`: `3`.
  - Trivial: `0`.
- [ ] `calc_dynkin_index`:
  - SU(3) fundamental: `1/2`.
  - SU(3) adjoint `(2,1,0)`: `3`.
- [ ] `calc_weight`:
  - For SU(2) fundamental: GT `[[1,0],[1]]` gives `'z' -> [1/2]`, `'p' -> [1,0]` (verify exact convention from `find_gt_patterns` output).
  - For SU(2) fundamental: GT `[[1,0],[0]]` gives `'z' -> [-1/2]`.
  - Invalid `kind` raises (verify whether it raises in source; if not, omit).
- [ ] `find_gt_patterns`:
  - Count check: `len(find_gt_patterns(R)) == calc_dimension(R)` for SU(2): `(1,0)`, `(2,0)`; for SU(3): `(1,0,0)`, `(2,1,0)`.
  - Index 0 is the highest-weight pattern (top of each row equals the i-weight entries).
- [ ] `ladder_op`:
  - For SU(2) fundamental, J(1)- on highest-weight gives the lowest-weight state with coefficient 1.
  - J(1)+ on highest weight returns empty/zero (annihilated).
  - For SU(3) adjoint, ladder ops preserve dimension count of nonzero matrix elements (sanity check, hand-checkable subset).
- [ ] `find_suN_basis`:
  - SU(2) fundamental: returns 3 sparse matrices (J_x, J_y, J_z forms) with expected sizes 2×2 and the correct commutation relation `[T_1,T_2] = i T_3` (or whichever convention the code uses — pick one fundamental case to verify).
  - Hermiticity of generators (or whichever conjugate-transpose convention is in use).
- [ ] `find_direct_sum`:
  - SU(2): `[(1,0),(1,0)]` → `{(2,0):1, (0,0):1}` (i.e. 2⊗2 = 3⊕1).
  - SU(3): `[(1,0,0),(1,1,0)]` → `{(2,1,0):1,(0,0,0):1}` (3⊗3̄ = 8⊕1).
  - SU(3): `[(1,0,0),(1,0,0)]` → `{(2,0,0):1,(1,1,0):1}` (3⊗3 = 6⊕3̄).
  - SU(3): `[(2,1,0),(2,1,0)]` (8⊗8) → includes `(2,1,0):2` (adjoint appears twice) and `(0,0,0):1`. **Mark slow.**
  - `sum_iweight` argument returns the int multiplicity for that irrep.
- [ ] `find_symmetry_direct_sum`:
  - SU(3) `[(1,0,0),(1,0,0)]` (two identical 3s): the 6 should sit under partition `(2,)` (symmetric), the 3̄ under `(1,1)` (antisymmetric).
  - Returns `(decomp_dict, idx_list)`; verify `idx_list == [[0,1]]`.
- [ ] `find_plethysms`:
  - SU(2) `(1,0)` with `n=2`: triplet `(2,0)` under `(2,)`, singlet `(0,0)` under `(1,1)`.
  - SU(3) `(1,0,0)` with `n=2`: 6 under `(2,)`, 3̄ under `(1,1)`.
  - SU(3) `(1,0,0)` with `n=3`: 10 under `(3,)`, 8 under `(2,1)` with multiplicity 2 (verify via reference) — **mark slow** if expensive.

## Phase 3: `symmetric_group/` tests (new file `tests/test_symmetric_group.py`)

- [ ] `find_partitions(n)`:
  - `n=0` → `[[]]` or `[[0]]` (verify exact behavior in source first; encode whatever it actually does as the spec).
  - `n=4` → `[[4],[3,1],[2,2],[2,1,1],[1,1,1,1]]`.
  - `len(list(find_partitions(n)))` matches OEIS partition counts for small n (e.g. n=5 → 7, n=6 → 11).
- [ ] `find_tableaux(partition)`:
  - `[3]` → 1 tableau (single row); `[1,1,1]` → 1 tableau (single column); `[2,1]` → 2 tableaux.
  - `[2,2]` → 2 tableaux; `[3,1]` → 3 tableaux.
  - Cell values are 0-indexed and exactly the set `{0,...,n-1}`.
- [ ] `partition_to_sequence` / `sequence_to_partition`: round-trip on `[3,2,1]`, `[4]`, `[1,1,1,1]`.
- [ ] `_product_of_hook_lengths` (already imported in `test_lattice_data.py`'s helper): verify against the hook length formula. `[2,1]` → `3·1·1 = 3`; `[3,1]` → `4·2·1·1 = 8`; `[2,2]` → `3·2·2·1 = 12`. Hook length formula: `dim(S_n irrep) = n! / hook_product`. Test that this gives integer dimensions matching known values: `[2,1]` → 2, `[3,1]` → 3, `[2,2]` → 2.
- [ ] `young_symmetrizer`:
  - Trivial partition `[1]` (n=1): yields the identity permutation with coefficient 1.
  - Symmetric tableau `[[0,1]]` for n=2: idempotent — applying it twice (in coefficient algebra) reproduces it. Check by exhausting the generator into a dict `{perm: coeff}` and verifying it's the (1/2)(e + (01)) symmetrizer up to normalization stated in the source.
  - Antisymmetric tableau `[[0],[1]]` for n=2: produces (1/2)(e - (01)).
  - Idempotency check for `[2,1]` standard tableaux.

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
