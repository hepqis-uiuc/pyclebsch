# Phase 2: `su_n_operators.py` tests — COMPLETE

## What was done

Created `tests/test_su_n_operators.py` with **50 tests** covering the public API of `pyclebsch/su_n_operators.py`. All pass on the first run (no retries needed).

Test groups:

- **`normalize_iweight`** (2 tests): subtraction of last entry; idempotence on already-normalized.
- **`calc_dimension`** (10 parametrized): SU(2) `(0,0)..(3,0)` and SU(3) `(0,0,0)..(3,0,0)` against Weyl formula.
- **`calc_casimir`** (5 tests): trivial=0; SU(3) fund=4/3; antifund=4/3; adjoint=3; normalization invariance.
- **`calc_dynkin_index`** (2 tests): SU(3) fund=1/2; adjoint=3.
- **`calc_weight`** (3 tests): SU(2) HW/LW z and p weights; invalid kind raises.
- **`find_gt_patterns`** (9 tests): count vs dim (7-parametric); HW first; sorted by p-weight descending.
- **`ladder_op`** (4 tests): J+ annihilates HW, J- annihilates LW, J- HW = LW (coeff 1), J+ LW = HW (coeff 1) for SU(2).
- **`find_suN_basis`** (5 tests): SU(2) generator count 3 (2×2); SU(3) generator count 8 (3×3); SU(2) generators are σ_a/2 in the expected (Tz, Tx, Ty) order; commutator `[T_x, T_y] = i T_z`; `Tr(T_a²) = 1/2`.
- **`find_direct_sum`** (5 tests): SU(2) 2⊗2; SU(3) 3⊗3̄, 3⊗3, 8⊗8; `sum_iweight` returns int multiplicity (including 0).
- **`find_symmetry_direct_sum`** (2 tests): SU(2) and SU(3) fund⊗fund.
- **`find_plethysms`** (3 tests): SU(2) n=2; SU(3) n=2; SU(3) n=3.

## Discrepancies from plan expectations (caught during writing, before running)

1. **Adjoint multiplicity under (2,1) in SU(3) fund n=3 plethysm.** The plan said multiplicity 2 — that was wrong. The correct multiplicity is 1; the factor of 2 in dim-counting (16 = 2 × 8) comes from `dim_{S_3}((2,1)) = 2`, not from the plethysm coefficient. Caught by re-deriving 27 = 10 + 2·8 + 1 before writing the assertion.

2. **`find_symmetry_direct_sum` output shape.** The plan said `{sum_irrep: {Sn_partition: mult}}` with keys like `(2,)`. The actual output nests the partition tuple inside an outer tuple — keys are `((2,),)`. The source iterates over groups of identical irreps in the product (here only one group), wraps each group's partition in a tuple, and uses the resulting tuple-of-partitions as the key. This is documented in SKILL.md §2/§4 as "Sn_partition_tuple". Adjusted test expectations accordingly.

3. **SU(3) 8⊗8 speed.** Plan said "mark slow." It ran in <1s on this machine, so `@pytest.mark.slow` was not added.

## Coverage gained

`pyclebsch/su_n_operators.py` is now exercised end-to-end. The only branch not directly tested: the `D==1` short-circuit in `find_suN_basis` (trivial rep). Could add a one-liner if desired but it's defensive code.

## Status

**COMPLETE.** Full suite: 66 passing (was 16 pre-Phase 2, was 14 pre-project). Ready for Phase 3.
