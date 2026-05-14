# Phase 3: `symmetric_group/` tests — COMPLETE

## What was done

Created `tests/test_symmetric_group.py` with **61 tests** covering `tableaux.py` and `young_symmetrizer.py`. All pass on the first run.

Test groups:

- **`find_partitions`** (12 tests): count matches OEIS A000041 for n=0..7 (parametrized); explicit content for n=4; descending-order and sum-to-n invariants; n=0 quirk (`[[0]]`).
- **`partition_to_sequence` / `sequence_to_partition`** (7 tests): round-trip across 6 partitions; explicit forward direction for `[3,2,1]→[1,0,1,0,1,0]`, `[4]→[1,1,1,1,0]`, `[1,1,1,1]→[1,0,0,0,0]`.
- **`_product_of_hook_lengths`** (13 tests): 8 hand-verified hook-product values; 5 cross-checks against the hook-length formula `dim = n!/hook_product`.
- **`find_tableaux`** (16 tests): 11 count assertions; cell-value set; row-length/partition match; row+column strict increase; explicit content for `[2,1]`; `ValueError` on unsorted input.
- **`young_symmetrizer`** (13 tests):
  - n=1 trivial yields `{(0,): 1.0}`.
  - Full symmetric and antisymmetric for n=2 and n=3 — exact coefficients verified.
  - **Idempotency Y² = Y verified on 8 tableaux** (parametrized): pure sym/antisym, row-ordered `[[0,1],[2]]`, **non-row/non-column-ordered `[[0,2],[1]]`** (exercises the MOLD descent in `young_symmetrizer.py`), and two n=4 mixed cases.

Group-algebra composition implemented in test as `_compose_symmetrizers`: `(Σ a_p [P]) · (Σ b_q [Q]) = Σ a_p b_q [P∘Q]` with `R[i] = P[Q[i]]` matching the source convention `A[i] → A[P[i]]`.

## Discrepancies from plan expectations

None caught during writing. Hand-derived values matched source behavior on every test.

The `find_partitions(0)` behavior (`[[0]]` rather than `[[]]`) was confirmed by tracing the accel_asc algorithm before writing the assertion. The test documents the actual behavior.

## Coverage gained

`pyclebsch/symmetric_group/tableaux.py` fully exercised, including the `_product_of_hook_lengths` and `_conjugate_partition` (indirectly via `find_tableaux`/`_product_of_hook_lengths`) helpers. `young_symmetrizer.py` exercised on all four code branches: total symmetric (`len(tableau)==1`), total antisymmetric (`len(tableau[0])==1`), row-ordered shortcut (Theorem 4), and the MOLD descent (Theorem 5, via the `[[0,2],[1]]` tableau which is neither row- nor column-ordered).

Not directly covered: `_rim_hooks` (private; only used inside `_class_character` in plethysm_utils, which is exercised indirectly via `find_plethysms` tests in Phase 2).

## Status

**COMPLETE.** Full suite: 127 passing (was 66 pre-Phase 3). Ready for Phase 4.
