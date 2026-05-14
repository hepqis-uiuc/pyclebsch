# Phase 4: `cgc.py` extended tests — COMPLETE

## What was done

Created `tests/test_cgc_api.py` with **21 tests** covering the public `calc_cgcs` / `print_cgcs` / `check_cgcs` API. All pass on the first run.

Test groups:

- **Return-shape filtering** (5 tests): one per number of optional args supplied (0..4). Each verifies the documented return type. The 0-arg case checks the outer dict keys are exactly `{(2,1,0), (0,0,0)}` for SU(3) 3⊗3̄.
- **Error paths** (2 tests): invalid `sum_state` raises `ValueError`; a `product_state` with no CGC returns `0` (not `KeyError`).
- **Singlet checks** (4 tests): exactly 3 nonzero CGCs in the 3⊗3̄ singlet; each has magnitude `1/√3`; Σ|CGC|² = 1; smallest tuple-ordered product_state with a nonzero CGC has positive sign — verifies the source's phase-fix convention.
- **Lower-weight algorithm path** (4 tests): SU(3) 3⊗3 → 6 has the expected 6 sum_states; each row of the CGC matrix is normalized (Σ|CGC|² = 1); the lowest-weight state (max sum_state index) — fully derived via repeated J(k)- descent — is also normalized; 3⊗3 → 3̄ has 3 sum_states.
- **`check_cgcs` SU(2)** (1 test): 2⊗2 orthogonality (SU(3) cases already in `test_cgc_cache.py`).
- **`print_cgcs` smoke tests** (5 tests, using `capsys`): non-empty stdout at all 5 filter levels; checks for expected formatting markers (`#` banner, "decomposition state:") where appropriate.

## Discrepancies from plan expectations

None. The source's behavior matched the SKILL.md documentation cleanly. One small adjustment from the plan: the plan said "all four args → float" and that's exactly what the source returns (not `np.float64` — looking at the code, `lower_dict[sum_state][P]` values come from solved linear systems, and they pass the `isinstance(..., float)` check; numpy scalar arithmetic with Python floats stays float at this level).

## Coverage gained

`pyclebsch/cgc.py` now exercised:
- All 5 branches of `calc_cgcs` filtering (lines 651–698).
- Error-handling paths (`raise ValueError` for invalid `sum_state`).
- Full `print_cgcs` (lines 701–782) at all 5 invocation levels.
- `calc_lower_weight_cgcs` indirectly via every test that requests a non-HW sum_state.
- `check_cgcs` previously tested for SU(3) 3⊗3̄ and 8⊗8; now also SU(2) 2⊗2.

Not directly tested in this phase: the cache miss/hit interaction within `calc_lower_weight_cgcs` (covered in Phase 1-era `test_cgc_cache.py`) and `calc_highest_weight_cgcs` directly (used internally by `calc_lower_weight_cgcs` for the HW state of each multiplicity — covered transitively by every test that requests `sum_state=0`).

## Status

**COMPLETE.** Full suite: 148 passing (was 127 pre-Phase 4). Ready for Phase 5.
