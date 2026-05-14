# Overview: pyclebsch Coverage Improvement

**Status: Phases 1–2 complete. Phases 3–5 not started.**

- **Phase 1 (DONE)** — `tests/test_plaquette_matrix_elements.py`: 2 tests (shape, parallelize-consistency). Hermiticity/symmetry test dropped after investigation (Wilson loop is unitary, not Hermitian; FORDER interaction makes the orientation-reversal symmetry non-trivial). Detail in `phase-1-plaquette-matrix-element-tests.md`.
- **Phase 2 (DONE)** — `tests/test_su_n_operators.py`: 50 tests, all passing first run. Caught two plan errors before they hit the test suite: (a) the adjoint multiplicity in SU(3) fund n=3 plethysm is 1, not 2, and (b) `find_symmetry_direct_sum` returns nested `((partition,),)` keys, not bare `(partition,)`. Detail in `phase-2-su-n-operators.md`.
- **Phases 3–5** — not started. Plan unchanged.

Full suite: 66/66 passing (started at 14).

Next: Phase 3 — `tests/test_symmetric_group.py` covering `tableaux.py` and `young_symmetrizer.py`.
