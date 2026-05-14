# Overview: pyclebsch Coverage Improvement

**Status: Phases 1–4 complete. Phase 5 not started.**

- **Phase 1 (DONE)** — `tests/test_plaquette_matrix_elements.py`: 2 tests (shape, parallelize-consistency). Hermiticity/symmetry dropped after investigation. Detail in `phase-1-plaquette-matrix-element-tests.md`.
- **Phase 2 (DONE)** — `tests/test_su_n_operators.py`: 50 tests. Caught two plan errors before assertions hit. Detail in `phase-2-su-n-operators.md`.
- **Phase 3 (DONE)** — `tests/test_symmetric_group.py`: 61 tests. Young symmetrizer idempotency verified on 8 representative tableaux including the MOLD-descent branch. Detail in `phase-3-symmetric-group.md`.
- **Phase 4 (DONE)** — `tests/test_cgc_api.py`: 21 tests. All 5 return-shape branches of `calc_cgcs`, error paths, singlet phase convention, lower-weight algorithm orthonormality, and `print_cgcs` smoke tests at all 5 filter levels. Detail in `phase-4-cgc-api.md`.
- **Phase 5** — not started.

Full suite: 148/148 passing (started at 14).

Next: Phase 5 — `matrix_elements/helpers.py` and `lattice_data.py` gap tests.
