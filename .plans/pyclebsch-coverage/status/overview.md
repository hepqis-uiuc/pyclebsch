# Overview: pyclebsch Coverage Improvement

**Status: Phases 1–3 complete. Phases 4–5 not started.**

- **Phase 1 (DONE)** — `tests/test_plaquette_matrix_elements.py`: 2 tests (shape, parallelize-consistency). Hermiticity/symmetry dropped after investigation. See `phase-1-plaquette-matrix-element-tests.md`.
- **Phase 2 (DONE)** — `tests/test_su_n_operators.py`: 50 tests. Caught two plan errors before assertions hit (adjoint plethysm multiplicity, `find_symmetry_direct_sum` key shape). See `phase-2-su-n-operators.md`.
- **Phase 3 (DONE)** — `tests/test_symmetric_group.py`: 61 tests, all passing first run. Covers partitions, hook lengths, standard Young tableaux, and the Young symmetrizer including the MOLD-descent branch via the `[[0,2],[1]]` tableau. Idempotency Y² = Y verified on 8 representative tableaux. See `phase-3-symmetric-group.md`.
- **Phases 4–5** — not started. Plan unchanged.

Full suite: 127/127 passing (started at 14).

Next: Phase 4 — extend `cgc.py` tests for `calc_cgcs` shape filtering, `print_cgcs`, lower-weight path, singlet phase convention.
