# Overview: pyclebsch Coverage Improvement

**Status: Phase 1 complete. Phases 2–5 not started.**

- **Phase 1 (DONE)** — `tests/test_plaquette_matrix_elements.py` added with two tests: shape/structure smoke test and `parallelize=True/False` consistency. Both pass. The originally-planned Hermiticity/symmetry test was dropped after investigation showed the function returns matrix elements of the unitary Wilson loop (not Hermitian) and that the full orientation-reversal symmetry interacts with FORDER in ways beyond Phase 1 scope. Full suite: 16/16 passing.
- **Phases 2–5** — not started. Plan unchanged at `plan.md`.

Next: await user direction on whether to proceed to Phase 2 (new `tests/test_su_n_operators.py`).
