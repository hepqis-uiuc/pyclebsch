# Overview: pyclebsch Coverage Improvement

**Status: awaiting approval.** Plan drafted at `plan.md` with 5 phases:

1. New `tests/test_plaquette_matrix_elements.py` — shape, Hermiticity, and `parallelize=False` checks on a tiny PBC d=2 lattice.
2. New `tests/test_su_n_operators.py` covering dim/casimir/Dynkin/GT-patterns/ladder ops/direct-sum/plethysms.
3. New `tests/test_symmetric_group.py` covering partitions, tableaux, hook lengths, Young symmetrizer.
4. Extend `cgc.py` tests: return-shape filtering, lower-weight path, `print_cgcs` smoke, singlet phase convention.
5. Fill gaps in `matrix_elements/`: `helpers.py`, `sites_links_and_plaquettes`, `irreps_and_singlets`, `physical_plaquette_states`.

No package code changes are planned. Tests will redirect CGC cache to `tmp_path` via an autouse fixture so the on-disk cache isn't polluted. Slow CGC computations marked `@pytest.mark.slow`.

Next: user reviews `plan.md` and approves before any implementation begins.
