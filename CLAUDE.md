# pyclebsch

SU(N) Clebsch-Gordan coefficient computation library with lattice gauge theory applications.

## Quick Reference

- **Language**: Python 3.14+
- **Package manager**: uv
- **Test framework**: pytest

## Commands

- Run tests: `uv run -m pytest -v`
- Run tests including slow: `uv run -m pytest -v --runslow`
- Run specific test: `uv run -m pytest tests/test_lattice_data.py::test_compute_plaquette_signature`
- Generate ymcirc JSON data: `uv run -m run.gen_ymcirc_data`
- Run demo: `uv run -m run.demo`

## Architecture

### Core (`pyclebsch/`)
- `cgc.py` — CGC calculation (highest-weight null space, ladder descent)
- `su_n_operators.py` — Irrep dimensions, GT patterns, ladder ops, decompositions, plethysms
- `symmetric_group/` — Young tableaux, symmetrizers, plethysm utilities

### Lattice Gauge Theory (`pyclebsch/matrix_elements/`)
- `lattice_data.py` — LatticeDef, plaquette geometry, truncation schemes, Gauss law, physical states
- `plaquette_matrix_elements.py` — Wilson loop matrix elements via site factor factorization
- `helpers.py` — Casimir, conjugate_irrep, get_irreps

### Scripts (`run/`)
- `gen_ymcirc_data.py` — Generates compressed JSON files for ymcirc consumption

## Key Conventions

- Irreps: i-weight tuples normalized to last component 0
- States: GT pattern indices, sorted by p-weight descending (index 0 = highest weight)
- CGC multiplicity: 1-indexed in `calc_cgcs` API, 0-indexed for singlet multiplicities in lattice code
- FORDER: permutation of `[1,2,3,-1,-2,-3]` defining half-link ordering at sites
- Plaquette states (internal): `(l1,l2,l3,l4, c1,c2,c3,c4, g1,g2,g3,g4)` — active links, controls, multiplicities
- Plaquette states (ymcirc): `((g1,g2,g3,g4), (l1,l2,l3,l4), (c1,c2,c3,c4))` — reordered

## Skill

Load `/pyclebsch` for the full reference when working on this codebase.
