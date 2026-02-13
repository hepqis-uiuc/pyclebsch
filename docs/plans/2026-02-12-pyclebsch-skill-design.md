# pyclebsch Skill Design

**Date**: 2026-02-12
**Audience**: Both pyclebsch developers and ymcirc consumers
**Format**: Single monolithic skill (`/pyclebsch`), companion to `/ymcirc` skill

## Sections

1. Package overview & structure
2. SU(N) representation conventions (i-weights, GT patterns, state ordering)
3. CGC computation API & algorithms (highest-weight null space, ladder descent, Young symmetrizers)
4. Tensor product decomposition & plethysm
5. Symmetric group (Young tableaux, symmetrizers)
6. Lattice gauge theory layer (LatticeDef, FORDER, truncation schemes, plaquette states, site factors)
7. Matrix elements (site factor decomposition, glueing, parallelization)
8. JSON output format for ymcirc (plaquette states & matrix elements files)
9. Key constants & precision

## Key Design Decisions

- **Full detail on algorithms**: The skill is self-contained; someone modifying pyclebsch internals should find everything they need.
- **Inline ymcirc divergence notes**: Short callouts where pyclebsch conventions differ from current ymcirc, enabling refactoring.
- **Companion to /ymcirc**: Assumes ymcirc skill is co-loaded; does not duplicate ymcirc reference material.
- **Covers the `feature/ymcirc_gen_json_scripts` branch specifically**: Includes FORDER, B truncation, JSON generation — features not yet in ymcirc.
