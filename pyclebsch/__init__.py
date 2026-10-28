"""
SU(N) Clebsch-Gordan coefficients, and plaquette matrix elements for lattice
gauge theory.

The modules:
- cgc: Clebsch-Gordan coefficients (calc_cgcs);
- su_n_operators: representation theory of SU(N), from i-weights and
  Gelfand-Tsetlin patterns to decompositions of tensor products;
- symmetric_group: partitions, Young tableaux and Young projection
  operators, which cgc and su_n_operators use;
- matrix_elements: lattice geometry, physical plaquette states and plaquette
  matrix elements;
- cache: the caches of Clebsch-Gordan coefficients, physical plaquette
  states and plaquette matrix elements.
This top-level namespace holds only the functions that configure and
inspect the caches (__all__). Everything else is imported from its module,
for example from pyclebsch.cgc import calc_cgcs.
"""

from pyclebsch.cache import (
    cgc_cache_stats,
    clear_memory_cache,
    get_cgc_cache_dir,
    physical_plaquette_states_cache_stats,
    plaquette_matrix_elements_cache_stats,
    set_cgc_cache_dir,
    set_cgc_memory_cache_limit,
    set_physical_plaquette_states_cache_limit,
    set_plaquette_matrix_elements_cache_limit,
)

__all__ = [
    "cgc_cache_stats",
    "clear_memory_cache",
    "get_cgc_cache_dir",
    "physical_plaquette_states_cache_stats",
    "plaquette_matrix_elements_cache_stats",
    "set_cgc_cache_dir",
    "set_cgc_memory_cache_limit",
    "set_physical_plaquette_states_cache_limit",
    "set_plaquette_matrix_elements_cache_limit",
]
