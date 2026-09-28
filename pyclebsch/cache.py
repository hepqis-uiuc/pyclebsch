"""Versioning of cached Clebsch-Gordan coefficient tables."""

# Version of the CGC *conventions* baked into cached tables. Bump it whenever a
# change alters any stored table for the same key. That covers:
#   - the ordering of Gelfand-Tsetlin patterns (stored states are indices into it);
#   - the labeling of multiplicity copies (S_n symmetrization, RREF/QR basis choice);
#   - the phase convention;
#   - the normalization / trivial-irrep rules that produce the key.
# tests/test_cgc_golden.py fails when any of these change; the fix is to bump
# this number and add golden values for the new version.
CGC_CACHE_VERSION: int = 1
