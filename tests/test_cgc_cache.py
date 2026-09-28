import copy
import dataclasses
import json
import pickle
import re
from collections import Counter
from importlib import metadata
from pathlib import Path

import pytest

import pyclebsch.cgc as cgc
from pyclebsch import cache as cgc_cache
from pyclebsch.matrix_elements.lattice_data import irreps_and_singlets, sites_links_and_plaquettes
from pyclebsch.matrix_elements.plaquette_matrix_elements import calc_plaquette_elements


@pytest.fixture(autouse=True)
def restore_cache_dir():
    original = cgc.get_cache_dir()
    yield
    cgc.set_cache_dir(original)


# --- API surface tests ---

def test_get_cache_dir_default():
    """Default cache dir is Path('./CGC_Data')."""
    assert cgc.get_cache_dir() == Path("./CGC_Data")


def test_set_cache_dir_custom(tmp_path):
    """set_cache_dir() changes what get_cache_dir() returns."""
    custom = tmp_path / "my_cache"
    cgc.set_cache_dir(custom)
    assert cgc.get_cache_dir() == custom


def test_set_cache_dir_none():
    """set_cache_dir(None) disables caching; get_cache_dir() returns None."""
    cgc.set_cache_dir(None)
    assert cgc.get_cache_dir() is None


def test_set_cache_dir_accepts_string(tmp_path):
    """set_cache_dir() accepts a str and converts it to a Path."""
    cgc.set_cache_dir(str(tmp_path / "str_cache"))
    assert isinstance(cgc.get_cache_dir(), Path)
    assert cgc.get_cache_dir() == tmp_path / "str_cache"


# --- Cache behavior tests ---

def test_cgcs_written_to_configured_dir(tmp_path):
    """When cache dir is set, pickle files appear there after calc_cgcs."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    cgc.calc_cgcs([(2, 1, 0), (2, 1, 0)])
    cache_contents = list((tmp_path / "CGC_Data").rglob("*"))
    assert len(cache_contents) > 0
    assert any("highest_weight_CGC" in str(p) for p in cache_contents)


def test_cgcs_not_written_when_cache_none(tmp_path):
    """When cache is None, calc_cgcs still returns correct results but writes nothing."""
    cgc.set_cache_dir(None)
    cgc.calc_cgcs([(2, 1, 0), (2, 1, 0)])
    assert not (tmp_path / "CGC_Data").exists()


def test_cgcs_read_from_cache(tmp_path):
    """A second call with the same args reads from cache (no recomputation)."""
    import time

    cache = tmp_path / "CGC_Data"
    cgc.set_cache_dir(cache)
    cgc.calc_cgcs([(2, 1, 0), (2, 1, 0)])
    files = list(cache.rglob("*"))
    mtimes = {f: f.stat().st_mtime for f in files if f.is_file()}
    time.sleep(0.05)
    cgc.calc_cgcs([(2, 1, 0), (2, 1, 0)])
    for f, mtime in mtimes.items():
        assert f.stat().st_mtime == mtime


# --- Correctness regression tests ---

def test_cgc_correctness_8x8(tmp_path):
    """CGCs for 8x8 in SU(3) pass orthogonality check."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    assert cgc.check_cgcs([(2, 1, 0), (2, 1, 0)])


def test_cgc_correctness_3x3bar(tmp_path):
    """CGCs for 3x3bar in SU(3) pass orthogonality check."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    assert cgc.check_cgcs([(1, 0, 0), (1, 1, 0)])


# --- Input types ---

def _product_dirs(cache: Path) -> set[Path]:
    """Directories that directly contain cache files, wherever they sit
    below the cache root."""
    return {f.parent for f in cache.rglob("*") if f.is_file()}


def test_numpy_and_int_iweights_share_entry(tmp_path):
    """numpy-integer i-weights reuse the entry written for plain ints, and
    results carry plain-int sum-irrep keys."""
    import numpy as np

    cache = tmp_path / "CGC_Data"
    cgc.set_cache_dir(cache)
    int_result = cgc.calc_cgcs([(1, 0, 0), (1, 1, 0)])
    before = cgc.cache_stats()
    numpy_result = cgc.calc_cgcs([tuple(np.array([1, 0, 0])), tuple(np.array([1, 1, 0]))])
    assert _stats_since(before)["computed"] == 0
    assert len(_product_dirs(cache)) == 1
    assert numpy_result == int_result
    assert all(type(entry) is int for key in numpy_result for entry in key)


def test_float_iweights_rejected_without_side_effects(tmp_path, monkeypatch):
    """Float i-weights raise TypeError before anything is written, either in
    the configured cache directory or in the working directory."""
    cache = tmp_path / "disk"
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    cgc.set_cache_dir(cache)
    with pytest.raises(TypeError):
        cgc.calc_cgcs([(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)])
    assert not cache.exists() or not any(cache.iterdir())
    assert not any(work.iterdir())


# --- Memory tier ---

# Lattice-code parameters, as in tests/test_plaquette_matrix_elements.py.
FORDER = [1, 2, 3, -1, -2, -3]
N_COLORS = 3
EPS = 1e-10
PRES = 10

# Distinct tables (highest- and lower-weight) that the L1 matrix elements
# need: a fresh disk cache holds this many files after one serial L1 run.
# The test checks the computation count against the files actually written
# as well, so a mismatch with this number means the workload changed.
L1_DISTINCT_TABLES = 19

# Largest difference allowed between two computations of the same CGCs by the
# same code. Repeated runs agree to ~1e-15 (differences of 4.4e-16 were seen
# between runs of the lattice workloads); a stale or corrupted table differs
# by O(0.1-1).
RECOMPUTE_TOL = 1e-12


def _stats_since(before: "cgc_cache.CacheStats") -> dict[str, int]:
    """Change in every cache counter since the snapshot before."""
    now = cgc.cache_stats()
    return {f.name: getattr(now, f.name) - getattr(before, f.name)
            for f in dataclasses.fields(now)}


def _max_abs_difference(a, b) -> float:
    """Largest absolute difference between two nested CGC results. A key
    missing on one side counts as an empty dict or a zero coefficient."""
    if isinstance(a, dict) or isinstance(b, dict):
        a = a if isinstance(a, dict) else {}
        b = b if isinstance(b, dict) else {}
        return max((_max_abs_difference(a.get(k), b.get(k)) for k in a.keys() | b.keys()),
                   default=0.0)
    return abs((a or 0.0) - (b or 0.0))


def _l1_matrix_elements() -> dict:
    """Serial matrix elements of every plaquette of the 2x2x1 lattice,
    periodic in x and y, T truncation cutoff 1."""
    sites, _links, plaquettes = sites_links_and_plaquettes([2, 2, 1], [True, True, False], FORDER)
    truncation_irreps, singlets, conj_dict = irreps_and_singlets(N_COLORS, sites, "T", 1)
    return {P: calc_plaquette_elements(N_COLORS, P, sites, plaquettes, truncation_irreps, singlets,
                                       conj_dict, FORDER, EPS, PRES, parallelize=False)
            for P in sorted(plaquettes)}


@pytest.fixture
def memory_limit_restored():
    """Remove any memory cap a test sets."""
    yield
    cgc_cache.set_memory_cache_limit(None)


def test_memory_tier_computes_each_entry_once(tmp_path):
    """Each distinct table is computed once per process, however many
    calc_cgcs calls need it."""
    cache = tmp_path / "CGC_Data"
    cgc.set_cache_dir(cache)
    before = cgc.cache_stats()
    first = _l1_matrix_elements()
    first_run = _stats_since(before)
    files = [f for f in cache.rglob("*") if f.is_file()]
    assert first_run["computed"] == len(files) == L1_DISTINCT_TABLES
    assert first_run["disk_writes"] == first_run["computed"]
    assert first_run["memory_hits"] > first_run["computed"]

    before = cgc.cache_stats()
    second = _l1_matrix_elements()
    second_run = _stats_since(before)
    assert second_run["computed"] == 0
    assert second_run["disk_hits"] == 0
    assert second_run["memory_hits"] > 0
    assert _max_abs_difference(first, second) == 0


def test_memory_results_match_fresh_computation():
    """A result served from memory equals one computed after the memory tier
    is cleared."""
    cgc.set_cache_dir(None)
    products = [
        [(2, 1, 0), (2, 1, 0)],
        [(1, 0, 0), (1, 0, 0), (1, 0, 0)],
        [(1, 1, 0), (1, 0, 0)],
        [(1, 0, 0), (1, 1, 0), (1, 0, 0), (1, 1, 0)],
    ]
    for product in products:
        cgc.calc_cgcs(product)
        before = cgc.cache_stats()
        cached = cgc.calc_cgcs(product)
        assert _stats_since(before)["computed"] == 0
        cgc.clear_memory_cache()
        before = cgc.cache_stats()
        fresh = cgc.calc_cgcs(product)
        assert _stats_since(before)["computed"] > 0
        assert _max_abs_difference(cached, fresh) <= RECOMPUTE_TOL


def _vandalize(result) -> None:
    """Overwrite every coefficient and add a junk key at every nesting level."""
    for key, value in list(result.items()):
        if isinstance(value, dict):
            _vandalize(value)
        else:
            result[key] = 99.0
    result["junk"] = 99.0


def test_calc_cgcs_results_do_not_alias_cache():
    """Mutating anything calc_cgcs returns leaves later results unchanged,
    for every call form that returns a dict."""
    cgc.set_cache_dir(None)
    product, sum_irrep, copy_idx, state = [(2, 1, 0), (2, 1, 0)], (2, 1, 0), 2, 3
    call_forms = {
        "all irreps": lambda: cgc.calc_cgcs(product),
        "one irrep": lambda: cgc.calc_cgcs(product, sum_irrep),
        "one copy": lambda: cgc.calc_cgcs(product, sum_irrep, copy_idx),
        "one state": lambda: cgc.calc_cgcs(product, sum_irrep, copy_idx, state),
    }
    snapshots = {name: copy.deepcopy(call()) for name, call in call_forms.items()}
    for call in call_forms.values():
        _vandalize(call())
    for name, call in call_forms.items():
        assert _max_abs_difference(call(), snapshots[name]) <= RECOMPUTE_TOL, name


def test_lower_level_functions_not_public():
    """calc_cgcs is the only public way to get CGCs."""
    assert not hasattr(cgc, "calc_highest_weight_cgcs")
    assert not hasattr(cgc, "calc_lower_weight_cgcs")


def test_set_cache_dir_clears_memory(tmp_path):
    """After switching directories, tables are written to the new one rather
    than served from memory."""
    cgc.set_cache_dir(tmp_path / "first")
    assert cgc.check_cgcs([(2, 1, 0), (2, 1, 0)])
    cgc.set_cache_dir(tmp_path / "second")
    assert cgc.check_cgcs([(2, 1, 0), (2, 1, 0)])
    assert any(f.is_file() for f in (tmp_path / "second").rglob("*"))


def _synthetic_key(label: int) -> "cgc_cache.CacheKey":
    """A distinct key per label; the tables stored under it are synthetic."""
    return cgc_cache.CacheKey.lower([(label, 0, 0), (label, 0, 0)], (2 * label, 0, 0), 1, EPS)


def test_memory_limit_evicts_least_recently_used(memory_limit_restored):
    """With a cap of 2, the least recently used entry is evicted, and a later
    request for it recomputes rather than serving anything stale."""
    cgc.set_cache_dir(None)
    cgc_cache.set_memory_cache_limit(2)
    computations = Counter()

    def get(label: int):
        def compute():
            computations[label] += 1
            return {0: {(0, 0): float(label)}}
        return cgc_cache._cache.get_or_compute(_synthetic_key(label), compute)

    before = cgc.cache_stats()
    get(1)
    get(2)
    get(1)          # hit; 1 becomes the most recently used entry
    get(3)          # evicts 2, the least recently used
    assert len(cgc_cache._cache._memory) == 2
    assert _stats_since(before)["evictions"] == 1
    assert get(1) == {0: {(0, 0): 1.0}}
    assert computations[1] == 1
    assert get(2) == {0: {(0, 0): 2.0}}
    assert computations[2] == 2


def test_memory_limit_results_unchanged(memory_limit_restored):
    """L1 matrix elements with a small cap equal those with no cap."""
    cgc.set_cache_dir(None)
    cgc_cache.set_memory_cache_limit(3)
    before = cgc.cache_stats()
    capped = _l1_matrix_elements()
    assert _stats_since(before)["evictions"] > 0
    cgc_cache.set_memory_cache_limit(None)
    cgc.clear_memory_cache()
    uncapped = _l1_matrix_elements()
    assert _max_abs_difference(capped, uncapped) <= RECOMPUTE_TOL


def test_memory_limit_rejects_invalid_values(memory_limit_restored):
    """The cap must be None or an integer of at least 1."""
    cgc_cache.set_memory_cache_limit(5)
    for invalid in (0, -1, 2.5):
        with pytest.raises(ValueError):
            cgc_cache.set_memory_cache_limit(invalid)
    assert cgc_cache._cache.max_memory_entries == 5


# --- Disk tier ---

# The 8 in 3x3bar: one highest-weight and one lower-weight entry on disk.
OCTET_PRODUCT = [(1, 0, 0), (1, 1, 0)]
OCTET = (2, 1, 0)


def _entry_files(cache: Path) -> list[Path]:
    """Every file below the cache root."""
    return sorted(f for f in cache.rglob("*") if f.is_file())


def _octet_lower_path() -> Path:
    """Where the lower-weight table of copy 1 of the octet is stored."""
    key = cgc_cache.CacheKey.lower(OCTET_PRODUCT, OCTET, 1, cgc.EPS)
    return cgc_cache._cache._entry_path(key)


def _fresh_octet() -> dict:
    """The octet's CGCs computed with no cache at all."""
    original = cgc.get_cache_dir()
    cgc.set_cache_dir(None)
    try:
        return cgc.calc_cgcs(OCTET_PRODUCT, OCTET, 1)
    finally:
        cgc.set_cache_dir(original)


def _load_envelope(path: Path) -> dict:
    with open(path, "rb") as fp:
        return pickle.load(fp)


def _dump_envelope(path: Path, envelope: dict) -> None:
    with open(path, "wb") as fp:
        pickle.dump(envelope, fp)


def test_disk_entries_written_under_version_namespace(tmp_path):
    """Every entry lives under v<CGC_CACHE_VERSION>/."""
    cache = tmp_path / "CGC_Data"
    cgc.set_cache_dir(cache)
    cgc.calc_cgcs([(2, 1, 0), (2, 1, 0)])
    files = _entry_files(cache)
    assert files
    assert all(f.relative_to(cache).parts[0] == f"v{cgc_cache.CGC_CACHE_VERSION}" for f in files)


def test_disk_entry_used_after_memory_clear(tmp_path):
    """With the memory tier cleared, entries are read back from disk rather
    than recomputed."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    first = cgc.calc_cgcs([(2, 1, 0), (2, 1, 0)])
    cgc.clear_memory_cache()
    before = cgc.cache_stats()
    second = cgc.calc_cgcs([(2, 1, 0), (2, 1, 0)])
    delta = _stats_since(before)
    assert delta["disk_hits"] > 0
    assert delta["computed"] == 0
    assert _max_abs_difference(first, second) == 0


def test_legacy_layout_entries_are_ignored(tmp_path):
    """An entry at the unversioned layout of pyclebsch <= 0.1 is never read,
    even when it sits in the configured directory."""
    fresh = _fresh_octet()
    cache = tmp_path / "CGC_Data"
    legacy = cache / "[(1, 0, 0), (1, 1, 0)]" / "lower_weight_CGC_((2, 1, 0), 1)"
    legacy.parent.mkdir(parents=True)
    altered = {state: {ps: 0.5 for ps in coefficients} for state, coefficients in fresh.items()}
    _dump_envelope(legacy, altered)
    cgc.set_cache_dir(cache)
    assert _max_abs_difference(cgc.calc_cgcs(OCTET_PRODUCT, OCTET, 1), fresh) <= RECOMPUTE_TOL


def test_previous_version_entries_are_ignored(tmp_path, monkeypatch):
    """After a CGC_CACHE_VERSION bump, entries written under the old version
    are not read, and new ones are written under the new version."""
    cache = tmp_path / "CGC_Data"
    cgc.set_cache_dir(cache)
    cgc.calc_cgcs(OCTET_PRODUCT, OCTET, 1)
    new_version = cgc_cache.CGC_CACHE_VERSION + 1
    monkeypatch.setattr(cgc_cache, "CGC_CACHE_VERSION", new_version)
    cgc.clear_memory_cache()
    before = cgc.cache_stats()
    cgc.calc_cgcs(OCTET_PRODUCT, OCTET, 1)
    delta = _stats_since(before)
    assert delta["disk_hits"] == 0
    assert delta["computed"] > 0
    assert any(f.relative_to(cache).parts[0] == f"v{new_version}" for f in _entry_files(cache))


def test_format_version_mismatch_is_rejected(tmp_path):
    """An entry whose envelope format differs is rejected with a warning,
    recomputed, and rewritten."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    cgc.calc_cgcs(OCTET_PRODUCT, OCTET, 1)
    path = _octet_lower_path()
    envelope = _load_envelope(path)
    envelope["format_version"] = cgc_cache.CACHE_FORMAT_VERSION + 1
    _dump_envelope(path, envelope)
    cgc.clear_memory_cache()
    before = cgc.cache_stats()
    with pytest.warns(cgc_cache.CGCCacheWarning):
        cgc.calc_cgcs(OCTET_PRODUCT, OCTET, 1)
    delta = _stats_since(before)
    assert delta["disk_rejected"] == 1
    assert delta["computed"] == 1
    assert _load_envelope(path)["format_version"] == cgc_cache.CACHE_FORMAT_VERSION


def test_entry_with_mismatched_key_is_rejected(tmp_path):
    """A valid entry copied onto another key's path is rejected with a
    warning, and the right table is computed and written there."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    fresh = cgc.calc_cgcs(OCTET_PRODUCT, OCTET, 1)
    singlet_key = cgc_cache.CacheKey.lower(OCTET_PRODUCT, (0, 0, 0), 1, cgc.EPS)
    cgc.calc_cgcs(OCTET_PRODUCT, (0, 0, 0), 1)
    octet_path = _octet_lower_path()
    octet_path.write_bytes(cgc_cache._cache._entry_path(singlet_key).read_bytes())
    cgc.clear_memory_cache()
    with pytest.warns(cgc_cache.CGCCacheWarning):
        result = cgc.calc_cgcs(OCTET_PRODUCT, OCTET, 1)
    assert _max_abs_difference(result, fresh) <= RECOMPUTE_TOL
    octet_key = cgc_cache.CacheKey.lower(OCTET_PRODUCT, OCTET, 1, cgc.EPS)
    assert _load_envelope(octet_path)["key"] == octet_key.as_header()


def test_eps_change_invalidates(tmp_path, monkeypatch):
    """Changing cgc.EPS changes the key, so neither the memory entry nor the
    disk entry written under the old value is served."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    cgc.calc_cgcs(OCTET_PRODUCT, OCTET, 1)
    path = _octet_lower_path()
    new_eps = cgc.EPS / 10
    monkeypatch.setattr(cgc, "EPS", new_eps)
    before = cgc.cache_stats()
    with pytest.warns(cgc_cache.CGCCacheWarning):
        cgc.calc_cgcs(OCTET_PRODUCT, OCTET, 1)
    delta = _stats_since(before)
    assert delta["memory_hits"] == 0
    assert delta["disk_hits"] == 0
    assert delta["computed"] > 0
    assert _load_envelope(path)["key"]["eps"] == new_eps


@pytest.mark.parametrize("damage", ["empty", "truncated"])
def test_corrupt_entry_is_recovered(tmp_path, damage):
    """An empty or truncated entry is rejected with a warning, the correct
    values are returned, and a valid entry replaces it."""
    fresh = _fresh_octet()
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    cgc.calc_cgcs(OCTET_PRODUCT, OCTET, 1)
    path = _octet_lower_path()
    contents = path.read_bytes()
    path.write_bytes(b"" if damage == "empty" else contents[: len(contents) // 2])
    cgc.clear_memory_cache()
    with pytest.warns(cgc_cache.CGCCacheWarning):
        result = cgc.calc_cgcs(OCTET_PRODUCT, OCTET, 1)
    assert _max_abs_difference(result, fresh) <= RECOMPUTE_TOL
    envelope = _load_envelope(path)
    assert envelope["key"] == cgc_cache.CacheKey.lower(OCTET_PRODUCT, OCTET, 1, cgc.EPS).as_header()


def test_failed_write_leaves_no_partial_file(tmp_path, monkeypatch):
    """If writing an entry fails part way, neither the entry nor a temporary
    file is left behind."""
    cache = tmp_path / "CGC_Data"
    cgc.set_cache_dir(cache)

    def failing_dump(obj, fp, *args, **kwargs):
        fp.write(b"partial")
        raise RuntimeError("simulated failure while writing")

    monkeypatch.setattr(cgc_cache.pickle, "dump", failing_dump)
    with pytest.raises(RuntimeError):
        cgc.calc_cgcs(OCTET_PRODUCT, OCTET, 1)
    assert _entry_files(cache) == []


def test_disk_entry_header_records_provenance(tmp_path):
    """Every entry records the pyclebsch version and commit; in this editable
    install the commit is unknown."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    cgc.calc_cgcs(OCTET_PRODUCT, OCTET, 1)
    envelope = _load_envelope(_octet_lower_path())
    assert envelope["pyclebsch_version"] is None or isinstance(envelope["pyclebsch_version"], str)
    commit = envelope["pyclebsch_commit"]
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit)
    assert commit is None


class _FakeDistribution:
    """Stands in for importlib.metadata.Distribution: only read_text is used."""

    def __init__(self, direct_url: str | None) -> None:
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        return self._direct_url if filename == "direct_url.json" else None


def test_installed_commit_handles_all_install_types(monkeypatch):
    """The commit is read from direct_url.json for git installs, and is None
    for every other install type and for any unreadable metadata."""
    git_hash = "cb35b4e" + "0" * 33
    cases = {
        "git install": (json.dumps({"url": "https://example.invalid/pyclebsch.git",
                                    "vcs_info": {"vcs": "git", "commit_id": git_hash}}), git_hash),
        "editable install": (json.dumps({"url": "file:///src/pyclebsch",
                                         "dir_info": {"editable": True}}), None),
        "index install": (None, None),
        "malformed JSON": ("{not json", None),
    }
    for label, (direct_url, expected) in cases.items():
        monkeypatch.setattr(metadata, "distribution", lambda name, d=direct_url: _FakeDistribution(d))
        cgc_cache._installed_commit.cache_clear()
        assert cgc_cache._installed_commit() == expected, label

    def not_installed(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", not_installed)
    cgc_cache._installed_commit.cache_clear()
    assert cgc_cache._installed_commit() is None
    cgc_cache._installed_commit.cache_clear()


def test_long_product_uses_hashed_name():
    """A name that would exceed MAX_ENTRY_NAME_BYTES is replaced by a hash,
    and short names stay readable."""
    long_key = cgc_cache.CacheKey.lower([(1, 0, 0)] * 30, (30, 0, 0), 1, EPS)
    long_path = long_key.relative_path()
    assert all(len(part.encode()) <= cgc_cache.MAX_ENTRY_NAME_BYTES for part in long_path.parts)
    assert long_path.parts[0] != str(list(long_key.product))
    short_key = cgc_cache.CacheKey.lower(OCTET_PRODUCT, OCTET, 1, EPS)
    assert short_key.relative_path() == Path("[(1, 0, 0), (1, 1, 0)]") / "lower_weight_CGC_((2, 1, 0), 1)"
