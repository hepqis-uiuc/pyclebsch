import pytest
from pathlib import Path

import pyclebsch.cgc as cgc


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
