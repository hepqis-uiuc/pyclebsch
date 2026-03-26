# Implementation Plan: Configurable CGC Cache Directory

## Context

The `CGC_Data/` cache is currently created relative to `__file__` (the installed package location) via `_create_cgc_data_directory()` in `pyclebsch/cgc.py`. For pip-installed users this would write inside `site-packages/`, which is inappropriate. We're implementing Option B from the refactoring proposal (stdlib-only, `./CGC_Data` default) with the `set_cache_dir()` / `get_cache_dir()` API from Option A.

## Key Files

- `pyclebsch/cgc.py` -- sole file containing all CGC_Data I/O (lines 17-22, 39-42, 410-411, 430-433, 546-547, 571-572)
- `pyclebsch/__init__.py` -- will re-export `set_cache_dir` / `get_cache_dir`
- `tests/test_cgc_cache.py` -- new test file (TDD, written first)
- `run/demo.py` -- update warning comment (lines 49-50)

## Phase 1: Write Tests (Red)

Create `tests/test_cgc_cache.py` with the following test cases. All tests use `pytest`'s `tmp_path` fixture to isolate from the real `CGC_Data/`.

### 1a. API surface tests

```python
import pyclebsch.cgc as cgc

def test_get_cache_dir_default():
    """Default cache dir is Path('./CGC_Data')."""

def test_set_cache_dir_custom(tmp_path):
    """set_cache_dir() changes what get_cache_dir() returns."""

def test_set_cache_dir_none():
    """set_cache_dir(None) disables caching; get_cache_dir() returns None."""

def test_set_cache_dir_accepts_string(tmp_path):
    """set_cache_dir() accepts a str and converts it to a Path."""
```

Each test must save and restore the original cache dir in a fixture to avoid polluting other tests:

```python
@pytest.fixture(autouse=True)
def restore_cache_dir():
    original = cgc.get_cache_dir()
    yield
    cgc.set_cache_dir(original)
```

### 1b. Cache behavior tests

These tests exercise `calc_cgcs` (a small case: `8x8` decomposition in SU(3)) and verify disk behavior.

```python
def test_cgcs_written_to_configured_dir(tmp_path):
    """When cache dir is set, pickle files appear there after calc_cgcs."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    result = cgc.calc_cgcs([(2,1,0), (2,1,0)])
    cache_contents = list((tmp_path / "CGC_Data").rglob("*"))
    assert len(cache_contents) > 0  # files were created
    assert any("highest_weight_CGC" in str(p) for p in cache_contents)

def test_cgcs_not_written_when_cache_none(tmp_path):
    """When cache is None, calc_cgcs still returns correct results but writes nothing."""
    cgc.set_cache_dir(None)
    result = cgc.calc_cgcs([(2,1,0), (2,1,0)])
    # Verify result is correct by checking orthogonality
    assert cgc.check_cgcs([(2,1,0), (2,1,0)])
    # Verify nothing was written to default location
    assert not (tmp_path / "CGC_Data").exists()

def test_cgcs_read_from_cache(tmp_path):
    """A second call with the same args reads from cache (no recomputation)."""
    cache = tmp_path / "CGC_Data"
    cgc.set_cache_dir(cache)
    result1 = cgc.calc_cgcs([(2,1,0), (2,1,0)])
    # Record modification times
    files = list(cache.rglob("*"))
    mtimes = {f: f.stat().st_mtime for f in files if f.is_file()}
    # Second call should not modify files
    import time; time.sleep(0.05)  # ensure mtime would differ
    result2 = cgc.calc_cgcs([(2,1,0), (2,1,0)])
    for f, mtime in mtimes.items():
        assert f.stat().st_mtime == mtime
```

### 1c. Correctness regression test

Use the demo's known-good output to verify CGC values are unchanged. Specifically, verify `check_cgcs` returns `True` for `8x8` under both cached and uncached modes:

```python
def test_cgc_correctness_8x8(tmp_path):
    """CGCs for 8x8 in SU(3) pass orthogonality check."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    assert cgc.check_cgcs([(2,1,0), (2,1,0)])

def test_cgc_correctness_3x3bar(tmp_path):
    """CGCs for 3x3bar in SU(3) pass orthogonality check."""
    cgc.set_cache_dir(tmp_path / "CGC_Data")
    assert cgc.check_cgcs([(1,0,0), (1,1,0)])
```

### 1d. Run tests, confirm they fail

```bash
cd pyclebsch && uv run -m pytest tests/test_cgc_cache.py -v
```

Expected: all tests fail because `set_cache_dir` and `get_cache_dir` don't exist yet.

## Phase 2: Implement (Green)

### 2a. Add API to `pyclebsch/cgc.py`

At module level (after imports, before `_create_cgc_data_directory`), add:

```python
_cgc_cache_dir: Path | None = Path("./CGC_Data")

def set_cache_dir(path: str | Path | None) -> None:
    """Set the directory where computed CGCs are cached.
    Pass None to disable caching entirely.
    """
    global _cgc_cache_dir
    _cgc_cache_dir = Path(path) if path is not None else None

def get_cache_dir() -> Path | None:
    """Return the current CGC cache directory, or None if caching is disabled."""
    return _cgc_cache_dir
```

### 2b. Replace `_create_cgc_data_directory` with `_resolve_cache_dir`

```python
def _resolve_cache_dir() -> Path | None:
    """Return the effective cache directory, creating it if needed.
    Returns None when caching is disabled.
    """
    if _cgc_cache_dir is None:
        return None
    _cgc_cache_dir.mkdir(parents=True, exist_ok=True)
    return _cgc_cache_dir
```

### 2c. Update `calc_highest_weight_cgcs` (lines 39-42, 410-411)

**Read path** (top of function):
```python
cache_dir = _resolve_cache_dir()
if cache_dir is not None:
    highest_weight_cgc_data_path = cache_dir / str(product_iweights) / ('highest_weight_CGC_' + str(sum_iweight))
    if highest_weight_cgc_data_path.exists():
        with open(highest_weight_cgc_data_path, 'rb') as fp:
            return load(fp)
```

**Write path** (bottom of function):
```python
if cache_dir is not None:
    highest_weight_cgc_data_path = cache_dir / str(product_iweights) / ('highest_weight_CGC_' + str(sum_iweight))
    highest_weight_cgc_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(highest_weight_cgc_data_path, 'wb') as fp:
        dump(cgc_dict, fp)
```

Note: `cache_dir` is captured at the top of the function. The `highest_weight_cgc_data_path` variable must be reachable by both the read and write blocks; if `cache_dir is None`, both blocks are skipped.

### 2d. Update `calc_lower_weight_cgcs` (lines 430-433, 546-547)

Same pattern as 2c, substituting the lower-weight filename.

### 2e. Update `calc_cgcs` (lines 571-572)

```python
cache_dir = _resolve_cache_dir()
if cache_dir is not None:
    cgc_data_directory = cache_dir / str(product_irreps)
    cgc_data_directory.mkdir(parents=True, exist_ok=True)
```

### 2f. Delete `_create_cgc_data_directory`

Remove the old function entirely (lines 17-22). All callers are replaced.

### 2g. Re-export from `pyclebsch/__init__.py`

```python
from pyclebsch.cgc import set_cache_dir, get_cache_dir
```

### 2h. Update `run/demo.py` comment

Replace lines 49-50:
```
(THIS WILL CAUSE WRITES TO YOUR FILE SYSTEM
IN A FOLDER CALLED CGC_Data WHICH IS IN THE PARENT FOLDER OF pyclebsch)
```
with:
```
(THIS WILL WRITE CACHED CGC DATA TO ./CGC_Data IN YOUR WORKING DIRECTORY.
USE pyclebsch.cgc.set_cache_dir() TO CHANGE THE LOCATION, OR PASS None TO DISABLE.)
```

## Phase 3: Verify (Green stays green)

### 3a. Run new tests
```bash
cd pyclebsch && uv run -m pytest tests/test_cgc_cache.py -v
```
All should pass.

### 3b. Run existing tests
```bash
cd pyclebsch && uv run -m pytest -v
```
Existing `test_lattice_data.py` tests should still pass (they don't touch CGC caching).

### 3c. Run demo end-to-end
```bash
cd pyclebsch && uv run -m run.demo
```
Output should match the known-good output captured earlier. `CGC_Data/` should appear in the working directory (not next to the installed package).

### 3d. Verify `None` mode
Quick manual check:
```bash
cd /tmp && python -c "
import pyclebsch.cgc as cgc
cgc.set_cache_dir(None)
print(cgc.calc_cgcs([(1,0,0),(1,1,0)]))
" && ls CGC_Data 2>&1  # should say "No such file or directory"
```
