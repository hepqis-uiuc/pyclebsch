# Refactoring Proposal: CGC_Data Cache Location

## Problem

The `CGC_Data/` directory is currently created as a sibling of the `pyclebsch/` package directory, determined by navigating up from `__file__`:

```python
# cgc.py:17-22
def _create_cgc_data_directory():
    script_directory = Path(__file__).resolve().parent.parent
    data_directory = PurePath(script_directory, 'CGC_Data')
    Path(data_directory).mkdir(exist_ok=True)
    return data_directory
```

This means:
- **For developers**: `CGC_Data/` appears in the repo root (acceptable, already gitignored).
- **For pip-installed users**: `CGC_Data/` would be created inside `site-packages/` (or wherever the package was installed), which is surprising, likely read-only, and may fail silently or raise `PermissionError`.

### Scope of the problem

File I/O is concentrated in a single module, `pyclebsch/cgc.py`. The three functions that touch `CGC_Data/` are:

| Function | Lines | What it does |
|---|---|---|
| `_create_cgc_data_directory()` | 17-22 | Creates `CGC_Data/` dir, returns its path |
| `calc_highest_weight_cgcs()` | 39-42, 410-411 | Reads/writes `highest_weight_CGC_*` pickle files |
| `calc_lower_weight_cgcs()` | 430-433, 546-547 | Reads/writes `lower_weight_CGC_*` pickle files |
| `calc_cgcs()` | 571-572 | Creates per-product-irrep subdirectory |

All other modules (`plaquette_matrix_elements.py`, `lattice_data.py`, `run/` scripts) access CGCs exclusively through `calc_cgcs()`, which calls the functions above. So the fix is entirely local to `cgc.py`.

The `run/gen_ymcirc_data.py` script also writes to `./out/`, but that's a script-level output directory, not a library-internal cache, so it's out of scope.

---

## Proposed Design

### Option A (recommended): Configurable cache with sensible default

Introduce a module-level cache path that the user can configure, with a default location following platform conventions.

```python
# pyclebsch/cgc.py (sketch)

import platformdirs  # or use stdlib approach below

_cgc_cache_dir: Path | None = None  # None = use default

def set_cache_dir(path: str | Path | None) -> None:
    """Set the directory where computed CGCs are cached.

    Pass None to disable caching (compute on every call).
    """
    global _cgc_cache_dir
    _cgc_cache_dir = Path(path) if path is not None else None

def get_cache_dir() -> Path | None:
    """Return the current cache directory, or None if caching is disabled."""
    return _cgc_cache_dir

def _resolve_cache_dir() -> Path | None:
    """Return the effective cache directory, creating it if needed."""
    if _cgc_cache_dir is not None:
        _cgc_cache_dir.mkdir(parents=True, exist_ok=True)
        return _cgc_cache_dir
    # Default: platform-appropriate user cache
    default = Path(platformdirs.user_cache_dir("pyclebsch")) / "CGC_Data"
    default.mkdir(parents=True, exist_ok=True)
    return default
```

**Default location** (via `platformdirs.user_cache_dir`):
- Linux: `~/.cache/pyclebsch/CGC_Data/`
- macOS: `~/Library/Caches/pyclebsch/CGC_Data/`
- Windows: `C:\Users\<user>\AppData\Local\pyclebsch\Cache\CGC_Data\`

**User overrides**:
```python
import pyclebsch.cgc as cgc

# Use a project-local cache (equivalent to current behavior if run from repo root)
cgc.set_cache_dir("./CGC_Data")

# Use a specific absolute path
cgc.set_cache_dir("/data/shared_cgc_cache")

# Disable caching entirely (always recompute)
cgc.set_cache_dir(None)
```

#### Changes to cache read/write functions

The current `_create_cgc_data_directory()` would be replaced by `_resolve_cache_dir()`. The read/write logic in `calc_highest_weight_cgcs` and `calc_lower_weight_cgcs` would become:

```python
cache_dir = _resolve_cache_dir()
if cache_dir is not None:
    data_path = cache_dir / str(product_iweights) / f'highest_weight_CGC_{sum_iweight}'
    if data_path.exists():
        with open(data_path, 'rb') as fp:
            return load(fp)

# ... computation ...

if cache_dir is not None:
    subdir = cache_dir / str(product_iweights)
    subdir.mkdir(parents=True, exist_ok=True)
    with open(subdir / f'highest_weight_CGC_{sum_iweight}', 'wb') as fp:
        dump(cgc_dict, fp)
```

Same pattern for `calc_lower_weight_cgcs` and the subdirectory creation in `calc_cgcs`.

#### New dependency

This adds one dependency: `platformdirs` (a lightweight, well-maintained package -- it's already a transitive dependency of many common tools). Alternatively, you could use the stdlib-only fallback described in Option B.

---

### Option B: Stdlib-only, user's working directory as default

If you want to avoid adding a dependency:

```python
_cgc_cache_dir: Path | None = Path("./CGC_Data")

def set_cache_dir(path: str | Path | None) -> None:
    global _cgc_cache_dir
    _cgc_cache_dir = Path(path) if path is not None else None
```

- **Default**: `./CGC_Data` relative to the user's working directory (not relative to `__file__`).
- **Pros**: No new dependency; users see the cache where they run from.
- **Cons**: The cache location changes depending on where the user runs their script from, which can lead to re-computation when the same code is invoked from different directories.

---

### Option C: In-memory LRU cache with optional disk persistence

For users who only need CGCs within a single session, an `@functools.lru_cache` on the computation functions would eliminate disk I/O entirely. Disk caching could then be opt-in rather than the default:

```python
@functools.lru_cache(maxsize=None)
def calc_highest_weight_cgcs(product_iweights, sum_iweight, multiplicity):
    ...  # pure computation, no file I/O

# Separate utility for explicit save/load
def save_cgc_cache(path: Path): ...
def load_cgc_cache(path: Path): ...
```

- **Pros**: Zero disk side-effects by default; fastest for repeated calls in a session.
- **Cons**: No persistence across sessions unless the user explicitly saves; large computations would need to be re-done each time. The arguments to `calc_highest_weight_cgcs` include `list[tuple]` and `dict` types which aren't hashable as-is, so you'd need to convert them to frozen/hashable types (e.g., tuple of tuples).

---

## Recommendation

**Option A** gives the best balance: a platform-appropriate default that just works for pip-installed users, full user control, and the ability to disable caching. It is also the most common pattern in the Python ecosystem (e.g., `huggingface_hub`, `pooch`, `joblib` all follow this pattern).

If minimizing dependencies matters more, **Option B** is a fine pragmatic choice -- the key fix is just changing the default from `Path(__file__).parent.parent / 'CGC_Data'` to `Path('./CGC_Data')` so it respects the user's working directory rather than writing next to the installed package.

## Migration Steps

1. Add `set_cache_dir()` / `get_cache_dir()` API to `cgc.py`.
2. Replace `_create_cgc_data_directory()` with the new `_resolve_cache_dir()`.
3. Guard all `open()`/`dump()` calls with a `cache_dir is not None` check.
4. Expose `set_cache_dir` in `pyclebsch/__init__.py` for convenience.
5. Update `run/demo.py` warning comment (line 49-50) to reflect the new behavior.
6. (Option A only) Add `platformdirs` to `pyproject.toml` dependencies.
7. Optionally add an environment variable override (e.g., `PYCLEBSCH_CACHE_DIR`) for CI/container use cases.

## Impact on Existing Users

- Existing `CGC_Data/` directories in the repo root will no longer be auto-discovered (the default location changes). Users who want to keep their existing cache can call `set_cache_dir("./CGC_Data")` or move the directory to the new default location.
- Since this is a `0.1.0` package, a breaking change to cache location is reasonable. A note in the changelog would suffice.

## Impact on Tests

The test file `tests/test_lattice_data.py` does not directly test caching behavior. If tests call `calc_cgcs`, they will use whatever cache directory is set. For test isolation, tests can call `set_cache_dir(tmp_path)` using pytest's `tmp_path` fixture, or `set_cache_dir(None)` to disable caching during tests.

## Impact on Parallelized Computation

`plaquette_matrix_elements.py` uses `multiprocessing.Pool` which spawns worker processes. The current design works because `_create_cgc_data_directory()` derives the path from `__file__`, which is the same in every process. With the proposed design, the module-level `_cgc_cache_dir` variable will be inherited by forked child processes (the default on Linux/macOS). However, if `spawn` start method is used (default on macOS with Python 3.14), the global won't be inherited. This is already partially addressed by the existing pattern where CGCs are pre-computed before parallelization. To be safe, the `set_cache_dir()` value could also be persisted via an environment variable that `_resolve_cache_dir()` checks as a fallback.
