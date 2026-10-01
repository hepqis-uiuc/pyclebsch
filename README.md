# pyclebsch
A Python package for calculating SU(N) Clebsch-Gordan coefficients (CGCs). It is largely based on the algorithm presented in https://homepages.physik.uni-muenchen.de/~vondelft/PapersVonDelft/Alex2011.pdf, with some modifications to account for residual symmetric group symmetries that can be present in computed CGCs.

Computed Clebsch-Gordan coefficients are cached in memory by default; to save to disk, see "Caching CGC data" below.

This codebase is currently at an 'alpha' stage of development. Breaking changes should be expected.

## Installation
This project uses [uv](https://docs.astral.sh/uv/getting-started/) for environment (packages, Python version) management. Ensure that you have uv installed (instructions for various operating systems available at the previously linked-to docs).

Once you have uv installed, use it to execute project scripts (collected in the `run` directory), and the correct virtual environment will automatically be used. For example:

``` shell
uv run -m run.demo
```
There is no need to manually activate or deactivate the virtual environment. Information about the virtual environment is documented in `pyproject.toml`, and can be viewed by running
```shell
uv pip list
```
### Installation for Windows subsystem for Linux (WSL)
After setting up WSL there is a checklist of programs you may need before proceeding with the regular installation instructions above:

1. Download / update Git by running `sudo apt-get install git`.
2. Download / update Python3 by running `sudo apt install python3 python3-pip`

## Caching CGC data
### In memory (always on)
Each table of CGCs is computed at most once per process and then kept in memory, however many times it is requested. The memory cache can be inspected and controlled with:
```Python
import pyclebsch

pyclebsch.cache_stats()               # hits, computations, disk reads/writes since start
pyclebsch.clear_memory_cache()        # drop every in-memory table
pyclebsch.set_memory_cache_limit(500) # keep at most 500 tables (least recently used are dropped); None = no limit (the default)
```

### On disk (opt-in)
By default nothing is written to disk, so every new process computes the CGC tables it needs again. For small tensor products this is often fine, but single large products can take much longer (for example, finding the singlet in 3 ⊗ 3̄ ⊗ 8 ⊗ 8 ⊗ 6 ⊗ 6̄ can take upwards of a minute on a consumer-grade laptop). To keep tables across runs, name a directory, either in code
```Python
import pyclebsch

pyclebsch.set_cache_dir("~/cgc-cache")   # a leading ~ is expanded; the path is resolved to an absolute path when set
pyclebsch.set_cache_dir(None)            # turn the disk cache off again
```
or through the environment, before Python starts:
```shell
export PYCLEBSCH_CACHE_DIR="$HOME/cgc-cache"
```
There is no default location: `set_cache_dir("")` raises `ValueError`, and an empty `PYCLEBSCH_CACHE_DIR` means that CGCs will not be written to disk. A relative path is resolved against the working directory at the time of the call, so a later `os.chdir` does not move the cache. `set_cache_dir` may be called at any time; it also clears the in-memory cache.

**Multiprocessing.** `set_cache_dir` records the directory in `PYCLEBSCH_CACHE_DIR`, so worker processes started afterwards use the same directory, and pyclebsch's own worker pools also receive the setting explicitly. One case needs care: with the `forkserver` start method (the default on Linux from Python 3.14), a `multiprocessing.Pool` you create yourself after the fork server has started does not see a later `set_cache_dir` call. Configure the cache before creating your first pool, or create pools with `Pool(n, **pyclebsch.cache.pool_kwargs())`.

**What is stored.** Entries live under `<cache dir>/v<N>/`, where `N` is the version of the CGC conventions (`pyclebsch.cache.CGC_CACHE_VERSION`), with one file per table. Each file records metadata alongside the CGCs: a metadata "envelope" format version, the CGC convention version, the table's key (including the zero threshold `EPS`), the installed pyclebsch version, and the git commit it was installed from. The commit is only known for installs made directly from a git repository; it is `None` for installs from a package index and for editable installs. An entry whose versions or key do not match, or that cannot be read, is deleted with a `CGCCacheWarning` and recomputed. Entries are written atomically, so several processes can share a cache directory.

**Trust.** Cache entries are Python pickle files, and loading a pickle can run arbitrary code. Only point pyclebsch at directories you trust, and do not commit a cache directory or share it with others.

### Upgrading from pyclebsch 0.1
- Nothing is written to disk by default, and `get_cache_dir()` returns `None` until a directory is set. Scripts that relied on `./CGC_Data` persisting across runs should call `set_cache_dir` or set `PYCLEBSCH_CACHE_DIR`.
- Old `CGC_Data` folders are never read; they can be deleted.
- `calc_highest_weight_cgcs` and `calc_lower_weight_cgcs` are no longer public. Use `calc_cgcs`, which covers every access pattern (all irreps, one irrep, one copy, one state, one coefficient) and sorts the product for you.
- i-weights must have integer entries. numpy integers are accepted and converted, so the sum-irrep keys in `calc_cgcs` results are plain `int` tuples; floats raise `TypeError`.

## Usage/citations
If you use this code in a paper, please cite:
```bibtex
@misc{balaji-2025-perturbat-su,
  author = {Balaji, Praveen and Conefrey-Shinozaki, Cianan and Draper, Patrick and Elhaderi, Jason K. and Gupta, Drishti and Hidalgo, Luis and Lytle, Andrew},
  title = {Perturbation theory, irrep truncations, and state preparation methods for quantum simulations of SU(3) lattice gauge theory},
  year = {2025},
  doi = {10.48550/ARXIV.2509.25865},
  url = {https://arxiv.org/abs/2509.25865},
}
```

## Adding, updating, and removing dependencies
To add a package to the project (for example, `numpy`):

``` shell
uv add numpy
```
This automatically updates the `pyproject.toml` file as well as the `uv.lock` file. The former is human readable/editable, while the latter is intended for consumption/editing by uv itself only. (Don't touch it!) Don't use `uv pip` to install packages, because this doesn't automatically update the lockfile.

To add a package to the project that's only needed for development (for example, `pytest`):
```shell
uv add --dev pytest
```

If you want to update a package to a newer version:
```shell
uv add --upgrade numpy
```

To remove a package:
```shell
uv remove numpy
```

### Locking and syncing
The uv tool should automatically lock (generate a machine-readable description of the environment) and sync (update your actual virtual environment) as needed. However, if you run in to issues environment issues, you might try executing these commands manually as debug steps:
```shell
uv lock
uv sync
```

## Tests
The project uses [pytest](https://docs.pytest.org/en/stable/).
There are also numerous useful [how-to](https://docs.pytest.org/en/stable/how-to/index.html#how-to) guides available.

### Writing new tests
When adding new tests, group the functionality being tested by file. For example, if you were writing tests for a class or function in a module named `pyclebsch.some_module` or `run.some_module`, then all tests for this class should go in a file `tests/test_some_module.py`.

Note that all test files must be named `test_[something].py`!

### Running tests
If you want to run all tests in the `test` directory, activate the virtual environment and then type:
```
uv run -m pytest -v
```
The `-v` flag is optional and simply outputs additional debug info. Another useful flag is `-s`, which enables displaying all print statements generated while tests are running.

If you want to run all tests in a specific file:
```
uv run -m pytest tests/test_[file].py
```

If you want to run *just one* test:
```
uv run -m pytest tests/test_mod.py::test_func.
```

There's also more complete documentation on [how to invoke pytest](https://docs.pytest.org/en/stable/how-to/usage.html) which presents some additional features.

### Slow tests
Tests which take a long time to run can be skipped by default. To do this, use the following decorator:
```
@pytest.mark.slow
def test_this_is_some_slow_test():
    [test logic here]
```

`conftest.py` is set up so that any test marked this way will be skipped by default. To include slow tests in a test run:
```
uv run -m pytest --runslow
```

## Usage
Various scripts which make use of the functionality in `pyclebsch` are gathered in the `run` directory. To execute any of them run (for example):
```shell
uv run -m run.demo
```
If the script `run/some_script.py` exists, then replace `demo` with `some_script` to execute it instead. Additional scripts can be added this way.
