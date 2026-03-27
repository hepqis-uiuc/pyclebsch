# pyclebsch
A Python package for calculating SU(N) Clebsch-Gordan coefficients (CGCs). It is largely based on the algorithm presented in https://homepages.physik.uni-muenchen.de/~vondelft/PapersVonDelft/Alex2011.pdf, with some modifications to account for residual symmetric group symmetries that can be present in computed CGCs.

Note that by default, Clebsch-Gordan coefficients are cached in a `CGC_Data` directory in whatever the current working directory is. See the "Caching" section below for more information on this behavior.

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

## Caching CGC data to disk
In the process of running, computed Clebsch-Gordan coefficients are by default cached in a `CGC_Data` folder in the current working directory. This is fine for some use cases, but if you anticipate running the same script from multiple directories, it can lead to unnecessary data duplication. To avoid such duplication, the cache directory can be set with an absolute path in the following way:
```Python
import pyclebsch.cgc as cgc

cgc.set_cache_dir("/path/to/cache/dir/desired-cgc-dir-name")
```
In this case, computed Clebsch-Gordan coefficients will be saved in the folder `desired-cgc-dir-name` at the absolute path specified. Alternatively, caching to disk can be disabled via
```Python
import pyclebsch.cgc as cgc

cgc.set_cache_dir(None)
```

If the default caching behavior doesn't work for your use case, configuring the cache directory should be done before using any other `pyclebsch` functionality.

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
