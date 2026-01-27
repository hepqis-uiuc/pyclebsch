# pyclebsch
A Python package for calculating SU(N) Clebsch-Gordan coefficients (CGCs). It is largely based on the algorithm presented in https://homepages.physik.uni-muenchen.de/~vondelft/PapersVonDelft/Alex2011.pdf, with some modifications to account for residual symmetric group symmetries that can be present in computed CGCs.

Note that the current version of the package reads/writes data to a folder 'CGC_Data' in the same directory as the `pyclesbsch` package. This folder is created if it is not already present.

This codebase is currently at an 'alpha' stage of development. Breaking changes should be expected.

## Installation
This project uses [uv](https://docs.astral.sh/uv/getting-started/) for environment (packages, Python version) management. Ensure that you have uv installed (instructions for various operating systems available at the previously linked-to docs).

Once you have uv installed, use it to execute project scripts, and the correct virtual environment will automatically be used. For example:

``` shell
uv run run.py
```
There is no need to manually activate or deactivate the virtual environment. Information about the virtual environment is documented in `pyproject.toml`, and can be viewed by running
```shell
uv pip list
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

### Installation for Windows subsystem for Linux (WSL)
After setting up WSL there is a checklist of programs you may need before proceeding with the regular installation instructions above:

1. Download / update Git by running `sudo apt-get install git`.
2. Download / update Python3 by running `sudo apt install python3 python3-pip`

## Usage
There is a small script called run.py which demonstrates standard usage of some of the functionality of this package. We plan on adding more documentation in the future.
