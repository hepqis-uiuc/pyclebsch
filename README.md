# pyclebsch
A Python package for calculating SU(N) Clebsch-Gordan coefficients (CGCs). It is largely based on the algorithm presented in https://homepages.physik.uni-muenchen.de/~vondelft/PapersVonDelft/Alex2011.pdf, with some modifications to account for residual symmetric group symmetries that can be present in computed CGCs.

Note that the current version of the package reads/writes data to a folder 'CGC_Data' in the same directory as the `pyclesbsch` package. This folder is created if it is not already present.

This codebase is currently at an 'alpha' stage of development. Breaking changes should be expected.

## Installation
The Makefile is configured to automatically set up a Python virtual environment. To use it:

1. Run `make venv` to create the Python virtual environment.
2. Run `source .venv/bin/activate` to activate the virtual environment.
3. If you want to deactivate the virtual environment, run `deactivate`.
4. Run `make clean` to remove the Python virtual environment. Make sure you deactive the virtual environment before doing this!

Alternatively, you can use the `requirements.txt` file to set up a virtual environment with your favored environment management tool.

### Installation for Windows subsystem for Linux (WSL)
After setting up WSL there is a checklist of programs you may need before proceeding with the regular installation instructions above (version numbers may differ):

1. Download / update Git by running `sudo apt-get install git`.
2. Download / update Python3 by running `sudo apt install python3 python3-pip`
3. Download the venv package by running `sudo apt install python3.10-venv`

## Usage
There is a small script called run.py which demonstrates standard usage of some of the functionality of this package. We plan on adding more documentation in the future.
