import os

import pytest

from pyclebsch.cache import ENV_VAR, set_cgc_cache_dir


# Tests marked slow are skipped unless pytest is run with --runslow. This is
# the recipe from pytest's documentation, "Control skipping of tests
# according to command line option".
def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session", autouse=True)
def no_cache_settings_from_the_shell():
    """Start the test session with the disk cache off, whatever
    PYCLEBSCH_CACHE_DIR a developer's shell sets. pyclebsch reads the
    variable at import, which happens during collection, so both the variable
    and the setting it produced are reset here."""
    os.environ.pop(ENV_VAR, None)
    set_cgc_cache_dir(None)
    yield
