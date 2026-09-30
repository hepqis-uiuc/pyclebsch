import os

import pytest

from pyclebsch.cache import ENV_VAR, set_cache_dir


@pytest.fixture(scope="session", autouse=True)
def no_cache_settings_from_the_shell():
    """Start the test session with the disk cache off, whatever
    PYCLEBSCH_CACHE_DIR a developer's shell sets. pyclebsch reads the
    variable at import, which happens during collection, so both the variable
    and the setting it produced are reset here."""
    os.environ.pop(ENV_VAR, None)
    set_cache_dir(None)
    yield
