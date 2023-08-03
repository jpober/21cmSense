import pytest

from pathlib import Path


@pytest.fixture(scope="session")
def tmpdirec(tmpdir_factory):
    return Path(tmpdir_factory.mktemp("configs"))
