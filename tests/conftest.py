import os
import tempfile

import pytest

from retain import Memory


@pytest.fixture
async def memory():
    m = Memory(storage="sqlite+aiosqlite:///:memory:")
    yield m


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db", prefix="retain-test-")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
