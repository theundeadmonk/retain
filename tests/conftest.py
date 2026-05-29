import pytest

from retain import Memory


@pytest.fixture
async def memory():
    m = Memory(storage="sqlite+aiosqlite:///:memory:")
    yield m
