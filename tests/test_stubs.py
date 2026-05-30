import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.cold_path import event_status


@pytest.mark.unit
class TestStubs:
    """Tests for methods that are not yet implemented."""

    async def test_event_status_returns_not_found(self, engine: AsyncEngine):
        s = await event_status(engine, "nonexistent-id")
        assert s == {"status": "not_found"}
