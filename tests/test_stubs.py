import pytest

from retain import Memory
from retain.types import ProcessRequest


@pytest.mark.unit
class TestStubs:
    async def test_process_raises_not_implemented(self, memory: Memory):
        req = ProcessRequest(
            entities=[{"entity_type": "user", "entity_id": "alice"}],
            transcript=[{"role": "user", "content": "hello"}],
        )
        with pytest.raises(NotImplementedError):
            await memory.process(req)

    async def test_event_status_raises_not_implemented(self, memory: Memory):
        with pytest.raises(NotImplementedError):
            await memory.event_status("some-event-id")

    async def test_search_raises_not_implemented(self, memory: Memory):
        with pytest.raises(NotImplementedError):
            await memory.search("user", "alice", "query")
