import pytest

from retain import Memory
from retain.errors import RetainConfigError
from retain.types import EntityRef, ProcessRequest


@pytest.mark.unit
class TestStubs:
    """Tests for methods that are not yet implemented."""

    async def test_process_raises_without_llm(self, memory: Memory):
        req = ProcessRequest(
            entities=[EntityRef(entity_type="user", entity_id="alice")],
            transcript=[{"role": "user", "content": "hello"}],
        )
        with pytest.raises(RetainConfigError, match="LLM provider"):
            await memory.process(req)

    async def test_event_status_returns_not_found(self, memory: Memory):
        s = await memory.event_status("nonexistent-id")
        assert s == {"status": "not_found"}

    async def test_search_raises_not_implemented(self, memory: Memory):
        with pytest.raises(NotImplementedError):
            await memory.search("user", "alice", "query")
