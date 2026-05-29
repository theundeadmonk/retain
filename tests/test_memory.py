import pytest

from retain import Memory


@pytest.mark.unit
class TestMemory:
    async def test_remember_returns_id(self, memory: Memory):
        mid = await memory.remember("user", "alice", "preference", {"theme": "dark"})
        assert isinstance(mid, str)
        assert len(mid) == 32

    async def test_multiple_remembers_accumulate(self, memory: Memory):
        for i in range(5):
            await memory.remember("user", "alice", "note", {"i": i})

        ctx = await memory.context("user", "alice")
        assert ctx.memory_count == 5
        assert len(ctx.recent_memories) == 5

    async def test_context_returns_recent_first(self, memory: Memory):
        for i in range(15):
            await memory.remember("user", "alice", "note", {"i": i})

        ctx = await memory.context("user", "alice")
        assert len(ctx.recent_memories) == 10
        values = [m.value["i"] for m in ctx.recent_memories]
        assert values == [14, 13, 12, 11, 10, 9, 8, 7, 6, 5]

    async def test_remember_with_metadata_and_source(self, memory: Memory):
        mid = await memory.remember(
            "user", "alice",
            "incident",
            {"severity": "high"},
            metadata={"category": "billing"},
            source="extraction",
        )
        assert isinstance(mid, str)

        ctx = await memory.context("user", "alice")
        assert ctx.memory_count == 1
        m = ctx.recent_memories[0]
        assert m.memory_type == "incident"
        assert m.value == {"severity": "high"}
        assert m.source == "extraction"

    async def test_memories_persist_across_context_calls(self, memory: Memory):
        await memory.remember("user", "alice", "preference", {"theme": "dark"})

        ctx1 = await memory.context("user", "alice")
        ctx2 = await memory.context("user", "alice")
        assert ctx2.memory_count == ctx1.memory_count
        assert ctx2.recent_memories[0].value == ctx1.recent_memories[0].value
