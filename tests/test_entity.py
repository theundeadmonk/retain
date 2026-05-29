import pytest

from retain import Memory


@pytest.mark.unit
class TestEntity:
    async def test_first_context_creates_entity(self, memory: Memory):
        ctx = await memory.context("user", "alice")
        assert ctx.entity_type == "user"
        assert ctx.entity_id == "alice"
        assert ctx.profile_blob is None
        assert ctx.memory_count == 0
        assert len(ctx.recent_memories) == 0
        assert len(ctx.open_tasks) == 0

    async def test_subsequent_context_returns_same_entity(self, memory: Memory):
        ctx1 = await memory.context("user", "alice")
        ctx2 = await memory.context("user", "alice")
        assert ctx2.entity_id == "alice"
        assert ctx2.memory_count == ctx1.memory_count

    async def test_different_entity_types_are_isolated(self, memory: Memory):
        await memory.remember("user", "a", "note", {"msg": "user data"})
        await memory.remember("device", "a", "note", {"msg": "device data"})

        user_ctx = await memory.context("user", "a")
        dev_ctx = await memory.context("device", "a")

        assert user_ctx.memory_count == 1
        assert user_ctx.recent_memories[0].value == {"msg": "user data"}
        assert dev_ctx.memory_count == 1
        assert dev_ctx.recent_memories[0].value == {"msg": "device data"}

    async def test_same_type_different_ids_are_isolated(self, memory: Memory):
        await memory.remember("user", "a", "note", {"msg": "hello"})
        await memory.remember("user", "b", "note", {"msg": "world"})

        ctx_a = await memory.context("user", "a")
        ctx_b = await memory.context("user", "b")

        assert ctx_a.memory_count == 1
        assert ctx_b.memory_count == 1
        assert ctx_a.recent_memories[0].value["msg"] == "hello"
        assert ctx_b.recent_memories[0].value["msg"] == "world"

    async def test_entity_fields_match(self, memory: Memory):
        await memory.remember("test", "e1", "pref", {"theme": "dark"})
        ctx = await memory.context("test", "e1")
        assert ctx.entity_type == "test"
        assert ctx.entity_id == "e1"
        assert ctx.first_seen is not None
        assert ctx.last_seen is not None
