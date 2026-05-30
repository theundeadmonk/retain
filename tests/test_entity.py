import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.hot_path import context, remember


@pytest.mark.unit
class TestEntity:
    async def test_first_context_creates_entity(self, engine: AsyncEngine):
        ctx = await context(engine, "user", "alice")
        assert ctx.entity_type == "user"
        assert ctx.entity_id == "alice"
        assert ctx.profile_blob is None
        assert ctx.memory_count == 0
        assert len(ctx.recent_memories) == 0
        assert len(ctx.open_tasks) == 0

    async def test_subsequent_context_returns_same_entity(self, engine: AsyncEngine):
        ctx1 = await context(engine, "user", "alice")
        ctx2 = await context(engine, "user", "alice")
        assert ctx2.entity_id == "alice"
        assert ctx2.memory_count == ctx1.memory_count

    async def test_different_entity_types_are_isolated(self, engine: AsyncEngine):
        await remember(engine, "user", "a", "note", {"msg": "user data"})
        await remember(engine, "device", "a", "note", {"msg": "device data"})

        user_ctx = await context(engine, "user", "a")
        dev_ctx = await context(engine, "device", "a")

        assert user_ctx.memory_count == 1
        assert user_ctx.recent_memories[0].value == {"msg": "user data"}
        assert dev_ctx.memory_count == 1
        assert dev_ctx.recent_memories[0].value == {"msg": "device data"}

    async def test_same_type_different_ids_are_isolated(self, engine: AsyncEngine):
        await remember(engine, "user", "a", "note", {"msg": "hello"})
        await remember(engine, "user", "b", "note", {"msg": "world"})

        ctx_a = await context(engine, "user", "a")
        ctx_b = await context(engine, "user", "b")

        assert ctx_a.memory_count == 1
        assert ctx_b.memory_count == 1
        assert ctx_a.recent_memories[0].value["msg"] == "hello"
        assert ctx_b.recent_memories[0].value["msg"] == "world"

    async def test_entity_fields_match(self, engine: AsyncEngine):
        await remember(engine, "test", "e1", "pref", {"theme": "dark"})
        ctx = await context(engine, "test", "e1")
        assert ctx.entity_type == "test"
        assert ctx.entity_id == "e1"
        assert ctx.first_seen is not None
        assert ctx.last_seen is not None
