import pytest

from retain import Memory


@pytest.mark.integration
class TestIntegration:
    async def test_full_lifecycle(self, memory: Memory):
        ctx = await memory.context("guest", "room_412")
        assert ctx.profile_blob is None

        mid = await memory.remember("guest", "room_412", "preference", {"pillows": "extra"})
        assert mid is not None

        ctx = await memory.context("guest", "room_412")
        assert ctx.memory_count == 1
        assert ctx.recent_memories[0].value == {"pillows": "extra"}

        task = await memory.create_task("guest", "room_412", "amenity", "Extra pillows to room 412")
        assert task.status == "open"

        tasks = await memory.list_tasks("guest", "room_412")
        assert len(tasks) == 1

        completed = await memory.complete_task(task.id)
        assert completed is not None
        assert completed.status == "resolved"

        tasks = await memory.list_tasks("guest", "room_412", status="open")
        assert len(tasks) == 0

    async def test_multiple_entities_interleaved(self, memory: Memory):
        await memory.remember("guest", "a", "note", {"msg": "guest a"})
        await memory.remember("guest", "b", "note", {"msg": "guest b"})
        await memory.remember("guest", "a", "note", {"msg": "guest a second"})

        ctx_a = await memory.context("guest", "a")
        assert ctx_a.memory_count == 2
        msgs_a = [m.value["msg"] for m in ctx_a.recent_memories]
        assert "guest a" in msgs_a
        assert "guest b" not in msgs_a

        ctx_b = await memory.context("guest", "b")
        assert ctx_b.memory_count == 1
        assert ctx_b.recent_memories[0].value["msg"] == "guest b"

    async def test_pure_read_does_not_mutate_entity(self, memory: Memory):
        ctx1 = await memory.context("user", "alice")
        ctx2 = await memory.context("user", "alice")
        ctx3 = await memory.context("user", "alice")

        assert ctx1.memory_count == ctx2.memory_count == ctx3.memory_count
        assert ctx1.task_count_open == ctx2.task_count_open == ctx3.task_count_open
        assert ctx1.recent_memories == ctx2.recent_memories == ctx3.recent_memories

    async def test_new_entity_context_is_empty(self, memory: Memory):
        ctx = await memory.context("user", "new_user")
        assert ctx.profile_blob is None
        assert ctx.memory_count == 0
        assert len(ctx.recent_memories) == 0
        assert ctx.task_count_open == 0
        assert len(ctx.open_tasks) == 0
