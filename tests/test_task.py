import pytest

from retain import Memory


@pytest.mark.unit
class TestTask:
    async def test_create_task_returns_open_task(self, memory: Memory):
        task = await memory.create_task("user", "alice", "support", "Cannot log in")
        assert task.status == "open"
        assert task.task_type == "support"
        assert task.description == "Cannot log in"
        assert task.resolved_at is None
        assert isinstance(task.id, str)
        assert len(task.id) == 32

    async def test_create_task_updates_entity_count(self, memory: Memory):
        ctx_before = await memory.context("user", "alice")
        assert ctx_before.task_count_open == 0

        await memory.create_task("user", "alice", "support", "issue 1")
        await memory.create_task("user", "alice", "support", "issue 2")

        ctx_after = await memory.context("user", "alice")
        assert ctx_after.task_count_open == 2

    async def test_complete_task_resolves(self, memory: Memory):
        task = await memory.create_task("user", "alice", "support", "issue 1")

        completed = await memory.complete_task(task.id)
        assert completed is not None
        assert completed.status == "resolved"
        assert completed.resolved_at is not None
        assert completed.id == task.id

    async def test_complete_task_returns_none_if_not_found(self, memory: Memory):
        result = await memory.complete_task("nonexistent")
        assert result is None

    async def test_complete_task_decrements_count(self, memory: Memory):
        task = await memory.create_task("user", "alice", "support", "issue 1")

        ctx_mid = await memory.context("user", "alice")
        assert ctx_mid.task_count_open == 1

        await memory.complete_task(task.id)

        ctx_end = await memory.context("user", "alice")
        assert ctx_end.task_count_open == 0

    async def test_complete_task_is_idempotent(self, memory: Memory):
        task = await memory.create_task("user", "alice", "support", "issue 1")

        r1 = await memory.complete_task(task.id)
        r2 = await memory.complete_task(task.id)

        assert r1 is not None
        assert r2 is not None
        assert r2.status == "resolved"
        assert r2.resolved_at is not None

    async def test_list_tasks_no_filter(self, memory: Memory):
        await memory.create_task("user", "alice", "support", "issue 1")
        await memory.create_task("user", "alice", "billing", "issue 2")

        tasks = await memory.list_tasks("user", "alice")
        assert len(tasks) == 2

    async def test_list_tasks_filter_by_status(self, memory: Memory):
        t1 = await memory.create_task("user", "alice", "support", "issue 1")
        await memory.create_task("user", "alice", "support", "issue 2")

        await memory.complete_task(t1.id)

        open_tasks = await memory.list_tasks("user", "alice", status="open")
        resolved_tasks = await memory.list_tasks("user", "alice", status="resolved")
        assert len(open_tasks) == 1
        assert len(resolved_tasks) == 1

    async def test_count_never_goes_below_zero(self, memory: Memory):
        task = await memory.create_task("user", "alice", "support", "issue 1")

        await memory.complete_task(task.id)
        await memory.complete_task(task.id)

        ctx = await memory.context("user", "alice")
        assert ctx.task_count_open == 0
