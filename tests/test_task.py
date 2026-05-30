import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.hot_path import complete_task, context, create_task, list_tasks


@pytest.mark.unit
class TestTask:
    async def test_create_task_returns_open_task(self, engine: AsyncEngine):
        task = await create_task(engine, "user", "alice", "support", "Cannot log in")
        assert task.status == "open"
        assert task.task_type == "support"
        assert task.description == "Cannot log in"
        assert task.resolved_at is None
        assert isinstance(task.id, str)
        assert len(task.id) == 32

    async def test_create_task_updates_entity_count(self, engine: AsyncEngine):
        ctx_before = await context(engine, "user", "alice")
        assert ctx_before.task_count_open == 0

        await create_task(engine, "user", "alice", "support", "issue 1")
        await create_task(engine, "user", "alice", "support", "issue 2")

        ctx_after = await context(engine, "user", "alice")
        assert ctx_after.task_count_open == 2

    async def test_complete_task_resolves(self, engine: AsyncEngine):
        task = await create_task(engine, "user", "alice", "support", "issue 1")
        completed = await complete_task(engine, task.id)
        assert completed is not None
        assert completed.status == "resolved"
        assert completed.resolved_at is not None
        assert completed.id == task.id

    async def test_complete_task_returns_none_if_not_found(self, engine: AsyncEngine):
        result = await complete_task(engine, "nonexistent")
        assert result is None

    async def test_complete_task_decrements_count(self, engine: AsyncEngine):
        task = await create_task(engine, "user", "alice", "support", "issue 1")
        ctx_mid = await context(engine, "user", "alice")
        assert ctx_mid.task_count_open == 1
        await complete_task(engine, task.id)
        ctx_end = await context(engine, "user", "alice")
        assert ctx_end.task_count_open == 0

    async def test_complete_task_is_idempotent(self, engine: AsyncEngine):
        task = await create_task(engine, "user", "alice", "support", "issue 1")
        r1 = await complete_task(engine, task.id)
        r2 = await complete_task(engine, task.id)
        assert r1 is not None
        assert r2 is not None
        assert r2.status == "resolved"
        assert r2.resolved_at is not None

    async def test_list_tasks_no_filter(self, engine: AsyncEngine):
        await create_task(engine, "user", "alice", "support", "issue 1")
        await create_task(engine, "user", "alice", "billing", "issue 2")
        tasks = await list_tasks(engine, "user", "alice")
        assert len(tasks) == 2

    async def test_list_tasks_filter_by_status(self, engine: AsyncEngine):
        t1 = await create_task(engine, "user", "alice", "support", "issue 1")
        await create_task(engine, "user", "alice", "support", "issue 2")
        await complete_task(engine, t1.id)

        open_tasks = await list_tasks(engine, "user", "alice", status="open")
        resolved_tasks = await list_tasks(engine, "user", "alice", status="resolved")
        assert len(open_tasks) == 1
        assert len(resolved_tasks) == 1

    async def test_count_never_goes_below_zero(self, engine: AsyncEngine):
        task = await create_task(engine, "user", "alice", "support", "issue 1")
        await complete_task(engine, task.id)
        await complete_task(engine, task.id)
        ctx = await context(engine, "user", "alice")
        assert ctx.task_count_open == 0
