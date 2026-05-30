"""HTTP tests for POST+GET+PATCH /v1/tasks."""

import pytest


@pytest.mark.unit
class TestRoutesTasks:
    async def test_create_task_returns_open_task(self, client):
        response = await client.post(
            "/v1/tasks",
            json={
                "entity_type": "customer",
                "entity_id": "cust_c",
                "task_type": "followup",
                "description": "Call back about refund",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "open"
        assert body["task_type"] == "followup"
        assert body["entity_type"] == "customer"

    async def test_complete_task_resolves(self, client):
        create = await client.post(
            "/v1/tasks",
            json={
                "entity_type": "customer",
                "entity_id": "cust_d",
                "task_type": "issue",
                "description": "Fix billing",
            },
        )
        task_id = create.json()["id"]

        patch = await client.patch(f"/v1/tasks/{task_id}")
        assert patch.status_code == 200
        assert patch.json()["status"] == "resolved"

    async def test_complete_nonexistent_task_returns_404(self, client):
        response = await client.patch("/v1/tasks/nonexistent_task_id_12345")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_list_tasks_filters_by_status(self, client):
        await client.post(
            "/v1/tasks",
            json={
                "entity_type": "customer",
                "entity_id": "cust_e",
                "task_type": "issue",
                "description": "Task 1",
            },
        )
        await client.post(
            "/v1/tasks",
            json={
                "entity_type": "customer",
                "entity_id": "cust_e",
                "task_type": "issue",
                "description": "Task 2",
            },
        )
        response = await client.get(
            "/v1/tasks",
            params={
                "entity_type": "customer",
                "entity_id": "cust_e",
                "status": "open",
            },
        )
        assert response.status_code == 200
        tasks = response.json()
        assert len(tasks) == 2
        assert all(t["status"] == "open" for t in tasks)
