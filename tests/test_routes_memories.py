"""HTTP tests for POST /v1/memories."""

import pytest


@pytest.mark.unit
class TestRoutesMemories:
    async def test_remember_returns_id(self, client):
        response = await client.post(
            "/v1/memories",
            json={
                "entity_type": "customer",
                "entity_id": "cust_a",
                "memory_type": "preference",
                "value": {"room": "412"},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "id" in body
        assert len(body["id"]) == 32

    async def test_remember_persists_in_context(self, client):
        await client.post(
            "/v1/memories",
            json={
                "entity_type": "customer",
                "entity_id": "cust_b",
                "memory_type": "preference",
                "value": {"pillows": "extra"},
            },
        )
        response = await client.post(
            "/v1/context",
            json={"entity_type": "customer", "entity_id": "cust_b"},
        )
        body = response.json()
        assert body["memory_count"] == 1
        assert len(body["recent_memories"]) == 1
        assert body["recent_memories"][0]["memory_type"] == "preference"
