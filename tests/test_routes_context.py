"""HTTP tests for POST /v1/context."""

import pytest


@pytest.mark.unit
class TestRoutesContext:
    async def test_returns_empty_context_for_new_entity(self, client):
        response = await client.post(
            "/v1/context",
            json={"entity_type": "customer", "entity_id": "new_cust"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["entity_type"] == "customer"
        assert body["entity_id"] == "new_cust"
        assert body["profile_blob"] is None
        assert body["recent_memories"] == []
        assert body["open_tasks"] == []
        assert body["memory_count"] == 0
