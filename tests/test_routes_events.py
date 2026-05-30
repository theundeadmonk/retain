"""HTTP tests for POST /v1/process + GET /v1/events/{id}."""

import pytest


@pytest.mark.unit
class TestRoutesEvents:
    async def test_nonexistent_event_returns_404(self, client):
        response = await client.get("/v1/events/nonexistent_event_id_12345")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_process_without_llm_returns_503(self, client):
        response = await client.post(
            "/v1/process",
            json={
                "entities": [
                    {"entity_type": "customer", "entity_id": "cust_z"}
                ],
                "transcript": [
                    {"role": "agent", "content": "Hello"},
                ],
            },
        )
        assert response.status_code == 503
        assert "llm" in response.json()["detail"].lower()
