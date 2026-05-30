"""HTTP-level API tests.

Tests actual request/response shapes, status codes, and error handling.
Uses httpx.AsyncClient with ASGITransport.
"""

import pytest
from sqlalchemy import insert

from retain.hot_path import _new_id, _utcnow
from retain.models import TranscriptChunk


class TestContext:
    @pytest.mark.unit
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


class TestMemories:
    @pytest.mark.unit
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

    @pytest.mark.unit
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


class TestTasks:
    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
    async def test_complete_nonexistent_task_returns_404(self, client):
        response = await client.patch("/v1/tasks/nonexistent_task_id_12345")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.unit
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


class TestEvents:
    @pytest.mark.unit
    async def test_nonexistent_event_returns_404(self, client):
        response = await client.get("/v1/events/nonexistent_event_id_12345")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestSearchErrors:
    @pytest.mark.unit
    async def test_empty_query_returns_400(self, client):
        response = await client.post("/v1/search", params={"query": "   "})
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    @pytest.mark.unit
    async def test_missing_provider_returns_503(self, client):
        client._transport.app.state.embedding_provider = None
        response = await client.post("/v1/search", params={"query": "hello"})
        assert response.status_code == 503
        client._transport.app.state.embedding_provider = _mock_embedding_provider(
            dim=1024
        )

    @pytest.mark.unit
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


class TestDenseSearch:
    @pytest.mark.unit
    async def test_returns_results_with_expected_shape(
        self, client, embedding_provider
    ):
        await _seed_chunk(
            client,
            embedding_provider,
            text="Customer reported a billing issue with double charge.",
            entity_type="customer",
            entity_id="cust_s",
        )

        response = await client.post(
            "/v1/search",
            params={
                "query": "billing issue",
                "entity_type": "customer",
                "entity_id": "cust_s",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "chunks" in body
        chunks = body["chunks"]
        assert len(chunks) >= 1
        chunk = chunks[0]
        assert "id" in chunk
        assert "transcript_id" in chunk
        assert "entity_type" in chunk
        assert "entity_id" in chunk
        assert chunk["entity_id"] == "cust_s"
        assert "chunk_index" in chunk
        assert "chunk_text" in chunk
        assert "score" in chunk
        assert 0.0 <= chunk["score"] <= 1.0

    @pytest.mark.unit
    async def test_empty_results_when_no_chunks(self, client):
        response = await client.post(
            "/v1/search",
            params={
                "query": "nothing should match",
                "entity_type": "nobody",
                "entity_id": "none",
            },
        )
        assert response.status_code == 200
        assert response.json()["chunks"] == []


# ── helpers ────────────────────────────────────────────────


def _mock_embedding_provider(dim: int = 1024):
    from tests.conftest import _MockEmbeddingProvider as Mock
    return Mock(dim=dim)


async def _seed_chunk(client, provider, *, text, entity_type, entity_id):
    vector = provider.encode_query_sync(text)
    engine = client._transport.app.state.engine
    async with engine.begin() as conn:
        await conn.execute(
            insert(TranscriptChunk).values(
                id=_new_id(),
                transcript_id="00000000000000000000000000000001",
                entity_type=entity_type,
                entity_id=entity_id,
                chunk_index=0,
                chunk_text=text,
                embedding=vector,
                created_at=_utcnow(),
                updated_at=_utcnow(),
            )
        )
