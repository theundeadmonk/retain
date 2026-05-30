"""HTTP tests for POST /v1/search."""

import pytest
from sqlalchemy import insert

from retain.hot_path import _new_id, _utcnow
from retain.models import TranscriptChunk


@pytest.mark.unit
class TestRoutesSearchErrors:
    async def test_empty_query_returns_400(self, client):
        response = await client.post("/v1/search", params={"query": "   "})
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    async def test_missing_provider_returns_503(self, client):
        client._transport.app.state.embedding_provider = None
        response = await client.post("/v1/search", params={"query": "hello"})
        assert response.status_code == 503
        client._transport.app.state.embedding_provider = _mock_embedding_provider(
            dim=1024
        )


@pytest.mark.unit
class TestRoutesSearchDense:
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
