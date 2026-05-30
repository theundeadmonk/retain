"""Integration tests for semantic search."""

import pytest

from retain.search import search


@pytest.mark.integration
class TestSearch:
    async def test_search_returns_empty_when_no_chunks(self, engine, embedding_provider):
        result = await search(engine, embedding_provider, "test query")
        assert result["chunks"] == []

    async def test_search_finds_relevant_chunk(self, engine, embedding_provider):
        await _insert_chunk(
            engine,
            embedding_provider,
            entity_type="customer",
            entity_id="cust_1",
            text="The customer's billing address is 123 Main Street.",
        )
        await _insert_chunk(
            engine,
            embedding_provider,
            entity_type="customer",
            entity_id="cust_1",
            text="The customer enjoys hiking on weekends.",
        )

        result = await search(
            engine,
            embedding_provider,
            "billing address",
            entity_type="customer",
            entity_id="cust_1",
            limit=2,
        )
        chunks = result["chunks"]
        assert len(chunks) == 2
        assert all(0.0 <= c["score"] <= 1.0 for c in chunks)
        assert all(c["entity_type"] == "customer" for c in chunks)

    async def test_search_respects_entity_filter(self, engine, embedding_provider):
        await _insert_chunk(
            engine,
            embedding_provider,
            entity_type="customer",
            entity_id="cust_a",
            text="Cust A has a refund request.",
        )
        await _insert_chunk(
            engine,
            embedding_provider,
            entity_type="customer",
            entity_id="cust_b",
            text="Cust B asked about shipping.",
        )

        result = await search(
            engine,
            embedding_provider,
            "refund",
            entity_type="customer",
            entity_id="cust_a",
            limit=3,
        )
        chunks = result["chunks"]
        assert all(c["entity_id"] == "cust_a" for c in chunks)

    async def test_search_respects_limit(self, engine, embedding_provider):
        for i in range(10):
            await _insert_chunk(
                engine,
                embedding_provider,
                entity_type="customer",
                entity_id="cust_1",
                text=f"Conversation topic {i}: the customer asked about product details.",
            )

        result = await search(engine, embedding_provider, "product", limit=3)
        assert len(result["chunks"]) == 3

    async def test_search_missing_embedding_skipped(self, engine, embedding_provider):
        from sqlalchemy import insert

        from retain.hot_path import _utcnow
        from retain.models import TranscriptChunk

        await _insert_chunk(
            engine,
            embedding_provider,
            entity_type="customer",
            entity_id="cust_1",
            text="This one has an embedding.",
        )
        async with engine.begin() as conn:
            await conn.execute(
                insert(TranscriptChunk).values(
                    id="deadbeef000000000000000000000001",
                    transcript_id="00000000000000000000000000000001",
                    entity_type="customer",
                    entity_id="cust_1",
                    chunk_index=99,
                    chunk_text="This one has no embedding.",
                    embedding=None,
                    created_at=_utcnow(),
                    updated_at=_utcnow(),
                ),
            )

        result = await search(engine, embedding_provider, "embedding", limit=10)
        assert len(result["chunks"]) == 1


async def _insert_chunk(engine, provider, *, entity_type, entity_id, text):
    import uuid

    from sqlalchemy import insert

    from retain.models import TranscriptChunk

    vector = provider.encode_query_sync(text)
    async with engine.begin() as conn:
        await conn.execute(
            insert(TranscriptChunk).values(
                id=uuid.uuid4().hex,
                transcript_id="00000000000000000000000000000001",
                entity_type=entity_type,
                entity_id=entity_id,
                chunk_index=0,
                chunk_text=text,
                embedding=vector,
            )
        )
