"""Synthetic retrieval benchmark — precision@5, recall@5, MRR.

Generates known facts, embeds them locally with BGE-large + SPLADE,
seeds them into the database with pre-computed vectors, then runs
search queries against the HTTP API. No LLM needed — tests pure
retrieval quality. Cleans up all bench_* data after run.

Usage:
    uv run python benchmarks/retrieval.py

Requires a running server (docker compose up) with BGE + SPLADE loaded.
Requires the embeddings dependency group (uv sync --group embeddings).
"""

import asyncio
import random
import uuid
from typing import Any

import httpx
from sqlalchemy import insert, text

from retain.embeddings.local import FastEmbedProvider, SparseEmbedProvider
from retain.hot_path import _new_id, _utcnow
from retain.models import TranscriptChunk
from retain.settings import settings
from retain.storage import create_engine

BASE = "http://localhost:8000"
RUN_ID = uuid.uuid4().hex[:8]
NUM_ENTITIES = 50
FACTS_PER_ENTITY = 5

FACT_TEMPLATES = [
    "{entity_id} email alias is {fact_word}@example.com",
    "{entity_id} prefers {fact_word} as their contact method",
    "{entity_id} order #{num} was for ${amount}",
    "{entity_id} last call was about {fact_word}",
    "{entity_id} has {num} open support tickets",
    "{entity_id} billing address is {num} {fact_word} Street",
    "{entity_id} account manager is {fact_word}",
    "{entity_id} signed up on {fact_word} 2024",
    "{entity_id} subscription tier is {fact_word}",
    "{entity_id} payment method on file is {fact_word}",
]


def _generate_facts() -> list[dict[str, Any]]:
    random.seed(42)
    facts: list[dict[str, Any]] = []
    used_fact_words: set[str] = set()

    for entity_idx in range(NUM_ENTITIES):
        entity_type = "customer"
        entity_id = f"bench_ret_{RUN_ID}_{entity_idx:02d}"

        for _ in range(FACTS_PER_ENTITY):
            template = random.choice(FACT_TEMPLATES)
            fact_word = f"item{random.randint(0, 9999):04d}"
            while fact_word in used_fact_words:
                fact_word = f"item{random.randint(0, 9999):04d}"
            used_fact_words.add(fact_word)

            text = template.format(
                entity_id=entity_id,
                fact_word=fact_word,
                num=random.randint(1000, 9999),
                amount=round(random.uniform(10, 500), 2),
            )
            query = fact_word

            facts.append({
                "entity_type": entity_type,
                "entity_id": entity_id,
                "text": text,
                "query": query,
            })

    return facts


async def _seed_with_embeddings(
    facts: list[dict[str, Any]],
    dense: FastEmbedProvider,
    sparse: SparseEmbedProvider,
) -> None:
    engine = create_engine(settings.database_url)
    try:
        texts = [f["text"] for f in facts]
        print(f"Embedding {len(texts)} chunks (BGE-large + SPLADE) ...")

        dense_embeddings = await dense.encode_documents(texts)
        sparse_embeddings = await sparse.encode_documents(texts)

        print("Inserting chunks into database ...")
        async with engine.begin() as conn:
            for fact, dense_vec, sparse_vec in zip(facts, dense_embeddings, sparse_embeddings):
                await conn.execute(
                    insert(TranscriptChunk).values(
                        id=_new_id(),
                        transcript_id=_new_id(),
                        entity_type=fact["entity_type"],
                        entity_id=fact["entity_id"],
                        chunk_index=0,
                        chunk_text=fact["text"],
                        embedding=dense_vec,
                        sparse_embedding=_to_sparse_str(sparse_vec),
                        created_at=_utcnow(),
                        updated_at=_utcnow(),
                    )
                )
    finally:
        await engine.dispose()


def _to_sparse_str(sparse: dict[str, object]) -> str:
    indices: list[int] = sparse["indices"]  # type: ignore[assignment]
    values: list[float] = sparse["values"]  # type: ignore[assignment]
    pairs = ",".join(f"{i}:{v}" for i, v in zip(indices, values))
    return "{" + pairs + "}/30522"


async def _run_queries(facts: list[dict[str, Any]]) -> dict[str, float]:
    async with httpx.AsyncClient(base_url=BASE, timeout=30) as client:
        entity_facts: dict[tuple[str, str], set[str]] = {}
        for f in facts:
            key = (f["entity_type"], f["entity_id"])
            entity_facts.setdefault(key, set()).add(f["text"])

        precision_hits = 0
        recall_scores: list[float] = []
        reciprocal_ranks: list[float] = []

        for i, fact in enumerate(facts):
            if i % 50 == 0:
                print(f"  Query {i}/{len(facts)} ...")

            response = await client.post(
                "/v1/search",
                params={
                    "query": fact["query"],
                    "entity_type": fact["entity_type"],
                    "entity_id": fact["entity_id"],
                    "limit": 5,
                },
                timeout=30,
            )
            body = response.json()
            chunks = body.get("chunks", [])

            found = False
            rank = 0
            for j, chunk in enumerate(chunks):
                if fact["text"] in chunk.get("chunk_text", ""):
                    found = True
                    rank = j + 1
                    break

            if found:
                precision_hits += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

            key = (fact["entity_type"], fact["entity_id"])
            all_facts = entity_facts.get(key, set())
            found_facts: set[str] = set()
            for chunk in chunks:
                for entity_fact in all_facts:
                    if entity_fact in chunk.get("chunk_text", ""):
                        found_facts.add(entity_fact)
            recall = len(found_facts) / len(all_facts) if all_facts else 1.0
            recall_scores.append(recall)

    return {
        "precision_at_5": round(precision_hits / len(facts) * 100, 1),
        "recall_at_5": round(sum(recall_scores) / len(recall_scores) * 100, 1),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
        "queries": len(facts),
        "entities": NUM_ENTITIES,
    }


async def _teardown() -> None:
    engine = create_engine()
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM transcript_chunks WHERE entity_id LIKE 'bench_ret_%'")
            )
    finally:
        await engine.dispose()


async def main() -> None:
    print(f"Run ID: {RUN_ID}")
    print(f"Generating {NUM_ENTITIES} entities x {FACTS_PER_ENTITY} facts ...")
    facts = _generate_facts()
    print(f"Generated {len(facts)} synthetic facts.")

    print("Loading embedding models (BGE-large + SPLADE) ...")
    dense = FastEmbedProvider(
        model_name=settings.embedding_query_model,
        batch_size=settings.embedding_batch_size,
    )
    sparse = SparseEmbedProvider(
        model_name=settings.embedding_sparse_model,
        batch_size=settings.embedding_batch_size,
    )
    _ = dense.dim
    sparse._load()
    print("Models loaded.")

    await _seed_with_embeddings(facts, dense, sparse)

    print(f"Running {len(facts)} search queries ...")
    metrics = await _run_queries(facts)

    print()
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Precision@5 | {metrics['precision_at_5']}% |")
    print(f"| Recall@5 | {metrics['recall_at_5']}% |")
    print(f"| MRR | {metrics['mrr']} |")
    print(f"| Queries | {metrics['queries']} |")
    print(f"| Entities | {metrics['entities']} |")

    await _teardown()
    print("Cleaned up bench_ret_* data.")


if __name__ == "__main__":
    asyncio.run(main())
