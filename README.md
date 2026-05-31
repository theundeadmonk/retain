# Retain

The memory layer built for real-time agents — FastAPI server.

PostgreSQL + pgvector 0.8.2. BGE-large dense + SPLADE sparse hybrid retrieval.
Half-precision embeddings, HNSW on both dense and sparse indexes.
Profile context in <5ms. Full-text search in <60ms. Extraction runs async post-call.

---

## Quickstart

```bash
# Start PostgreSQL + server
docker compose up -d

# Store a typed fact
curl -s -X POST localhost:8000/v1/memories \
  -H 'Content-Type: application/json' \
  -d '{"entity_type":"customer","entity_id":"alice","memory_type":"preference","value":{"room":"412"}}'

# Look up everything the agent needs
curl -s -X POST localhost:8000/v1/context \
  -H 'Content-Type: application/json' \
  -d '{"entity_type":"customer","entity_id":"alice"}' | jq .

# Search past conversations
curl -s -X POST "localhost:8000/v1/search?query=billing+issue&entity_type=customer&entity_id=alice" | jq .
```

---

## Architecture

### Path Tiers

Retain splits work into three latency tiers so the agent never waits on
the database:

| Tier | Budget | What runs | Endpoints |
|------|--------|-----------|-----------|
| **Hot** | <50ms | Pure PostgreSQL reads/writes. No IO beyond the DB. | `/v1/context`, `/v1/memories`, `/v1/tasks` |
| **Warm** | <60ms | Local embedding inference + pgvector HNSW + RRF fusion. | `/v1/search` |
| **Cold** | 5-15s async | LLM extraction + proposition decomposition + dense/sparse embedding. Client submits → gets `event_id` → polls. | `/v1/process`, `/v1/events/{id}` |

### Latency Budget

| Endpoint | Target | Actual | Composition |
|----------|--------|--------|-------------|
| `POST /v1/context` | <50ms | ~5ms | Single-row PostgreSQL upsert + 3 indexed SELECTs |
| `POST /v1/memories` | <5ms | ~2ms | Fire-and-forget INSERT |
| `POST /v1/tasks` | <5ms | ~2ms | Fire-and-forget INSERT |
| `PATCH /v1/tasks/{id}` | <5ms | ~2ms | Single-row UPDATE |
| `GET /v1/tasks` | <50ms | ~5ms | Indexed SELECT with optional status filter |
| `POST /v1/search` | <80ms | ~50ms | BGE encode (25ms) + SPLADE encode (15ms) + RRF query (10ms) |
| `POST /v1/process` | <10ms | ~5ms | INSERT transcript + INSERT event + spawn background task |
| `GET /v1/events/{id}` | <5ms | ~2ms | Single-row SELECT |

---

## Retrieval Pipeline

### Cold path — post-call (async, 5-15s)

```
process() returns event_id immediately, then:

1. LLM extract     → typed memory facts (memories table)        ~2-5s
2. Dedup            → rule-based exact-match de-duplication       ~5ms
3. LLM decompose    → atomic self-contained propositions          ~1-2s
4. BGE-large embed  → halfvec dense vectors (1024d)              ~100ms
5. SPLADE embed     → sparse vectors (30522d vocab)              ~50ms
6. LLM synthesize   → caller profile blob (<80 tokens)            ~1-2s
7. Store             → transactional INSERT into transcript_chunks ~10ms
```

### Warm path — mid-call (blocking, ~50ms)

```
search() returns results to the agent:

1. BGE encode query    → halfvec dense vector                    ~25ms
2. SPLADE encode query → sparse vector                          ~15ms
3. HNSW dense search   → pgvector cosine on indexed column       ~3ms
4. HNSW sparse search  → pgvector L2 on indexed column           ~3ms
5. RRF fusion          → reciprocal rank fusion of both lists     ~2ms
6. SPLADE tiebreak     → raw sparse distance secondary sort      ~0ms
7. Query weight        → adapt dense/sparse balance by query type ~0ms
```

---

## Retrieval Techniques

### Adopted

| Technique | Paper | Year | What it does | Where |
|-----------|-------|------|-------------|-------|
| **Proposition-Based Chunking** | Chen et al. | 2024 | LLM decomposes transcripts into atomic self-contained facts. "Agent will refund $49.99 for order #98765 in 3-5 days" — each chunk is independently meaningful. Eliminates chunk ambiguity. Subsumes semantic chunking, metadata prefixes, and contextual retrieval. | Cold path |
| **Hybrid Dense+Sparse** | RRF + SPLADE | 2025 | Dense vectors (BGE-large) catch semantics ("billing problem" ≈ "payment issue"). SPLADE sparse vectors catch keywords ("order #12345", "refund policy"). Reciprocal Rank Fusion merges rankings. Beat either alone. | Warm path |
| **SPLADE Sparse Embeddings** | Formal et al. | 2021 | Learned sparse representations — term importance from neural model, not just TF-IDF. Native in fastembed + pgvector `sparsevec`. | Cold (embed) + warm (query) |
| **Query-Adaptive Weighting** | — | — | Queries containing order numbers, ticket IDs, or long digit sequences get higher sparse weight (0.6 vs 1.0). Exact terms dominate when identifiers are present. Zero latency cost. | Warm path |
| **SPLADE Tiebreaking** | — | — | Within RRF score range, raw SPLADE L2 distance breaks ties between rows with similar fused scores. Zero latency — distance is already computed for ranking. | Warm path |
| **Half-Precision Vectors** | pgvector 0.7+ | 2023 | Dense embeddings at 2 bytes/float instead of 4. 50% storage reduction, faster I/O. `<=>` operator is type-transparent. | Cold (embed) |
| **HNSW with Iterative Scan** | pgvector 0.8.0 | 2025 | Default HNSW can miss results when WHERE clauses filter by `entity_type`/`entity_id`. `relaxed_order` + `max_scan_tuples` guarantees full recall for filtered queries. | Warm path |
| **HNSW ef_search Tuning** | pgvector 0.8+ | 2025 | `ef_search = 100` probes 2.5x more graph nodes than default (40). +2-5% recall at ~1ms cost. | Warm path |
| **Three-Tier Path Model** | — | — | Hot (<50ms) pure DB, warm (<60ms) embedding+search, cold (5-15s) LLM batch. | Architecture |

### Evaluated but not adopted

| Technique | Why we skipped it |
|-----------|-------------------|
| **Contextual Retrieval** (Anthropic, 2024) | 1 LLM call per chunk to generate context prefix. Subsumed by proposition-based chunking — self-contained facts don't need external context. |
| **Semantic Chunking** (LangChain, 2024) | Splits at embedding divergence. Subsumed by proposition-based chunking — LLM decomposition IS boundary detection. |
| **Late Chunking / ColBERT** (Jina, 2024) | Token-level MaxSim scoring. Requires Qdrant for multi-vector storage + different model (ColBERT, not BGE). Planned for GPU upgrade path. |
| **Metadata Prefix** | Prepending entity context per chunk. Subsumed by proposition-based chunking — LLM includes entity references in each atomic fact. |
| **Binary Quantization** (pgvector 0.7+) | 1-bit vector pre-filter for 100x speedup. Premature — HNSW is not yet the bottleneck. |

### Model & Infrastructure Choices

| Choice | Selected | Why over alternatives |
|--------|----------|----------------------|
| **Dense embedding** | BAAI/bge-large-en-v1.5 (1024d) | MTEB 65.1, Apache 2.0. 3x faster ONNX vs PyTorch. 2.4% behind Arctic-v2 SOTA but 7x lighter. |
| **Sparse embedding** | prithivida/Splade_PP_en_v1 (30522d) | Learned sparse — better than BM25 for keyword AND semantic term matching. Same fastembed ONNX pipeline. |
| **Inference runtime** | fastembed (ONNX) | No PyTorch, no CUDA toolkit. ~500 MB Docker overhead vs 3.5 GB. 1-2s cold start vs 5-10s. |
| **Vector storage** | pgvector 0.8.2 in PostgreSQL | Colocated with entity/task/memory tables. No extra service to deploy/monitor/backup. HNSW handles millions of vectors. |
| **Search fusion** | Reciprocal Rank Fusion (RRF) | Parameter-free, combines arbitrary ranked lists. Works natively in PostgreSQL CTEs. Outperforms linear score combination. |
| **Chunking** | Proposition-based (LLM) | Self-contained atomic facts. Eliminates chunk ambiguity. 1 LLM call per transcript (not per chunk). |

### GPU Upgrade Path

When a GPU node is provisioned:
1. Export `Snowflake/snowflake-arctic-embed-l-v2.0` to ONNX
2. Swap `RETAIN_EMBEDDING_QUERY_MODEL` in settings
3. ONNX Runtime auto-detects `CUDAExecutionProvider`
4. **+2.4% MTEB recall, 3-5ms GPU inference** — zero code changes
5. Revisit: Late Chunking (ColBERT) and Contextual Retrieval (faster per-chunk LLM calls)

---

## API Reference

### Hot path

```
POST /v1/context
{"entity_type": "customer", "entity_id": "cust_123"}

Returns: {profile_blob, recent_memories, open_tasks, memory_count, first_seen, last_seen}
```

```
POST /v1/memories
{"entity_type": "customer", "entity_id": "cust_123", "memory_type": "preference", "value": {...}}

Returns: {"id": "..."}
```

```
POST /v1/tasks
{"entity_type": "customer", "entity_id": "cust_123", "task_type": "followup", "description": "..."}

Returns: TaskRecord
```

```
PATCH /v1/tasks/{id}
{}

Returns: TaskRecord | 404
```

```
GET /v1/tasks?entity_type=customer&entity_id=cust_123&status=open

Returns: [TaskRecord, ...]
```

### Warm path

```
POST /v1/search?query=double+charge+refund&entity_type=customer&entity_id=cust_123&limit=5

Returns: {chunks: [{id, transcript_id, entity_type, entity_id, chunk_index, chunk_text, score}]}
```

### Cold path

```
POST /v1/process
{
  "entities": [{"entity_type": "customer", "entity_id": "cust_123"}],
  "transcript": [{"role": "agent", "content": "..."}, {"role": "customer", "content": "..."}],
  "instructions": "Focus on billing issues"
}

Returns: {"event_id": "..."}
```

```
GET /v1/events/{event_id}

Returns: {event_id, event_type, status, payload, result, created_at, completed_at} | 404
```

---

## Database Schema

| Table | Purpose |
|-------|---------|
| `entities` | Per-entity profile blob + aggregate counters (memory_count, task_count_open) |
| `memories` | Typed dimensional facts — append-only. Immutable once written. |
| `tasks` | Tracked task lifecycle: `open` → `in_progress` → `resolved` / `cancelled` |
| `transcripts` | Raw conversation transcripts — source of truth for extraction |
| `transcript_chunks` | Proposition-based chunks with `halfvec(1024)` dense + `sparsevec(30522)` sparse vectors. HNSW indexes on both. |
| `events` | Async processing status: `pending` → `processing` → `completed` / `failed` |

---

## Configuration

All via environment variables with `RETAIN_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `RETAIN_DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection |
| `RETAIN_LLM_API_KEY` | — | LLM API key (required for extraction) |
| `RETAIN_LLM_BASE_URL` | — | LLM base URL (OpenAI-compatible) |
| `RETAIN_LLM_MODEL` | `gpt-4o-mini` | LLM model for extraction + decomposition |
| `RETAIN_EMBEDDING_QUERY_MODEL` | `BAAI/bge-large-en-v1.5` | Dense embedding model |
| `RETAIN_EMBEDDING_SPARSE_MODEL` | `prithivida/Splade_PP_en_v1` | Sparse embedding model |
| `RETAIN_EMBEDDING_BATCH_SIZE` | `32` | Batch size for embedding inference |

---

## Development

```bash
uv sync --group dev --group embeddings

docker compose up -d postgres
uv run alembic upgrade head
uv run uvicorn retain.main:app --reload

uv run pytest tests/ -m unit
```

---

## License

Apache 2.0
