"""Latency benchmark — measures p50/p95/p99 for all endpoints.

Usage:
    uv run python benchmarks/latency.py

Requires a running server (docker compose up) with BGE + SPLADE loaded.
Each endpoint uses a dedicated entity to isolate measurements.
Cache: a cleanup phase after each run deletes bench_* data so
consecutive runs do not accumulate garbage.

Design:
- Uses `engine.begin()` autocommit — no manual transaction management
- Each endpoint gets a fresh entity to prevent data accumulation skewing latency
- Teardown phase clears all bench_* rows across all tables
"""

import asyncio
import os
import statistics
import time
import uuid
from collections.abc import Sequence

import httpx
from sqlalchemy import text

from retain.storage import create_engine

BASE = os.environ.get("RETAIN_BENCH_URL", "http://localhost:8000")
RUN_ID = uuid.uuid4().hex[:8]
WARMUP = 30
MEASURED = 200

E_CTX = ("customer", f"bench_ctx_{RUN_ID}")
E_TASK = ("customer", f"bench_task_{RUN_ID}")
E_SEARCH = ("customer", f"bench_search_{RUN_ID}")
E_PROCESS = ("customer", f"bench_proc_{RUN_ID}")


async def measure_endpoint(
    client: httpx.AsyncClient,
    name: str,
    fn,
) -> dict[str, float]:
    timings: list[float] = []

    for _ in range(WARMUP):
        await fn(client)
    for _ in range(MEASURED):
        start = time.perf_counter()
        await fn(client)
        timings.append((time.perf_counter() - start) * 1000)

    sorted_t = sorted(timings)
    return {
        "endpoint": name,
        "p50": round(statistics.median(sorted_t), 1),
        "p95": round(_percentile(sorted_t, 0.95), 1),
        "p99": round(_percentile(sorted_t, 0.99), 1),
        "mean": round(statistics.mean(timings), 1),
    }


def _percentile(values: Sequence[float], p: float) -> float:
    idx = max(0, int(len(values) * p) - 1)
    return sorted(values)[idx]


async def _setup(client: httpx.AsyncClient) -> None:
    """Seed minimal isolated data for each endpoint."""

    et, eid = E_CTX
    r = await client.post(
        "/v1/memories",
        json={
            "entity_type": et,
            "entity_id": eid,
            "memory_type": "bench",
            "value": {"key": "val"},
        },
    )
    r.raise_for_status()

    et, eid = E_TASK
    r = await client.post(
        "/v1/tasks",
        json={
            "entity_type": et,
            "entity_id": eid,
            "task_type": "bench",
            "description": "seed task",
        },
    )
    r.raise_for_status()


async def _teardown() -> None:
    engine = create_engine()
    try:
        # Only tables with an entity_id column
        tables = ["tasks", "memories", "transcript_chunks", "entities"]
        async with engine.begin() as conn:
            for table in tables:
                await conn.execute(
                    text(f"DELETE FROM {table} WHERE entity_id LIKE 'bench_%'")
                )
    finally:
        await engine.dispose()


async def main() -> None:
    print(f"Run ID: {RUN_ID}")
    print(f"Warmup: {WARMUP} | Measured: {MEASURED}")
    print()

    async with httpx.AsyncClient(base_url=BASE, timeout=30) as client:
        await _setup(client)

        # ── context ──────────────────────────────────────

        et, eid = E_CTX

        async def ctx(client: httpx.AsyncClient) -> None:
            r = await client.post(
                "/v1/context",
                json={"entity_type": et, "entity_id": eid},
            )
            r.raise_for_status()

        print("Benchmarking POST /v1/context ...")
        ctx_result = await measure_endpoint(client, "POST /v1/context", ctx)

        # ── remember ─────────────────────────────────────

        async def remember(client: httpx.AsyncClient) -> None:
            r = await client.post(
                "/v1/memories",
                json={
                    "entity_type": et,
                    "entity_id": eid,
                    "memory_type": "bench",
                    "value": {"ts": time.time()},
                },
            )
            r.raise_for_status()

        print("Benchmarking POST /v1/memories ...")
        mem_result = await measure_endpoint(client, "POST /v1/memories", remember)

        # ── create task ───────────────────────────────────

        et_t, eid_t = E_TASK

        async def create_task(client: httpx.AsyncClient) -> None:
            r = await client.post(
                "/v1/tasks",
                json={
                    "entity_type": et_t,
                    "entity_id": eid_t,
                    "task_type": "bench",
                    "description": str(uuid.uuid4()),
                },
            )
            r.raise_for_status()

        print("Benchmarking POST /v1/tasks (create) ...")
        task_result = await measure_endpoint(client, "POST /v1/tasks", create_task)

        # ── list tasks ────────────────────────────────────

        async def list_tasks(client: httpx.AsyncClient) -> None:
            r = await client.get(
                "/v1/tasks",
                params={"entity_type": et_t, "entity_id": eid_t},
            )
            r.raise_for_status()

        print("Benchmarking GET /v1/tasks ...")
        list_result = await measure_endpoint(client, "GET /v1/tasks", list_tasks)

        # ── search ────────────────────────────────────────

        et_s, eid_s = E_SEARCH

        async def search(client: httpx.AsyncClient) -> None:
            r = await client.post(
                "/v1/search",
                params={
                    "query": "billing question",
                    "entity_type": et_s,
                    "entity_id": eid_s,
                    "limit": 5,
                },
            )
            r.raise_for_status()

        print("Benchmarking POST /v1/search ...")
        search_result = await measure_endpoint(client, "POST /v1/search", search)

        # ── complete task ─────────────────────────────────

        create_resp = await client.post(
            "/v1/tasks",
            json={
                "entity_type": et_t,
                "entity_id": eid_t,
                "task_type": "bench",
                "description": "complete me",
            },
        )
        create_resp.raise_for_status()
        task_id = create_resp.json()["id"]

        async def complete_task(client: httpx.AsyncClient) -> None:
            r = await client.patch(f"/v1/tasks/{task_id}")
            r.raise_for_status()

        print("Benchmarking PATCH /v1/tasks/{id} ...")
        complete_result = await measure_endpoint(client, "PATCH /v1/tasks/{id}", complete_task)

        # ── event status ──────────────────────────────────

        async def event_status(client: httpx.AsyncClient) -> None:
            await client.get(f"/v1/events/{uuid.uuid4().hex}")

        print("Benchmarking GET /v1/events/{id} ...")
        event_result = await measure_endpoint(client, "GET /v1/events/{id}", event_status)

        # ── process ───────────────────────────────────────

        et_p, eid_p = E_PROCESS

        async def process_req(client: httpx.AsyncClient) -> None:
            await client.post(
                "/v1/process",
                json={
                    "entities": [{"entity_type": et_p, "entity_id": eid_p}],
                    "transcript": [{"role": "agent", "content": "test"}],
                },
                timeout=30,
            )

        print("Benchmarking POST /v1/process ...")
        process_result = await measure_endpoint(client, "POST /v1/process", process_req)

    # ── human output ─────────────────────────────────────
    print()
    print("| Endpoint | p50 | p95 | p99 | Mean |")
    print("|----------|-----|-----|-----|------|")
    for r in [ctx_result, mem_result, task_result, list_result, search_result,
              complete_result, event_result, process_result]:
        print(
            f"| {r['endpoint']:32s} | "
            f"{r['p50']:4.1f}ms | "
            f"{r['p95']:4.1f}ms | "
            f"{r['p99']:4.1f}ms | "
            f"{r['mean']:4.1f}ms |"
        )

    await _teardown()
    print("Cleaned up bench_* data.")


if __name__ == "__main__":
    asyncio.run(main())
