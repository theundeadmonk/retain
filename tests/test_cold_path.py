"""Tests for async process() + event_status() pipeline."""

import asyncio
import json

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from retain.cold_path import event_status, process
from retain.errors import RetainLLMError
from retain.hot_path import context
from retain.llm import MockProvider
from retain.types import EntityRef, ProcessRequest

_EXTRACT_FACTS = json.dumps([
    {
        "entity_type": "guest",
        "entity_id": "abc123",
        "memory_type": "preference",
        "value": {"room": "412", "pillows": "extra"},
        "source": "extraction",
    },
])

_PROFILE = "Guest prefers room 412 with extra pillows."

_ENTITY = EntityRef(entity_type="guest", entity_id="abc123")
_TRANSCRIPT = [{"role": "user", "content": "I like room 412 with extra pillows"}]


async def _poll(engine: AsyncEngine, event_id: str, timeout: float = 3.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        s = await event_status(engine, event_id)
        if s["status"] in ("completed", "failed"):
            return s
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError(f"Event {event_id} did not complete in {timeout}s")
        await asyncio.sleep(0.02)


@pytest.mark.integration
class TestProcessPipeline:
    """Full end-to-end: process → poll → context returns profile."""

    async def test_full_pipeline(self, engine: AsyncEngine) -> None:
        provider = MockProvider(response=[_EXTRACT_FACTS, _PROFILE])
        event_id = await process(
            engine, provider,
            ProcessRequest(transcript=_TRANSCRIPT, entities=[_ENTITY]),
        )
        s = await _poll(engine, event_id)
        assert s["status"] == "completed"

        ctx = await context(engine, "guest", "abc123")
        assert ctx.profile_blob == _PROFILE
        assert ctx.memory_count == 1
        assert ctx.recent_memories[0].value == {"room": "412", "pillows": "extra"}
        assert ctx.recent_memories[0].source == "extraction"

    async def test_multiple_entities(self, engine: AsyncEngine) -> None:
        facts = json.dumps([
            {"entity_type": "guest", "entity_id": "a", "memory_type": "pref",
             "value": {"key": "a_val"}, "source": "extraction"},
            {"entity_type": "guest", "entity_id": "b", "memory_type": "pref",
             "value": {"key": "b_val"}, "source": "extraction"},
        ])
        provider = MockProvider(response=[facts, "Profile for both."])
        event_id = await process(
            engine, provider,
            ProcessRequest(
                transcript=[{"role": "user", "content": "hi"}],
                entities=[
                    EntityRef(entity_type="guest", entity_id="a"),
                    EntityRef(entity_type="guest", entity_id="b"),
                ],
            ),
        )
        s = await _poll(engine, event_id)
        assert s["status"] == "completed"
        assert s["result"]["entities_processed"] == 2

    async def test_dedup_across_extractions(self, engine: AsyncEngine) -> None:
        provider = MockProvider(response=[
            _EXTRACT_FACTS, _PROFILE,
            _EXTRACT_FACTS, _PROFILE,
        ])
        eid1 = await process(
            engine, provider,
            ProcessRequest(transcript=_TRANSCRIPT, entities=[_ENTITY]),
        )
        await _poll(engine, eid1)
        eid2 = await process(
            engine, provider,
            ProcessRequest(transcript=_TRANSCRIPT, entities=[_ENTITY]),
        )
        await _poll(engine, eid2)
        ctx = await context(engine, "guest", "abc123")
        assert ctx.memory_count == 1

    async def test_context_returns_profile(self, engine: AsyncEngine) -> None:
        provider = MockProvider(response=[_EXTRACT_FACTS, "Profile after first call"])
        eid = await process(
            engine, provider,
            ProcessRequest(transcript=_TRANSCRIPT, entities=[_ENTITY]),
        )
        await _poll(engine, eid)
        ctx = await context(engine, "guest", "abc123")
        assert ctx.profile_blob == "Profile after first call"

    async def test_empty_transcript_completes(self, engine: AsyncEngine) -> None:
        provider = MockProvider(response=["[]", ""])
        event_id = await process(
            engine, provider,
            ProcessRequest(transcript=[], entities=[_ENTITY]),
        )
        s = await _poll(engine, event_id)
        assert s["status"] == "completed"
        assert s["result"]["facts_extracted"] == 0

    async def test_empty_entities_completes(self, engine: AsyncEngine) -> None:
        provider = MockProvider(response=["[]", ""])
        event_id = await process(
            engine, provider,
            ProcessRequest(transcript=_TRANSCRIPT, entities=[]),
        )
        s = await _poll(engine, event_id)
        assert s["status"] == "completed"

    async def test_instructions_passed_to_extraction(self, engine: AsyncEngine) -> None:
        provider = MockProvider(response=[_EXTRACT_FACTS, _PROFILE])
        event_id = await process(
            engine, provider,
            ProcessRequest(
                transcript=_TRANSCRIPT,
                entities=[_ENTITY],
                instructions="Focus on billing issues only",
            ),
        )
        await _poll(engine, event_id)
        call = provider.calls[0]
        system = call["messages"][0]["content"]
        assert "Focus on billing issues only" in system

    async def test_event_status_nonexistent(self, engine: AsyncEngine) -> None:
        s = await event_status(engine, "nonexistent")
        assert s["status"] == "not_found"


@pytest.mark.integration
class TestProcessFailure:
    """Background task error handling."""

    async def test_llm_failure_marks_event_failed(self, engine: AsyncEngine) -> None:
        class FailingProvider(MockProvider):
            async def complete(self, messages, **kwargs):
                raise RetainLLMError("extraction crashed")

        provider = FailingProvider()
        event_id = await process(
            engine, provider,
            ProcessRequest(transcript=_TRANSCRIPT, entities=[_ENTITY]),
        )
        s = await _poll(engine, event_id)
        assert s["status"] == "failed"
        assert "extraction crashed" in s["result"]["error"]

    async def test_llm_error_propagates(self, engine: AsyncEngine) -> None:
        class FailingProvider(MockProvider):
            async def complete(self, messages, **kwargs):
                raise RetainLLMError("API down")

        provider = FailingProvider()
        event_id = await process(
            engine, provider,
            ProcessRequest(transcript=_TRANSCRIPT, entities=[_ENTITY]),
        )
        s = await _poll(engine, event_id)
        assert s["status"] == "failed"

    async def test_extraction_timeout(self, engine: AsyncEngine) -> None:
        class SlowProvider(MockProvider):
            async def complete(self, messages, **kwargs):
                await asyncio.sleep(5.0)
                return await super().complete(messages, **kwargs)

        provider = SlowProvider(response=[_EXTRACT_FACTS, _PROFILE])
        event_id = await process(
            engine, provider,
            ProcessRequest(transcript=_TRANSCRIPT, entities=[_ENTITY]),
            extraction_timeout=0.1,
        )
        s = await _poll(engine, event_id)
        assert s["status"] == "failed"
        assert "timed out" in s["result"]["error"]
        assert s["payload"]["extraction_timeout"] == 0.1
