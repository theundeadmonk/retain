"""Tests for async process() + event_status() pipeline."""

import asyncio
import json

import pytest

from retain import Memory
from retain.errors import RetainConfigError, RetainLLMError
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


async def _poll(ms: Memory, event_id: str, timeout: float = 3.0) -> dict:
    """Poll event_status until completed or failed."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        s = await ms.event_status(event_id)
        if s["status"] in ("completed", "failed"):
            return s
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError(f"Event {event_id} did not complete in {timeout}s")
        await asyncio.sleep(0.02)


@pytest.mark.integration
class TestProcessPipeline:
    """Full end-to-end: process → poll → context returns profile."""

    async def test_full_pipeline(self, temp_db: str) -> None:
        provider = MockProvider(response=[_EXTRACT_FACTS, _PROFILE])
        ms = Memory(storage=f"sqlite+aiosqlite:///{temp_db}", llm=provider)
        event_id = await ms.process(ProcessRequest(
            transcript=_TRANSCRIPT, entities=[_ENTITY],
        ))
        s = await _poll(ms, event_id)
        assert s["status"] == "completed"
        assert s["event_type"] == "extraction"

        ctx = await ms.context("guest", "abc123")
        assert ctx.profile_blob == _PROFILE
        assert ctx.memory_count == 1
        assert ctx.recent_memories[0].value == {"room": "412", "pillows": "extra"}
        assert ctx.recent_memories[0].source == "extraction"

    async def test_multiple_entities(self, temp_db: str) -> None:
        facts = json.dumps([
            {"entity_type": "guest", "entity_id": "a", "memory_type": "pref",
             "value": {"key": "a_val"}, "source": "extraction"},
            {"entity_type": "guest", "entity_id": "b", "memory_type": "pref",
             "value": {"key": "b_val"}, "source": "extraction"},
        ])
        provider = MockProvider(response=[facts, "Profile for both."])
        ms = Memory(storage=f"sqlite+aiosqlite:///{temp_db}", llm=provider)
        event_id = await ms.process(ProcessRequest(
            transcript=[{"role": "user", "content": "hi"}],
            entities=[
                EntityRef(entity_type="guest", entity_id="a"),
                EntityRef(entity_type="guest", entity_id="b"),
            ],
        ))
        s = await _poll(ms, event_id)
        assert s["status"] == "completed"
        assert s["result"]["entities_processed"] == 2

        ctx_a = await ms.context("guest", "a")
        ctx_b = await ms.context("guest", "b")
        assert ctx_a.memory_count == 1
        assert ctx_b.memory_count == 1

    async def test_dedup_across_extractions(self, temp_db: str) -> None:
        provider = MockProvider(response=[
            _EXTRACT_FACTS, _PROFILE,
            _EXTRACT_FACTS, _PROFILE,
        ])
        ms = Memory(storage=f"sqlite+aiosqlite:///{temp_db}", llm=provider)
        eid1 = await ms.process(ProcessRequest(
            transcript=_TRANSCRIPT, entities=[_ENTITY],
        ))
        await _poll(ms, eid1)

        eid2 = await ms.process(ProcessRequest(
            transcript=_TRANSCRIPT, entities=[_ENTITY],
        ))
        await _poll(ms, eid2)

        ctx = await ms.context("guest", "abc123")
        assert ctx.memory_count == 1

    async def test_context_returns_profile_when_stored(self, temp_db: str) -> None:
        provider = MockProvider(response=[_EXTRACT_FACTS, "Profile after first call"])
        ms = Memory(storage=f"sqlite+aiosqlite:///{temp_db}", llm=provider)
        eid = await ms.process(ProcessRequest(
            transcript=_TRANSCRIPT, entities=[_ENTITY],
        ))
        await _poll(ms, eid)
        ctx = await ms.context("guest", "abc123")
        assert ctx.profile_blob == "Profile after first call"

    async def test_empty_transcript_completes_cleanly(self, temp_db: str) -> None:
        provider = MockProvider(response=["[]", ""])
        ms = Memory(storage=f"sqlite+aiosqlite:///{temp_db}", llm=provider)
        event_id = await ms.process(ProcessRequest(
            transcript=[], entities=[_ENTITY],
        ))
        s = await _poll(ms, event_id)
        assert s["status"] == "completed"
        assert s["result"]["facts_extracted"] == 0

    async def test_empty_entities_completes_cleanly(self, temp_db: str) -> None:
        provider = MockProvider(response=["[]", ""])
        ms = Memory(storage=f"sqlite+aiosqlite:///{temp_db}", llm=provider)
        event_id = await ms.process(ProcessRequest(
            transcript=_TRANSCRIPT, entities=[],
        ))
        s = await _poll(ms, event_id)
        assert s["status"] == "completed"

    async def test_instructions_passed_to_extraction(self, temp_db: str) -> None:
        provider = MockProvider(response=[_EXTRACT_FACTS, _PROFILE])
        ms = Memory(storage=f"sqlite+aiosqlite:///{temp_db}", llm=provider)
        event_id = await ms.process(ProcessRequest(
            transcript=_TRANSCRIPT,
            entities=[_ENTITY],
            instructions="Focus on billing issues only",
        ))
        await _poll(ms, event_id)
        call = provider.calls[0]
        system = call["messages"][0]["content"]
        assert "Focus on billing issues only" in system

    async def test_event_status_for_nonexistent_event(self, temp_db: str) -> None:
        ms = Memory(storage=f"sqlite+aiosqlite:///{temp_db}")
        s = await ms.event_status("nonexistent")
        assert s["status"] == "not_found"


@pytest.mark.unit
class TestProcessGuards:
    """process() input validation."""

    async def test_raises_without_llm_provider(self) -> None:
        ms = Memory()
        with pytest.raises(RetainConfigError, match="LLM provider"):
            await ms.process(ProcessRequest(
                transcript=_TRANSCRIPT, entities=[_ENTITY],
            ))


@pytest.mark.integration
class TestProcessFailure:
    """Background task error handling."""

    async def test_background_failure_marks_event_failed(self, temp_db: str) -> None:
        class FailingProviderOnce(MockProvider):
            async def complete(self, messages, **kwargs):
                raise RetainLLMError("extraction crashed")

        provider = FailingProviderOnce()
        ms = Memory(storage=f"sqlite+aiosqlite:///{temp_db}", llm=provider)
        event_id = await ms.process(ProcessRequest(
            transcript=_TRANSCRIPT, entities=[_ENTITY],
        ))
        s = await _poll(ms, event_id)
        assert s["status"] == "failed"
        assert "extraction crashed" in s["result"]["error"]

    async def test_llm_error_propagates_to_event(self, temp_db: str) -> None:
        class FailingProvider(MockProvider):
            async def complete(self, messages, **kwargs):
                raise RetainLLMError("API down")

        provider = FailingProvider()
        ms = Memory(storage=f"sqlite+aiosqlite:///{temp_db}", llm=provider)
        event_id = await ms.process(ProcessRequest(
            transcript=_TRANSCRIPT, entities=[_ENTITY],
        ))
        s = await _poll(ms, event_id)
        assert s["status"] == "failed"

    async def test_extraction_timeout(self, temp_db: str) -> None:
        class SlowProvider(MockProvider):
            async def complete(self, messages, **kwargs):
                await asyncio.sleep(5.0)
                return await super().complete(messages, **kwargs)

        provider = SlowProvider(response=[_EXTRACT_FACTS, _PROFILE])
        ms = Memory(storage=f"sqlite+aiosqlite:///{temp_db}", llm=provider)
        event_id = await ms.process(
            ProcessRequest(transcript=_TRANSCRIPT, entities=[_ENTITY]),
            extraction_timeout=0.1,
        )
        s = await _poll(ms, event_id)
        assert s["status"] == "failed"
        assert "timed out" in s["result"]["error"]
        assert s["payload"]["extraction_timeout"] == 0.1
