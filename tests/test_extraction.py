"""Tests for LLM-based fact extraction."""

import json

import pytest

from retain.errors import RetainLLMError
from retain.extraction import extract
from retain.extraction.prompts import format_extraction_prompt
from retain.llm import MockProvider
from retain.types import EntityRef

_ENTITY = EntityRef(entity_type="guest", entity_id="abc123")


@pytest.mark.unit
class TestExtractBasic:
    """Core extraction behavior."""

    async def test_returns_facts_from_valid_json(self) -> None:
        facts = [
            {
                "entity_type": "guest",
                "entity_id": "abc123",
                "memory_type": "preference",
                "value": {"room": "412", "pillows": "extra"},
                "source": "extraction",
            }
        ]
        provider = MockProvider(response=json.dumps(facts))
        result = await extract(provider, [{"role": "user", "content": "hi"}], [_ENTITY])
        assert len(result) == 1
        assert result[0].entity_type == "guest"
        assert result[0].entity_id == "abc123"
        assert result[0].memory_type == "preference"
        assert result[0].value == {"room": "412", "pillows": "extra"}
        assert result[0].source == "extraction"
        assert result[0].id is None

    async def test_returns_multiple_facts(self) -> None:
        facts = [
            {
                "entity_type": "guest",
                "entity_id": "abc123",
                "memory_type": "personal_info",
                "value": {"name": "Alice"},
                "source": "extraction",
            },
            {
                "entity_type": "guest",
                "entity_id": "abc123",
                "memory_type": "issue",
                "value": {"category": "billing"},
                "source": "extraction",
            },
        ]
        provider = MockProvider(response=json.dumps(facts))
        result = await extract(provider, [{"role": "user", "content": "hi"}], [_ENTITY])
        assert len(result) == 2
        assert result[0].memory_type == "personal_info"
        assert result[1].memory_type == "issue"

    async def test_empty_transcript_returns_empty(self) -> None:
        provider = MockProvider(response="[]")
        result = await extract(provider, [], [_ENTITY])
        assert result == []

    async def test_empty_entities_returns_empty(self) -> None:
        provider = MockProvider(response="[]")
        result = await extract(
            provider, [{"role": "user", "content": "hi"}], []
        )
        assert result == []

    async def test_passes_transcript_to_llm(self) -> None:
        provider = MockProvider(response="[]")
        transcript = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        await extract(provider, transcript, [_ENTITY])
        assert len(provider.calls) == 1
        sent = provider.calls[0]
        assert sent["temperature"] == 0.0
        assert sent["max_tokens"] == 2048

    async def test_llm_failure_raises(self) -> None:
        async def failing_complete(messages, **kwargs):
            raise RetainLLMError("API unavailable")

        provider = MockProvider()
        provider.complete = failing_complete  # type: ignore[assignment]
        with pytest.raises(RetainLLMError):
            await extract(provider, [{"role": "user", "content": "hi"}], [_ENTITY])


@pytest.mark.unit
class TestExtractWithInstructions:
    """Instructions passed through to the prompt."""

    async def test_instructions_appear_in_prompt(self) -> None:
        provider = MockProvider(response="[]")
        instructions = "Focus on billing issues"
        await extract(
            provider,
            [{"role": "user", "content": "hi"}],
            [_ENTITY],
            instructions=instructions,
        )
        sent = provider.calls[0]["messages"]
        system_msg = sent[0]["content"]
        assert "Focus on billing issues" in system_msg

    async def test_empty_instructions_omits_section(self) -> None:
        provider = MockProvider(response="[]")
        await extract(
            provider,
            [{"role": "user", "content": "hi"}],
            [_ENTITY],
        )
        sent = provider.calls[0]["messages"]
        system_msg = sent[0]["content"]
        assert "Additional extraction guidance" not in system_msg


@pytest.mark.unit
class TestExtractEdgeCases:
    """Parse resilience against various LLM outputs."""

    async def test_empty_response_string(self) -> None:
        provider = MockProvider(response="")
        result = await extract(provider, [{"role": "user", "content": "hi"}], [_ENTITY])
        assert result == []

    async def test_malformed_json_returns_empty(self) -> None:
        provider = MockProvider(response="this is not json")
        result = await extract(provider, [{"role": "user", "content": "hi"}], [_ENTITY])
        assert result == []

    async def test_code_fence_json(self) -> None:
        facts = [
            {
                "entity_type": "guest",
                "entity_id": "abc123",
                "memory_type": "preference",
                "value": {"room": "412"},
                "source": "extraction",
            }
        ]
        provider = MockProvider(
            response=f"```json\n{json.dumps(facts)}\n```"
        )
        result = await extract(provider, [{"role": "user", "content": "hi"}], [_ENTITY])
        assert len(result) == 1

    async def test_response_not_a_list(self) -> None:
        provider = MockProvider(response=json.dumps({"not": "a list"}))
        result = await extract(provider, [{"role": "user", "content": "hi"}], [_ENTITY])
        assert result == []

    async def test_skips_items_with_missing_fields(self) -> None:
        facts = [
            {"entity_type": "guest", "entity_id": "abc123"},
            {
                "entity_type": "guest",
                "entity_id": "abc123",
                "memory_type": "valid",
                "value": {"key": "val"},
                "source": "extraction",
            },
        ]
        provider = MockProvider(response=json.dumps(facts))
        result = await extract(provider, [{"role": "user", "content": "hi"}], [_ENTITY])
        assert len(result) == 1
        assert result[0].memory_type == "valid"

    async def test_skips_entity_not_in_expected_list(self) -> None:
        facts = [
            {
                "entity_type": "other",
                "entity_id": "unknown",
                "memory_type": "info",
                "value": {"key": "val"},
                "source": "extraction",
            }
        ]
        provider = MockProvider(response=json.dumps(facts))
        result = await extract(provider, [{"role": "user", "content": "hi"}], [_ENTITY])
        assert result == []

    async def test_skips_non_dict_value(self) -> None:
        facts = [
            {
                "entity_type": "guest",
                "entity_id": "abc123",
                "memory_type": "info",
                "value": "string value",
                "source": "extraction",
            }
        ]
        provider = MockProvider(response=json.dumps(facts))
        result = await extract(provider, [{"role": "user", "content": "hi"}], [_ENTITY])
        assert result == []

    async def test_skips_non_dict_item(self) -> None:
        provider = MockProvider(response=json.dumps(["not a dict"]))
        result = await extract(provider, [{"role": "user", "content": "hi"}], [_ENTITY])
        assert result == []

    async def test_code_fence_without_lang(self) -> None:
        facts = [
            {
                "entity_type": "guest",
                "entity_id": "abc123",
                "memory_type": "preference",
                "value": {"key": "val"},
                "source": "extraction",
            }
        ]
        provider = MockProvider(response=f"```\n{json.dumps(facts)}\n```")
        result = await extract(provider, [{"role": "user", "content": "hi"}], [_ENTITY])
        assert len(result) == 1


@pytest.mark.unit
class TestFormatPrompt:
    """Prompt formatting helper."""

    def test_includes_entities(self) -> None:
        entities = [
            EntityRef(entity_type="guest", entity_id="alice"),
            EntityRef(entity_type="guest", entity_id="bob"),
        ]
        prompt = format_extraction_prompt(entities)
        assert "guest: alice" in prompt
        assert "guest: bob" in prompt

    def test_includes_instructions(self) -> None:
        prompt = format_extraction_prompt([_ENTITY], "Look for preferences only")
        assert "Look for preferences only" in prompt

    def test_empty_instructions_no_section(self) -> None:
        prompt = format_extraction_prompt([_ENTITY])
        assert "Additional extraction guidance" not in prompt
