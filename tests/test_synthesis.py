"""Tests for deduplication and profile synthesis."""

import pytest

from retain.errors import RetainLLMError
from retain.extraction import deduplicate, synthesize_profile
from retain.llm import MockProvider
from retain.types import MemoryRecord

_FACT = MemoryRecord(
    entity_type="guest",
    entity_id="abc123",
    memory_type="preference",
    value={"room": "412", "pillows": "extra"},
    source="extraction",
)

_FACT2 = MemoryRecord(
    entity_type="guest",
    entity_id="abc123",
    memory_type="personal_info",
    value={"name": "Alice"},
    source="extraction",
)


@pytest.mark.unit
class TestDeduplicate:
    """Rule-based dedup on exact match of type + value."""

    async def test_identical_fact_is_removed(self) -> None:
        result = deduplicate([_FACT], [_FACT])
        assert result == []

    async def test_different_facts_are_kept(self) -> None:
        result = deduplicate([_FACT2], [_FACT])
        assert result == [_FACT2]

    async def test_partial_overlap_keeps_only_novel(self) -> None:
        novel = MemoryRecord(
            entity_type="guest",
            entity_id="abc123",
            memory_type="preference",
            value={"room": "413"},
            source="extraction",
        )
        result = deduplicate([_FACT, novel], [_FACT])
        assert result == [novel]

    async def test_empty_new_facts(self) -> None:
        result = deduplicate([], [_FACT])
        assert result == []

    async def test_empty_existing_facts(self) -> None:
        result = deduplicate([_FACT], [])
        assert result == [_FACT]

    async def test_skips_new_facts_with_missing_fields(self) -> None:
        bad = MemoryRecord(
            entity_type="",
            entity_id="",
            memory_type="",
            value={},
            source="extraction",
        )
        result = deduplicate([bad], [])
        assert result == []

    async def test_same_values_different_key_order(self) -> None:
        a = MemoryRecord(
            entity_type="guest", entity_id="abc", memory_type="pref",
            value={"a": "1", "b": "2"}, source="extraction",
        )
        b = MemoryRecord(
            entity_type="guest", entity_id="abc", memory_type="pref",
            value={"b": "2", "a": "1"}, source="extraction",
        )
        result = deduplicate([a], [b])
        assert result == []

    async def test_different_entity_same_value_both_kept(self) -> None:
        a = MemoryRecord(
            entity_type="guest", entity_id="a", memory_type="pref",
            value={"key": "val"}, source="extraction",
        )
        b = MemoryRecord(
            entity_type="guest", entity_id="b", memory_type="pref",
            value={"key": "val"}, source="extraction",
        )
        result = deduplicate([a, b], [])
        assert len(result) == 2

    async def test_dedup_within_new_facts(self) -> None:
        result = deduplicate([_FACT, _FACT], [])
        assert result == [_FACT]


@pytest.mark.unit
class TestSynthesizeProfile:
    """Profile synthesis via LLM."""

    async def test_returns_llm_response(self) -> None:
        provider = MockProvider(response="Guest Alice prefers room 412.")
        profile = await synthesize_profile(provider, "guest", "abc123", [_FACT])
        assert profile == "Guest Alice prefers room 412."

    async def test_passes_facts_in_prompt(self) -> None:
        provider = MockProvider(response="")
        await synthesize_profile(provider, "guest", "abc123", [_FACT, _FACT2])
        assert len(provider.calls) == 1
        sent = provider.calls[0]["messages"][0]["content"]
        assert "preference" in sent
        assert "personal_info" in sent
        assert "room" in sent
        assert "Alice" in sent

    async def test_respects_token_budget(self) -> None:
        provider = MockProvider(response="")
        await synthesize_profile(
            provider, "guest", "abc123", [_FACT],
            budget_tokens=50,
        )
        sent = provider.calls[0]["messages"][0]["content"]
        assert "50" in sent
        assert provider.calls[0]["max_tokens"] == 50

    async def test_empty_facts_returns_fallback(self) -> None:
        result = await synthesize_profile(
            MockProvider(), "guest", "abc123", []
        )
        assert "no stored information" in result

    async def test_llm_failure_propagates(self) -> None:
        async def failing(messages, **kwargs):
            raise RetainLLMError("API error")

        provider = MockProvider()
        provider.complete = failing  # type: ignore[assignment]
        with pytest.raises(RetainLLMError):
            await synthesize_profile(provider, "guest", "abc123", [_FACT])

    async def test_uses_some_temperature(self) -> None:
        provider = MockProvider(response="")
        await synthesize_profile(provider, "guest", "abc123", [_FACT])
        assert provider.calls[0]["temperature"] == 0.3

    async def test_strips_whitespace(self) -> None:
        provider = MockProvider(response="  Hello profile.  ")
        result = await synthesize_profile(provider, "guest", "abc123", [_FACT])
        assert result == "Hello profile."
