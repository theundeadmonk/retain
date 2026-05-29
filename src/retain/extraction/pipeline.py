"""LLM-based fact extraction from conversation transcripts."""

import json
import logging
from typing import Any

from retain.errors import RetainLLMError
from retain.extraction.prompts import format_extraction_prompt
from retain.llm.base import LLMProvider
from retain.types import EntityRef, MemoryRecord

__all__ = [
    "extract",
]

logger = logging.getLogger("retain")


async def extract(
    provider: LLMProvider,
    transcript: list[dict[str, Any]],
    entities: list[EntityRef],
    *,
    instructions: str = "",
) -> list[MemoryRecord]:
    """Extract structured facts from a transcript using an LLM.

    Args:
        provider: LLM provider to use for extraction.
        transcript: Conversation messages in OpenAI format
            (``[{"role": "user"|"assistant", "content": "..."}]``).
        entities: Entities to extract facts about.
        instructions: Optional domain-specific extraction guidance
            (e.g. "Focus on billing issues and payment preferences").

    Returns:
        List of candidate :class:`MemoryRecord` objects with no ``id``
        (assigned at storage time).

    Raises:
        RetainLLMError: LLM call failed or returned unparseable output.
    """
    if not transcript or not entities:
        return []

    system_prompt = format_extraction_prompt(entities, instructions)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(transcript, indent=2)},
    ]

    try:
        raw = await provider.complete(
            messages,
            temperature=0.0,
            max_tokens=2048,
        )
    except RetainLLMError:
        raise
    except Exception as exc:
        raise RetainLLMError(f"LLM call failed during extraction: {exc}") from exc

    return _parse_extraction_response(raw, entities)


def _parse_extraction_response(
    raw: str,
    expected_entities: list[EntityRef],
) -> list[MemoryRecord]:
    """Parse the LLM response into MemoryRecord objects.

    Handles common edge cases: response wrapped in markdown code fences,
    empty response, missing fields, non-JSON output.
    """
    cleaned = raw.strip()
    if not cleaned:
        return []

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse extraction output as JSON: %s", exc)
        return []

    if not isinstance(data, list):
        logger.warning(
            "Expected JSON array from extraction, got %s", type(data).__name__
        )
        return []

    expected = {(e.entity_type, e.entity_id) for e in expected_entities}
    facts: list[MemoryRecord] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        entity_type = item.get("entity_type", "")
        entity_id = item.get("entity_id", "")
        memory_type = item.get("memory_type", "")
        value = item.get("value")
        source = item.get("source", "extraction")

        if not entity_type or not entity_id or not memory_type or not value:
            continue

        if (entity_type, entity_id) not in expected:
            continue

        if not isinstance(value, dict):
            continue

        facts.append(MemoryRecord(
            entity_type=entity_type,
            entity_id=entity_id,
            memory_type=memory_type,
            value=value,
            source=source,
        ))

    return facts
