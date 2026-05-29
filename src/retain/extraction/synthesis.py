"""Deduplication and profile synthesis."""

import json

from retain.errors import RetainLLMError
from retain.llm.base import LLMProvider
from retain.types import MemoryRecord

__all__ = [
    "deduplicate",
    "synthesize_profile",
]

_SYNTHESIS_PROMPT = """\
Synthesize a concise caller profile for a {entity_type} (ID: {entity_id}) \
from structured facts gathered across conversations.

Facts about this entity:
{fact_lines}

Write a concise natural-language profile of no more than {budget} tokens. \
Focus on the most important information: identity, preferences, active needs, \
and recurring patterns. Omit trivial or outdated details. Write in third person.

Profile:
"""


def _fact_key(fact: MemoryRecord) -> tuple[str, str, str, str] | None:
    if not fact.entity_type or not fact.entity_id or not fact.memory_type or not fact.value:
        return None
    return (
        fact.entity_type,
        fact.entity_id,
        fact.memory_type,
        json.dumps(fact.value, sort_keys=True),
    )


def deduplicate(
    new_facts: list[MemoryRecord],
    existing_facts: list[MemoryRecord],
) -> list[MemoryRecord]:
    """Return only novel facts not already in ``existing_facts``.

    Uses exact-match dedup on ``(entity_type, entity_id, memory_type,
    sorted value JSON)``. Two facts with the same type and identical
    structured values are considered duplicates.
    """
    existing_keys: set[tuple[str, str, str, str]] = set()
    for f in existing_facts:
        key = _fact_key(f)
        if key is not None:
            existing_keys.add(key)

    novel: list[MemoryRecord] = []
    seen: set[tuple[str, str, str, str]] = set()
    for f in new_facts:
        key = _fact_key(f)
        if key is None:
            continue
        if key not in existing_keys and key not in seen:
            seen.add(key)
            novel.append(f)

    return novel


async def synthesize_profile(
    provider: LLMProvider,
    entity_type: str,
    entity_id: str,
    facts: list[MemoryRecord],
    *,
    budget_tokens: int = 80,
) -> str:
    """Synthesize a concise profile blob from all known facts.

    Args:
        provider: LLM provider.
        entity_type: Entity type for the profile.
        entity_id: Entity ID for the profile.
        facts: All known facts about this entity.
        budget_tokens: Maximum token budget (default 80).

    Returns:
        A concise natural-language profile string.
    """
    if not facts:
        return f"A {entity_type} with no stored information yet."

    fact_lines = "\n".join(
        f"  - [{f.memory_type}] {json.dumps(f.value)}"
        for f in facts
    )

    prompt = _SYNTHESIS_PROMPT.format(
        entity_type=entity_type,
        entity_id=entity_id,
        fact_lines=fact_lines,
        budget=budget_tokens,
    )

    try:
        profile = await provider.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=budget_tokens,
        )
    except RetainLLMError:
        raise
    except Exception as exc:
        raise RetainLLMError(
            f"Profile synthesis failed: {exc}"
        ) from exc

    return profile.strip()
