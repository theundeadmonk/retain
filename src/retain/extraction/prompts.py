"""Extraction prompts for LLM-based fact extraction."""

from retain.types import EntityRef

__all__ = [
    "DEFAULT_EXTRACTION_PROMPT",
    "format_extraction_prompt",
]


DEFAULT_EXTRACTION_PROMPT = """\
You are analyzing a conversation transcript to extract structured facts about \
specific entities.

Entities to extract facts about:
{entities}

{instructions}
Extract all facts that can be learned about each entity from this conversation. \
Be thorough — include preferences, personal details, issues, requests, behaviors, \
and any other information that would help understand or serve this entity on a \
future call.

For each fact, provide:
- entity_type: The type of entity this fact is about
- entity_id: The ID of the entity this fact is about
- memory_type: A category for this fact (e.g., "preference", "personal_info", \
"issue", "request", "behavior", "relationship")
- value: A JSON object containing the fact details. Use keys that make sense for \
the fact type (e.g., {{"key": "room_preference", "value": "top floor"}}).
- source: Always "extraction"

Respond with ONLY a JSON array of fact objects — no markdown, no explanation. \
If no facts can be extracted, respond with an empty array [].

[
  {{
    "entity_type": "...",
    "entity_id": "...",
    "memory_type": "...",
    "value": {{...}},
    "source": "extraction"
  }}
]
"""


def format_extraction_prompt(
    entities: list[EntityRef],
    instructions: str = "",
) -> str:
    """Format the extraction prompt with entity info and custom instructions."""
    entity_lines = "\n".join(
        f"  - {e.entity_type}: {e.entity_id}" for e in entities
    )

    instr_section = ""
    if instructions:
        instr_section = (
            f"Additional extraction guidance from the caller:\n{instructions}\n"
        )

    return DEFAULT_EXTRACTION_PROMPT.format(
        entities=entity_lines,
        instructions=instr_section,
    )
