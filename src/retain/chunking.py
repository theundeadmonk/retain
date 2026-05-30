"""Proposition-based transcript decomposition — cold path.

Uses LLM to decompose transcripts into atomic, self-contained propositions.
Each proposition becomes a standalone chunk for embedding and retrieval.

Technique: Proposition-Based Chunking (Chen et al., 2024)
Why: Self-contained atomic facts eliminate chunk ambiguity, making
     contextual retrieval (#4) and metadata prefixes (#1) unnecessary.
"""

from retain.llm.base import LLMProvider
from retain.types import EntityRef

__all__ = [
    "DECOMPOSE_PROMPT",
    "decompose_transcript",
]


DECOMPOSE_PROMPT = """\
You are analyzing a conversation transcript. Decompose it into a list of \
atomic, self-contained propositions.

Each proposition must:
- Be a single, complete fact (one piece of information)
- Include all necessary context — who said what, about whom, regarding what
- Be understandable without reading any other proposition or the original \
transcript
- Use specific details where available (names, amounts, dates, order numbers)

Return ONLY a JSON array of strings — no markdown, no explanation.

Example transcript:
  agent: Hello, how can I help?
  customer: I was double-charged $49.99 for order #4567 on May 1st
  agent: I see the charge. I'll process a refund — it will take 3-5 business days.

Example output:
[
  "Customer contacted support about a billing issue",
  "Customer was double-charged $49.99 for order #4567",
  "The double charge occurred on May 1st",
  "Agent confirmed the double charge for order #4567",
  "Agent will process a refund for the $49.99 double charge",
  "The refund will take 3-5 business days to process"
]

Now decompose this transcript:

{transcript}

Propositions (JSON array):"""


async def decompose_transcript(
    llm: LLMProvider,
    transcript: list[dict[str, str]],
    *,
    entities: list[EntityRef] | None = None,
) -> list[str]:
    """Decompose a transcript into atomic propositions via LLM.

    Args:
        llm: The LLM provider for decomposition.
        transcript: List of messages with ``role`` and ``content`` keys.
        entities: Optional list of entities to contextualize propositions.

    Returns:
        List of self-contained proposition strings ready for embedding.
    """
    transcript_text = _transcript_to_text(transcript)
    if entities:
        entity_context = "\n".join(
            f"  - {e.entity_type}: {e.entity_id}" for e in entities
        )
        prompt_prefix = (
            f"Relevant entities:\n{entity_context}\n\n"
        )
    else:
        prompt_prefix = ""
    prompt = prompt_prefix + DECOMPOSE_PROMPT.format(transcript=transcript_text)

    response = await llm.complete(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2048,
    )

    return _parse_propositions(response)


def _transcript_to_text(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _parse_propositions(response: str) -> list[str]:
    """Parse the LLM JSON array response into a list of strings."""
    import json

    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        response = "\n".join(lines[1:-1])

    try:
        parsed = json.loads(response)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if isinstance(item, str) and item.strip()]
    except json.JSONDecodeError:
        pass

    return []
