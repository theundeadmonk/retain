"""Transcript chunking for embedding storage."""


__all__ = ["chunk_transcript"]


def chunk_transcript(
    transcript: list[dict[str, str]],
    *,
    max_chars: int = 2000,
    overlap_sentences: int = 2,
) -> list[str]:
    """Split a transcript into overlapping sentence-aware chunks.

    Args:
        transcript: List of messages with ``role`` and ``content`` keys.
        max_chars: Maximum characters per chunk (soft limit).
        overlap_sentences: Number of sentences to overlap between chunks.

    Returns:
        List of chunk strings ready for embedding.
    """
    text = _transcript_to_text(transcript)
    sentences = _split_sentences(text)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current and current_len + len(sentence) > max_chars:
            chunks.append(" ".join(current))
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = list(overlap)
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks


def _transcript_to_text(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _split_sentences(text: str) -> list[str]:
    import re

    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]
