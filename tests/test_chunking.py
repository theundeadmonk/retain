"""Tests for transcript chunking."""


from retain.chunking import chunk_transcript


class TestChunkTranscript:
    def test_single_message(self):
        transcript = [{"role": "user", "content": "Hello, I need help with my order."}]
        chunks = chunk_transcript(transcript)
        assert len(chunks) == 1
        assert "Hello, I need help with my order." in chunks[0]

    def test_short_transcript_single_chunk(self):
        transcript = [
            {"role": "user", "content": "Hi there."},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ]
        chunks = chunk_transcript(transcript)
        assert len(chunks) == 1
        assert "user:" in chunks[0]
        assert "assistant:" in chunks[0]

    def test_long_transcript_splits(self):
        long_msg = ". ".join([f"Sentence number {i}" for i in range(100)])
        transcript = [{"role": "user", "content": long_msg}]
        chunks = chunk_transcript(transcript, max_chars=300)
        assert len(chunks) > 1

    def test_empty_transcript(self):
        chunks = chunk_transcript([])
        assert chunks == []

    def test_messages_with_no_content_skipped(self):
        transcript = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Actual content."},
        ]
        chunks = chunk_transcript(transcript)
        assert len(chunks) == 1
        assert "system:" not in chunks[0]

    def test_overlap_preserves_context(self):
        transcript = [
            {"role": "user", "content": "A. " * 50 + "B. " * 50},
        ]
        chunks = chunk_transcript(transcript, max_chars=80, overlap_sentences=2)
        assert len(chunks) > 1
        last_bit_of_first = chunks[0].strip().split(". ")[-3]
        first_bit_of_second = chunks[1].strip().split(". ")[0]
        assert last_bit_of_first in first_bit_of_second or first_bit_of_second in last_bit_of_first
