"""Tests for proposition-based transcript decomposition."""


from retain.chunking import _parse_propositions, _transcript_to_text
from retain.llm.mock import MockProvider


class TestTranscriptToText:
    def test_converts_messages_with_role(self):
        text = _transcript_to_text([
            {"role": "agent", "content": "Hello"},
            {"role": "customer", "content": "Hi there"},
        ])
        assert "agent: Hello" in text
        assert "customer: Hi there" in text

    def test_skips_empty_content(self):
        text = _transcript_to_text([
            {"role": "system", "content": ""},
            {"role": "agent", "content": "Only content"},
        ])
        assert "system:" not in text
        assert "agent: Only content" in text


class TestParsePropositions:
    def test_parses_json_array(self):
        response = '["Fact one", "Fact two", "Fact three"]'
        result = _parse_propositions(response)
        assert result == ["Fact one", "Fact two", "Fact three"]

    def test_parses_code_fenced_json(self):
        response = """```json
["Fact one", "Fact two"]
```"""
        result = _parse_propositions(response)
        assert result == ["Fact one", "Fact two"]

    def test_empty_response_returns_empty(self):
        assert _parse_propositions("") == []

    def test_malformed_json_returns_empty(self):
        assert _parse_propositions("not json at all") == []

    def test_skips_non_string_items(self):
        response = '["good", 123, null, "also good"]'
        result = _parse_propositions(response)
        assert "good" in result
        assert "also good" in result
        assert len(result) == 2  # 123 and null skipped


class TestDecomposeTranscript:
    async def test_returns_propositions_from_llm(self):
        provider = MockProvider(
            response='["Customer contacted support", "Customer had billing issue"]'
        )

        from retain.chunking import decompose_transcript

        propositions = await decompose_transcript(
            provider,
            [{"role": "customer", "content": "I was double-charged"}],
        )
        assert len(propositions) == 2
        assert "Customer contacted support" in propositions

    async def test_empty_transcript_returns_empty(self):
        provider = MockProvider(response="[]")

        from retain.chunking import decompose_transcript

        propositions = await decompose_transcript(provider, [])
        assert propositions == []

    async def test_passes_entity_context_to_prompt(self):
        provider = MockProvider(response='["Fact about customer"]')

        from retain.chunking import decompose_transcript
        from retain.types import EntityRef

        await decompose_transcript(
            provider,
            [{"role": "agent", "content": "test"}],
            entities=[EntityRef(entity_type="customer", entity_id="cust_1")],
        )
        call = provider.calls[0]
        assert "customer: cust_1" in call["messages"][0]["content"]
