"""Tests for LLM provider protocol and implementations."""

import json
from typing import Any
from unittest import mock

import httpx
import pytest

from retain.errors import RetainLLMError
from retain.llm import LLMProvider, MockProvider, OpenAIProvider


@pytest.mark.unit
class TestLLMProviderABC:
    """LLMProvider is an ABC — can't be instantiated directly."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]


@pytest.mark.unit
class TestMockProvider:
    """MockProvider is deterministic and records calls."""

    async def test_returns_set_response(self) -> None:
        provider = MockProvider(response="hello world")
        result = await provider.complete([{"role": "user", "content": "hi"}])
        assert result == "hello world"

    async def test_default_response_is_empty(self) -> None:
        provider = MockProvider()
        result = await provider.complete([{"role": "user", "content": "hi"}])
        assert result == ""

    async def test_records_call(self) -> None:
        provider = MockProvider(response="ok")
        await provider.complete(
            [{"role": "user", "content": "hello"}],
            model="test-model",
            max_tokens=512,
            temperature=0.5,
        )
        assert len(provider.calls) == 1
        call = provider.calls[0]
        assert call["messages"] == [{"role": "user", "content": "hello"}]
        assert call["model"] == "test-model"
        assert call["max_tokens"] == 512
        assert call["temperature"] == 0.5

    async def test_accumulates_multiple_calls(self) -> None:
        provider = MockProvider(response="ok")
        await provider.complete([{"role": "user", "content": "first"}])
        await provider.complete([{"role": "user", "content": "second"}])
        assert len(provider.calls) == 2
        assert provider.calls[0]["messages"][0]["content"] == "first"
        assert provider.calls[1]["messages"][0]["content"] == "second"

    async def test_passes_extra_kwargs(self) -> None:
        provider = MockProvider(response="ok")
        await provider.complete(
            [{"role": "user", "content": "hi"}],
            top_p=0.9,
            stop=["\n"],
        )
        assert provider.calls[0]["kwargs"]["top_p"] == 0.9
        assert provider.calls[0]["kwargs"]["stop"] == ["\n"]


@pytest.mark.unit
class TestOpenAIProvider:
    """Tests for the OpenAI-compatible provider."""

    async def test_constructor_defaults(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
        assert provider._api_key == "test-key"
        assert provider._base_url == "https://api.openai.com/v1"

    async def test_constructor_custom_base_url(self) -> None:
        provider = OpenAIProvider(
            api_key="key",
            base_url="https://api.deepseek.com",
        )
        assert provider._base_url == "https://api.deepseek.com"

    async def test_constructor_strips_trailing_slash(self) -> None:
        provider = OpenAIProvider(
            api_key="key",
            base_url="https://api.deepseek.com/",
        )
        assert provider._base_url == "https://api.deepseek.com"

    async def test_raises_on_connection_error(self) -> None:
        provider = OpenAIProvider(api_key="key")
        with mock.patch.object(provider._client, "post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("connection refused")
            with pytest.raises(RetainLLMError):
                await provider.complete([{"role": "user", "content": "hi"}])

    async def test_raises_on_bad_status(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text='{"error": "unauthorized"}')

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider(api_key="bad", client=client)
        with pytest.raises(RetainLLMError, match="401"):
            await provider.complete([{"role": "user", "content": "hi"}])

    async def test_raises_on_empty_choices(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider(api_key="key", client=client)
        with pytest.raises(RetainLLMError, match="missing choices"):
            await provider.complete([{"role": "user", "content": "hi"}])

    async def test_returns_content(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "hello from mock"}}],
            })

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider(api_key="key", client=client)
        result = await provider.complete([{"role": "user", "content": "hi"}])
        assert result == "hello from mock"

    async def test_sends_expected_request_body(self) -> None:
        sent: list[dict[str, Any]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}}],
            })

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider(api_key="sk-test", model="deepseek-chat", client=client)
        await provider.complete(
            [{"role": "user", "content": "hello"}],
            max_tokens=256,
            temperature=0.7,
        )

        assert len(sent) == 1
        body = sent[0]
        assert body["model"] == "deepseek-chat"
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        assert body["max_tokens"] == 256
        assert body["temperature"] == 0.7

    async def test_includes_auth_header(self) -> None:
        headers: dict[str, str] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            headers.update(dict(request.headers))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}}],
            })

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider(api_key="sk-test-key", client=client)
        await provider.complete([{"role": "user", "content": "hi"}])

        assert headers.get("authorization") == "Bearer sk-test-key"
        assert headers.get("content-type") == "application/json"

    async def test_sends_extra_kwargs(self) -> None:
        sent: list[dict[str, Any]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}}],
            })

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider(api_key="key", client=client)
        await provider.complete(
            [{"role": "user", "content": "hi"}],
            top_p=0.9,
            presence_penalty=0.1,
        )

        assert sent[0]["top_p"] == 0.9
        assert sent[0]["presence_penalty"] == 0.1
