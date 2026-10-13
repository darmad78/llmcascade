from __future__ import annotations

import pytest

from llmcascade.adapters.base import LLMResponse
from llmcascade.openai_http import (
    ChatCompletionRequest,
    ChatMessage,
    llm_response_to_openai,
    messages_to_prompt,
    resolve_model_pin,
)


def test_messages_to_prompt():
    prompt = messages_to_prompt(
        [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hi"),
        ]
    )
    assert "System: You are helpful." in prompt
    assert "User: Hi" in prompt
    assert prompt.endswith("Assistant:")


def test_resolve_model_pin_auto():
    body = ChatCompletionRequest(messages=[ChatMessage(role="user", content="x")], model="auto")
    assert resolve_model_pin(body) == (None, [], None)


def test_resolve_model_pin_explicit():
    body = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="x")],
        model="llama-3.3-70b-versatile",
        include_free_cascade=True,
    )
    pin, failovers, cascade = resolve_model_pin(body)
    assert pin == "llama-3.3-70b-versatile"
    assert cascade is True


def test_openai_response_shape():
    body = ChatCompletionRequest(messages=[ChatMessage(role="user", content="x")], user="app1")
    out = llm_response_to_openai(body, LLMResponse(text="ok", model="m", tokens_used=10, latency_ms=3.0))
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"]["content"] == "ok"
    assert out["model"] == "m"
    assert out["usage"]["total_tokens"] >= 1


def test_stream_rejected():
    with pytest.raises(ValueError, match="streaming"):
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="x")],
            stream=True,
        )
