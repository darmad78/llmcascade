"""OpenAI-compatible /v1/chat/completions (non-streaming) for drop-in clients."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from llmcascade.adapters.base import LLMResponse


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"] = "user"
    content: str | list[Any] | None = None

    def text_content(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for block in self.content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(str(block["text"]))
                elif block.get("text"):
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool | None = False
    user: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    # llmcascade extension when pinning a registry model
    include_free_cascade: bool | None = None
    failover_models: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _reject_stream(self) -> ChatCompletionRequest:
        if self.stream:
            raise ValueError("streaming is not supported on /v1/chat/completions yet; use POST /v1/complete with stream=true")
        return self


def messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Flatten chat messages into a single prompt for the native complete path."""
    lines: list[str] = []
    for msg in messages:
        text = msg.text_content()
        if not text:
            continue
        role = msg.role
        if role == "developer":
            role = "system"
        label = role.capitalize()
        lines.append(f"{label}: {text}")
    if not lines:
        raise ValueError("messages must include at least one non-empty content")
    lines.append("Assistant:")
    return "\n".join(lines)


def resolve_model_pin(body: ChatCompletionRequest) -> tuple[str | None, list[str], bool | None]:
    raw = (body.model or "").strip()
    if not raw or raw.lower() in ("auto", "llmcascade", "llmcascade/auto"):
        return None, [], None
    pin = raw
    failovers = [m.strip() for m in body.failover_models if m and m.strip()]
    cascade = body.include_free_cascade
    if cascade is None:
        cascade = False
    return pin, failovers, cascade


def llm_response_to_openai(body: ChatCompletionRequest, result: LLMResponse) -> dict[str, Any]:
    created = int(time.time())
    total = max(result.tokens_used, 0)
    completion_est = max(len(result.text) // 4, 1) if result.text else 0
    prompt_est = max(total - completion_est, 0)
    notes = (body.notes or body.user or "").strip() or None
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": created,
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_est,
            "completion_tokens": completion_est,
            "total_tokens": prompt_est + completion_est,
        },
        "llmcascade": {
            "latency_ms": result.latency_ms,
            "notes": notes,
            "skipped_models": result.skipped_models,
        },
    }


def chat_completion_params(body: ChatCompletionRequest) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if body.temperature is not None:
        params["temperature"] = body.temperature
    if body.max_tokens is not None:
        params["max_tokens"] = body.max_tokens
    if body.top_p is not None:
        params["top_p"] = body.top_p
    return params
