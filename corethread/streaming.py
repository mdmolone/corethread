"""OpenAI-compatible chat-completion stream helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from corethread.models import ChatCompletionRequest, ChatCompletionResponse

__all__ = [
    "format_sse_data",
    "format_sse_done",
    "include_usage_requested",
    "response_to_stream_chunks",
]


def include_usage_requested(request: ChatCompletionRequest) -> bool:
    """Return true when the OpenAI `stream_options.include_usage` flag is set."""
    stream_options = getattr(request, "stream_options", None)
    if stream_options is None and request.model_extra:
        stream_options = request.model_extra.get("stream_options")
    return isinstance(stream_options, dict) and stream_options.get("include_usage") is True


def _base_chunk(response: ChatCompletionResponse) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": response.id,
        "object": "chat.completion.chunk",
        "created": response.created,
        "model": response.model,
    }
    if response.system_fingerprint is not None:
        chunk["system_fingerprint"] = response.system_fingerprint
    return chunk


def _message_content(response: ChatCompletionResponse) -> str:
    if not response.choices:
        return ""
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def response_to_stream_chunks(
    response: ChatCompletionResponse,
    *,
    include_usage: bool = False,
) -> Iterator[dict[str, Any]]:
    """Convert a completed chat response into a short OpenAI-style chunk stream."""
    role_chunk = _base_chunk(response)
    role_chunk["choices"] = [
        {
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }
    ]
    yield role_chunk

    content = _message_content(response)
    if content:
        content_chunk = _base_chunk(response)
        content_chunk["choices"] = [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ]
        yield content_chunk

    finish_reason = response.choices[0].finish_reason if response.choices else "stop"
    finish_chunk = _base_chunk(response)
    finish_chunk["choices"] = [
        {
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }
    ]
    yield finish_chunk

    if include_usage:
        usage_chunk = _base_chunk(response)
        usage_chunk["choices"] = []
        usage_chunk["usage"] = response.usage.model_dump(mode="json")
        yield usage_chunk


def format_sse_data(payload: dict[str, Any]) -> str:
    """Format one JSON payload as an OpenAI-compatible SSE data frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def format_sse_done() -> str:
    """Format the OpenAI stream terminator."""
    return "data: [DONE]\n\n"
