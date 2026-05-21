"""OpenAI compatibility hardening tests for chat-completion streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from corethread.errors import CoreThreadError, ProviderHTTPError, ProviderTimeout
from corethread.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    JudgeVerdict,
)
from corethread.providers.base import ChatOptions
from corethread.streaming import include_usage_requested, response_to_stream_chunks
from tests._fakes import FakeJudge, FakeProvider, make_happy_local_response


def _sse_data_values(text: str) -> list[str]:
    values: list[str] = []
    for frame in text.split("\n\n"):
        for line in frame.splitlines():
            if line.startswith("data: "):
                values.append(line.removeprefix("data: "))
    return values


def _sse_json_values(text: str) -> list[dict[str, Any]]:
    return [json.loads(value) for value in _sse_data_values(text) if value != "[DONE]"]


class _BufferedStreamingOrchestrator:
    async def handle(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return make_happy_local_response(
            model=request.model,
            content="local raw stream",
            prompt_tokens=7,
            completion_tokens=2,
        )

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[dict[str, Any]]:
        response = await self.handle(request)
        for chunk in response_to_stream_chunks(
            response,
            include_usage=include_usage_requested(request),
        ):
            yield chunk


class _FrontierStreamingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__(
            name="frontier",
            response=make_happy_local_response(model="gpt-4o"),
        )
        self.stream_call_count = 0

    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        *,
        options: ChatOptions | None = None,
        model_override: str | None = None,
        timeout_s: float = 120.0,
        on_final: Callable[[ChatCompletionResponse], None] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        self.stream_call_count += 1
        yield {
            "id": "chatcmpl-frontier-stream",
            "object": "chat.completion.chunk",
            "created": 1731990317,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield {
            "id": "chatcmpl-frontier-stream",
            "object": "chat.completion.chunk",
            "created": 1731990317,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "frontier raw stream"},
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "id": "chatcmpl-frontier-stream",
            "object": "chat.completion.chunk",
            "created": 1731990317,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        if on_final is not None:
            on_final(
                make_happy_local_response(
                    model="gpt-4o",
                    content="frontier raw stream",
                    prompt_tokens=11,
                    completion_tokens=4,
                )
            )


class _FrontierStreamingErrorProvider(FakeProvider):
    def __init__(self, exc: CoreThreadError) -> None:
        super().__init__(
            name="frontier",
            response=make_happy_local_response(model="gpt-4o"),
        )
        self._exc = exc

    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        *,
        options: ChatOptions | None = None,
        model_override: str | None = None,
        timeout_s: float = 120.0,
        on_final: Callable[[ChatCompletionResponse], None] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        raise self._exc
        yield {}


def _install_orchestrator(
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    local: FakeProvider,
    frontier: FakeProvider,
    verdict: JudgeVerdict | None = None,
) -> None:
    from corethread import orchestrator as orch_mod
    from corethread.orchestrator import Orchestrator

    if verdict is None:
        verdict = JudgeVerdict(
            pass_=True,
            confidence_score=0.9,
            reasoning=(
                "[answered_core_q=true, no_disclaimers=true, no_contradictions=true] test verdict."
            ),
        )
    fake_judge = FakeJudge(verdicts=[verdict])
    monkeypatch.setattr(orch_mod.judge, "grade", fake_judge)
    main_module.app.state.orchestrator = Orchestrator(
        local=local,
        judge_provider=local,
        frontier=frontier,
        cfg=main_module.app.state.config,
    )


def test_raw_stream_local_accept_has_openai_sse_frames_and_headers(
    app_with_fake_orchestrator: TestClient,
) -> None:
    with app_with_fake_orchestrator as client:
        from corethread import main as m

        m.app.state.orchestrator = _BufferedStreamingOrchestrator()
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1:8b",
                "messages": [{"role": "user", "content": "stream locally"}],
                "stream": True,
            },
        )

    assert response.status_code == 200, response.text
    assert "text/event-stream" in response.headers["content-type"]
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    data_values = _sse_data_values(response.text)
    assert data_values.count("[DONE]") == 1
    chunks = _sse_json_values(response.text)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[1]["choices"][0]["delta"] == {"content": "local raw stream"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_raw_stream_local_include_usage_emits_one_usage_chunk(
    app_with_fake_orchestrator: TestClient,
) -> None:
    with app_with_fake_orchestrator as client:
        from corethread import main as m

        m.app.state.orchestrator = _BufferedStreamingOrchestrator()
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1:8b",
                "messages": [{"role": "user", "content": "stream locally"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

    chunks = _sse_json_values(response.text)
    usage_chunks = [chunk for chunk in chunks if chunk.get("usage") is not None]
    assert response.status_code == 200, response.text
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["choices"] == []
    assert usage_chunks[0]["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
    }


def test_raw_stream_pivoted_live_frontier_preserves_chunk_shape(
    app_with_fake_orchestrator: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontier = _FrontierStreamingProvider()
    with app_with_fake_orchestrator as client:
        from corethread import main as m

        _install_orchestrator(
            m,
            monkeypatch,
            local=FakeProvider(
                name="local",
                response=make_happy_local_response(content="local rejected"),
            ),
            frontier=frontier,
            verdict=JudgeVerdict(
                pass_=False,
                confidence_score=0.3,
                reasoning="[answered_core_q=false] test pivot.",
            ),
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1:8b",
                "messages": [{"role": "user", "content": "force pivot"}],
                "stream": True,
            },
        )

    assert response.status_code == 200, response.text
    assert frontier.stream_call_count == 1
    data_values = _sse_data_values(response.text)
    assert data_values.count("[DONE]") == 1
    chunks = _sse_json_values(response.text)
    assert {chunk["id"] for chunk in chunks} == {"chatcmpl-frontier-stream"}
    assert chunks[1]["choices"][0]["delta"]["content"] == "frontier raw stream"


def test_stream_prefirst_local_timeout_returns_json_504_not_sse(
    app_with_fake_orchestrator: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with app_with_fake_orchestrator as client:
        from corethread import main as m

        _install_orchestrator(
            m,
            monkeypatch,
            local=FakeProvider(
                name="local",
                raise_on_chat=ProviderTimeout("local", 2.0),
            ),
            frontier=FakeProvider(
                name="frontier",
                response=make_happy_local_response(model="gpt-4o"),
            ),
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1:8b",
                "messages": [{"role": "user", "content": "timeout"}],
                "stream": True,
            },
        )

    assert response.status_code == 504
    assert "application/json" in response.headers["content-type"]
    assert response.json()["error"]["code"] == "provider_timeout"


@pytest.mark.parametrize(
    ("exc", "expected_status", "expected_code"),
    [
        (
            ProviderHTTPError("frontier", 429, "rate limited"),
            502,
            "internal_error",
        ),
        (
            ProviderTimeout("frontier", 2.0),
            504,
            "provider_timeout",
        ),
    ],
)
def test_stream_prefirst_frontier_errors_return_json_error(
    app_with_fake_orchestrator: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    exc: CoreThreadError,
    expected_status: int,
    expected_code: str,
) -> None:
    with app_with_fake_orchestrator as client:
        from corethread import main as m

        _install_orchestrator(
            m,
            monkeypatch,
            local=FakeProvider(
                name="local",
                response=make_happy_local_response(content="local rejected"),
            ),
            frontier=_FrontierStreamingErrorProvider(exc),
            verdict=JudgeVerdict(
                pass_=False,
                confidence_score=0.3,
                reasoning="[answered_core_q=false] test pivot.",
            ),
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1:8b",
                "messages": [{"role": "user", "content": "frontier error"}],
                "stream": True,
            },
        )

    assert response.status_code == expected_status
    assert "application/json" in response.headers["content-type"]
    assert response.json()["error"]["code"] == expected_code
