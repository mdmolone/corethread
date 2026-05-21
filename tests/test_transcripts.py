from __future__ import annotations

import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

from corethread.api_ui import router as ui_router
from corethread.models import ChatCompletionRequest, ChatMessage
from corethread.transcripts import TranscriptStore, instrument_provider
from tests._fakes import FakeProvider, make_happy_local_response


def _request(content: str = "hello") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="meta/llama-3.3-70b",
        messages=[ChatMessage(role="user", content=content)],
    )


def test_transcript_store_evicts_oldest() -> None:
    store = TranscriptStore(maxlen=2)
    store.record_request("req-1", _request("one"))
    store.record_request("req-2", _request("two"))
    store.record_request("req-3", _request("three"))

    assert store.get("req-1") is None
    assert [item["request_id"] for item in store.latest()] == ["req-3", "req-2"]


@pytest.mark.asyncio
async def test_instrument_provider_captures_local_request_and_response() -> None:
    store = TranscriptStore(maxlen=5)
    provider = FakeProvider(response=make_happy_local_response(content="local answer"))
    instrument_provider(provider, store=store, role="local", judge_model="judge-model")

    structlog.contextvars.bind_contextvars(request_id="req-local")
    try:
        await provider.chat(_request("What is 18 + 24?"))
    finally:
        structlog.contextvars.clear_contextvars()

    item = store.get("req-local")
    assert item is not None
    assert item["request_messages"][0]["content"] == "What is 18 + 24?"
    assert item["local_response"] == "local answer"
    assert item["final_response"] == "local answer"
    assert item["final_response_source"] == "local"


@pytest.mark.asyncio
async def test_instrument_provider_forces_configured_frontier_model() -> None:
    store = TranscriptStore(maxlen=5)
    provider = FakeProvider(response=make_happy_local_response(content="frontier answer"))
    instrument_provider(
        provider,
        store=store,
        role="frontier",
        default_model_override="gpt-4o",
    )

    structlog.contextvars.bind_contextvars(request_id="req-frontier")
    try:
        await provider.chat(_request("force a pivot"))
    finally:
        structlog.contextvars.clear_contextvars()

    assert provider.calls[0][2] == "gpt-4o"
    item = store.get("req-frontier")
    assert item is not None
    assert item["frontier_response"] == "frontier answer"
    assert item["final_response"] == "frontier answer"
    assert item["final_response_source"] == "frontier"


def test_transcript_endpoint_returns_retained_body() -> None:
    app = FastAPI()
    app.include_router(ui_router)
    store = TranscriptStore(maxlen=5)
    store.record_request("req-api", _request("show me the body"))
    store.record_response(
        "req-api",
        stage="local",
        response=make_happy_local_response(content="local endpoint answer"),
    )
    app.state.transcript_store = store
    app.state.transcript_capture_enabled = True

    with TestClient(app) as client:
        response = client.get("/v1/traces/req-api/transcript")

    assert response.status_code == 200
    body = response.json()
    assert body["request_messages"][0]["content"] == "show me the body"
    assert body["local_response"] == "local endpoint answer"


def test_transcript_endpoint_disabled_by_default() -> None:
    app = FastAPI()
    app.include_router(ui_router)
    app.state.transcript_store = None
    app.state.transcript_capture_enabled = False

    with TestClient(app) as client:
        response = client.get("/v1/traces/req-api/transcript")

    assert response.status_code == 403
    assert "Transcript capture is disabled" in response.json()["detail"]
