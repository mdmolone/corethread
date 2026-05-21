from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from corethread.config import ControlsConfig, ModelPricing
from corethread.controls import RequestControls
from corethread.errors import QuotaExceeded, RateLimitExceeded
from corethread.models import ChatCompletionRequest, ChatMessage
from corethread.obs import RequestTrace


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role="user", content="private prompt body")],
        user="local-operator",
    )


def _trace(request_id: str = "req-controls-001") -> RequestTrace:
    return {
        "request_id": request_id,
        "selected_local_model": "llama3.1:8b",
        "judge_model": "qwen2.5:7b",
        "frontier_model": "gpt-4o",
        "confidence_score": 0.2,
        "pivoted": True,
        "local_latency_ms": 10,
        "judge_latency_ms": 5,
        "frontier_latency_ms": 100,
        "input_tokens": 1000,
        "output_tokens": 500,
        "frontier_cost_est": None,
        "judge_parse_failed": False,
        "pivot_reason": "low_score",
        "local_error_class": None,
    }


async def test_request_controls_rate_limit_blocks_second_request() -> None:
    controls = RequestControls(ControlsConfig(requests_per_minute=1), clock=lambda: 1000.0)

    await controls.admit(request_id="req-1", request=_request())

    with pytest.raises(RateLimitExceeded):
        await controls.admit(request_id="req-2", request=_request())


async def test_request_controls_cost_quota_blocks_after_recorded_usage() -> None:
    controls = RequestControls(
        ControlsConfig(
            daily_cost_quota_usd=0.01,
            pricing={
                "gpt-4o": ModelPricing(
                    input_per_1m_tokens=10.0,
                    output_per_1m_tokens=20.0,
                )
            },
        ),
        clock=lambda: 1000.0,
    )

    await controls.admit(request_id="req-1", request=_request())
    await controls.record_trace(_trace("req-1"))

    snapshot = controls.snapshot()
    assert snapshot["daily_cost_usd"] == pytest.approx(0.02)
    with pytest.raises(QuotaExceeded):
        await controls.admit(request_id="req-2", request=_request())


async def test_audit_log_is_body_free_by_default(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    controls = RequestControls(
        ControlsConfig(
            audit_enabled=True,
            audit_path=str(audit_path),
            pricing={
                "gpt-4o": ModelPricing(
                    input_per_1m_tokens=10.0,
                    output_per_1m_tokens=20.0,
                )
            },
        ),
        clock=lambda: 1000.0,
    )

    await controls.admit(request_id="req-1", request=_request())
    await controls.record_trace(_trace("req-1"))

    text = audit_path.read_text(encoding="utf-8")
    assert "private prompt body" not in text
    events = [json.loads(line) for line in text.splitlines()]
    assert [event["event"] for event in events] == [
        "request.accepted",
        "request.completed",
    ]
    assert events[0]["requested_model"] == "gpt-4o"
    assert events[1]["cost_usd"] == pytest.approx(0.02)


def test_chat_route_rate_limit_error_is_openai_shaped() -> None:
    from corethread import main as m

    importlib.reload(m)

    class _RejectingControls:
        async def admit(self, *, request_id: str, request: ChatCompletionRequest) -> None:
            raise RateLimitExceeded("Request rate limit exceeded.")

    class _NeverCalledOrchestrator:
        async def handle(self, request: ChatCompletionRequest) -> Any:
            raise AssertionError("orchestrator should not be called")

    m.app.state.controls = _RejectingControls()
    m.app.state.orchestrator = _NeverCalledOrchestrator()

    client = TestClient(m.app, raise_server_exceptions=False)
    try:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Say OK."}],
            },
        )
    finally:
        client.close()

    assert resp.status_code == 429, resp.text
    body = resp.json()
    assert body["error"]["type"] == "rate_limit_error"
    assert body["error"]["code"] == "rate_limit_exceeded"
    assert "Say OK" not in resp.text
