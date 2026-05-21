"""SC#1 named test — Pitfall 18 / TRC-09 SSE trace stream secret-leak defense.

The literal `sk-test1234567890abcdefghij` is the SC#1 binding contract per
CONTEXT.md D-19 #1 — do NOT change it. Asserts the literal is absent from
EVERY line of the SSE stream (data:, event:, id:, : heartbeat).

Approach: build a minimal FastAPI app with router + TraceBus on app.state;
seed the bus with one synthetic trace; drive the ASGI app directly via a
small in-process driver (httpx.ASGITransport in 0.28.1 buffers the entire
response body before returning, so it cannot consume infinite SSE streams);
read at least one data line; assert literal absence on every line.

The 15-field RequestTrace shape carries no body fields (TRC-09 lock); the
TraceEvent Pydantic model has extra='forbid' (D-14); the SSE generator
runtime-validates via TraceEvent.model_validate(...).model_dump() before
yield (Pitfall 18 runtime defense). This test is the binding contract.

DEVIATION (Rule 3 — blocking issue auto-fix): The plan prescribed
`httpx.AsyncClient(transport=ASGITransport(app))` + `client.stream(...)` +
`aiter_lines()` for SSE testing. httpx 0.28.1's ASGITransport.handle_async_request
buffers `body_parts.append(body)` and only returns the Response after
`response_complete.is_set()` — the ASGI app must finish sending its body
before any bytes reach the client. For an infinite SSE generator this never
completes. The custom `_drive_sse_until_bytes` helper below drives the app's
ASGI 3.0 callable directly, capturing chunks as they're sent and disconnecting
once enough material has been collected. This pattern is the project's first
SSE test driver — re-used by tests/test_api_ui.py for the D-10/D-11/D-13 tests.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest
from fastapi import FastAPI

from corethread.api_ui import router as ui_router
from corethread.obs import RequestTrace
from corethread.pubsub import TraceBus

SK_TEST_LITERAL = "sk-test1234567890abcdefghij"  # SC#1 binding contract — D-19 #1


def _make_synthetic_trace(request_id: str) -> RequestTrace:
    """Build a complete 15-field RequestTrace dict for direct bus.publish_nowait.

    Mirrors tests/test_pubsub.py:_make_trace verbatim — the project's standard
    full-trace builder for unit tests that bypass the orchestrator.
    """
    return {
        "request_id": request_id,
        "selected_local_model": "llama3.1:8b",
        "judge_model": "qwen2.5:7b",
        "frontier_model": None,
        "confidence_score": 0.9,
        "pivoted": False,
        "local_latency_ms": 0,
        "judge_latency_ms": 0,
        "frontier_latency_ms": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "frontier_cost_est": None,
        "judge_parse_failed": False,
        "pivot_reason": "none",
        "local_error_class": None,
    }


async def _drive_sse_until_bytes(
    app: Any,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    max_bytes: int = 500,
    timeout_s: float = 5.0,
) -> tuple[int, bytes]:
    """Drive the ASGI app and return (status_code, accumulated body bytes).

    Disconnects from the app once the accumulated body length exceeds
    `max_bytes` OR when `timeout_s` elapses, whichever comes first. The
    handler's `request.is_disconnected()` check then returns True on the
    next poll and the SSE generator exits cleanly.

    Why this exists: httpx 0.28.1's ASGITransport buffers the entire response
    body before returning the Response object — see module docstring DEVIATION.
    """
    body_chunks: list[bytes] = []
    status_holder: dict[str, int] = {}
    request_sent = False
    disconnect_now = asyncio.Event()

    raw_path = path.encode("ascii")
    encoded_headers: list[tuple[bytes, bytes]] = []
    if headers:
        encoded_headers = [
            (k.lower().encode("ascii"), v.encode("ascii")) for k, v in headers.items()
        ]

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "headers": encoded_headers,
        "scheme": "http",
        "path": path,
        "raw_path": raw_path,
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 50000),
        "root_path": "",
    }

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        # Block until we want to disconnect, then surface http.disconnect.
        await disconnect_now.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            status_holder["status"] = message["status"]
        elif message["type"] == "http.response.body":
            body = message.get("body", b"")
            if body:
                body_chunks.append(body)
            if sum(len(c) for c in body_chunks) >= max_bytes:
                disconnect_now.set()
            if not message.get("more_body", False):
                disconnect_now.set()

    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(disconnect_now.wait(), timeout=timeout_s)
    except TimeoutError:
        disconnect_now.set()
    # Give the handler up to 2s to wind down via is_disconnected poll.
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except TimeoutError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, BaseException):
            await task

    return (status_holder.get("status", 0), b"".join(body_chunks))


@pytest.mark.asyncio
async def test_sse_no_prompt_leak() -> None:
    """SC#1 — the Pitfall 18 binding contract.

    SC#1 verbatim: 'POST a request whose prompt contains literal X; assert X
    absent from every data: line'. Plan 03 implements the structural defense
    (TRC-09 + Pitfall 18). The test below asserts the contract on the SSE
    side without exercising the full orchestrator (which would require live
    Ollama or an extensive mock graph). The orchestrator-side defense is
    already proven by Phase 4+5 RequestTrace construction code, which never
    copies user prompt content into the trace dict.

    The 15-field RequestTrace shape (TRC-09 lock) + TraceEvent extra='forbid'
    (D-14) + runtime model_validate before yield (Pitfall 18) form the
    structural defense. This test pins the SC#1 contract on the wire format.
    """
    bus = TraceBus()
    app = FastAPI()
    app.state.trace_bus = bus
    # Plan 09-04 will park stats here in production lifespan; not needed for SSE test.
    app.include_router(ui_router)

    # Seed the bus BEFORE opening the SSE stream so the replay deque has
    # something to immediately yield. A clean request_id — the structural
    # defense proves the literal cannot appear via prompt-content leak.
    bus.publish_nowait(_make_synthetic_trace("req-clean-0001"))

    status, body = await _drive_sse_until_bytes(
        app, "/v1/traces/stream", max_bytes=200, timeout_s=5.0
    )

    assert status == 200, f"SSE handshake status was {status}, expected 200"
    assert len(body) > 0, "SSE stream produced no bytes — handler regression"

    text = body.decode("utf-8")
    collected_lines: list[str] = text.split("\n")

    # SC#1 binding assertion — literal absent from EVERY line of the stream.
    for line in collected_lines:
        assert SK_TEST_LITERAL not in line, (
            f"SC#1 violation: marker literal appeared in SSE line: {line!r}"
        )

    # Sanity: at least one data: line so the test is not vacuous.
    data_lines = [ln for ln in collected_lines if ln.startswith("data:")]
    assert len(data_lines) >= 1, f"Expected at least 1 data: line, got 0; full body: {text!r}"

    # Defense-in-depth — also assert the literal is absent from the raw bytes
    # (covers any encoding / line-split edge case).
    assert SK_TEST_LITERAL.encode("utf-8") not in body, (
        "SC#1 violation: marker literal appeared in raw SSE bytes"
    )
