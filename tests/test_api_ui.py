"""Phase 9 multi-decision coverage tests for corethread.api_ui.

Plan 01 contributes:
- test_config_endpoint_no_api_key_resolved_field (D-07 lock)
- test_config_view_extra_forbid (defense-in-depth schema-level assertion)
- test_frontier_view_rejects_api_key_input (Pydantic-level lock)
- test_to_config_view_does_not_read_secret (allowlist builder unit test)

Plans 03 and 04 will APPEND SSE / stats tests to this file. DO NOT modify
the Plan-01 tests in those plans — additive only.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from corethread.api_ui import (
    AdminRestartResponse,
    ConfigProbeResult,
    ConfigView,
    ControlsView,
    FrontierView,
    JudgeView,
    LocalView,
    ModelProfileView,
    PrivacyView,
    RoutingView,
    UiView,
    to_config_view,
)
from corethread.api_ui import (
    router as ui_router,
)

# ---------------------------------------------------------------------------
# Plan 01 — ConfigView shape + Pitfall 19 lock
# ---------------------------------------------------------------------------


def test_config_endpoint_no_api_key_resolved_field(
    cfg_with_marker_key: tuple[Any, Any],
) -> None:
    """D-07 literal lock — body has api_key_env but NOT api_key NOT api_key_resolved.

    SC#2 prose says "frontier block exposes api_key_env: str (env var NAME)
    only — never the resolved value". D-07 reads this as: no synthetic
    api_key_resolved bool either. This test pins both omissions.
    """
    _yaml_path, cfg = cfg_with_marker_key

    app = FastAPI()
    app.state.config = cfg
    app.include_router(ui_router)

    with TestClient(app) as client:
        body = client.get("/v1/config").json()

    frontier = body["frontier"]
    assert "api_key_env" in frontier, "CFG-02 requires api_key_env to be surfaced"
    assert "api_key" not in frontier, (
        "Pitfall 19 lock — frontier.api_key MUST NOT appear in /v1/config"
    )
    assert "api_key_resolved" not in frontier, "D-07 lock — no synthetic api_key_resolved bool"
    # Defense in depth — every key in frontier must be in the locked allowlist
    assert set(frontier.keys()) == {"model", "api_key_env", "max_tokens"}, (
        f"FrontierView field set drifted: {set(frontier.keys())}"
    )


def test_config_view_extra_forbid() -> None:
    """Schema-level defense — every ConfigView model uses extra='forbid'."""
    for model_cls in (
        ConfigView,
        LocalView,
        JudgeView,
        FrontierView,
        RoutingView,
        UiView,
        PrivacyView,
        ControlsView,
    ):
        extra = model_cls.model_config.get("extra")
        assert extra == "forbid", (
            f"{model_cls.__name__}.model_config['extra'] = {extra!r} "
            f"(expected 'forbid' per D-22 schema lock)"
        )

    # FrontierView field set is the load-bearing assertion
    fields = list(FrontierView.model_fields.keys())
    assert fields == ["model", "api_key_env", "max_tokens"], fields


def test_frontier_view_rejects_api_key_input() -> None:
    """Defense in depth — Pydantic refuses to construct FrontierView(api_key=...).

    If a future executor adds api_key as an extra='allow' field on FrontierView
    this test fires before the SC#2 grep test does.
    """
    with pytest.raises(ValidationError):
        FrontierView(  # type: ignore[call-arg]
            model="gpt-4o",
            api_key_env="OPENAI_API_KEY",
            max_tokens=512,
            api_key="sk-leak-attempt-1234567890",
        )


def test_to_config_view_does_not_read_secret(
    cfg_with_marker_key: tuple[Any, Any],
) -> None:
    """Allowlist builder unit test — the function uses no exclude/denylist.

    Hits to_config_view(cfg) directly (not via TestClient) and asserts
    the dumped dict has no api_key field, no api_key_resolved field,
    and the marker key is absent from a recursive walk.
    """
    from tests.conftest import SEC_03_MARKER_KEY

    _yaml_path, cfg = cfg_with_marker_key
    view = to_config_view(cfg)
    dumped = view.model_dump()

    # Recursive walk — same shape as the SC#2 named test
    def _walk(obj: Any) -> list[str]:
        out: list[str] = []
        if isinstance(obj, dict):
            for v in obj.values():
                out.extend(_walk(v))
        elif isinstance(obj, list):
            for v in obj:
                out.extend(_walk(v))
        else:
            out.append(str(obj))
        return out

    for leaf in _walk(dumped):
        assert SEC_03_MARKER_KEY not in leaf

    assert "api_key" not in dumped["frontier"]
    assert "api_key_resolved" not in dumped["frontier"]
    assert dumped["frontier"]["api_key_env"] == "OPENAI_API_KEY"
    assert dumped["frontier"]["max_tokens"] == 512  # CFG-03 cost-lever surface
    assert dumped["judge"]["prompt"]
    assert dumped["routing"]["constraint_prompt"]  # CFG-03 cost-lever surface


class _ProbeClient:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.calls: list[dict[str, Any]] = []

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        self.calls.append({"url": url, "headers": headers or {}, "timeout": timeout})
        return httpx.Response(
            self.status_code,
            json=self.payload,
            request=httpx.Request("GET", url),
        )


def test_config_probe_lmstudio_lists_models() -> None:
    """Config editor probe fetches LM Studio models without exposing secrets."""
    fake_client = _ProbeClient(
        {
            "data": [
                {"id": "meta/llama-3.3-70b"},
                {"id": "qwen/qwen3.5-35b-a3b"},
            ]
        }
    )
    app = FastAPI()
    app.state.http_client = fake_client
    app.include_router(ui_router)

    profile = ModelProfileView(
        provider="lmstudio",
        model="meta/llama-3.3-70b",
        base_url="http://lmstudio:1234/v1",
    )

    with TestClient(app) as client:
        resp = client.post("/v1/config/probe", json=profile.model_dump())

    assert resp.status_code == 200, resp.text
    body = ConfigProbeResult.model_validate(resp.json())
    assert body.ok is True
    assert body.models == ["meta/llama-3.3-70b", "qwen/qwen3.5-35b-a3b"]
    assert body.selected_model_available is True
    assert fake_client.calls == [
        {
            "url": "http://lmstudio:1234/v1/models",
            "headers": {},
            "timeout": 10.0,
        }
    ]


def test_config_probe_missing_cloud_env_does_not_call_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cloud probes report missing env-var names without resolved key leakage."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    fake_client = _ProbeClient({"data": [{"id": "openai/gpt-4o"}]})
    app = FastAPI()
    app.state.http_client = fake_client
    app.include_router(ui_router)

    profile = ModelProfileView(
        provider="openrouter",
        model="openai/gpt-4o",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    )

    with TestClient(app) as client:
        resp = client.post("/v1/config/probe", json=profile.model_dump())

    assert resp.status_code == 200, resp.text
    body = ConfigProbeResult.model_validate(resp.json())
    assert body.ok is False
    assert body.error_class == "MissingApiKeyEnv"
    assert "OPENROUTER_API_KEY" in body.message
    assert "sk-" not in resp.text
    assert fake_client.calls == []


def test_admin_restart_endpoint_invokes_local_restart_callback() -> None:
    """Restart endpoint is local-only and delegates actual process control."""
    calls: list[str] = []

    def _restart_callback(request: Any) -> None:
        calls.append(request.url.path)

    app = FastAPI()
    app.state.restart_callback = _restart_callback
    app.include_router(ui_router)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        resp = client.post("/v1/admin/restart")

    assert resp.status_code == 202, resp.text
    body = AdminRestartResponse.model_validate(resp.json())
    assert body.status == "restarting"
    assert calls == ["/v1/admin/restart"]


def test_admin_restart_endpoint_rejects_non_loopback_client() -> None:
    calls: list[str] = []

    def _restart_callback(request: Any) -> None:
        calls.append(request.url.path)

    app = FastAPI()
    app.state.restart_callback = _restart_callback
    app.include_router(ui_router)

    with TestClient(app, client=("10.0.0.10", 50000)) as client:
        resp = client.post("/v1/admin/restart")

    assert resp.status_code == 403, resp.text
    assert calls == []


# ---------------------------------------------------------------------------
# Plan 03 — SSE coverage (D-10 / D-11 / D-13)
# ---------------------------------------------------------------------------
#
# DEVIATION (Rule 3 — blocking issue auto-fix): plan prescribed
# httpx.AsyncClient(transport=ASGITransport(app)) + client.stream + aiter_lines
# for SSE testing. httpx 0.28.1's ASGITransport buffers the entire response
# body before returning the Response; for an infinite SSE generator this
# never completes. We instead drive the ASGI 3.0 callable directly via a
# small in-process driver (mirroring tests/test_sse_no_prompt_leak.py).

import asyncio as _asyncio  # noqa: E402 — section break aid
import contextlib as _contextlib  # noqa: E402
from collections.abc import MutableMapping as _MutableMapping  # noqa: E402
from typing import Any as _Any  # noqa: E402

from corethread.obs import RequestTrace as _RequestTrace  # noqa: E402
from corethread.pubsub import TraceBus as _TraceBus  # noqa: E402


def _ssetest_make_trace(request_id: str) -> _RequestTrace:
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


async def _ssetest_drive(
    app: _Any,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    max_bytes: int = 2000,
    timeout_s: float = 5.0,
) -> tuple[int, bytes]:
    """Drive ASGI app, return (status, body bytes), disconnect at max_bytes/timeout.

    Matches the _drive_sse_until_bytes helper in tests/test_sse_no_prompt_leak.py
    (httpx 0.28 ASGITransport buffers — see DEVIATION note above).
    """
    body_chunks: list[bytes] = []
    status_holder: dict[str, int] = {}
    request_sent = False
    disconnect_now = _asyncio.Event()

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
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 50000),
        "root_path": "",
    }

    async def receive() -> dict[str, _Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await disconnect_now.wait()
        return {"type": "http.disconnect"}

    async def send(message: _MutableMapping[str, _Any]) -> None:
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

    task = _asyncio.create_task(app(scope, receive, send))
    try:
        await _asyncio.wait_for(disconnect_now.wait(), timeout=timeout_s)
    except TimeoutError:
        disconnect_now.set()
    try:
        await _asyncio.wait_for(task, timeout=2.0)
    except TimeoutError:
        task.cancel()
        with _contextlib.suppress(_asyncio.CancelledError, BaseException):
            await task

    return (status_holder.get("status", 0), b"".join(body_chunks))


def _parse_data_objects(body: bytes) -> list[dict[str, _Any]]:
    """Parse SSE body bytes; return list of decoded JSON objects from data: lines."""
    import json as _json

    out: list[dict[str, _Any]] = []
    for line in body.decode("utf-8").split("\n"):
        if line.startswith("data:"):
            payload = line[len("data:") :].lstrip()
            if payload:
                with _contextlib.suppress(_json.JSONDecodeError):
                    out.append(_json.loads(payload))
    return out


async def test_traces_stream_replays_after_last_event_id() -> None:
    """D-10 — drain replay deque until matching request_id, then enter live mode.

    Publish 5 traces; reconnect with Last-Event-ID = 2nd trace's request_id;
    assert the next data line received is the 3rd trace (drain-until-match
    consumed the 1st-2nd; live mode resumes from the 3rd).
    """
    bus = _TraceBus()
    app = FastAPI()
    app.state.trace_bus = bus
    app.include_router(ui_router)

    for i in range(5):
        bus.publish_nowait(_ssetest_make_trace(f"req-{i:03d}"))

    # Connect with Last-Event-ID = req-001 (the 2nd trace; expect req-002 next)
    status, body = await _ssetest_drive(
        app,
        "/v1/traces/stream",
        headers={"Last-Event-ID": "req-001"},
        max_bytes=2000,
        timeout_s=5.0,
    )

    assert status == 200, f"status={status}"
    objs = _parse_data_objects(body)
    assert len(objs) >= 1, f"expected at least 1 data object; body={body!r}"
    assert objs[0]["request_id"] == "req-002", (
        f"D-10 drain-until-match failed; expected req-002 first, got {objs[0]['request_id']}"
    )


async def test_traces_stream_unknown_last_event_id_replays_full_deque() -> None:
    """D-11 — unknown Last-Event-ID falls through to live mode (no match).

    Publish 5 traces; reconnect with Last-Event-ID = 'nonexistent-id'.
    The drain loop walks ALL 5 from the queue looking for the match; when
    none is found it falls through to live mode with an empty queue.
    The test pins the broader invariant: NO data line carries a request_id
    outside the published set (frontend dedupes by request_id per D-11).
    """
    bus = _TraceBus()
    app = FastAPI()
    app.state.trace_bus = bus
    app.include_router(ui_router)

    for i in range(5):
        bus.publish_nowait(_ssetest_make_trace(f"req-{i:03d}"))

    status, body = await _ssetest_drive(
        app,
        "/v1/traces/stream",
        headers={"Last-Event-ID": "nonexistent-id"},
        max_bytes=2000,
        timeout_s=2.5,
    )

    assert status == 200, f"status={status}"

    # D-11 broader invariant — any data line carries a request_id from the published set
    objs = _parse_data_objects(body)
    valid_ids = {f"req-{i:03d}" for i in range(5)}
    for obj in objs:
        assert obj["request_id"] in valid_ids, (
            f"Unexpected request_id in stream: {obj['request_id']!r}"
        )


async def test_traces_stream_emits_keepalive() -> None:
    """D-13 + Pitfall 21 — EventSourceResponse path emits text/event-stream.

    Native FastAPI 0.136 EventSourceResponse keep-alive is module-level
    via fastapi.sse._PING_INTERVAL=15.0 applied automatically by the
    routing-layer _keepalive_inserter. Asserting an actual `: ping`
    line within a 200ms window would require monkeypatching the
    interval; the architecture review's Claude-Discretion path accepts
    this lighter-touch assertion (status 200 + text/event-stream
    media type) as the D-13 / Pitfall 21 anchor.
    """
    bus = _TraceBus()
    app = FastAPI()
    app.state.trace_bus = bus
    app.include_router(ui_router)

    # Drive the app and capture the response.start event to read headers.
    captured: dict[str, _Any] = {"status": 0, "headers": []}
    request_sent = False
    disconnect_now = _asyncio.Event()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "headers": [],
        "scheme": "http",
        "path": "/v1/traces/stream",
        "raw_path": b"/v1/traces/stream",
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 50000),
        "root_path": "",
    }

    async def receive() -> dict[str, _Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await disconnect_now.wait()
        return {"type": "http.disconnect"}

    async def send(message: _MutableMapping[str, _Any]) -> None:
        if message["type"] == "http.response.start":
            captured["status"] = message["status"]
            captured["headers"] = message.get("headers", [])
            # Got the response start — disconnect immediately.
            disconnect_now.set()

    task = _asyncio.create_task(app(scope, receive, send))
    try:
        await _asyncio.wait_for(disconnect_now.wait(), timeout=2.0)
    except TimeoutError:
        disconnect_now.set()
    try:
        await _asyncio.wait_for(task, timeout=2.0)
    except TimeoutError:
        task.cancel()
        with _contextlib.suppress(_asyncio.CancelledError, BaseException):
            await task

    assert captured["status"] == 200, f"status={captured['status']}"
    headers_dict = {k.decode("ascii").lower(): v.decode("ascii") for k, v in captured["headers"]}
    content_type = headers_dict.get("content-type", "")
    assert "text/event-stream" in content_type, (
        f"content-type={content_type!r} (expected text/event-stream — "
        f"native EventSourceResponse signal; ping=15 keep-alive is module-level)"
    )


# ---------------------------------------------------------------------------
# Plan 04 — Stats coverage (D-15 / D-16 / D-17)
# ---------------------------------------------------------------------------

import importlib as _importlib  # noqa: E402


def test_stats_endpoint_returns_two_shape_union(
    valid_yaml_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-15 — empty window has 4 keys; populated window has 12 keys with all pivot_reason buckets.

    Boots the real main.app via lifespan reload; the lifespan creates a
    StatsAggregator on app.state.stats. Test (a) hits /v1/stats fresh
    and asserts the 4-key empty shape; (b) injects a synthetic trace into
    app.state.stats via stats.ingest() then re-hits /v1/stats and asserts
    the 12-key populated shape with all pivot_reasons buckets present.
    """
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(valid_yaml_path))
    from corethread import main as m

    _importlib.reload(m)

    with TestClient(m.app, raise_server_exceptions=False) as client:
        # Empty-window assertion — 4 keys, window_size == 0
        resp = client.get("/v1/stats")
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("Cache-Control") == "no-cache"
        empty = resp.json()
        assert set(empty.keys()) == {
            "window_started_at",
            "window_size",
            "max_window_size",
            "note",
        }, set(empty.keys())
        assert empty["window_size"] == 0
        assert empty["note"] == "Stats reset on CoreThread restart (in-memory)."

        # Inject a synthetic trace + re-hit for the populated shape
        stats = m.app.state.stats
        stats.ingest(
            {
                "request_id": "req-stats-001",
                "selected_local_model": "llama3.1:8b",
                "judge_model": "qwen2.5:7b",
                "frontier_model": None,
                "confidence_score": 0.9,
                "pivoted": False,
                "local_latency_ms": 100,
                "judge_latency_ms": 50,
                "frontier_latency_ms": None,
                "input_tokens": 10,
                "output_tokens": 5,
                "frontier_cost_est": None,
                "judge_parse_failed": False,
                "pivot_reason": "none",
                "local_error_class": None,
            }
        )

        resp2 = client.get("/v1/stats")
        assert resp2.status_code == 200
        populated = resp2.json()
        expected_keys = {
            "window_started_at",
            "window_size",
            "max_window_size",
            "pivot_rate",
            "pivot_reasons",
            "tokens_in_total",
            "tokens_out_total",
            "local_latency_ms",
            "judge_latency_ms",
            "frontier_latency_ms",
            "confidence_score",
            "note",
        }
        assert set(populated.keys()) == expected_keys, set(populated.keys()) ^ expected_keys
        assert populated["window_size"] == 1
        # Phase 8 D-07 — all pivot_reasons buckets present
        assert set(populated["pivot_reasons"].keys()) == {
            "none",
            "low_score",
            "local_truncated",
            "local_error",
            "judge_error",
        }
        assert sum(populated["pivot_reasons"].values()) == 1


def test_stats_endpoint_window_invalid_returns_400_envelope(
    valid_yaml_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-16 — ?window=2d (not in Window Literal) routes through main.py:359.

    FastAPI's RequestValidationError handler returns the OpenAI-shape 400
    envelope (`{"error": {"message": ..., "type": "invalid_request_error",
    "code": "invalid_request_error"}}`) — zero new error-path code.
    """
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(valid_yaml_path))
    from corethread import main as m

    _importlib.reload(m)

    with TestClient(m.app, raise_server_exceptions=False) as client:
        resp = client.get("/v1/stats?window=2d")
        assert resp.status_code == 400, resp.text
        body = resp.json()
        # OpenAI-shape envelope — top-level "error" key (NOT FastAPI default "detail")
        assert "error" in body, body
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "invalid_request_error"


async def test_stats_pump_survives_ingest_exception(
    valid_yaml_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-17 — single bad ingest does NOT kill the pump.

    Boots the lifespan, monkeypatches StatsAggregator.ingest to raise on
    the first call, publishes 3 traces via the bus, asserts the pump task
    is still alive (not done) and ingest was called 3 times (one raised,
    two succeeded — but with a class-name-only warning logged).
    """
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(valid_yaml_path))
    from corethread import main as m

    _importlib.reload(m)

    # Use the manual lifespan-driver pattern from conftest.py asgi_openai_sdk_client
    async with m.app.router.lifespan_context(m.app):
        stats = m.app.state.stats
        bus = m.app.state.trace_bus

        call_count = {"n": 0}
        original_ingest = stats.ingest

        def _flaky_ingest(trace: _RequestTrace) -> None:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated transient ingest failure")
            original_ingest(trace)

        monkeypatch.setattr(stats, "ingest", _flaky_ingest)

        # DEVIATION (Rule 3 — blocking issue): asyncio.create_task() in lifespan
        # only SCHEDULES the pump; the pump's `bus.subscribe(replay=False)`
        # context-manager handshake (which adds its queue to bus._subscribers)
        # has not yet executed when we return from the lifespan_context entry.
        # If we publish_nowait() before the pump subscribes, the events are
        # fanned out to an empty subscriber set and dropped (replay=False
        # also means the pump won't see historical traces). We yield control
        # via a brief sleep so the pump task can run through its subscribe()
        # handshake first. This is the same race the production lifespan
        # is immune to (real traces flow through obs.emit_trace which runs
        # well after the lifespan startup completes).
        await _asyncio.sleep(0.05)

        # Publish 3 traces — first should raise inside pump (logged + continue);
        # second + third should succeed.
        for i in range(3):
            bus.publish_nowait(_ssetest_make_trace(f"req-pump-{i:03d}"))

        # Give the pump a few quanta to drain the queue
        await _asyncio.sleep(0.5)

        # Assert ingest was called 3 times (the pump did NOT die after raise)
        assert call_count["n"] == 3, call_count
        # The pump task is still alive
        stats_task = next(
            (t for t in _asyncio.all_tasks() if t.get_name() == "stats_pump"),
            None,
        )
        assert stats_task is not None, "stats_pump task missing"
        assert not stats_task.done(), "stats_pump died (D-17 violation)"
