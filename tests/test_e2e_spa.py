"""SC#5 milestone-closer E2E test (Phase 11 / Plan 04 / D-14 + D-15).

Boots `corethread.main:app` via FastAPI TestClient (in-process — D-14: no
uvicorn subprocess; matches Phase 5 test_e2e.py and Phase 9 test_sse_no_prompt_leak.py).
Pytest fixture monkeypatches `app.state.orchestrator` to a stub (never actually
called — the test exercises mounted endpoints + bus-replay only) and emits 5
scripted RequestTrace events via `obs.emit_trace()` — the canonical Phase 7
fan-out point publishes through `_TRACE_BUS` that the lifespan wired at startup.

Asserts (D-14 verbatim — 5 distinct surface areas closed in 1 file):
  (a) GET / returns SPA index.html with <div id="root"></div> Vite marker + Cache-Control: no-cache
  (b) GET /v1/traces/stream yields all 5 scripted events (parsed via ASGI-direct SSE driver)
  (c) GET /v1/config returns valid JSON containing no sk- substring + Cache-Control: no-cache
  (d) GET /assets/<discovered>.js returns Cache-Control: public, max-age=31536000, immutable
  (e) GET /assets/<discovered>.js Content-Type is application/javascript or text/javascript (D-07)

Module-level skip (D-15) when frontend/dist/index.html is missing locally;
CI's frontend-check job runs `pnpm --dir frontend build` before pytest so the
skip never fires there.

Test count delta: +1 file with 1 test function (CONTEXT.md D-21: 311 → 312).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from corethread.obs import RequestTrace

# D-15: module-level skip if frontend/dist/ is missing locally. CI builds first.
pytestmark = pytest.mark.skipif(
    not (Path("frontend/dist") / "index.html").exists(),
    reason="frontend/dist/ not built — run 'pnpm --dir frontend build' first",
)


def _make_trace(request_id: str) -> RequestTrace:
    """15-field RequestTrace builder — mirrors tests/test_pubsub.py:30-53.

    All fields populated to match the TraceEvent extra='forbid' shape (TRC-09 lock,
    Phase 9 / Plan 03). Values are deterministic for E2E reproducibility.
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


@pytest.fixture
def e2e_client_with_traces(
    valid_yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Boot main.app and emit 5 scripted traces via obs.emit_trace().

    Real lifespan wires obs._TRACE_BUS, app.state.trace_bus, app.state.stats,
    app.state.config — all four are needed by /v1/traces/stream + /v1/config +
    /v1/stats endpoints. Plan 11-02's lifespan-time SPA mount is also active
    because the lifespan runs (frontend/dist/ exists per the module-level skipif).

    Mirrors `tests/conftest.py::app_with_fake_orchestrator` (lines 456-509):
    monkeypatch.setenv → importlib.reload → with TestClient(...) → emit traces.
    """
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(valid_yaml_path))
    from corethread import main as m
    from corethread import obs

    importlib.reload(m)

    with TestClient(m.app, raise_server_exceptions=False) as client:
        # Emit 5 scripted traces via the canonical fan-out point. Phase 7
        # obs.emit_trace tees into _TRACE_BUS, which lifespan wired at boot
        # (corethread/main.py:359). The /v1/traces/stream handler subscribes
        # with replay=True, so all 5 are visible to the SSE consumer.
        for i in range(5):
            obs.emit_trace(_make_trace(f"req-e2e-{i:03d}"))
        yield client


async def _drive_sse_until_bytes(
    app: Any,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    max_bytes: int = 2000,
    timeout_s: float = 5.0,
) -> tuple[int, bytes]:
    """In-process ASGI-direct SSE driver — mirrors test_sse_no_prompt_leak.py:70-149.

    httpx 0.28 ASGITransport buffers the entire response body before returning,
    so an infinite SSE generator never completes through TestClient.get(). This
    helper drives the ASGI 3.0 protocol directly, capturing chunks as they're
    sent and signaling http.disconnect once enough bytes have been collected.

    Inlined here (rather than imported from test_sse_no_prompt_leak.py /
    test_api_ui.py) because both source helpers are private (underscore-prefixed)
    and importing private helpers across test files is fragile. This is the
    third reuse of the project pattern, consistent with the project standard.
    """
    body_chunks: list[bytes] = []
    status_holder: dict[str, int] = {}
    request_sent = False
    disconnect_now = asyncio.Event()

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

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
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


def _parse_sse_data_objects(body: bytes) -> list[dict[str, Any]]:
    """Parse SSE body bytes; return list of decoded JSON objects from data: lines.

    Mirrors tests/test_api_ui.py:249-262 _parse_data_objects.
    """
    out: list[dict[str, Any]] = []
    for line in body.decode("utf-8", errors="replace").split("\n"):
        if line.startswith("data:"):
            payload = line[len("data:") :].lstrip()
            if payload:
                with contextlib.suppress(json.JSONDecodeError):
                    out.append(json.loads(payload))
    return out


def test_e2e_spa_milestone_close(e2e_client_with_traces: TestClient) -> None:
    """SC#5 — full E2E milestone closer (5 distinct assertion areas).

    (a) GET / returns SPA index.html (contains <div id="root"></div>) + Cache-Control: no-cache
    (b) GET /v1/traces/stream yields all 5 scripted events
    (c) GET /v1/config returns valid JSON, no sk- substring + Cache-Control: no-cache
    (d) GET /assets/<discovered>.js returns Cache-Control: public, max-age=31536000, immutable
    (e) GET /assets/<discovered>.js Content-Type starts with application/javascript (D-07)
    """
    client = e2e_client_with_traces

    # ---- (a) SPA index.html on / with no-cache (Plan 11-02 D-06) ---------------
    resp_root = client.get("/")
    assert resp_root.status_code == 200, (
        f"SC#5(a): GET / returned {resp_root.status_code}, expected 200. "
        f"Body (first 200): {resp_root.text[:200]!r}"
    )
    assert '<div id="root"></div>' in resp_root.text, (
        "SC#5(a): GET / body missing Vite marker '<div id=\"root\"></div>' — "
        "frontend/dist/index.html may not be the produced bundle"
    )
    assert resp_root.headers["Cache-Control"] == "no-cache", (
        f"SC#5(a) / Plan 11-02 D-06: GET / Cache-Control = "
        f"{resp_root.headers.get('Cache-Control')!r}, expected 'no-cache'"
    )

    # ---- (b) SSE stream — 5 scripted events (Phase 9 D-10 replay-from-bus) ----
    # The lifespan-wired _TRACE_BUS replays the 5 emitted traces to any new
    # subscriber (replay=True in api_ui.py:288). The ASGI-direct driver drains
    # bytes until either max_bytes hit or 5s timeout, then signals disconnect.
    from corethread import main as m

    status, body = asyncio.run(
        _drive_sse_until_bytes(m.app, "/v1/traces/stream", max_bytes=4000, timeout_s=5.0)
    )
    assert status == 200, f"SC#5(b): /v1/traces/stream handshake status={status}, expected 200"

    parsed = _parse_sse_data_objects(body)
    parsed_ids = {obj.get("request_id") for obj in parsed}
    expected_ids = {f"req-e2e-{i:03d}" for i in range(5)}
    assert expected_ids.issubset(parsed_ids), (
        f"SC#5(b): expected request_ids {sorted(expected_ids)} not all present "
        f"in SSE stream. Got: {sorted(x for x in parsed_ids if x is not None)}. "
        f"Body (first 400): {body[:400]!r}"
    )

    # ---- (c) /v1/config no sk- substring + Cache-Control: no-cache ------------
    resp_config = client.get("/v1/config")
    assert resp_config.status_code == 200, (
        f"SC#5(c): /v1/config returned {resp_config.status_code}, expected 200"
    )
    # CFG-02 / SEC-03 carry-forward: no resolved API key, anywhere in the body.
    assert "sk-" not in resp_config.text, (
        f"SC#5(c) / CFG-02 / SEC-03: /v1/config response contains 'sk-' substring. "
        f"Body (first 300): {resp_config.text[:300]!r}"
    )
    # Phase 9 D-09 carry-forward: deploys instantly invalidate config.
    assert resp_config.headers["Cache-Control"] == "no-cache", (
        f"SC#5(c): /v1/config Cache-Control = "
        f"{resp_config.headers.get('Cache-Control')!r}, expected 'no-cache'"
    )

    # ---- (d) Hashed asset has immutable cache header (Plan 11-02 D-06) ---------
    # Discover an actual hashed JS file in frontend/dist/assets/ — Vite hashes
    # are non-deterministic across builds, so we walk the directory rather than
    # pin a specific hash. The skipif at module level guarantees dist/ exists.
    asset_dir = Path("frontend/dist/assets")
    sample_js = next(asset_dir.glob("*.js"), None)
    assert sample_js is not None, (
        f"SC#5(d): no .js file found in {asset_dir}. "
        f"Either pnpm build did not produce hashed JS bundles, or the dist "
        f"layout has changed (Vite 7 default is assets/<name>-<hash>.js)."
    )
    resp_asset = client.get(f"/assets/{sample_js.name}")
    assert resp_asset.status_code == 200, (
        f"SC#5(d): /assets/{sample_js.name} returned {resp_asset.status_code}, expected 200"
    )
    assert resp_asset.headers["Cache-Control"] == "public, max-age=31536000, immutable", (
        f"SC#5(d) / Plan 11-02 D-06: /assets/{sample_js.name} Cache-Control = "
        f"{resp_asset.headers.get('Cache-Control')!r}, "
        f"expected 'public, max-age=31536000, immutable'"
    )

    # ---- (e) D-07 mimetypes.add_type — .js gets the right Content-Type --------
    # The module-level mimetypes.add_type calls in corethread/main.py register
    # .mjs/.wasm/.map BEFORE app instantiation (Pitfalls 29 + 30). For .js, the
    # default Python mimetypes db returns either `application/javascript` or
    # `text/javascript` depending on the platform — both are acceptable for
    # browser-loadable JS bundles.
    content_type = resp_asset.headers.get("content-type", "")
    assert content_type.startswith(("application/javascript", "text/javascript")), (
        f"SC#5(e) / Plan 11-02 D-07: /assets/{sample_js.name} Content-Type = "
        f"{content_type!r}, expected application/javascript or text/javascript"
    )
