"""FastAPI smoke tests — Phase 1 Success Criteria #3 (API-03) + #4 (API-06 + OBS-02 e2e).

Locks:
- ``GET /health`` returns 200 with the locked ``{status, version, providers}`` shape.
  Phase 2 grew ``providers`` from ``{}`` into a per-provider object with a ``local``
  slot (kind/state/last_error) per D-10; ``status`` is the aggregate ``ok|degraded``.
- ``POST /v1/chat/completions`` and ``GET /v1/models`` return 503 with the verbatim
  CONTEXT.md envelope.
- A typed ``CoreThreadError`` (e.g., ``ProviderTimeout``) returns the correct
  ``http_status`` AND the OpenAI-shape envelope built by ``error_envelope()``.
- A deliberately raised ``RuntimeError`` whose message includes a fake ``sk-...``
  key AND ``Authorization: Bearer ...`` returns the GENERIC envelope (``"Internal
  error"``) AND a regex grep for ``sk-|Authorization: Bearer`` on the response
  body AND captured logger output finds zero unredacted matches (OBS-02 e2e).
- Every error response uses the top-level ``error`` key, NOT FastAPI's default
  ``detail`` key (API-06).

App-construction strategy:
    The FastAPI app reads ``CORETHREAD_CONFIG_PATH`` at lifespan-startup time. We
    set the env var, ``importlib.reload(main)`` so the module re-evaluates, and
    register two test-only routes on the reloaded app before passing it to
    ``TestClient``. The lifespan runs when the test enters the ``with`` block.

Capture strategy:
    The lifespan's ``setup_logging()`` installs a stdout StreamHandler bound to
    the current ``sys.stdout``. We capture by swapping that handler's
    ``.stream`` attribute (NOT by reassigning ``sys.stdout``, which the handler
    captured at construction). Same pattern as ``tests/test_logging.py``.
"""

from __future__ import annotations

import importlib
import io
import logging
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

# Phase 4 lifespan dispatch tests need direct access to provider classes for
# isinstance assertions. Importing here keeps the imports section consolidated;
# the existing app_with_config fixture is unchanged.


# Regex from CONTEXT.md "Specifics" — verbatim shape used in success criterion #4.
# Matches `sk-` followed by >= 5 chars (more permissive than the production filter
# so the test would catch even a developer typo like sk-leak123) OR
# `Authorization: Bearer <something>`.
_LEAK_PATTERN = re.compile(
    r"sk-[A-Za-z0-9_\-]{5,}|Authorization:\s*Bearer\s+\S+",
    re.IGNORECASE,
)

# A fake key with >= 20 chars after sk- so it would be redacted by the production
# filter; we intentionally raise it inside an exception to verify Pitfall #12.
_FAKE_SK = "sk-leak-1234567890ABCDEF"
_FAKE_BEARER_KEY = "sk-bearer-leak-77777890ABCDE"


@pytest.fixture
def app_with_config(valid_yaml_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Reload main with CORETHREAD_CONFIG_PATH pointing at a valid tmpdir YAML
    and add two test-only routes for the exception-handler tests."""
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(valid_yaml_path))
    from corethread import main as m

    importlib.reload(m)

    from corethread.errors import ProviderTimeout

    @m.app.get("/__test_typed_error")
    async def _te() -> None:  # type: ignore[no-untyped-def]
        raise ProviderTimeout("test-provider", 5.0)

    @m.app.get("/__test_generic_error_with_fake_key")
    async def _tg() -> None:  # type: ignore[no-untyped-def]
        raise RuntimeError(f"config: {_FAKE_SK} and Authorization: Bearer {_FAKE_BEARER_KEY}")

    # raise_server_exceptions=False — Starlette's ServerErrorMiddleware re-raises
    # unhandled exceptions out of the test request by default (so the test would
    # see the exception instead of the 500 response from our handler). For
    # SC#4 we MUST observe the response our exception handler produces, not the
    # raw RuntimeError. With raise_server_exceptions=False the response is
    # returned to the test as a real 500 with the generic envelope body.
    yield TestClient(m.app, raise_server_exceptions=False)


@pytest.fixture
def lmstudio_yaml_path(tmp_path: Path, valid_yaml_path: Path) -> Path:
    """Build a minimal lmstudio-kind config YAML in a tmp_path.

    Reuses the structure of valid_yaml_path (the existing ollama-kind fixture)
    but flips local.kind to lmstudio and points base_url at the canonical
    test host `http://lmstudio:1234/v1` so respx routes can intercept.
    """
    yaml_text = (
        "local:\n"
        "  kind: lmstudio\n"
        "  base_url: http://lmstudio:1234/v1\n"
        "  model: granite-3.0-2b-instruct\n"
        "  num_ctx_default: 8192\n"
        "  num_ctx_overrides: {}\n"
        "judge:\n"
        "  model: qwen2.5:7b\n"
        "frontier:\n"
        "  api_key_env: OPENAI_API_KEY\n"
        "  base_url: https://api.openai.com/v1\n"
        "  model: gpt-4o\n"
        "  max_tokens: 512\n"
        "routing:\n"
        "  threshold: 0.7\n"
        '  constraint_prompt: "be concise."\n'
    )
    p = tmp_path / "config.lmstudio.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    return p


def _capture_handler_output_during(
    client: TestClient, fn: Callable[[TestClient], httpx.Response]
) -> tuple[httpx.Response, str]:
    """Enter the TestClient lifespan, swap the StreamHandler's stream, run fn(c),
    return (response, captured_output_string).

    The lifespan installs the handler on enter; we swap its stream right after,
    so anything emitted during the request is captured. Restore on exit.
    """
    buf = io.StringIO()
    with client as c:
        root = logging.getLogger()
        handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        saved = [(h, h.stream) for h in handlers]
        for h in handlers:
            h.stream = buf
        try:
            response = fn(c)
            for h in handlers:
                h.flush()
        finally:
            for h, s in saved:
                h.stream = s
    return response, buf.getvalue()


# ---------------------------------------------------------------------------
# SC#3 — /health returns 200 with locked shape
# ---------------------------------------------------------------------------


def test_health_returns_200_with_locked_shape(
    app_with_config: TestClient, respx_mock: respx.MockRouter
) -> None:
    """SC#3 / API-03: /health returns 200 with {status, version, providers:{local, frontier}}.

    Phase 2 grew providers:{} into a per-provider shape. Phase 4 / D-26 grows
    it further to surface BOTH local AND frontier slots. Top-level keys are
    still locked at {status, version, providers}; status is now an aggregate
    ("ok" if all providers are "ready", else "degraded"); providers always
    has a "local" slot AND (per Phase 4 D-26) a "frontier" slot with
    kind/state/last_error.

    Phase 5 / D-14: the Phase-4 bridging frontier stub has been retired;
    OpenAIProvider reports state="ready" whenever its ctor succeeds (no
    billable liveness probe). For the aggregate to be "ok" the local
    provider must ALSO report "ready", so this test registers a respx
    route for the ollama warmup ping (POST /api/chat returns a canned
    200). OpenAIProvider.warmup is a no-op (D-14) so no outbound respx
    route is needed for api.openai.com.
    """
    # Local-side warmup happiness: mock ollama's POST /api/chat so the
    # lifespan warmup ping completes successfully and local state flips
    # to "ready" (see tests/test_providers_ollama.py::STUB_OLLAMA_RESPONSE
    # for the canonical happy-path payload).
    stub_ollama = {
        "model": "llama3.1:8b",
        "created_at": "2026-04-21T00:00:00Z",
        "message": {"role": "assistant", "content": "ok"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 5,
        "eval_count": 1,
    }
    respx_mock.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json=stub_ollama)
    )
    with app_with_config as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    # Phase 5 / D-14: aggregate is "ok" when local is ready AND frontier is ready.
    assert body["status"] == "ok"
    assert isinstance(body["version"], str) and body["version"]
    # D-26 / D-14: providers has BOTH "local" AND "frontier" slots.
    assert set(body["providers"].keys()) == {"local", "frontier"}
    local = body["providers"]["local"]
    assert local["kind"] == "ollama"
    assert local["state"] == "ready"
    assert local["last_error"] is None
    assert set(local.keys()) == {"kind", "state", "last_error"}
    frontier = body["providers"]["frontier"]
    # Phase 5 / D-14: frontier kind is "openai" (OpenAIProvider.name), state
    # is "ready" (always-ready contract — no liveness probe), last_error None.
    assert frontier["kind"] == "openai"
    assert frontier["state"] == "ready"
    assert frontier["last_error"] is None
    assert set(frontier.keys()) == {"kind", "state", "last_error"}
    # Locked: only those three keys at the top level (extra keys would be a contract change)
    assert set(body.keys()) == {"status", "version", "providers"}


# ---------------------------------------------------------------------------
# Phase 5 / D-13 — chat route is wired; streaming returns OpenAI-style SSE
# ---------------------------------------------------------------------------


def test_chat_completions_route_streams_openai_sse_with_fake_orchestrator(
    app_with_fake_orchestrator: TestClient,
) -> None:
    """POST /v1/chat/completions with ``stream=true`` returns OpenAI-style
    SSE through the real FastAPI route surface.
    """
    with app_with_fake_orchestrator as c:
        from corethread import main as m

        m.app.state.orchestrator = (
            app_with_fake_orchestrator._fake_orchestrator  # type: ignore[attr-defined]
        )
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert '"object":"chat.completion.chunk"' in r.text
    assert "fake-orchestrator-response" in r.text
    assert "data: [DONE]" in r.text


def test_models_endpoint_returns_locked_list_shape_with_local_and_frontier(
    app_with_config: TestClient,
) -> None:
    """SC#4 / API-04 / D-08 + D-09 + D-10: GET /v1/models returns 200 with
    ``{"object": "list", "data": [...]}`` containing the configured local
    AND frontier model ids — with the judge model EXCLUDED per D-08.

    Phase 4 precursor (``test_models_endpoint_stub_returns_503_envelope``)
    asserted that this route was 503-stubbed. Plan 05-04 wired the real
    handler per D-08 (local + frontier, judge EXCLUDED, dedup via
    ``dict.fromkeys``) + D-09 (``owned_by='corethread'``, shared
    ``created`` timestamp from ``app.state.models_created_at``).

    The smoke-surface probe preserves its position in the test file; the
    assertion pivots from 503 stub envelope to the real list-shape lock.
    """
    with app_with_config as c:
        r = c.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
    # D-08: local + frontier model ids present; judge model EXCLUDED.
    ids = {entry["id"] for entry in body["data"]}
    assert "llama3.1:8b" in ids  # local model from valid_yaml_path
    assert "gpt-4o" in ids  # frontier model from valid_yaml_path
    assert "qwen2.5:7b" not in ids, "D-08 violation: judge model leaked into /v1/models"
    # D-09: every entry has owned_by='corethread', object='model', int created.
    for entry in body["data"]:
        assert entry["owned_by"] == "corethread"
        assert entry["object"] == "model"
        assert isinstance(entry["created"], int)


# ---------------------------------------------------------------------------
# API-06 — typed CoreThreadError flows through to envelope + correct status
# ---------------------------------------------------------------------------


def test_typed_error_returns_envelope_with_correct_status(
    app_with_config: TestClient,
) -> None:
    """API-06: typed CoreThreadError -> envelope + http_status from the exception."""
    with app_with_config as c:
        r = c.get("/__test_typed_error")
    assert r.status_code == 504  # ProviderTimeout.http_status
    assert r.json() == {
        "error": {
            "message": "Provider 'test-provider' timed out after 5.0s",
            "type": "provider_timeout",
            "code": "provider_timeout",
        }
    }


# ---------------------------------------------------------------------------
# SC#4 — generic error does not leak sk-/Bearer to body or stdout
# ---------------------------------------------------------------------------


def test_generic_error_does_not_leak_secret_to_body_or_stdout(
    app_with_config: TestClient,
) -> None:
    """SC#4 (CONTEXT.md Specifics + Pitfall #12 + OBS-02 end-to-end):
    deliberately raise an exception with `sk-...` and `Authorization: Bearer ...`
    in its message; the response body must contain zero `sk-` substrings and the
    captured stdout/handler output must contain zero unredacted matches."""
    response, stdout_text = _capture_handler_output_during(
        app_with_config, lambda c: c.get("/__test_generic_error_with_fake_key")
    )

    # 500 with the generic envelope — exc.message NOT echoed
    assert response.status_code == 500
    body = response.json()
    assert body == {
        "error": {
            "message": "Internal error",
            "type": "internal_error",
            "code": "internal_error",
        }
    }

    # SC#4 verbatim: no `sk-` or `Authorization: Bearer ...` in body
    body_text = response.text
    body_hits = _LEAK_PATTERN.findall(body_text)
    assert not body_hits, f"BODY LEAK: {body_hits} in {body_text!r}"

    # SC#4 verbatim: no UNREDACTED matches in captured handler output. The
    # production redaction filter rewrites matches to `***REDACTED***`, so any
    # match here that retains the original key suffix is a real leak. We
    # filter out matches that contain the redaction marker as a sanity guard.
    stdout_hits = [h for h in _LEAK_PATTERN.findall(stdout_text) if "***REDACTED***" not in h]
    assert not stdout_hits, f"STDOUT LEAK: {stdout_hits} in {stdout_text!r}"

    # Defense in depth: the literal fake keys must not appear ANYWHERE in
    # captured output (not just by regex match — by literal substring).
    assert _FAKE_SK not in stdout_text, f"LITERAL LEAK: {_FAKE_SK} in {stdout_text!r}"
    assert _FAKE_BEARER_KEY not in stdout_text, (
        f"LITERAL LEAK: {_FAKE_BEARER_KEY} in {stdout_text!r}"
    )
    assert _FAKE_SK not in body_text, f"LITERAL LEAK: {_FAKE_SK} in body {body_text!r}"


# ---------------------------------------------------------------------------
# API-06 — every error uses top-level `error` key, NOT FastAPI's `detail`
# ---------------------------------------------------------------------------


def test_error_responses_use_error_key_not_fastapi_detail(
    app_with_config: TestClient,
) -> None:
    """API-06: every error response has top-level `error` key, NOT FastAPI's
    default `detail` key. This is the contract that makes the router a true
    drop-in for OpenAI-shaped clients.

    Phase 5 / Plan 05-04 wired ``GET /v1/models`` to a real 200 handler
    (D-08/D-09/D-10), so the Phase-4 case that poked /v1/models expecting a
    4xx no longer applies. The /v1/models 200-shape lock lives in its own
    smoke test (``test_models_endpoint_returns_locked_list_shape_...``).
    Here we keep two body-validation failure paths on the real chat route
    — empty-body POST and an invalid message role, both rendered through the
    sanitized 400 envelope — plus the two test-only probes that
    prove typed + generic exception-handler coverage. All four MUST NOT
    leak FastAPI's default ``detail`` key.
    """
    cases: list[tuple[str, str, dict[str, Any]]] = [
        ("/v1/chat/completions", "post", {"json": {}}),
        (
            "/v1/chat/completions",
            "post",
            {
                "json": {
                    "model": "gpt-4o",
                    "messages": [{"role": "not-a-role", "content": "hi"}],
                }
            },
        ),
        ("/__test_typed_error", "get", {}),
        ("/__test_generic_error_with_fake_key", "get", {}),
    ]
    with app_with_config as c:
        for path, method, kwargs in cases:
            r = c.post(path, **kwargs) if method == "post" else c.get(path, **kwargs)
            assert r.status_code >= 400, f"{path}: expected error status, got {r.status_code}"
            body = r.json()
            assert "error" in body, f"{path}: missing 'error' key in {body}"
            assert "detail" not in body, f"{path}: leaked FastAPI 'detail' key in {body}"


# ---------------------------------------------------------------------------
# Phase 4 / SC#4 — lifespan dispatch routes to the correct provider class
# ---------------------------------------------------------------------------


def test_sc4_lifespan_dispatches_to_ollama_when_local_kind_is_ollama(
    valid_yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SC#4 / D-24: with cfg.local.kind=='ollama' the dispatch creates an
    OllamaProvider on app.state.local_provider; orchestrator is constructed.
    """
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(valid_yaml_path))
    from corethread import main as m

    importlib.reload(m)

    from corethread.orchestrator import Orchestrator
    from corethread.providers.ollama import OllamaProvider
    from corethread.providers.openai import OpenAIProvider

    with TestClient(m.app) as _c:
        assert isinstance(m.app.state.local_provider, OllamaProvider)
        # Phase 5 / D-14: the bridging frontier stub has been retired —
        # OpenAIProvider is the canonical frontier adapter in the "frontier" slot.
        assert isinstance(m.app.state.frontier_provider, OpenAIProvider)
        assert isinstance(m.app.state.orchestrator, Orchestrator)
        # D-02: orchestrator was constructed with the SAME provider instances
        # parked on app.state (no copies, no new constructions).
        assert m.app.state.orchestrator._local is m.app.state.local_provider
        assert m.app.state.orchestrator._frontier is m.app.state.frontier_provider


def test_sc4_lifespan_dispatches_to_lmstudio_when_local_kind_is_lmstudio(
    lmstudio_yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SC#4 / D-24: with cfg.local.kind=='lmstudio' the dispatch creates an
    LMStudioProvider on app.state.local_provider; orchestrator is constructed.

    respx routes mock both LM Studio warmup endpoints (GET /v1/models for
    D-14 model validation + POST /v1/chat/completions for the 1-token cold-
    start ping) so warmup() succeeds and lifespan boots cleanly.
    """
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(lmstudio_yaml_path))

    # Inline payload — deliberately NOT imported from tests.conftest.
    # Plan 04-05 ships `stub_lmstudio_models_response` as a pytest FIXTURE
    # (not a module-level constant), so it cannot be imported here at the
    # `with respx.mock()` scope. Inlining keeps this lifespan test self-
    # contained and decouples it from Plan 04-05's symbol naming.
    models_payload: dict[str, object] = {
        "object": "list",
        "data": [
            {
                "id": "granite-3.0-2b-instruct",
                "object": "model",
                "created": 1234567890,
                "owned_by": "lmstudio",
            },
            {
                "id": "qwen2-vl-7b-instruct",
                "object": "model",
                "created": 1234567890,
                "owned_by": "lmstudio",
            },
        ],
    }
    chat_stub = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "granite-3.0-2b-instruct",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    with respx.mock(assert_all_called=False) as router:
        router.get("http://lmstudio:1234/v1/models").respond(200, json=models_payload)
        router.post("http://lmstudio:1234/v1/chat/completions").respond(200, json=chat_stub)

        from corethread import main as m

        importlib.reload(m)

        from corethread.orchestrator import Orchestrator
        from corethread.providers.lmstudio import LMStudioProvider
        from corethread.providers.openai import OpenAIProvider

        with TestClient(m.app) as _c:
            assert isinstance(m.app.state.local_provider, LMStudioProvider)
            # Phase 5 / D-14: the bridging frontier stub has been retired —
            # OpenAIProvider is the canonical frontier adapter in the "frontier" slot.
            assert isinstance(m.app.state.frontier_provider, OpenAIProvider)
            assert isinstance(m.app.state.orchestrator, Orchestrator)


def test_d_24_lifespan_constructs_orchestrator(app_with_config: TestClient) -> None:
    """D-24 step 5: app.state.orchestrator is set; ctor was keyword-only
    (D-02) so local/frontier are bound to the correct slots.
    """
    from corethread.orchestrator import Orchestrator

    with app_with_config as c:
        from corethread import main as m

        assert hasattr(m.app.state, "orchestrator"), (
            "D-24 violation: app.state.orchestrator missing — lifespan did not "
            "instantiate the Orchestrator after warmup"
        )
        assert isinstance(m.app.state.orchestrator, Orchestrator)
        # Smoke /health to prove the orchestrator's existence didn't break
        # the route surface.
        r = c.get("/health")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Phase 4 / D-26 — /health surfaces local + frontier slots
# ---------------------------------------------------------------------------


def test_d_26_health_reports_local_and_frontier_slots(
    app_with_config: TestClient,
) -> None:
    """D-26: /health.providers has BOTH 'local' AND 'frontier' keys after
    Phase 4. Each slot has the locked {kind, state, last_error} shape.
    """
    with app_with_config as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body["providers"].keys()) == {"local", "frontier"}, (
        f"D-26 violation: expected providers={{local, frontier}}, "
        f"got {set(body['providers'].keys())}"
    )
    for slot_name in ("local", "frontier"):
        slot = body["providers"][slot_name]
        assert set(slot.keys()) == {"kind", "state", "last_error"}, (
            f"D-26 violation: {slot_name} slot has unexpected keys {slot.keys()}"
        )
        assert slot["state"] in {"ready", "warming", "unhealthy"}


def test_d_14_health_aggregate_is_ok_when_openai_frontier_is_ready(
    app_with_config: TestClient, respx_mock: respx.MockRouter
) -> None:
    """D-14 (Phase 5): aggregate status is 'ok' when every provider reports
    state='ready'. OpenAIProvider.health() always returns state='ready' as
    soon as the client constructs successfully (no liveness probe — a probe
    would be slow AND billable); the local warmup against the respx-mocked
    ollama endpoint also reports 'ready'. So /health now flips from the
    Phase-4 'degraded' aggregate (frontier-stub forced it) to 'ok'
    end-to-end — the observable signal that the service is pivot-capable.

    REWRITTEN IN PLACE (not deleted-and-re-added): preserves the 13-test
    baseline count ``grep -c "^def test_\\|^async def test_" == 13``
    that Plan 05-07's +10-delta acceptance criterion depends on.
    """
    # Local-side warmup happiness (ollama POST /api/chat returns a canned 200
    # so local state flips to "ready"). OpenAIProvider.warmup is a no-op per
    # D-14 so no outbound api.openai.com respx route is needed.
    stub_ollama = {
        "model": "llama3.1:8b",
        "created_at": "2026-04-21T00:00:00Z",
        "message": {"role": "assistant", "content": "ok"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 5,
        "eval_count": 1,
    }
    respx_mock.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json=stub_ollama)
    )
    with app_with_config as c:
        r = c.get("/health")
    body = r.json()
    # Phase 5 / D-14: both providers are "ready", so aggregate is "ok".
    assert body["status"] == "ok", (
        f"D-14 violation: expected status='ok' with OpenAIProvider frontier "
        f"(ready state), got {body['status']!r} with providers={body['providers']}"
    )
    assert body["providers"]["frontier"]["kind"] == "openai"
    assert body["providers"]["frontier"]["state"] == "ready"
    assert body["providers"]["frontier"]["last_error"] is None
    assert body["providers"]["local"]["state"] == "ready"


# ---------------------------------------------------------------------------
# Phase 5 / D-13 — /v1/chat/completions route is wired; no more Phase 1 503
# ---------------------------------------------------------------------------


def test_d_13_chat_completions_route_is_wired_not_503(
    app_with_fake_orchestrator: TestClient,
) -> None:
    """D-13 (Phase 5): the chat route body is
    ``return await app.state.orchestrator.handle(request)`` — the Phase 1
    503 stub is retired (Plan 05-04 wired it). This guards against the
    inverse regression from the Phase 4 precursor
    (``test_d_25_chat_completions_route_still_returns_503``, rewritten in
    place here): a future refactor that accidentally re-introduces a
    503-stub body would now break the SC#1 end-to-end contract.

    To avoid an outbound ollama/openai call inside this smoke test, we
    swap in a fake orchestrator and probe the route with ``stream=true``.
    The meaningful invariant is that the response is **not** the Phase-4
    503 stub envelope and is returned as OpenAI-style SSE.
    """
    with app_with_fake_orchestrator as c:
        from corethread import main as m

        m.app.state.orchestrator = (
            app_with_fake_orchestrator._fake_orchestrator  # type: ignore[attr-defined]
        )
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    # D-13: no longer 503; stream=true now returns OpenAI-style SSE.
    assert r.status_code != 503, (
        "D-13 regression: chat route returned the retired Phase-1 503 stub; "
        "Plan 05-04 wired the real orchestrator handler."
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "data: [DONE]" in r.text


# ---------------------------------------------------------------------------
# Phase 4 / D-08 regression guard — lmstudio warmup failure does not crash
# ---------------------------------------------------------------------------


def test_lifespan_warmup_failure_for_lmstudio_kind_does_not_crash_startup(
    lmstudio_yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-08 / Pitfall G regression guard for the NEW lmstudio path.

    With respx returning httpx.ConnectError on /v1/models, LMStudioProvider.warmup()
    raises ProviderUnavailable. The lifespan MUST catch it (per the existing
    Phase 2 D-08 try/except for (ProviderUnavailable, ProviderTimeout)) and
    boot degraded — not re-raise + crash. /health then returns degraded as
    expected. Without this test, a future refactor that narrows the lifespan
    catch-tuple could silently break the lmstudio degraded-start path.
    """
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(lmstudio_yaml_path))

    with respx.mock(assert_all_called=False) as router:
        # ConnectError on the /v1/models GET → LMStudioProvider.warmup()
        # maps it to ProviderUnavailable per D-11 (Phase 2 Research-Delta-3
        # except-clause ordering — Connect* before Timeout*-trio).
        import httpx as _httpx

        router.get("http://lmstudio:1234/v1/models").mock(
            side_effect=_httpx.ConnectError("connection refused")
        )

        from corethread import main as m

        importlib.reload(m)

        # The CRITICAL assertion: lifespan does NOT raise.
        with TestClient(m.app) as c:
            r = c.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    # Local provider is unhealthy because warmup failed.
    assert body["providers"]["local"]["state"] == "unhealthy"
    # last_error is the class name of the typed exception (Pitfall #12 — class only).
    assert body["providers"]["local"]["last_error"] in {
        "ProviderUnavailable",
        "ConnectError",  # if the adapter parks the inner cause
    }


def test_lifespan_lmstudio_warmup_http_error_boots_degraded(
    lmstudio_yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LM Studio warmup HTTP errors should not strand local restart flows."""
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(lmstudio_yaml_path))

    with respx.mock(assert_all_called=False) as router:
        router.get("http://lmstudio:1234/v1/models").respond(
            200,
            json={"data": [{"id": "granite-3.0-2b-instruct"}]},
        )
        router.post("http://lmstudio:1234/v1/chat/completions").respond(
            400,
            json={"error": {"message": "bad warmup"}},
        )

        from corethread import main as m

        importlib.reload(m)

        with TestClient(m.app) as c:
            r = c.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["providers"]["local"]["state"] == "unhealthy"
    assert body["providers"]["local"]["last_error"] in {
        "ProviderHTTPError",
        "HTTPStatusError",
    }


# ===========================================================================
# Phase 5 / Plan 05-07 — D-19 smoke coverage extensions (APPEND-ONLY)
# ===========================================================================
#
# Everything below this banner was added by Plan 05-07 per CONTEXT.md D-19:
# streaming support, /v1/models shape + judge-exclusion (4 tests),
# real chat-completions route smoke (1 test), lifespan frontier-type guard
# (1 test), /health ok-flip (1 test) — 10 tests total. The 13 Phase 4 /
# Plan-05-05-locked baseline tests above this banner are UNCHANGED by this
# plan's diff; Plan 05-05 owns every prior in-place rewrite.


# ---------------------------------------------------------------------------
# Phase 5 — streaming support
# ---------------------------------------------------------------------------


def test_streaming_true_returns_openai_sse_via_fake_orchestrator(
    app_with_fake_orchestrator: TestClient,
) -> None:
    """stream=True returns OpenAI-style SSE via the fake orchestrator."""
    with app_with_fake_orchestrator as c:
        from corethread import main as m

        m.app.state.orchestrator = (
            app_with_fake_orchestrator._fake_orchestrator  # type: ignore[attr-defined]
        )
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    assert r.status_code == 200, r.text
    assert "text/event-stream" in r.headers["content-type"]
    assert '"delta":{"role":"assistant"}' in r.text
    assert "fake-orchestrator-response" in r.text
    assert "data: [DONE]" in r.text


def test_streaming_false_returns_200_via_fake_orchestrator(
    app_with_fake_orchestrator: TestClient,
) -> None:
    """stream=False reaches the normal orchestrator response path."""
    with app_with_fake_orchestrator as c:
        from corethread import main as m

        m.app.state.orchestrator = (
            app_with_fake_orchestrator._fake_orchestrator  # type: ignore[attr-defined]
        )
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "fake-orchestrator-response"


def test_unknown_field_tools_forwarded_through(
    app_with_fake_orchestrator: TestClient,
) -> None:
    """API-01 + D-11 forward-compat: unknown fields (tools=[], reasoning_effort,
    seed, logprobs, etc.) pass through extra="allow" and do NOT trigger a 400.

    If a future refactor accidentally narrows ChatCompletionRequest's
    ``model_config`` from ``extra="allow"`` to ``extra="forbid"`` or
    ``extra="ignore"``, this test fires: tools=[] + reasoning_effort +
    seed would either 422 (forbid) or silently drop (ignore), both of
    which violate API-01.
    """
    with app_with_fake_orchestrator as c:
        from corethread import main as m

        m.app.state.orchestrator = (
            app_with_fake_orchestrator._fake_orchestrator  # type: ignore[attr-defined]
        )
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [],
                "reasoning_effort": "high",
                "seed": 42,
            },
        )
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# Phase 5 / D-08 + D-09 + D-10 — GET /v1/models
# ---------------------------------------------------------------------------


def test_models_endpoint_returns_openai_list_shape(
    app_with_config: TestClient,
) -> None:
    """D-09: GET /v1/models returns {"object": "list", "data": [...]} with
    entries for local + frontier models. Each entry has the D-10 schema
    (id, object="model", created:int, owned_by="corethread"). ModelEntry
    uses extra="forbid" so only the four locked keys may appear per entry.
    """
    with app_with_config as c:
        r = c.get("/v1/models")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list"
    assert "data" in body
    data = body["data"]
    # D-08: dedupe. Default config has local.model != frontier.model, so len==2.
    # If future configs collide them, len==1 is also acceptable.
    assert len(data) in {1, 2}, f"D-08 violation: expected 1 or 2 entries, got {len(data)}"
    for entry in data:
        assert entry["object"] == "model"
        assert entry["owned_by"] == "corethread"
        assert isinstance(entry["created"], int)
        assert isinstance(entry["id"], str) and entry["id"]
        # D-10 strict-forbid: only these four keys per ModelEntry schema
        assert set(entry.keys()) == {"id", "object", "created", "owned_by"}


def test_models_endpoint_excludes_judge(
    app_with_config: TestClient,
) -> None:
    """D-08 binding lock: Judge model is EXCLUDED from /v1/models.

    The judge is an internal control-loop detail, not a routable model — a
    client requesting model=<judge_model> would be meaningless. This overrides
    the ROADMAP SC#4 bullet that reads "primary local + judge + frontier".
    """
    with app_with_config as c:
        r = c.get("/v1/models")
    assert r.status_code == 200, r.text
    body = r.json()
    ids = [entry["id"] for entry in body["data"]]
    # Read the judge model from the live app config — binds to whatever
    # valid_yaml_path fixture produced.
    from corethread import main as m

    judge_model = m.app.state.config.judge.model
    assert judge_model not in ids, (
        f"D-08 violation: judge model {judge_model!r} leaked into /v1/models data: {ids}"
    )


def test_models_created_at_is_stable_within_lifespan(
    app_with_config: TestClient,
) -> None:
    """D-09: `created` is captured ONCE at lifespan time; same value across
    every entry in the list AND stable across repeated GET calls within one
    service lifetime (changes only on restart).
    """
    with app_with_config as c:
        r1 = c.get("/v1/models")
        r2 = c.get("/v1/models")
    assert r1.status_code == 200 and r2.status_code == 200
    # Same value across every entry in r1.
    created_values = [e["created"] for e in r1.json()["data"]]
    assert len(set(created_values)) == 1, (
        f"D-09 violation: expected same `created` across all entries, got {created_values}"
    )
    # Same value across repeated calls.
    assert r1.json() == r2.json(), "D-09 violation: response changed between calls"
    # And it matches app.state.models_created_at.
    from corethread import main as m

    assert hasattr(m.app.state, "models_created_at"), "D-10 violation: models_created_at not set"
    assert r1.json()["data"][0]["created"] == m.app.state.models_created_at


def test_models_endpoint_includes_local_and_frontier_ids(
    app_with_config: TestClient,
) -> None:
    """D-08: data contains cfg.local.model AND cfg.frontier.model (or the
    single entry if they collide)."""
    with app_with_config as c:
        r = c.get("/v1/models")
    assert r.status_code == 200
    ids = {e["id"] for e in r.json()["data"]}
    from corethread import main as m

    expected = {m.app.state.config.local.model, m.app.state.config.frontier.model}
    assert ids == expected, f"D-08 violation: ids={ids}, expected={expected}"


# ---------------------------------------------------------------------------
# Phase 5 / D-13 — real POST /v1/chat/completions route
# ---------------------------------------------------------------------------


def test_real_chat_completions_route_returns_200_via_fake_orchestrator(
    app_with_fake_orchestrator: TestClient,
) -> None:
    """D-13: route body is `return await app.state.orchestrator.handle(request)`.
    Swap a fake orchestrator and verify end-to-end: request validates via
    Pydantic, routes to the fake, response is serialized and returned 200.
    """
    with app_with_fake_orchestrator as c:
        from corethread import main as m

        m.app.state.orchestrator = (
            app_with_fake_orchestrator._fake_orchestrator  # type: ignore[attr-defined]
        )
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "roundtrip"}],
                "max_tokens": 50,
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    # ChatCompletionResponse shape: id/object/created/model/choices/usage
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert body["choices"][0]["message"]["content"] == "fake-orchestrator-response"
    assert body["choices"][0]["finish_reason"] in {"stop", "length", "tool_calls", "content_filter"}
    assert body["usage"]["completion_tokens"] >= 0


# ---------------------------------------------------------------------------
# Phase 5 / D-14 — lifespan frontier is OpenAIProvider, NOT stub
# ---------------------------------------------------------------------------


def test_lifespan_frontier_is_openai_not_stub(
    valid_yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-14: after Plan 05-05 deletes StubFrontierProvider, the lifespan
    constructs OpenAIProvider per Plan 05-04. Verify app.state.frontier_provider
    is an OpenAIProvider instance (not the deleted stub).

    This assertion is the hard landmark that Plan 05-05 cleanup actually
    landed — if someone accidentally re-adds StubFrontierProvider, this
    fires before any semantic test.
    """
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(valid_yaml_path))
    from corethread import main as m

    importlib.reload(m)

    from corethread.providers.openai import OpenAIProvider

    with TestClient(m.app) as _c:
        assert isinstance(m.app.state.frontier_provider, OpenAIProvider), (
            f"D-14 violation: expected OpenAIProvider on app.state.frontier_provider, "
            f"got {type(m.app.state.frontier_provider).__name__}"
        )
        # D-16 preservation: orchestrator still holds the same frontier instance.
        assert m.app.state.orchestrator._frontier is m.app.state.frontier_provider


# ---------------------------------------------------------------------------
# Phase 5 / D-14 — /health aggregate flips degraded → ok
# ---------------------------------------------------------------------------


def test_health_flips_to_ok(valid_yaml_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-14 /health ok-flip: Phase 4 D-26 reported status='degraded' because
    StubFrontierProvider.health() returned unhealthy. Phase 5 OpenAIProvider.
    health() returns state='ready' per D-14, so the aggregate flips to 'ok'
    (assuming local is also ready — tests already cover the local-unhealthy
    case, inverted here for Phase 5).

    Requires respx to stub Ollama warmup so local also reports ready.
    """
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(valid_yaml_path))

    # Stub Ollama warmup routes so local lifespan boots clean.
    ollama_chat_stub = {
        "model": "llama3.1:8b",
        "created_at": "2026-04-21T00:00:00Z",
        "message": {"role": "assistant", "content": "ok"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 1,
        "eval_count": 1,
    }

    with respx.mock(assert_all_called=False) as router:
        router.post("http://localhost:11434/api/chat").respond(
            200,
            json=ollama_chat_stub,
        )

        from corethread import main as m

        importlib.reload(m)

        with TestClient(m.app) as c:
            r = c.get("/health")

    assert r.status_code == 200, r.text
    body = r.json()
    # Phase 5 D-14 invariant: both slots ready → aggregate "ok".
    assert body["status"] == "ok", (
        f"D-14 violation: expected status='ok' with OpenAIProvider frontier, "
        f"got {body['status']!r}; full body={body}"
    )
    # Frontier slot reports ready per D-14.
    assert body["providers"]["frontier"]["state"] == "ready"
    assert body["providers"]["frontier"]["kind"] == "openai"
    # Local slot also ready (ollama warmup succeeded via respx).
    assert body["providers"]["local"]["state"] == "ready"
