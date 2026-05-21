"""Tests for providers/ollama.py — respx-driven transport + semantics tests.

Respx strategy:
- respx.mock (via the pytest-respx `respx_mock` fixture) patches httpx's
  transport globally. The `ollama_http_client` fixture yields a fresh
  AsyncClient inside the test's event loop so respx's patch takes effect
  (Landmine 5 mitigation).
- Routes are registered against `http://ollama:11434/api/chat` to match
  the `cfg_ollama` fixture's base_url.

Locks (by decision):
- D-01: POST targets `/api/chat`, NOT `/api/generate`; body has `messages`
  (list), not `prompt` (string).
- D-02 / Delta-1: OllamaChatResponse accepts done_reason Optional; schema
  drift raises ProviderHTTPError(200, field_paths).
- D-03: `options={"format": ...}` forwards `format` opaquely in the body.
- D-09: 10 concurrent first-requests trigger exactly 1 warmup (double-
  checked lock). Warmup failure does NOT stick — next request retries
  (Pitfall G).
- D-12: Warmup body shape (num_predict=1, num_ctx=<configured>, hi message).
- D-13: keep_alive=-1 on EVERY request (chat + warmup).
- D-14: response.model == request.model (echo, not backend).
- D-15: system_fingerprint == f"fp_corethread_{request.model[:10]}".
- D-16: to_openai_chat_completion is the only response path — grep-gate
  + assertion that resp.created is int, resp.id starts with "chatcmpl-".
- D-17: num_ctx_overrides[backend_model] wins; fallback to num_ctx_default.
- D-18: Heuristic warning emitted when estimated_tokens > num_ctx AND
  prompt_eval_count < estimated_tokens * 0.9.
- D-19: ConnectError/ConnectTimeout -> ProviderUnavailable (503);
  Read/Write/PoolTimeout -> ProviderTimeout (504); HTTPStatusError -> ProviderHTTPError;
  ValidationError on response -> ProviderHTTPError(200, field_paths) with
  field paths only (Pitfall E).
- D-08: Lifespan catches warmup ProviderUnavailable/ProviderTimeout,
  logs warmup.failed, proceeds to boot. /health returns "degraded".
- D-16 grep-gate: ChatCompletionResponse( never appears in the adapter.

Tests that require log capture use the `caplog` pytest fixture — structlog
is configured in Phase 1 to route through stdlib logging
(`structlog.stdlib.LoggerFactory`), so `caplog.records` captures the
emitted entries. Each log-capture test calls ``setup_logging()`` first so
the structlog->stdlib pipe is live, then sets caplog to the right level.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import textwrap
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from corethread.errors import ProviderHTTPError, ProviderTimeout, ProviderUnavailable
from corethread.models import ChatCompletionRequest, ChatMessage, LocalConfig
from corethread.providers.ollama import (
    OllamaChatResponse,
    OllamaProvider,
    _build_ollama_payload,
)

# Canonical Ollama /api/chat success body — mirrored from conftest
# STUB_OLLAMA_RESPONSE so tests can reference it directly without a
# cross-file import dance. Kept in sync with conftest.STUB_OLLAMA_RESPONSE.
STUB_OLLAMA_RESPONSE: dict[str, Any] = {
    "model": "llama3.1:8b",
    "created_at": "2026-04-21T00:00:00Z",
    "message": {"role": "assistant", "content": "ok"},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 5,
    "eval_count": 1,
}


# ---------------------------------------------------------------------------
# Helpers — used across multiple tests
# ---------------------------------------------------------------------------


def _basic_request(model: str = "gpt-4o", content: str = "hi") -> ChatCompletionRequest:
    """Construct a minimal one-turn ChatCompletionRequest."""
    return ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role="user", content=content)],
    )


def _setup_logging_for_caplog() -> None:
    """Ensure structlog -> stdlib pipe is live so caplog captures events."""
    from corethread.logging_config import setup_logging

    setup_logging(level="DEBUG")


# ===========================================================================
# Success Criterion #1 — populated ChatCompletionResponse (OpenAI shape)
# ===========================================================================


async def test_chat_happy_path_returns_openai_shape(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """SC#1 + D-14 + D-15 + D-16 + Delta-2: happy path returns a fully
    populated OpenAI-shape response. Model echoes request.model (not
    backend cfg.local.model). created is an int synthesized by
    to_openai_chat_completion (Delta-2). id has chatcmpl- prefix.
    system_fingerprint = f"fp_corethread_{request.model[:10]}" (D-15).
    """
    body = {**STUB_OLLAMA_RESPONSE, "prompt_eval_count": 12, "eval_count": 3}
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=body)
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True  # skip lazy warmup
    req = _basic_request(model="gpt-4o")

    resp = await provider.chat(req)

    assert route.called
    # D-14 echo: response.model is request.model (NOT backend llama3.1:8b)
    assert resp.model == "gpt-4o"
    # D-15: system_fingerprint = f"fp_corethread_{request.model[:10]}"
    assert resp.system_fingerprint == "fp_corethread_gpt-4o"
    # Delta-2: created synthesized as int by the helper
    assert isinstance(resp.created, int)
    # D-16: id has chatcmpl- prefix from to_openai_chat_completion
    assert resp.id.startswith("chatcmpl-")
    # Usage wiring: prompt_eval_count -> prompt_tokens, eval_count -> completion_tokens
    assert resp.usage.prompt_tokens == 12
    assert resp.usage.completion_tokens == 3
    assert resp.usage.total_tokens == 15
    # finish_reason: "stop" -> "stop" via OLLAMA_FINISH_MAP
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].message.content == "ok"


# ===========================================================================
# Success Criterion #2 — /api/chat (not /api/generate), multi-turn support
# ===========================================================================


async def test_chat_uses_api_chat_endpoint_not_generate(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-01: POST goes to /api/chat, NOT /api/generate. Body has a messages
    list (not a flattened prompt string), preserving multi-turn history."""
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=STUB_OLLAMA_RESPONSE)
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True
    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[
            ChatMessage(role="user", content="My name is Alice."),
            ChatMessage(role="assistant", content="Nice to meet you."),
            ChatMessage(role="user", content="What is my name?"),
        ],
    )

    await provider.chat(req)

    assert route.called
    captured = route.calls[0].request
    assert captured.url.path == "/api/chat"
    body = json.loads(captured.content.decode("utf-8"))
    # Multi-turn support: messages list, NOT a flattened prompt
    assert "prompt" not in body
    assert isinstance(body["messages"], list)
    assert len(body["messages"]) == 3
    assert body["messages"][0]["content"] == "My name is Alice."
    assert body["messages"][2]["role"] == "user"


# ===========================================================================
# Success Criterion #3 — num_ctx explicit (Pitfall #4 defense)
# ===========================================================================


async def test_chat_sends_num_ctx_from_config_default(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-17: when num_ctx_overrides has no match, fallback is num_ctx_default
    (8192 here). The body ALWAYS carries an explicit options.num_ctx —
    Pitfall #4 defense against Ollama's silent 4096 truncation."""
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=STUB_OLLAMA_RESPONSE)
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    await provider.chat(_basic_request())

    body = json.loads(route.calls[0].request.content.decode("utf-8"))
    assert body["options"]["num_ctx"] == cfg_ollama.num_ctx_default
    assert body["options"]["num_ctx"] == 8192


async def test_chat_num_ctx_override_by_backend_model_name(
    respx_mock: respx.MockRouter,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-17: exact-match override on the BACKEND model name (cfg.local.model).
    No prefix / family fallback. No case-folding. Here cfg.model="llama3.1:70b"
    and override for that exact key is 16384 — must win over 8192 default."""
    cfg = LocalConfig(
        kind="ollama",
        base_url="http://ollama:11434",
        model="llama3.1:70b",
        num_ctx_default=8192,
        num_ctx_overrides={"llama3.1:70b": 16384},
    )
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=STUB_OLLAMA_RESPONSE)
    )
    provider = OllamaProvider(cfg, ollama_http_client)
    provider._warmed = True

    await provider.chat(_basic_request())

    body = json.loads(route.calls[0].request.content.decode("utf-8"))
    assert body["options"]["num_ctx"] == 16384


# ===========================================================================
# Success Criterion #4 — warmup + keep_alive=-1 (D-12 + D-13)
# ===========================================================================


async def test_warmup_sends_keep_alive_minus_one_and_num_predict_one(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-12 + D-13: warmup POSTs /api/chat with num_predict=1, explicit
    num_ctx, keep_alive=-1, stream=False, and a one-message {"role":"user",
    "content":"hi"} messages list. On success, _warmup_state becomes "ready".
    """
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=STUB_OLLAMA_RESPONSE)
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)

    await provider.warmup()

    assert route.called
    body = json.loads(route.calls[0].request.content.decode("utf-8"))
    assert body["keep_alive"] == -1  # D-13
    assert body["options"]["num_predict"] == 1  # D-12
    assert body["options"]["num_ctx"] == 8192  # D-17 default
    assert body["messages"] == [{"role": "user", "content": "hi"}]  # D-12
    assert body["stream"] is False  # v1 never streams
    # State transitions: initial unhealthy -> warming -> ready on success
    assert provider._warmup_state == "ready"
    assert provider._last_warmup_error is None


async def test_chat_sends_keep_alive_minus_one(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-13: EVERY chat body carries keep_alive=-1, hardcoded — never from
    OLLAMA_KEEP_ALIVE env var (the weakest override)."""
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=STUB_OLLAMA_RESPONSE)
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    await provider.chat(_basic_request())

    body = json.loads(route.calls[0].request.content.decode("utf-8"))
    assert body["keep_alive"] == -1


# ===========================================================================
# Success Criterion #5 — config-swappable provider dispatch (LOC-05)
# ===========================================================================


@pytest.fixture
def yaml_path_ollama_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a valid config.yaml with base_url=http://ollama:11434 (so
    respx_mock can intercept it) and return the path. Distinct from
    conftest.valid_yaml_path which uses localhost."""
    yaml_body = (
        textwrap.dedent(
            """
        local:
          kind: ollama
          base_url: http://ollama:11434
          model: llama3.1:8b
          num_ctx_default: 8192
          num_ctx_overrides: {}
        judge:
          model: qwen2.5:7b
        frontier:
          api_key_env: OPENAI_API_KEY
          base_url: https://api.openai.com/v1
          model: gpt-4o
          max_tokens: 512
        routing:
          threshold: 0.7
          constraint_prompt: Be concise.
        """
        ).strip()
        + "\n"
    )
    p = tmp_path / "config.yaml"
    p.write_text(yaml_body, encoding="utf-8")
    return p


def test_provider_dispatch_from_config_is_ollama(
    respx_mock: respx.MockRouter,
    yaml_path_ollama_host: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOC-05: lifespan dispatches cfg.local.kind="ollama" to OllamaProvider
    with zero code changes here. Verified via app.state.local_provider
    being an OllamaProvider instance AND /health reporting providers.local.kind
    == "ollama"."""
    # Mock warmup so TestClient lifespan completes cleanly (D-08 would boot
    # degraded on failure, but we want to verify the ready-path wiring too).
    respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=STUB_OLLAMA_RESPONSE)
    )
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(yaml_path_ollama_host))
    from corethread import main as m

    importlib.reload(m)
    with TestClient(m.app) as client:
        assert isinstance(m.app.state.local_provider, OllamaProvider)
        assert m.app.state.local_provider.name == "ollama"
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["providers"]["local"]["kind"] == "ollama"


# ===========================================================================
# D-09 — lazy double-checked warmup (concurrency correctness)
# ===========================================================================


async def test_lazy_warmup_called_once_for_concurrent_first_requests(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-09: 10 concurrent first-requests against a cold provider trigger
    EXACTLY ONE warmup call. The double-checked lock collapses the fan-in."""
    # Single route matches both warmup and chat (both POST /api/chat).
    # Count the num_predict=1 subset to isolate warmups.
    respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=STUB_OLLAMA_RESPONSE)
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    assert provider._warmed is False

    # Fan out 10 concurrent first-requests.
    await asyncio.gather(*(provider.chat(_basic_request()) for _ in range(10)))

    # Count warmups: num_predict=1 in the options dict.
    warmup_count = 0
    for call in respx_mock.calls:
        body = json.loads(call.request.content.decode("utf-8"))
        if body.get("options", {}).get("num_predict") == 1:
            warmup_count += 1
    assert warmup_count == 1, f"expected exactly 1 warmup, got {warmup_count}"
    assert provider._warmed is True


async def test_warmup_failure_does_not_stick_warmed_stays_false(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """Pitfall G: a failed warmup must NOT mark _warmed=True. Lazy warmup
    inside chat() calls warmup() which raises ProviderUnavailable on
    ConnectError; _warmed stays False so the next request retries."""
    respx_mock.post("http://ollama:11434/api/chat").mock(side_effect=httpx.ConnectError("down"))
    provider = OllamaProvider(cfg_ollama, ollama_http_client)

    with pytest.raises(ProviderUnavailable):
        await provider.chat(_basic_request())

    assert provider._warmed is False


async def test_second_chat_after_warmup_failure_retries_warmup(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """Extension of Pitfall G: second chat call MUST re-attempt warmup after
    the first failed. Asserted by hitting the mock N+1 times across two
    failing requests (one warmup attempt each)."""
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        side_effect=httpx.ConnectError("down")
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)

    with pytest.raises(ProviderUnavailable):
        await provider.chat(_basic_request())
    with pytest.raises(ProviderUnavailable):
        await provider.chat(_basic_request())

    # Two warmup attempts, one per request (each failed before chat could run).
    assert route.call_count == 2
    assert provider._warmed is False


async def test_warmup_read_error_raises_provider_unavailable(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """A mid-read transport drop during warmup is unavailable, not an uncaught 500."""
    respx_mock.post("http://ollama:11434/api/chat").mock(
        side_effect=httpx.ReadError("read dropped")
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)

    with pytest.raises(ProviderUnavailable) as info:
        await provider.warmup()

    assert info.value.http_status == 503
    assert provider._warmup_state == "unhealthy"
    assert provider._last_warmup_error == "ReadError"


# ===========================================================================
# D-19 — error mapping + specific-before-general ordering (Research-Delta-3)
# ===========================================================================


@pytest.mark.parametrize(
    "httpx_exc_factory, expected_core_exc, expected_status",
    [
        (lambda: httpx.ConnectError("refused"), ProviderUnavailable, 503),
        (lambda: httpx.ConnectTimeout("connect timeout"), ProviderUnavailable, 503),
        (lambda: httpx.ReadTimeout("read timeout"), ProviderTimeout, 504),
        (lambda: httpx.WriteTimeout("write timeout"), ProviderTimeout, 504),
        (lambda: httpx.PoolTimeout("pool timeout"), ProviderTimeout, 504),
        (lambda: httpx.ReadError("read dropped"), ProviderUnavailable, 503),
        (
            lambda: httpx.RemoteProtocolError("server disconnected"),
            ProviderUnavailable,
            503,
        ),
    ],
    ids=[
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ReadError",
        "RemoteProtocolError",
    ],
)
async def test_chat_maps_httpx_errors(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    httpx_exc_factory,
    expected_core_exc,
    expected_status,
) -> None:
    """D-19 + Research-Delta-3: httpx transport exceptions map to typed
    CoreThread errors with the right HTTP-status tier. The ConnectTimeout
    case is the regression guard — ConnectTimeout IS-A TimeoutException in
    httpx 0.28.1's MRO, so except-clause ordering MUST put ConnectTimeout
    BEFORE the (ReadTimeout, WriteTimeout, PoolTimeout) catch, else it would
    misroute to ProviderTimeout (504) instead of ProviderUnavailable (503)."""
    respx_mock.post("http://ollama:11434/api/chat").mock(side_effect=httpx_exc_factory())
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True  # skip warmup — we're testing chat's error paths

    with pytest.raises(expected_core_exc) as info:
        await provider.chat(_basic_request())

    assert info.value.http_status == expected_status


async def test_chat_maps_4xx_to_provider_http_error(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-19: upstream 4xx -> ProviderHTTPError with status_code captured on
    the instance (for internal logging) and http_status=502 (the generic
    bad-gateway tier)."""
    respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    with pytest.raises(ProviderHTTPError) as info:
        await provider.chat(_basic_request())

    assert info.value.status_code == 404
    assert info.value.http_status == 502


async def test_chat_maps_5xx_to_provider_http_error(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-19: upstream 5xx also maps to ProviderHTTPError(502). The
    ProviderHTTPError tier is the same for every non-2xx — the orchestrator
    does NOT differentiate internally between 4xx and 5xx; both mean "the
    local path is off the table, surface to client as 502"."""
    respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    with pytest.raises(ProviderHTTPError) as info:
        await provider.chat(_basic_request())

    assert info.value.status_code == 500
    assert info.value.http_status == 502


async def test_chat_invalid_json_maps_to_provider_http_error(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """A 200 with non-JSON content is provider response-shape failure, not a 500."""
    respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, content=b"not-json")
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    with pytest.raises(ProviderHTTPError) as info:
        await provider.chat(_basic_request())

    assert info.value.status_code == 200
    assert info.value.http_status == 502
    assert info.value.body == "invalid JSON response"


async def test_chat_schema_drift_raises_http_error_with_field_paths(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-19 + Pitfall E: when Ollama returns a body that fails
    OllamaChatResponse validation, the adapter raises ProviderHTTPError with
    status_code=200 (2xx transport, schema drift) and the body field
    containing ONLY joined field PATHS — NEVER raw body VALUES (which could
    echo user content, API keys, or secrets).

    Note: Pydantic's extra="forbid" surfaces the rejected KEY NAME as the
    field path, so ``UNEXPECTED_TOP_LEVEL`` itself IS a legitimate field
    path — it appears in info.value.body by design. The Pitfall E invariant
    is specifically that VALUES never leak; the values ``drift`` and
    ``SECRET_VALUE_IN_BODY`` are what must be scrubbed."""
    drift_body = {
        "UNEXPECTED_TOP_LEVEL": "SECRET_VALUE_IN_BODY",
        "another_bad_key": "another_secret",
    }
    respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=drift_body)
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    with pytest.raises(ProviderHTTPError) as info:
        await provider.chat(_basic_request())

    assert info.value.status_code == 200
    assert info.value.http_status == 502
    # Pitfall E: raw body VALUES must NOT appear in the exception body field
    # (field paths are structural identifiers, which is fine; values echo
    # user content or upstream secrets, which is not).
    assert "SECRET_VALUE_IN_BODY" not in info.value.body
    assert "another_secret" not in info.value.body


# ===========================================================================
# D-03 — format opaque passthrough (Phase 3 Judge populates it)
# ===========================================================================


async def test_chat_forwards_format_opaquely(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-03: options={"format": <JSON Schema dict>} is forwarded VERBATIM as
    top-level body["format"]. The adapter does NOT interpret it. Phase 3's
    Judge will pass JudgeVerdict.model_json_schema() here to force strict-JSON
    decode."""
    json_schema = {"type": "object", "properties": {}}
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=STUB_OLLAMA_RESPONSE)
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    await provider.chat(_basic_request(), options={"format": json_schema})

    body = json.loads(route.calls[0].request.content.decode("utf-8"))
    assert body["format"] == json_schema


async def test_chat_forwards_format_json_string(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-03: the format="json" string form is also forwarded verbatim."""
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=STUB_OLLAMA_RESPONSE)
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    await provider.chat(_basic_request(), options={"format": "json"})

    body = json.loads(route.calls[0].request.content.decode("utf-8"))
    assert body["format"] == "json"


# ===========================================================================
# D-16 — to_openai_chat_completion is the ONLY response path (grep-gate)
# ===========================================================================


def test_ollama_provider_never_constructs_chatcompletionresponse_directly() -> None:
    """D-16 grep-gate: providers/ollama.py MUST construct ALL responses via
    to_openai_chat_completion(...) and NEVER by calling ChatCompletionResponse(
    ... ) directly. This is the Pitfall #1 / Pitfall #2 structural defense —
    the adapter cannot forget to synthesize id/created/usage/finish_reason
    because a centralized helper owns those invariants."""
    src = Path("corethread/providers/ollama.py").read_text(encoding="utf-8")
    assert "ChatCompletionResponse(" not in src
    assert "to_openai_chat_completion(" in src


# ===========================================================================
# D-18 — heuristic truncation warning (no tiktoken)
# ===========================================================================


async def test_prompt_ctx_suspect_truncation_warning_emitted(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """D-18: when estimated_tokens > num_ctx AND prompt_eval_count <
    estimated * 0.9, emit a prompt.ctx.suspect_truncation warning with
    (provider, model, estimated_tokens, prompt_eval_count, num_ctx). No
    tiktoken — char/4 estimator is the signal to operators that num_ctx_default
    is too low."""
    _setup_logging_for_caplog()
    caplog.set_level(logging.WARNING)
    # Construct the test so estimated > num_ctx and prompt_eval_count <<< estimated.
    # num_ctx=1024 (override), content length of 50000 chars -> estimated=12500.
    # prompt_eval_count=800 (< 12500*0.9=11250) triggers the warning.
    cfg = LocalConfig(
        kind="ollama",
        base_url="http://ollama:11434",
        model="llama3.1:8b",
        num_ctx_default=1024,
        num_ctx_overrides={},
    )
    body = {**STUB_OLLAMA_RESPONSE, "prompt_eval_count": 800}
    respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=body)
    )
    provider = OllamaProvider(cfg, ollama_http_client)
    provider._warmed = True
    long_content = "x" * 50_000
    req = _basic_request(content=long_content)

    await provider.chat(req)

    # Find the warning record. structlog renders the event as JSON into
    # record.message, so we substring-match on the event name.
    suspect_records = [
        r for r in caplog.records if "prompt.ctx.suspect_truncation" in (r.message or "")
    ]
    assert suspect_records, (
        f"prompt.ctx.suspect_truncation warning was not emitted. "
        f"Records: {[r.message for r in caplog.records]}"
    )
    # Verify key kwargs serialized into the JSON event.
    msg = suspect_records[0].message
    assert '"provider": "ollama"' in msg
    assert '"model": "llama3.1:8b"' in msg
    assert '"num_ctx": 1024' in msg
    assert '"prompt_eval_count": 800' in msg
    assert '"estimated_tokens":' in msg


async def test_prompt_ctx_no_warning_when_not_truncated(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """D-18 negative: when estimated_tokens <= num_ctx, NO warning is emitted.
    This guards against false-positive noise for normal workloads."""
    _setup_logging_for_caplog()
    caplog.set_level(logging.WARNING)
    respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=STUB_OLLAMA_RESPONSE)
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    await provider.chat(_basic_request(content="hi"))  # estimated tokens ~0

    suspect_records = [
        r for r in caplog.records if "prompt.ctx.suspect_truncation" in (r.message or "")
    ]
    assert not suspect_records, (
        f"expected no suspect_truncation warning for short prompt, got: "
        f"{[r.message for r in suspect_records]}"
    )


# ===========================================================================
# D-08 — lifespan boots degraded on warmup failure (end-to-end)
# ===========================================================================


def test_lifespan_warmup_failure_boots_degraded(
    respx_mock: respx.MockRouter,
    yaml_path_ollama_host: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-08: when warmup fails with ProviderUnavailable during lifespan
    startup, main.py catches ProviderUnavailable/ProviderTimeout, logs
    warmup.failed, and boots degraded. /health returns status="degraded"
    with providers.local.state="unhealthy" and last_error="ConnectError"."""
    respx_mock.post("http://ollama:11434/api/chat").mock(side_effect=httpx.ConnectError("down"))
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(yaml_path_ollama_host))
    from corethread import main as m

    importlib.reload(m)
    with TestClient(m.app) as client:
        resp = client.get("/health")
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["providers"]["local"]["state"] == "unhealthy"
        assert body["providers"]["local"]["last_error"] == "ConnectError"


def test_lifespan_warmup_success_boots_ok(
    respx_mock: respx.MockRouter,
    yaml_path_ollama_host: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complement to D-08: when local warmup SUCCEEDS, providers.local.state
    is "ready". Proves the happy-path local-warmup logic works.

    Phase 5 / D-14 flip: OpenAIProvider replaces the Phase-4 bridging
    frontier stub and reports state="ready" whenever its ctor succeeds
    (no liveness probe — would be slow AND billable). With local also
    reporting "ready" after a successful warmup ping, the aggregate now
    flips to "ok" end-to-end — the observable signal that the service
    is pivot-capable. The Phase-4 assertion (`status == "degraded" by
    design because frontier-stub forces it`) retires with this flip.
    """
    respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=STUB_OLLAMA_RESPONSE)
    )
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(yaml_path_ollama_host))
    from corethread import main as m

    importlib.reload(m)
    with TestClient(m.app) as client:
        resp = client.get("/health")
        body = resp.json()
        # Phase 5 / D-14: aggregate is "ok" — both local and frontier are ready.
        assert body["status"] == "ok"
        # Local-side contract: warmup succeeded, so local is ready with no error.
        assert body["providers"]["local"]["state"] == "ready"
        assert body["providers"]["local"]["last_error"] is None
        # Frontier-side contract per D-14: OpenAIProvider always reports ready.
        assert body["providers"]["frontier"]["kind"] == "openai"
        assert body["providers"]["frontier"]["state"] == "ready"
        assert body["providers"]["frontier"]["last_error"] is None


# ===========================================================================
# Pitfall E — schema-drift log contains field paths only (no raw body)
# ===========================================================================


async def test_schema_drift_log_event_contains_field_paths_only(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pitfall E: the ollama.schema_drift log event MUST include a
    field_paths kwarg built from structural identifiers only — NEVER the
    raw upstream body content or the ValidationError().errors()[i]["input"]
    subfield."""
    _setup_logging_for_caplog()
    caplog.set_level(logging.WARNING)
    drift_body = {"UNEXPECTED_TOP_LEVEL": "SECRET_VALUE_IN_BODY"}
    respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=drift_body)
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    with pytest.raises(ProviderHTTPError):
        await provider.chat(_basic_request())

    drift_records = [r for r in caplog.records if "ollama.schema_drift" in (r.message or "")]
    assert drift_records, "ollama.schema_drift event was not emitted"
    msg = drift_records[0].message
    # Pitfall E: raw body content MUST NOT appear in the log record.
    assert "SECRET_VALUE_IN_BODY" not in msg
    assert "UNEXPECTED_TOP_LEVEL" in msg or "field_paths" in msg  # field path is fine


# ===========================================================================
# warmup() err_class-only log discipline (Pitfall #12 / T-02-01)
# ===========================================================================


async def test_warmup_failed_log_has_class_name_only(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pitfall #12 / T-02-01: warmup.failed log records the exception CLASS
    NAME only, never str(exc) or repr(exc). A specific exception message
    string MUST NOT leak into the captured log."""
    _setup_logging_for_caplog()
    caplog.set_level(logging.WARNING)
    secret_msg = "should-not-leak-secret-12345"
    respx_mock.post("http://ollama:11434/api/chat").mock(side_effect=httpx.ConnectError(secret_msg))
    provider = OllamaProvider(cfg_ollama, ollama_http_client)

    with pytest.raises(ProviderUnavailable):
        await provider.warmup()

    warmup_records = [r for r in caplog.records if "warmup.failed" in (r.message or "")]
    assert warmup_records, "warmup.failed log was not emitted"
    msg = warmup_records[0].message
    assert '"err_class": "ConnectError"' in msg
    # Secret message MUST NOT leak into the log
    assert secret_msg not in msg


# ===========================================================================
# _build_ollama_payload — pure function tests (no mock needed)
# ===========================================================================


def test_build_payload_includes_keep_alive_and_stream_and_num_ctx() -> None:
    """D-13 + D-17: the pure payload builder always sets keep_alive=-1,
    stream=False, and options.num_ctx = the passed num_ctx. model field is
    the explicit model_override arg (D-14 echo happens at RESPONSE layer)."""
    req = _basic_request(model="gpt-4o", content="hi")
    payload = _build_ollama_payload(req, model_override="llama3.1:8b", num_ctx=8192)
    assert payload["keep_alive"] == -1
    assert payload["stream"] is False
    assert payload["options"]["num_ctx"] == 8192
    assert payload["model"] == "llama3.1:8b"


def test_build_payload_ignores_options_keep_alive_and_num_ctx() -> None:
    """D-13 + D-17: options["keep_alive"] and options["num_ctx"] are
    IGNORED by the builder — server-side resolution wins. options other
    fields (temperature, top_p, seed, num_predict, stop) are forwarded
    into the nested options dict."""
    req = _basic_request()
    payload = _build_ollama_payload(
        req,
        model_override="m",
        num_ctx=8192,
        options={
            "keep_alive": 300,  # must be IGNORED
            "num_ctx": 99999,  # must be IGNORED
            "temperature": 0.1,  # must be forwarded
        },
    )
    assert payload["keep_alive"] == -1  # hardcoded wins
    assert payload["options"]["num_ctx"] == 8192  # arg wins, not 99999
    assert payload["options"]["temperature"] == 0.1


def test_build_payload_forwards_format_top_level() -> None:
    """D-03: format is passed at the TOP LEVEL of the body (Ollama's wire
    contract), not under options. And it's forwarded opaquely — a string
    or dict both survive untouched."""
    req = _basic_request()
    payload = _build_ollama_payload(
        req,
        model_override="m",
        num_ctx=8192,
        options={"format": "json"},
    )
    assert payload["format"] == "json"
    assert "format" not in payload["options"]


# ===========================================================================
# OllamaChatResponse round-trip (Delta-1: done_reason Optional)
# ===========================================================================


def test_ollama_chat_response_accepts_body_without_done_reason() -> None:
    """Research-Delta-1: done_reason is conditionally present in Ollama
    responses (docs example omits it for one body, includes it for another).
    OllamaChatResponse MUST accept bodies WITHOUT done_reason (default=None)."""
    body = {
        "model": "llama3.1:8b",
        "created_at": "2026-04-21T00:00:00Z",
        "message": {"role": "assistant", "content": "hello"},
        "done": True,
        # no done_reason
    }
    parsed = OllamaChatResponse.model_validate(body)
    assert parsed.done_reason is None


def test_ollama_chat_response_forbid_rejects_extra_keys() -> None:
    """D-02: OllamaChatResponse is extra="forbid" — schema drift raises
    ValidationError. This is the signal the adapter converts to
    ProviderHTTPError(200)."""
    import pydantic

    body = {**STUB_OLLAMA_RESPONSE, "NEW_OLLAMA_FIELD": "drift"}
    with pytest.raises(pydantic.ValidationError):
        OllamaChatResponse.model_validate(body)


# ---------------------------------------------------------------------------
# Phase 3 / Plan 01 (D-09 / JDG-06) — model_override forwarding
# ---------------------------------------------------------------------------


async def test_chat_model_override_forwarded_to_payload(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-09: when chat(..., model_override="qwen2.5:7b") is called, the
    outbound /api/chat body's `model` field MUST be "qwen2.5:7b", NOT
    cfg_ollama.model ("llama3.1:8b"). The ChatCompletionRequest.model stays
    "llama3.1:8b" — D-14 echo is preserved on the RESPONSE, not in the
    request payload.

    Closes JDG-06: judge model dispatch without provider duplication.
    """
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=STUB_OLLAMA_RESPONSE)
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True  # skip lazy warmup — separate test covers it

    request = ChatCompletionRequest(
        model="llama3.1:8b",  # CLIENT-facing model — D-14 echoes this back
        messages=[ChatMessage(role="user", content="hi")],
    )

    resp = await provider.chat(request, model_override="qwen2.5:7b")

    # Outbound /api/chat payload dispatches to the OVERRIDE model:
    body = json.loads(route.calls[0].request.content.decode("utf-8"))
    assert body["model"] == "qwen2.5:7b", body

    # Response still echoes the CLIENT model (D-14):
    assert resp.model == "llama3.1:8b"
    # system_fingerprint derives from request.model (D-15):
    assert resp.system_fingerprint == "fp_corethread_llama3.1:8"


async def test_chat_model_override_absent_uses_cfg_model(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-09 backward-compat: chat(...) with NO model_override (Phase 2
    default) still dispatches to cfg.local.model. Regression guard — every
    Phase 2 test relies on this behavior."""
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=STUB_OLLAMA_RESPONSE)
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    request = ChatCompletionRequest(
        model="llama3.1:8b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    await provider.chat(request)  # no model_override kwarg

    body = json.loads(route.calls[0].request.content.decode("utf-8"))
    assert body["model"] == "llama3.1:8b", body  # cfg.local.model fallback


async def test_chat_model_override_resolves_num_ctx_for_override_model(
    respx_mock: respx.MockRouter,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-09 + D-17: num_ctx lookup resolves against the OVERRIDE model, not
    cfg.local.model. If cfg.local.num_ctx_overrides has an entry for the
    judge model, that value must land in the outbound /api/chat body's
    options.num_ctx field."""
    cfg = LocalConfig(
        kind="ollama",
        base_url="http://ollama:11434",
        model="llama3.1:8b",
        num_ctx_default=8192,
        num_ctx_overrides={"qwen2.5:7b": 32768},  # override-model-specific entry
    )
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=STUB_OLLAMA_RESPONSE)
    )
    provider = OllamaProvider(cfg, ollama_http_client)
    provider._warmed = True

    request = ChatCompletionRequest(
        model="llama3.1:8b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    await provider.chat(request, model_override="qwen2.5:7b")

    body = json.loads(route.calls[0].request.content.decode("utf-8"))
    assert body["model"] == "qwen2.5:7b", body
    # D-17: exact-match against the RESOLVED backend model:
    assert body["options"]["num_ctx"] == 32768, body
