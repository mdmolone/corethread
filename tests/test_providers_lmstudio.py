"""Tests for providers/lmstudio.py — respx-driven transport + semantics tests
for the LM Studio adapter.

Respx strategy:
- respx.mock (via the pytest-respx `respx_mock` fixture) patches httpx's
  transport globally. The `ollama_http_client` fixture (provider-agnostic —
  see conftest.py docstring) yields a fresh AsyncClient inside the test's
  event loop so respx's patch takes effect (Landmine 5 mitigation).
- Routes are registered against `http://lmstudio:1234/v1/chat/completions`
  and `http://lmstudio:1234/v1/models` to match the `cfg_lmstudio` fixture's
  base_url.

Locks (by decision):
- D-07: raw httpx (NOT openai.AsyncOpenAI). Adapter inherits the shared
  AsyncClient via DI.
- D-08: POST targets `/chat/completions` appended to a base_url that ALREADY
  includes `/v1`. Pitfall P4-4: NO double-`/v1/` path.
- D-09: LMStudioChatResponse(extra="forbid") on the intermediate model;
  schema drift -> ProviderHTTPError(200, field_paths). Response goes through
  to_openai_chat_completion(...) ONLY (D-09 closing + D-16 grep-gate).
- D-10: num_ctx_default + num_ctx_overrides are SILENTLY IGNORED; one-time
  startup WARN `lmstudio.num_ctx.ignored` if non-default.
- D-11: ConnectError/ConnectTimeout -> ProviderUnavailable (503);
  ReadTimeout/WriteTimeout/PoolTimeout -> ProviderTimeout (504);
  HTTPStatusError -> ProviderHTTPError. Research-Delta-3 ordering REPLICATED.
- D-12: Lazy double-checked warmup + asyncio.Lock. 10 concurrent first-
  requests trigger exactly 1 warmup body (mirrors Phase 2 D-09 idiom).
- D-13: NO keep_alive, NO options block in chat OR warmup payloads.
- D-14: Warmup does TWO steps — (1) GET /v1/models, (2) POST /chat/completions
  with max_tokens=1. Step 1 validates cfg.model is in data list.
- D-15: Model-not-in-list raises ConfigError (NOT caught by lifespan ->
  uvicorn exits non-zero, Phase 1 CFG-03 pattern). Connection refused
  -> ProviderUnavailable; read timeout -> ProviderTimeout; 4xx/5xx ->
  ProviderHTTPError; malformed JSON -> ProviderHTTPError(200).

Tests that require log capture use the `caplog` pytest fixture — structlog
is configured in Phase 1 to route through stdlib logging, so caplog.records
captures the emitted entries (Phase 2 test_providers_ollama.py:93-97 pattern).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import typing
from pathlib import Path
from typing import Any

import httpx
import pydantic
import pytest
import respx

from corethread.errors import ConfigError, ProviderHTTPError, ProviderTimeout, ProviderUnavailable
from corethread.models import ChatCompletionRequest, ChatMessage, LocalConfig
from corethread.providers.base import Provider
from corethread.providers.lmstudio import (
    LMStudioChatResponse,
    LMStudioProvider,
    _build_lmstudio_payload,
)
from tests.conftest import STUB_LMSTUDIO_RESPONSE

# ---------------------------------------------------------------------------
# Helpers — used across multiple tests (mirrors test_providers_ollama.py:80-97)
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
# Pure-Python introspection — mirrors test_providers_base.py shape
# (no respx, no fixtures beyond direct class import)
# ===========================================================================


def test_lmstudio_provider_is_a_provider_subclass() -> None:
    """LMStudioProvider must subclass the Provider ABC so the orchestrator's
    dependency-injection of `local: Provider` accepts it without runtime
    isinstance gymnastics."""
    assert issubclass(LMStudioProvider, Provider)


def test_lmstudio_provider_name_class_attribute_is_lmstudio() -> None:
    """D-04: subclasses set `name: str` as a class attribute. The value
    surfaces in /health and structured log events; tests grep against
    `kind="lmstudio"` so this literal is load-bearing."""
    assert LMStudioProvider.name == "lmstudio"


def test_lmstudio_provider_abstract_methods_are_satisfied() -> None:
    """LMStudioProvider implements every Provider abstractmethod, so
    Provider.__abstractmethods__ contains zero entries on the subclass and
    instantiation does NOT raise TypeError."""
    assert LMStudioProvider.__abstractmethods__ == frozenset()


def test_lmstudio_provider_chat_signature_matches_abc_keyword_only() -> None:
    """D-04 + D-06: chat has (self, request, *, options=None,
    model_override=None, timeout_s=120.0) with options/model_override/
    timeout_s keyword-only and matching defaults — the ABC contract."""
    sig = inspect.signature(LMStudioProvider.chat)
    params = sig.parameters
    assert list(params.keys()) == [
        "self",
        "request",
        "options",
        "model_override",
        "timeout_s",
    ]
    assert params["options"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["model_override"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["timeout_s"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["options"].default is None
    assert params["model_override"].default is None
    assert params["timeout_s"].default == 120.0
    assert inspect.iscoroutinefunction(LMStudioProvider.chat)


def test_lmstudio_provider_warmup_signature_matches_abc_keyword_only() -> None:
    """D-06: warmup has (self, *, timeout_s=300.0)."""
    sig = inspect.signature(LMStudioProvider.warmup)
    params = sig.parameters
    assert list(params.keys()) == ["self", "timeout_s"]
    assert params["timeout_s"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["timeout_s"].default == 300.0
    assert inspect.iscoroutinefunction(LMStudioProvider.warmup)


def test_lmstudio_provider_health_signature_matches_abc_keyword_only() -> None:
    """D-06: health has (self, *, timeout_s=2.0)."""
    sig = inspect.signature(LMStudioProvider.health)
    params = sig.parameters
    assert list(params.keys()) == ["self", "timeout_s"]
    assert params["timeout_s"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["timeout_s"].default == 2.0
    assert inspect.iscoroutinefunction(LMStudioProvider.health)


# ===========================================================================
# LMStudioChatResponse intermediate model — Pydantic shape lock (D-09)
# ===========================================================================


def test_lmstudio_chat_response_validates_canonical_stub() -> None:
    """The canonical STUB_LMSTUDIO_RESPONSE round-trips through the
    intermediate validator without error. Schema-parity guard: if Plan 04-04
    ever changes LMStudioChatResponse and the conftest stub drifts, this
    test fires before the suite even reaches a respx route."""
    parsed = LMStudioChatResponse.model_validate(STUB_LMSTUDIO_RESPONSE)
    assert parsed.id.startswith("chatcmpl-")
    assert parsed.object == "chat.completion"
    assert isinstance(parsed.created, int)
    assert parsed.choices[0].finish_reason == "stop"
    assert parsed.usage is not None
    assert parsed.usage.total_tokens == 6


def test_lmstudio_chat_response_forbid_rejects_extra_keys() -> None:
    """D-09: LMStudioChatResponse is extra="forbid" — schema drift raises
    ValidationError. This is the signal the adapter converts to
    ProviderHTTPError(200, field_paths)."""
    body = {**STUB_LMSTUDIO_RESPONSE, "NEW_LMSTUDIO_FIELD": "drift"}
    with pytest.raises(pydantic.ValidationError):
        LMStudioChatResponse.model_validate(body)


def test_lmstudio_chat_response_accepts_current_lmstudio_extensions() -> None:
    """LM Studio may include non-core OpenAI-compatible extension fields.

    These are accepted so current LM Studio responses do not fail validation,
    while unknown extras are still rejected by the surrounding schema-drift
    guard.
    """
    choices = typing.cast(list[dict[str, Any]], STUB_LMSTUDIO_RESPONSE["choices"])
    body = {
        **STUB_LMSTUDIO_RESPONSE,
        "choices": [
            {
                **choices[0],
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "reasoning_content": "",
                    "tool_calls": [],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "total_tokens": 6,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        "stats": {},
    }

    parsed = LMStudioChatResponse.model_validate(body)

    assert parsed.choices[0].message.content == "ok"
    assert parsed.choices[0].message.tool_calls == []
    assert parsed.usage is not None
    assert parsed.usage.completion_tokens_details == {"reasoning_tokens": 0}
    assert parsed.stats == {}


def test_lmstudio_chat_response_accepts_body_without_usage() -> None:
    """Pitfall P4-8: some LM Studio configurations / older versions return
    responses WITHOUT the `usage` block. usage is Optional; absence must not
    raise. The adapter then defaults prompt/completion tokens to 0."""
    body = {k: v for k, v in STUB_LMSTUDIO_RESPONSE.items() if k != "usage"}
    parsed = LMStudioChatResponse.model_validate(body)
    assert parsed.usage is None


def test_lmstudio_chat_response_accepts_body_without_system_fingerprint() -> None:
    """system_fingerprint is Optional in the OpenAI surface. Absence (or
    null) must not raise."""
    body = {k: v for k, v in STUB_LMSTUDIO_RESPONSE.items() if k != "system_fingerprint"}
    parsed = LMStudioChatResponse.model_validate(body)
    assert parsed.system_fingerprint is None


# ===========================================================================
# _build_lmstudio_payload — pure helper (no respx)
# ===========================================================================


def test_build_lmstudio_payload_includes_model_messages_stream_false() -> None:
    """The payload always sets `model = model_override`, the messages list
    derived from request.messages, and stream=False (v1 never streams)."""
    req = _basic_request(model="gpt-4o", content="hi")
    payload = _build_lmstudio_payload(req, model_override="granite-3.0-2b-instruct")
    assert payload["model"] == "granite-3.0-2b-instruct"
    assert payload["stream"] is False
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_build_lmstudio_payload_omits_keep_alive() -> None:
    """D-13 negative: NO keep_alive directive — Ollama-specific."""
    req = _basic_request()
    payload = _build_lmstudio_payload(req, model_override="m")
    assert "keep_alive" not in payload


def test_build_lmstudio_payload_omits_options_block() -> None:
    """D-13 negative: NO nested options block — LM Studio uses top-level
    OpenAI-shape fields. NO `num_ctx` either (D-10 — LM Studio context window
    is configured in the LM Studio UI)."""
    req = _basic_request()
    payload = _build_lmstudio_payload(req, model_override="m")
    assert "options" not in payload
    assert "num_ctx" not in payload


def test_build_lmstudio_payload_forwards_request_top_level_openai_fields() -> None:
    """Standard generation params from the request body land at the TOP
    LEVEL of the payload (OpenAI shape), NOT nested under options."""
    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.3,
        top_p=0.9,
        max_tokens=128,
        stop=["END"],
    )
    payload = _build_lmstudio_payload(req, model_override="m")
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 128
    assert payload["stop"] == ["END"]


def test_build_lmstudio_payload_uses_model_override_for_model_field() -> None:
    """D-09 model_override forwarding: the outbound payload's `model` field
    is the explicit model_override arg, NOT request.model. (D-14 echo on
    response is a separate concern.)"""
    req = _basic_request(model="gpt-4o")
    payload = _build_lmstudio_payload(req, model_override="qwen2.5:7b")
    assert payload["model"] == "qwen2.5:7b"


# ===========================================================================
# chat() happy path + endpoint path lock + OpenAI-shape construction
# ===========================================================================


async def test_chat_happy_path_routes_through_to_openai_helper(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-09 closing + D-14 echo + D-15 fingerprint: happy path returns a
    fully populated OpenAI-shape response, constructed via
    to_openai_chat_completion. id starts with chatcmpl- (synthesized,
    NOT echoed from STUB_LMSTUDIO_RESPONSE.id), created is an int,
    system_fingerprint = f"fp_corethread_{request.model[:10]}"."""
    route = respx_mock.post("http://lmstudio:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=STUB_LMSTUDIO_RESPONSE)
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)
    provider._warmed = True  # skip lazy warmup
    req = _basic_request(model="gpt-4o")

    resp = await provider.chat(req)

    assert route.called
    # D-14 echo on response: response.model == request.model
    assert resp.model == "gpt-4o"
    # D-15: system_fingerprint = f"fp_corethread_{request.model[:10]}"
    assert resp.system_fingerprint == "fp_corethread_gpt-4o"
    # to_openai_chat_completion synthesizes id (NOT echoed) and created (int)
    assert resp.id.startswith("chatcmpl-")
    assert isinstance(resp.created, int)
    # Usage wiring: STUB_LMSTUDIO_RESPONSE.usage = {prompt:5, completion:1}
    assert resp.usage.prompt_tokens == 5
    assert resp.usage.completion_tokens == 1
    assert resp.usage.total_tokens == 6
    # finish_reason: "stop" -> "stop" via OLLAMA_FINISH_MAP
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].message.content == "ok"


async def test_chat_posts_to_v1_chat_completions_path_not_v1_v1(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-08 + Pitfall P4-4: the adapter appends `/chat/completions` to a
    base_url that ALREADY ends with `/v1`. The outbound request URL must
    have a SINGLE `/v1/` segment, NEVER `/v1/v1/`. .rstrip('/') defeats
    operator base_urls with a trailing slash."""
    route = respx_mock.post("http://lmstudio:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=STUB_LMSTUDIO_RESPONSE)
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)
    provider._warmed = True
    await provider.chat(_basic_request())

    assert route.called
    captured = route.calls[0].request
    assert captured.url.path == "/v1/chat/completions"
    # Pitfall P4-4: NEVER `/v1/v1/`
    assert "/v1/v1/" not in str(captured.url)


async def test_chat_response_model_echoes_request_model_not_backend(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-14 echo: response.model is the CLIENT-facing request.model
    (e.g. 'gpt-4o'), NOT the backend cfg.local.model
    ('granite-3.0-2b-instruct'). Lets a stock OpenAI SDK client receive
    its expected model string regardless of what local backend served it."""
    respx_mock.post("http://lmstudio:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=STUB_LMSTUDIO_RESPONSE)
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)
    provider._warmed = True
    req = _basic_request(model="gpt-4o")

    resp = await provider.chat(req)

    assert resp.model == "gpt-4o"
    # The backend model from STUB_LMSTUDIO_RESPONSE is granite-3.0-2b-instruct;
    # confirm it does NOT leak as the response model.
    assert resp.model != cfg_lmstudio.model
    assert resp.model != "granite-3.0-2b-instruct"


def test_lmstudio_provider_never_constructs_chatcompletionresponse_directly() -> None:
    """D-09 closing grep-gate: providers/lmstudio.py MUST construct ALL
    responses via to_openai_chat_completion(...) and NEVER by calling
    ChatCompletionResponse(...) directly. This is the Pitfall #1 / #2
    structural defense — the adapter cannot forget to synthesize id/created/
    usage/finish_reason because a centralized helper owns those invariants.

    Mirrors test_providers_ollama.py:596-604 verbatim, target swapped."""
    src = Path("corethread/providers/lmstudio.py").read_text(encoding="utf-8")
    assert "ChatCompletionResponse(" not in src
    assert "to_openai_chat_completion(" in src


# ===========================================================================
# chat() error mapping + Research-Delta-3 except-clause ordering (D-11)
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
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    httpx_exc_factory,
    expected_core_exc,
    expected_status,
) -> None:
    """D-11 + Research-Delta-3: httpx transport exceptions map to typed
    CoreThread errors with the right HTTP-status tier. The ConnectTimeout
    case is the load-bearing regression guard — ConnectTimeout IS-A
    TimeoutException in httpx 0.28.1's MRO, so except-clause ordering MUST
    put ConnectTimeout BEFORE the (ReadTimeout, WriteTimeout, PoolTimeout)
    catch, else it would misroute to ProviderTimeout (504) instead of
    ProviderUnavailable (503)."""
    respx_mock.post("http://lmstudio:1234/v1/chat/completions").mock(
        side_effect=httpx_exc_factory()
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)
    provider._warmed = True  # skip warmup — testing chat()'s error paths

    with pytest.raises(expected_core_exc) as info:
        await provider.chat(_basic_request())

    assert info.value.http_status == expected_status


async def test_chat_maps_4xx_to_provider_http_error(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-11: upstream 4xx -> ProviderHTTPError with status_code captured
    on the instance (for internal logging) and http_status=502 (the generic
    bad-gateway tier)."""
    respx_mock.post("http://lmstudio:1234/v1/chat/completions").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)
    provider._warmed = True

    with pytest.raises(ProviderHTTPError) as info:
        await provider.chat(_basic_request())

    assert info.value.status_code == 404
    assert info.value.http_status == 502


async def test_chat_maps_5xx_to_provider_http_error(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-11: upstream 5xx also maps to ProviderHTTPError(502)."""
    respx_mock.post("http://lmstudio:1234/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)
    provider._warmed = True

    with pytest.raises(ProviderHTTPError) as info:
        await provider.chat(_basic_request())

    assert info.value.status_code == 500
    assert info.value.http_status == 502


async def test_chat_invalid_json_maps_to_provider_http_error(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """A 200 with non-JSON content is provider response-shape failure, not a 500."""
    respx_mock.post("http://lmstudio:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=b"not-json")
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)
    provider._warmed = True

    with pytest.raises(ProviderHTTPError) as info:
        await provider.chat(_basic_request())

    assert info.value.status_code == 200
    assert info.value.http_status == 502
    assert info.value.body == "invalid JSON response"


async def test_pitfall_e_schema_drift_value_scrubbing(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """D-11 + Pitfall E: when LM Studio returns a body that fails
    LMStudioChatResponse validation, the adapter raises ProviderHTTPError
    with status_code=200 (2xx transport, schema drift) and the body field
    containing ONLY joined field PATHS — NEVER raw body VALUES (which
    could echo user content, API keys, or secrets).

    Dual assertion: (a) the ProviderHTTPError.body field has no value
    leak; (b) the captured `lmstudio.schema_drift` log record likewise
    has no value leak — only field path identifiers.

    Note: extra="forbid" surfaces the rejected KEY NAME as the field path,
    so `UNEXPECTED_TOP_LEVEL` itself IS a legitimate field path identifier
    and may appear. The Pitfall E invariant is specifically that VALUES
    (`SECRET_VALUE_IN_BODY`, `another_secret`) never leak."""
    _setup_logging_for_caplog()
    caplog.set_level(logging.WARNING)
    drift_body = {
        "UNEXPECTED_TOP_LEVEL": "SECRET_VALUE_IN_BODY",
        "another_bad_key": "another_secret",
    }
    respx_mock.post("http://lmstudio:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=drift_body)
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)
    provider._warmed = True

    with pytest.raises(ProviderHTTPError) as info:
        await provider.chat(_basic_request())

    assert info.value.status_code == 200
    assert info.value.http_status == 502
    # Pitfall E (body): raw VALUES must NOT appear in the exception body field.
    assert "SECRET_VALUE_IN_BODY" not in info.value.body
    assert "another_secret" not in info.value.body

    # Pitfall E (log): raw VALUES must NOT appear in the lmstudio.schema_drift log.
    drift_records = [r for r in caplog.records if "lmstudio.schema_drift" in (r.message or "")]
    assert drift_records, (
        f"lmstudio.schema_drift event was not emitted; records: "
        f"{[r.message for r in caplog.records]}"
    )
    msg = drift_records[0].message
    assert "SECRET_VALUE_IN_BODY" not in msg
    assert "another_secret" not in msg


# ===========================================================================
# warmup() — D-14 sequencing + D-15 error-class distinction
# ===========================================================================


async def test_warmup_step1_get_models_then_step2_one_token_post(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    stub_lmstudio_models_response: dict[str, Any],
) -> None:
    """D-14: warmup() MUST call GET /v1/models FIRST (step 1) and ONLY
    proceed to POST /chat/completions (step 4) after the model-in-list
    validation passes. Asserts:
    - call ordering — GET /v1/models is the FIRST respx call, POST is the SECOND
    - step 4 payload shape (D-13): max_tokens=1, messages=[{role:user,
      content:hi}], stream=False, NO keep_alive, NO options block
    - state transitions: unhealthy -> warming -> ready
    - _last_warmup_error is None on success
    """
    models_route = respx_mock.get("http://lmstudio:1234/v1/models").mock(
        return_value=httpx.Response(200, json=stub_lmstudio_models_response),
    )
    chat_route = respx_mock.post("http://lmstudio:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=STUB_LMSTUDIO_RESPONSE),
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)
    assert provider._warmup_state == "unhealthy"  # initial state

    await provider.warmup()

    # D-14: both routes called exactly once
    assert models_route.called and models_route.call_count == 1
    assert chat_route.called and chat_route.call_count == 1

    # D-14: ordering — GET /v1/models comes BEFORE POST /v1/chat/completions.
    all_calls = list(respx_mock.calls)
    assert len(all_calls) == 2
    assert all_calls[0].request.method == "GET"
    assert all_calls[0].request.url.path == "/v1/models"
    assert all_calls[1].request.method == "POST"
    assert all_calls[1].request.url.path == "/v1/chat/completions"

    # D-13 + D-14 step 4 payload shape
    chat_body = json.loads(all_calls[1].request.content.decode("utf-8"))
    assert chat_body["max_tokens"] == 1  # D-14 step 4
    assert chat_body["messages"] == [{"role": "user", "content": "hi"}]  # D-14 step 4
    assert chat_body["stream"] is False  # v1 never streams
    assert "keep_alive" not in chat_body  # D-13 negative
    assert "options" not in chat_body  # D-13 negative

    # State transitions on success
    assert provider._warmup_state == "ready"
    assert provider._last_warmup_error is None


async def test_warmup_step2_payload_uses_max_tokens_1_no_keep_alive_no_options_block(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    stub_lmstudio_models_response: dict[str, Any],
) -> None:
    """D-13 + D-14 step 4: warmup payload uses max_tokens=1 (the canary),
    has NO keep_alive directive (Ollama-only), and NO nested options block
    (LM Studio uses top-level OpenAI-shape fields)."""
    respx_mock.get("http://lmstudio:1234/v1/models").mock(
        return_value=httpx.Response(200, json=stub_lmstudio_models_response),
    )
    chat_route = respx_mock.post("http://lmstudio:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=STUB_LMSTUDIO_RESPONSE),
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)

    await provider.warmup()

    body = json.loads(chat_route.calls[0].request.content.decode("utf-8"))
    assert body["max_tokens"] == 1
    assert "keep_alive" not in body
    assert "options" not in body
    # And the model field is the cfg.local.model (D-14 step 4 uses cfg.model directly)
    assert body["model"] == cfg_lmstudio.model


async def test_warmup_success_flips_state_to_ready_and_clears_last_error(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    stub_lmstudio_models_response: dict[str, Any],
) -> None:
    """On warmup success: _warmup_state transitions unhealthy -> warming
    -> ready; _last_warmup_error returns to None."""
    respx_mock.get("http://lmstudio:1234/v1/models").mock(
        return_value=httpx.Response(200, json=stub_lmstudio_models_response),
    )
    respx_mock.post("http://lmstudio:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=STUB_LMSTUDIO_RESPONSE),
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)
    # Pre-condition: initial state is "unhealthy" (no warmup yet)
    assert provider._warmup_state == "unhealthy"
    assert provider._last_warmup_error is None

    await provider.warmup()

    assert provider._warmup_state == "ready"
    assert provider._last_warmup_error is None


async def test_warmup_get_models_connect_error_raises_provider_unavailable(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-15: GET /v1/models connect refused -> ProviderUnavailable.
    Lifespan CATCHES this per Phase 2 D-08 (service boots degraded).
    _warmup_state remains 'unhealthy', _last_warmup_error == 'ConnectError'."""
    respx_mock.get("http://lmstudio:1234/v1/models").mock(side_effect=httpx.ConnectError("down"))
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)

    with pytest.raises(ProviderUnavailable) as info:
        await provider.warmup()

    assert info.value.http_status == 503
    assert provider._warmup_state == "unhealthy"
    assert provider._last_warmup_error == "ConnectError"


async def test_research_delta_3_connect_timeout_maps_to_provider_unavailable(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-15 + Research-Delta-3 (warmup-side regression guard): on the
    warmup() code path, GET /v1/models with a ConnectTimeout MUST map to
    ProviderUnavailable (503), NOT ProviderTimeout (504).

    The ordering bug would surface as ProviderTimeout because ConnectTimeout
    IS-A TimeoutException in httpx 0.28.1's MRO. The except-clause order
    in providers/lmstudio.py MUST put ConnectTimeout BEFORE the
    (ReadTimeout, WriteTimeout, PoolTimeout) catch — failing this test is
    the regression signal."""
    respx_mock.get("http://lmstudio:1234/v1/models").mock(side_effect=httpx.ConnectTimeout("ct"))
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)

    with pytest.raises(ProviderUnavailable) as info:
        await provider.warmup()

    assert info.value.http_status == 503
    # Specifically NOT ProviderTimeout (504) — the misordering signal:
    assert not isinstance(info.value, ProviderTimeout)
    assert provider._warmup_state == "unhealthy"
    assert provider._last_warmup_error == "ConnectTimeout"


async def test_warmup_get_models_read_timeout_raises_provider_timeout(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-15: GET /v1/models read timeout -> ProviderTimeout (504)."""
    respx_mock.get("http://lmstudio:1234/v1/models").mock(side_effect=httpx.ReadTimeout("read"))
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)

    with pytest.raises(ProviderTimeout) as info:
        await provider.warmup()

    assert info.value.http_status == 504
    assert provider._warmup_state == "unhealthy"
    assert provider._last_warmup_error == "ReadTimeout"


async def test_warmup_get_models_read_error_raises_provider_unavailable(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """A mid-read transport drop is unavailable, not an uncaught 500."""
    respx_mock.get("http://lmstudio:1234/v1/models").mock(
        side_effect=httpx.ReadError("read dropped")
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)

    with pytest.raises(ProviderUnavailable) as info:
        await provider.warmup()

    assert info.value.http_status == 503
    assert provider._warmup_state == "unhealthy"
    assert provider._last_warmup_error == "ReadError"


async def test_warmup_get_models_4xx_5xx_raises_provider_http_error(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-15: GET /v1/models 4xx/5xx -> ProviderHTTPError. Not caught by
    lifespan — kills startup (legitimate server-side bug operators must see)."""
    respx_mock.get("http://lmstudio:1234/v1/models").mock(
        return_value=httpx.Response(503, json={"error": "model loader busy"})
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)

    with pytest.raises(ProviderHTTPError) as info:
        await provider.warmup()

    assert info.value.status_code == 503
    assert info.value.http_status == 502
    assert provider._warmup_state == "unhealthy"
    assert provider._last_warmup_error == "HTTPStatusError"


async def test_warmup_step2_post_chat_completions_validation_error_raises_provider_http_error(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    stub_lmstudio_models_response: dict[str, Any],
) -> None:
    """D-15: warmup step 4 POST returns a body that fails
    LMStudioChatResponse validation -> ProviderHTTPError(200, field_paths).
    The adapter exercises the validator on the warmup response specifically
    so schema drift is caught at startup, not at first real request."""
    respx_mock.get("http://lmstudio:1234/v1/models").mock(
        return_value=httpx.Response(200, json=stub_lmstudio_models_response),
    )
    drift_body = {"unexpected_top": "drift"}
    respx_mock.post("http://lmstudio:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=drift_body),
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)

    with pytest.raises(ProviderHTTPError) as info:
        await provider.warmup()

    assert info.value.status_code == 200
    assert info.value.http_status == 502
    assert provider._warmup_state == "unhealthy"
    assert provider._last_warmup_error == "ValidationError"


# ===========================================================================
# SC#3 — the explicit fail-fast closure (the LOC-03 critical test)
# ===========================================================================


async def test_sc3_lmstudio_wrong_model_config_error(
    respx_mock: respx.MockRouter,
    ollama_http_client: httpx.AsyncClient,
    stub_lmstudio_models_response: dict[str, Any],
) -> None:
    """SC#3 + D-15 + LOC-03: when cfg.model is NOT present in the list of
    models returned by GET /v1/models, LMStudioProvider.warmup() MUST raise
    ConfigError (NOT ProviderUnavailable, NOT ProviderHTTPError). This is
    the Phase 4 fail-fast closure — uvicorn's lifespan does NOT catch
    ConfigError per Phase 1 CFG-03; the service exits non-zero so the
    operator sees the misconfiguration BEFORE any request is served.

    Per D-15: defeats LM Studio bug #619 (single-loaded-model silently
    ignores the `model` field in request body). If the configured model
    isn't loaded, refuse to start.

    Asserts:
    - pytest.raises(ConfigError) on warmup()
    - the exception message contains the configured model name (so the
      operator knows WHICH model is misconfigured)
    - the exception message contains the available_ids list (so the
      operator knows WHAT IS loaded — corrective action signal)
    - respx GET /v1/models was called exactly 1 time (D-14 step 1)
    - respx POST /chat/completions was NEVER called (we fail before step 4)
    """
    cfg = LocalConfig(
        kind="lmstudio",
        base_url="http://lmstudio:1234/v1",
        model="not-loaded-model",  # NOT in stub data
        num_ctx_default=8192,
        num_ctx_overrides={},
    )
    models_route = respx_mock.get("http://lmstudio:1234/v1/models").mock(
        return_value=httpx.Response(200, json=stub_lmstudio_models_response),
    )
    chat_route = respx_mock.post("http://lmstudio:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=STUB_LMSTUDIO_RESPONSE),
    )
    provider = LMStudioProvider(cfg, ollama_http_client)

    with pytest.raises(ConfigError) as info:
        await provider.warmup()

    # D-15: ConfigError message contains the configured model name AND
    # the available_ids list — both are corrective signals for the operator.
    msg = str(info.value)
    assert "not-loaded-model" in msg, msg
    assert "granite-3.0-2b-instruct" in msg, msg  # from stub data
    assert "qwen2-vl-7b-instruct" in msg, msg  # from stub data

    # D-14: GET /v1/models was called once (step 1). POST /chat/completions
    # was NEVER called (we fail-fast BEFORE step 4 1-token generation).
    assert models_route.called and models_route.call_count == 1
    assert not chat_route.called, "MUST fail-fast BEFORE the step 4 1-token generation"

    # State: warmup_state is "unhealthy"; _last_warmup_error is "ConfigError"
    assert provider._warmup_state == "unhealthy"
    assert provider._last_warmup_error == "ConfigError"
    assert provider._warmed is False


# ===========================================================================
# D-09 / D-12 — lazy double-checked warmup concurrency
# ===========================================================================


async def test_d_09_lazy_warmup_only_one_concurrent_request_warms(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    stub_lmstudio_models_response: dict[str, Any],
) -> None:
    """D-12 (mirrors Phase 2 D-09): 10 concurrent first-requests against a
    cold LMStudioProvider trigger EXACTLY ONE warmup sequence. The double-
    checked lock collapses the fan-in. Asserts:
    - GET /v1/models route called exactly 1 time (warmup step 1 fired once)
    - POST /chat/completions route received: 1 warmup body (max_tokens=1 +
      content="hi") + 10 chat bodies (no max_tokens=1 signature) = 11 total
    - provider._warmed flips True after the first warmup
    """
    models_route = respx_mock.get("http://lmstudio:1234/v1/models").mock(
        return_value=httpx.Response(200, json=stub_lmstudio_models_response),
    )
    chat_route = respx_mock.post("http://lmstudio:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=STUB_LMSTUDIO_RESPONSE),
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)
    assert provider._warmed is False

    # Fan out 10 concurrent first-requests
    await asyncio.gather(*(provider.chat(_basic_request()) for _ in range(10)))

    # D-12: warmup step 1 (GET /v1/models) fired EXACTLY ONCE despite 10 calls
    assert models_route.call_count == 1, (
        f"expected exactly 1 GET /v1/models from the lazy double-check, "
        f"got {models_route.call_count}"
    )
    # POST count: 1 warmup (max_tokens=1) + 10 chat (no max_tokens) = 11
    assert chat_route.call_count == 11, (
        f"expected 1 warmup-body + 10 chat-body POSTs, got {chat_route.call_count}"
    )
    # Verify the warmup body fired exactly once (max_tokens=1 signature)
    warmup_body_count = 0
    for call in chat_route.calls:
        body = json.loads(call.request.content.decode("utf-8"))
        if body.get("max_tokens") == 1:
            warmup_body_count += 1
    assert warmup_body_count == 1, (
        f"expected exactly 1 warmup body (max_tokens=1) across 11 POSTs, got {warmup_body_count}"
    )
    assert provider._warmed is True


async def test_warmup_failure_does_not_stick_warmed_stays_false(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """Pitfall G: a failed warmup must NOT mark _warmed=True. Lazy warmup
    inside chat() calls warmup() which raises ProviderUnavailable on
    ConnectError; _warmed stays False so the next request retries."""
    respx_mock.get("http://lmstudio:1234/v1/models").mock(side_effect=httpx.ConnectError("down"))
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)

    with pytest.raises(ProviderUnavailable):
        await provider.chat(_basic_request())

    assert provider._warmed is False


async def test_second_chat_after_warmup_failure_retries_warmup(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """Extension of Pitfall G: second chat call MUST re-attempt warmup
    after the first failed. Asserted by hitting the GET /v1/models mock
    twice across two failing requests (one warmup attempt each)."""
    models_route = respx_mock.get("http://lmstudio:1234/v1/models").mock(
        side_effect=httpx.ConnectError("down")
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)

    with pytest.raises(ProviderUnavailable):
        await provider.chat(_basic_request())
    with pytest.raises(ProviderUnavailable):
        await provider.chat(_basic_request())

    # Two warmup attempts, one per request (each failed before chat could run)
    assert models_route.call_count == 2
    assert provider._warmed is False


# ===========================================================================
# D-10 — num_ctx ignored startup warning
# ===========================================================================


def test_d_10_num_ctx_ignored_logs_once_on_init(
    ollama_http_client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """D-10: when LocalConfig has num_ctx_default != 8192 OR num_ctx_overrides
    non-empty, LMStudioProvider.__init__ emits a structured
    `lmstudio.num_ctx.ignored` log event EXACTLY ONCE on instantiation. The
    event payload contains provider='lmstudio', num_ctx_default=<value>,
    overrides_count=<len>."""
    _setup_logging_for_caplog()
    caplog.set_level(logging.WARNING)
    cfg = LocalConfig(
        kind="lmstudio",
        base_url="http://lmstudio:1234/v1",
        model="granite-3.0-2b-instruct",
        num_ctx_default=4096,  # NON-default
        num_ctx_overrides={"some-model": 16384},  # NON-default
    )
    LMStudioProvider(cfg, ollama_http_client)

    ignored_records = [r for r in caplog.records if "lmstudio.num_ctx.ignored" in (r.message or "")]
    assert len(ignored_records) == 1, (
        f"expected exactly 1 lmstudio.num_ctx.ignored event, got "
        f"{len(ignored_records)}; records: {[r.message for r in caplog.records]}"
    )
    msg = ignored_records[0].message
    assert '"provider": "lmstudio"' in msg
    assert '"num_ctx_default": 4096' in msg
    assert '"overrides_count": 1' in msg


def test_d_10_num_ctx_default_values_emit_no_warning(
    ollama_http_client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """D-10 negative: at default values (num_ctx_default=8192,
    num_ctx_overrides={}), the lmstudio.num_ctx.ignored event MUST NOT be
    emitted — guards against false-positive noise for operators who use
    the LocalConfig defaults."""
    _setup_logging_for_caplog()
    caplog.set_level(logging.WARNING)
    cfg = LocalConfig(
        kind="lmstudio",
        base_url="http://lmstudio:1234/v1",
        model="granite-3.0-2b-instruct",
        num_ctx_default=8192,
        num_ctx_overrides={},
    )
    LMStudioProvider(cfg, ollama_http_client)

    ignored_records = [r for r in caplog.records if "lmstudio.num_ctx.ignored" in (r.message or "")]
    assert not ignored_records, (
        f"expected zero lmstudio.num_ctx.ignored events at default values, "
        f"got: {[r.message for r in ignored_records]}"
    )


# ===========================================================================
# Pitfall #12 — warmup err_class-only log discipline
# ===========================================================================


async def test_warmup_failed_log_has_class_name_only(
    respx_mock: respx.MockRouter,
    cfg_lmstudio: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pitfall #12 / T-02-01: warmup.failed log records the exception
    CLASS NAME only, never str(exc) or repr(exc). A specific exception
    message string MUST NOT leak into the captured log."""
    _setup_logging_for_caplog()
    caplog.set_level(logging.WARNING)
    secret_msg = "should-not-leak-secret-12345"
    respx_mock.get("http://lmstudio:1234/v1/models").mock(
        side_effect=httpx.ConnectError(secret_msg)
    )
    provider = LMStudioProvider(cfg_lmstudio, ollama_http_client)

    with pytest.raises(ProviderUnavailable):
        await provider.warmup()

    warmup_records = [r for r in caplog.records if "warmup.failed" in (r.message or "")]
    assert warmup_records, "warmup.failed log was not emitted"
    msg = warmup_records[0].message
    assert '"err_class": "ConnectError"' in msg
    assert '"provider": "lmstudio"' in msg
    # Secret message MUST NOT leak into the log
    assert secret_msg not in msg


# Avoid unused-import on `typing` module; kept imported for future helpers.
_: Any = typing
