"""Shared pytest fixtures for Phase 1 tests.

Fixtures here are auto-discovered by pytest for every sibling test file.

- ``valid_yaml_path``: writes a valid ``config.yaml`` into a tmpdir and yields the path.
  Tests that need a parseable config (happy-path, override, app smoke) consume it.

- ``set_test_api_key``: autouse fixture that sets
  ``OPENAI_API_KEY=sk-test-1234567890ABCDEFGHIJ`` (>=20 chars after ``sk-`` so the
  production ``_SK_PATTERN`` redaction regex would actually fire on it — see
  REVIEW WR-02) for every test, restored via ``monkeypatch`` teardown. Tests that
  need to test the unset-env-var case explicitly call ``monkeypatch.delenv(...)``
  inside the test.

- ``reset_logging_state``: autouse fixture that resets ``logging_config._SETUP_DONE``
  AND clears all root-logger handlers/filters BEFORE each test so:
    * ``setup_logging()`` re-runs cleanly per test (handler stream binds to the
      test's current ``sys.stdout``, important for capture in test_app_smoke.py).
    * The ``test_setup_logging_idempotent`` test gets a clean slate to verify the
      no-duplicate-handlers invariant.
"""

from __future__ import annotations

import importlib
import logging
import textwrap
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from corethread.models import LocalConfig

# Mirrors config.yaml.example but trimmed to the fields AppConfig actually requires.
# All four sub-models (LocalConfig / JudgeConfigModel / FrontierConfig / RoutingConfig)
# validate against this body, and the loader resolves frontier.api_key_env into the
# SecretStr api_key from os.environ["OPENAI_API_KEY"].
VALID_CONFIG_YAML = (
    textwrap.dedent(
        """
    local:
      kind: ollama
      base_url: http://localhost:11434
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


@pytest.fixture
def valid_yaml_path(tmp_path: Path) -> Path:
    """Write a valid config.yaml in a tmpdir and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(VALID_CONFIG_YAML, encoding="utf-8")
    return p


#: Test fixture API key. >=20 chars after ``sk-`` so the production ``_SK_PATTERN``
#: redaction regex would actually scrub it if it leaked into a log path during a
#: test (REVIEW WR-02). Tests that assert ``TEST_OPENAI_API_KEY not in <output>``
#: now exercise BOTH the SecretStr ``__repr__`` mask AND the regex filter on the
#: same string — two independent leak signals from one assertion. Update this
#: constant (and any literal-string references in tests) together if you need to
#: change the value; do not introduce parallel literals.
TEST_OPENAI_API_KEY = "sk-test-1234567890ABCDEFGHIJ"


@pytest.fixture(autouse=True)
def set_test_api_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Autouse: set ``OPENAI_API_KEY`` for every test, restore on teardown.

    Uses ``TEST_OPENAI_API_KEY`` (>=20 chars after ``sk-``) so the production
    redaction filter would actually fire on this value (REVIEW WR-02). Tests
    that need to test the unset case use ``monkeypatch.delenv(...)`` inside
    the test body (monkeypatch handles the restore).
    """
    monkeypatch.setenv("OPENAI_API_KEY", TEST_OPENAI_API_KEY)
    yield


# ---------------------------------------------------------------------------
# Phase 9 / Plan 01 — SEC-03 marker fixture (shared with Plans 03 + 04)
# ---------------------------------------------------------------------------
#
# Distinct from set_test_api_key autouse fixture: this fixture sets a DIFFERENT
# marker value (sk-marker-must-not-leak) named verbatim in REQUIREMENTS.md
# SEC-03 + CONTEXT.md D-19 #2. Tests that hit /v1/config + /v1/traces/stream
# consume this fixture explicitly to override the autouse marker for the
# duration of the test, then assert the literal substring is absent from
# the entire response body. The literal differs from the autouse fixture's
# value so a test author cannot conflate the two.

SEC_03_MARKER_KEY = "sk-marker-must-not-leak"  # SEC-03 / D-19 #2 verbatim


@pytest.fixture
def cfg_with_marker_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Path, Any]]:
    """Yields (yaml_path, AppConfig) for SEC-03 / Pitfall 19 secret-leak tests.

    Sets OPENAI_API_KEY=sk-marker-must-not-leak (overriding the autouse
    set_test_api_key default), writes a tmp_path config.yaml using the
    canonical VALID_CONFIG_YAML body, and loads it via load_config.
    Tests then assert SEC_03_MARKER_KEY is absent from /v1/config response
    text + recursive .json() walk.
    """
    from corethread.config import load_config

    monkeypatch.setenv("OPENAI_API_KEY", SEC_03_MARKER_KEY)
    p = tmp_path / "config.yaml"
    p.write_text(VALID_CONFIG_YAML, encoding="utf-8")
    cfg = load_config(p)
    yield (p, cfg)


@pytest.fixture(autouse=True)
def reset_logging_state() -> Iterator[None]:
    """Autouse: reset logging_config singletons + root logger around each test.

    Why: ``setup_logging`` is gated by a module-level ``_SETUP_DONE`` flag; if a
    prior test ran setup and bound the StreamHandler to that test's ``sys.stdout``,
    a later test cannot rebind. We force a clean slate so each test that calls
    ``setup_logging(...)`` (or triggers it via the FastAPI lifespan) gets a fresh
    handler bound to the current ``sys.stdout``.

    Setup is aggressive (clears ALL root handlers/filters) — we genuinely want a
    blank slate so ``setup_logging`` can re-attach the production handler bound
    to the current ``sys.stdout``.

    Teardown is narrow (REVIEW WR-03): only remove handlers/filters this fixture
    plus ``setup_logging`` are responsible for — exact-class
    ``logging.StreamHandler`` and ``RedactingFilter`` instances. This avoids
    silently stripping third-party handlers (e.g., a coverage tool, a tracer
    agent) that another fixture or test import attaches at runtime. The
    exact-class check (``type(h) is logging.StreamHandler``) deliberately
    excludes subclasses such as pytest's ``LogCaptureHandler`` — same pattern
    used by ``test_setup_logging_idempotent`` to count production handlers.
    """
    from corethread import logging_config
    from corethread.logging_config import RedactingFilter

    logging_config._SETUP_DONE = False
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    for f in list(root.filters):
        root.removeFilter(f)
    yield
    # Teardown — narrow scope: only remove what setup_logging owns.
    # Don't strip third-party handlers/filters that may have been attached
    # mid-test by other fixtures or import-time side effects.
    logging_config._SETUP_DONE = False
    root = logging.getLogger()
    for h in list(root.handlers):
        if type(h) is logging.StreamHandler:
            root.removeHandler(h)
    for f in list(root.filters):
        if isinstance(f, RedactingFilter):
            root.removeFilter(f)


# ---------------------------------------------------------------------------
# Phase 2 additions — Ollama provider test fixtures
# ---------------------------------------------------------------------------
#
# These fixtures are module-level so any Phase 2+ test that imports from
# `conftest` gets them for free. Phase 4's LMStudio test file will reuse
# `ollama_http_client` (renamed to `shared_http_client` if Phase 4 needs a
# different base_url — for now it's Ollama-agnostic, it's just a fresh
# AsyncClient).

# Canonical Ollama /api/chat success body — reusable stub for any test that
# needs "a plausible good response" without caring about the exact content.
# Matches the OllamaChatResponse Pydantic model in providers/ollama.py exactly.
STUB_OLLAMA_RESPONSE: dict[str, object] = {
    "model": "llama3.1:8b",
    "created_at": "2026-04-21T00:00:00Z",  # ISO-8601 string (D-02 / Delta-2)
    "message": {"role": "assistant", "content": "ok"},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 5,
    "eval_count": 1,
}


@pytest.fixture
def cfg_ollama() -> LocalConfig:
    """A realistic `LocalConfig` for Phase 2 Ollama tests.

    base_url is deliberately `http://ollama:11434` (NOT `localhost`) so that
    respx patterns don't have to care about IPv4/IPv6 resolution. The tests
    all mount respx against this exact base_url.
    """
    return LocalConfig(
        kind="ollama",
        base_url="http://ollama:11434",
        model="llama3.1:8b",
        num_ctx_default=8192,
        num_ctx_overrides={},
    )


@pytest_asyncio.fixture
async def ollama_http_client() -> AsyncIterator[httpx.AsyncClient]:
    """Fresh httpx.AsyncClient per test, matching the Phase 1 lifespan shape.

    respx's `respx_mock` fixture (pytest-respx) will intercept requests made
    against this client automatically because respx patches httpx's transport
    globally when the `respx_mock` fixture is active (Landmine 5 mitigation:
    always construct the client INSIDE the test's event loop / after the
    respx fixture is live — this async fixture handles both).
    """
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# Phase 4 additions — LMStudio provider test fixtures (Plan 04-05, D-29)
# ---------------------------------------------------------------------------
#
# Mirror shape of the Phase 2 cfg_ollama + STUB_OLLAMA_RESPONSE conventions.
# Plan 04-05 uses these to drive tests/test_providers_lmstudio.py:
#   - cfg_lmstudio: LocalConfig with kind='lmstudio' for LMStudioProvider ctor
#   - STUB_LMSTUDIO_RESPONSE: canned /v1/chat/completions success body
#   - stub_lmstudio_models_response: canned /v1/models success body that lists
#     cfg_lmstudio.model (so warmup() validation passes by default)
#
# IMPORTANT: cfg_lmstudio.model MUST appear in stub_lmstudio_models_response
# so the default fixture combination doesn't trip D-15 fail-fast. Tests that
# want to exercise the SC#3 wrong-model failure construct a different cfg
# inline (or override cfg.model) so the model-not-in-list path fires
# deterministically.

# Canonical LM Studio /v1/chat/completions success body — mirrored from the
# Phase 2 conftest STUB_OLLAMA_RESPONSE pattern. Matches LMStudioChatResponse
# (Plan 04-04 providers/lmstudio.py) exactly: id starts with chatcmpl-,
# object='chat.completion', created is unix int (NOT Ollama's ISO string),
# choices[0] has finish_reason + message{role,content}, usage block present
# (Pitfall P4-8 Optional handling is exercised in dedicated tests).
STUB_LMSTUDIO_RESPONSE: dict[str, object] = {
    "id": "chatcmpl-i3gkjwthhw96whukek9tz",
    "object": "chat.completion",
    "created": 1731990317,  # unix int (D-09 + Pattern 3)
    "model": "granite-3.0-2b-instruct",
    "choices": [
        {
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"},
        },
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 1,
        "total_tokens": 6,
    },
    "system_fingerprint": None,
}


@pytest.fixture
def cfg_lmstudio() -> LocalConfig:
    """A realistic LocalConfig for Phase 4 LMStudio tests (D-29).

    base_url is `http://lmstudio:1234/v1` — note the `/v1` SUFFIX is part of
    the base_url for LM Studio (D-08 + Pitfall P4-4 defense). The adapter
    appends `/chat/completions` and `/models` WITHOUT another `/v1`. respx
    routes mount against the joined URLs:
      - GET  http://lmstudio:1234/v1/models
      - POST http://lmstudio:1234/v1/chat/completions

    `model='granite-3.0-2b-instruct'` matches the model id present in
    `stub_lmstudio_models_response` so the default fixture combination
    passes warmup() validation by default. Tests that exercise the D-15
    wrong-model failure construct a different cfg inline.

    `num_ctx_default=8192` and `num_ctx_overrides={}` are the LocalConfig
    defaults — at these values, LMStudioProvider.__init__ does NOT emit the
    `lmstudio.num_ctx.ignored` warning (D-10 negative-case default). The
    dedicated D-10 test constructs a non-default cfg to trigger the warning.
    """
    return LocalConfig(
        kind="lmstudio",
        base_url="http://lmstudio:1234/v1",
        model="granite-3.0-2b-instruct",
        num_ctx_default=8192,
        num_ctx_overrides={},
    )


@pytest.fixture
def stub_lmstudio_models_response() -> dict[str, object]:
    """Canned `/v1/models` GET response payload (D-14 + D-29 + Pattern 4).

    Shape: `{"object": "list", "data": [{"id": <str>, "object": "model",
    ...}, ...]}`. Plan 04-05 tests use this fixture as the respx return value
    for `respx_mock.get('http://lmstudio:1234/v1/models')`.

    `data[0].id == 'granite-3.0-2b-instruct'` matches `cfg_lmstudio.model`
    so the default fixture combination passes the D-14 model-in-list check
    cleanly. Additional models (`qwen2-vl-7b-instruct`) populate the list to
    prove the in-list lookup is `in` not `==[0]`.

    Per D-14 the adapter only requires `data[i].id` to be the model-identifier
    key — the LM-Studio-specific extension fields (`type`, `publisher`,
    `arch`, `state`, `max_context_length`) are NOT validated and are omitted
    from this canned payload to keep test assertions narrow.
    """
    return {
        "object": "list",
        "data": [
            {"id": "granite-3.0-2b-instruct", "object": "model"},
            {"id": "qwen2-vl-7b-instruct", "object": "model"},
        ],
    }


# ---------------------------------------------------------------------------
# Phase 5 additions — OpenAI frontier provider test fixtures (Plan 05-02, D-18)
# ---------------------------------------------------------------------------
#
# Mirror shape of the Phase 2 cfg_ollama + STUB_OLLAMA_RESPONSE and the Phase 4
# cfg_lmstudio + STUB_LMSTUDIO_RESPONSE pairs. Plan 05-02 uses these to drive
# tests/test_providers_openai.py respx-mocked adapter tests:
#   - cfg_openai_frontier: FrontierResolved (NOT FrontierConfig — see below)
#     instance for OpenAIProvider ctor
#   - STUB_OPENAI_CHAT_RESPONSE: canonical /v1/chat/completions success body
#     shaped to mirror the OpenAI typed response (openai.types.ChatCompletion)
#
# IMPORTANT: FrontierConfig (models.py:407-423) has model_config=extra="forbid"
# and the fields api_key_env / base_url / model / max_tokens ONLY — it does NOT
# carry an `api_key` field. Instantiating FrontierConfig(api_key=SecretStr(...))
# would raise `ValidationError: Extra inputs are not permitted`.
#
# FrontierResolved (config.py:86-106) SUBCLASSES FrontierConfig and ADDS the
# resolved `api_key: SecretStr` field; it is the shape load_config returns and
# the shape every downstream consumer (including OpenAIProvider.__init__)
# types against. Phase 5 tests therefore instantiate FrontierResolved DIRECTLY
# so api_key is present without relying on the env-var resolution path. See
# 05-PATTERNS.md lines 789-797 for the rationale lock.

# Canonical OpenAI /v1/chat/completions success body — mirrors STUB_LMSTUDIO_RESPONSE
# in shape but carries a versioned `gpt-4o-2024-08-06` model id (OpenAI echoes
# the versioned id, not the client-requested alias) and a non-null
# `system_fingerprint` so tests asserting its presence have material to work
# with. Matches openai.types.ChatCompletion structurally: id starts with
# chatcmpl-, object=='chat.completion', created is unix int, choices[0] has
# finish_reason='stop' + message{role,content}, usage block with all three
# token counts.
STUB_OPENAI_CHAT_RESPONSE: dict[str, object] = {
    "id": "chatcmpl-openai-upstream-id",
    "object": "chat.completion",
    "created": 1731990317,
    "model": "gpt-4o-2024-08-06",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "frontier answer"},
            "finish_reason": "stop",
        },
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    "system_fingerprint": "fp_openai_abc",
}


@pytest.fixture
def cfg_openai_frontier():
    """FrontierResolved fixture for Phase 5 OpenAIProvider tests (D-18 layer 1).

    Uses a sentinel sk-test-... api_key so OBS-02 redaction tests fire on it.
    base_url=https://api.openai.com/v1 so respx routes mount at the real OpenAI
    path for production-like test shape.

    Locked to FrontierResolved (config.py:86-106) per 05-PATTERNS.md lines
    789-797: FrontierConfig (models.py:407-423) has extra='forbid' and lacks
    an `api_key` field, so instantiating FrontierConfig(api_key=...) would
    raise ValidationError. FrontierResolved is the subclass that carries the
    resolved SecretStr api_key and is the shape OpenAIProvider.__init__
    types against.
    """
    from pydantic import SecretStr

    from corethread.config import FrontierResolved

    return FrontierResolved(
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        max_tokens=512,
        api_key=SecretStr("sk-test-1234567890ABCDEFGHIJ"),
    )


# ---------------------------------------------------------------------------
# Phase 5 / Plan 05-07 — FakeProvider-backed orchestrator fixture for the
# D-19 smoke-test extensions in tests/test_app_smoke.py
# ---------------------------------------------------------------------------
#
# This fixture boots `main.app` via the existing `valid_yaml_path` +
# `set_test_api_key` autouse flow and yields a `TestClient` that the caller
# enters in its own `with` block. After the lifespan has booted (which
# constructs the real OllamaProvider + OpenAIProvider + Orchestrator on
# `app.state`), callers MONKEYPATCH `app.state.orchestrator` with the
# FakeProvider-backed shim parked on `client._fake_orchestrator`. This lets
# Plan 05-07 tests smoke the real POST /v1/chat/completions route (D-13) +
# Pydantic stream validator (D-11/D-12) wiring through the real ASGI stack
# WITHOUT hitting Ollama or OpenAI.
#
# Why a post-lifespan swap (not a ctor-level injection): Phase 1's lifespan
# constructs providers + Orchestrator as part of its atomic startup block;
# there is no DI seam. Starlette's `app.state` is a mutable namespace, so
# the cleanest test-surface override is "let lifespan run, then rebind
# `state.orchestrator` for the duration of the `with` block". The real
# frontier_provider + local_provider instances remain on `app.state.{
# frontier,local}_provider` so D-14 lifespan-type assertions still see
# OpenAIProvider in that slot.
#
# Callers use the fixture like this:
#
#     def test_foo(app_with_fake_orchestrator):
#         with app_with_fake_orchestrator as c:
#             from corethread import main as m
#             m.app.state.orchestrator = (
#                 app_with_fake_orchestrator._fake_orchestrator
#             )
#             r = c.post("/v1/chat/completions", json={...})
#
# The swap MUST happen AFTER the TestClient enters its `with` block
# (which is when the lifespan fires and `app.state.orchestrator` is first
# set); before that point there is no attribute to overwrite.


@pytest.fixture
def app_with_fake_orchestrator(
    valid_yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Boot main.app with a FakeProvider-backed orchestrator shim (Plan 05-07).

    The real lifespan runs (constructs real OpenAIProvider + OllamaProvider
    on app.state.{frontier,local}_provider, sets app.state.models_created_at,
    etc.) but callers monkeypatch app.state.orchestrator with a minimal
    stand-in whose .handle(request) returns a canned ChatCompletionResponse.
    This lets Plan 05-07 tests smoke the real route wiring (Plan 05-04) WITHOUT
    hitting Ollama or OpenAI.

    The real Ollama warmup still runs at lifespan-enter time; callers that
    want the local side to report "ready" must register a respx route for
    the Ollama `POST /api/chat` warmup ping (see test_app_smoke.py
    `test_health_flips_to_ok` for the inline pattern). Tests that only need
    to assert the chat route returns 200 via the fake orchestrator tolerate
    a degraded local-warmup (the /v1/chat/completions route itself is still
    reachable).

    Returns a TestClient that the caller enters with a `with` block. A
    duck-typed `_FakeOrchestrator` instance is parked on
    `client._fake_orchestrator` so callers can swap it into
    `m.app.state.orchestrator` inside their own `with` block (which is
    when the attribute first exists after lifespan boot).
    """
    import importlib

    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(valid_yaml_path))
    from corethread import main as m

    importlib.reload(m)

    from corethread.models import ChatCompletionRequest, ChatCompletionResponse
    from corethread.streaming import response_to_stream_chunks
    from tests._fakes import make_happy_local_response

    class _FakeOrchestrator:
        """Duck-type of orchestrator.Orchestrator — only .handle() is called
        by the POST /v1/chat/completions route per Plan 05-04 D-13."""

        async def handle(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
            return make_happy_local_response(
                model=request.model,
                content="fake-orchestrator-response",
            )

        async def stream(self, request: ChatCompletionRequest):
            response = await self.handle(request)
            for chunk in response_to_stream_chunks(response):
                yield chunk

    client = TestClient(m.app, raise_server_exceptions=False)
    # Callers swap this into `m.app.state.orchestrator` inside their
    # `with` block (the lifespan enters at `with` time and parks the real
    # Orchestrator there; this attribute gives callers the replacement).
    client._fake_orchestrator = _FakeOrchestrator()  # type: ignore[attr-defined]

    yield client


# ---------------------------------------------------------------------------
# Phase 5 / Plan 05-08 — stock openai.AsyncOpenAI SDK bridged onto CoreThread's
# real FastAPI ASGI app for D-18 layer 3 end-to-end tests.
# ---------------------------------------------------------------------------
#
# This fixture is the "No Analog Found" pattern flagged in
# .planning/phases/05-frontier-end-to-end-route/05-PATTERNS.md: it is the
# first in-repo example of wiring a stock `openai.AsyncOpenAI` SDK client
# through `httpx.ASGITransport(app=main.app)` at CoreThread's real FastAPI
# route layer — NO real network, NO uvicorn, NO localhost socket.
# respx (installed via per-test `with respx.mock(...)` blocks) intercepts
# the OUTBOUND `https://api.openai.com/*` calls only; CoreThread itself
# runs through the real ASGI stack for maximal production fidelity.
#
# Lifespan handling (IMPORTANT):
# --------------------------------
# The plan's original prescription was
#   httpx.ASGITransport(app=m.app, lifespan="on")
# but httpx 0.28.1's ASGITransport.__init__ does NOT accept a `lifespan`
# kwarg (verified via `inspect.signature`), and the transport's
# `handle_async_request` only emits `type: "http"` scopes — it never
# dispatches the `lifespan.startup` / `lifespan.shutdown` ASGI events.
# Consequence: `httpx.AsyncClient(transport=ASGITransport(app))` as an
# async context manager does NOT fire FastAPI's lifespan, so
# `app.state.orchestrator` / `app.state.frontier_provider` would never
# be populated and every subsequent chat-completions call would raise
# AttributeError inside the real route.
#
# Deviation (Rule 3 — blocking issue auto-fix): drive the lifespan
# MANUALLY via `m.app.router.lifespan_context(m.app)` (the Starlette
# router's async context manager that emits lifespan.startup → body →
# lifespan.shutdown deterministically). This gives the same effect as
# the plan's prescription without requiring a missing httpx feature or
# a new `asgi-lifespan` dependency.
#
# Yields (sdk_client, main_module) so tests can both call
# `sdk_client.chat.completions.create(...)` AND manipulate
# `main_module.app.state.orchestrator` for fake-injection while still
# calling the stock SDK against the real FastAPI ASGI stack.


@pytest_asyncio.fixture
async def asgi_openai_sdk_client(
    valid_yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[Any, Any]]:
    """ASGI-bridged stock openai.AsyncOpenAI client for D-18 layer 3 e2e tests.

    Pattern (net-new — first in-repo example per 05-PATTERNS.md
    "No Analog Found"):

        # lifespan driven manually (see module-level comment for the
        # httpx-0.28-lacks-lifespan deviation)
        async with m.app.router.lifespan_context(m.app):
            httpx.ASGITransport(app=main.app)
            → httpx.AsyncClient(transport=transport, base_url="http://testserver")
            → openai.AsyncOpenAI(base_url="http://testserver/v1",
                                  api_key="sk-test-...",
                                  http_client=<async client>,
                                  max_retries=0)

    Yields (sdk_client, main_module) so tests can manipulate
    main_module.app.state.orchestrator for fake injection while still
    calling the stock SDK against the real FastAPI ASGI stack.

    CoreThread's config is loaded from valid_yaml_path (Ollama local +
    gpt-4o frontier + qwen2.5:7b judge). The set_test_api_key autouse
    fixture provides OPENAI_API_KEY=sk-test-... for lifespan boot.

    ``max_retries=0`` on the openai SDK client guarantees deterministic
    single-attempt behavior — critical for tests that assert
    ``respx.calls[0]`` and would fail non-deterministically if the SDK
    silently retried on transient ASGI-transport hiccups.

    ``base_url="http://testserver/v1"`` — the ``/v1`` suffix is IMPORTANT
    because the SDK prepends ``/chat/completions`` to the base_url; if
    base_url is ``http://testserver/``, the final URL is
    ``http://testserver/chat/completions`` which does NOT match
    CoreThread's ``POST /v1/chat/completions`` route.
    """
    import openai

    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(valid_yaml_path))
    from corethread import main as m

    importlib.reload(m)

    # Drive the FastAPI lifespan manually (httpx 0.28 ASGITransport does
    # not trigger lifespan events). This populates app.state.orchestrator,
    # app.state.frontier_provider, app.state.local_provider, app.state.config,
    # app.state.models_created_at — everything the real routes depend on.
    async with m.app.router.lifespan_context(m.app):
        asgi_transport = httpx.ASGITransport(app=m.app)
        async with httpx.AsyncClient(
            transport=asgi_transport, base_url="http://testserver"
        ) as http_client:
            sdk_client = openai.AsyncOpenAI(
                base_url="http://testserver/v1",
                api_key="sk-test-1234567890ABCDEFGHIJ",
                http_client=http_client,
                max_retries=0,  # deterministic test behavior; no SDK retry loops
            )
            yield (sdk_client, m)
