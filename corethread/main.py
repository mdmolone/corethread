"""FastAPI application entry point — lifespan, shared httpx client, global error handler.

Phase 1 / Plan 06 — wires the FastAPI skeleton:

- **Lifespan startup order (load-bearing):** `setup_logging()` runs FIRST so that any
  log records emitted by the subsequent `load_config()` failure path are scrubbed by
  the root `RedactingFilter` (PLAN-05). Then `load_config()` runs; on `ConfigError`
  the OpenAI-shaped error envelope is printed to stdout (CFG-03 — operator visibility)
  and the exception is re-raised so uvicorn exits non-zero. Finally a SINGLE shared
  `httpx.AsyncClient` is constructed with the locked 4-axis timeouts (5/120/10/5
  for connect / read / write / pool) per CONTEXT.md and parked on
  `app.state.http_client` for Phase 2+ providers to consume (Pitfall #11 mitigation).

- **Global exception handlers:** TWO `@app.exception_handler` registrations override
  FastAPI's default `{"detail": ...}` envelope. The typed `CoreThreadError` handler
  pattern-matches `exc.error_type` / `exc.error_code` / `exc.http_status` and returns
  the trusted `error_envelope(exc)` body. The generic `Exception` handler returns the
  generic `"Internal error"` envelope WITHOUT ever including the raw exception's
  string representation, repr, or args (Pitfall #12 — defends against
  `httpx.HTTPStatusError` and arbitrary upstream exception text leaking the
  `Authorization: Bearer sk-...` header).

- **Routes:**
    * `GET /health` — returns `{"status": "ok", "version": <str>, "providers": {}}`
      (CONTEXT.md decision: service-only at Phase 1; the empty `providers` dict is the
      hook Phase 2/4/5 will populate).
    * `POST /v1/chat/completions` — 503 stub with the verbatim CONTEXT.md envelope
      (`type=provider_unavailable`, `code=not_implemented`). Does NOT bind a Pydantic
      body — at Phase 1 there is no provider, so request validation is intentionally
      deferred to Phase 5 to keep the route's error-shape contract clean (T-01-32).
    * `GET /v1/models` — same 503 stub envelope.

- **CORETHREAD_CONFIG_PATH env var (ALL CAPS):** Operator override for the config path
  (default `config.yaml`). Mixed-case env-var names are brittle on Windows shells +
  CI runners; this stays consistent with `OPENAI_API_KEY`, `LOG_LEVEL`.

Closes Phase 1 success criteria #2 (CFG-03 fail-fast on bad config), #3 (API-03
`/health`), and #4 (API-06 OpenAI-shaped error envelope, no leakage).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import mimetypes
import os
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import SecretStr
from starlette.routing import Mount

from corethread import obs
from corethread.api_ui import router as ui_router
from corethread.config import FrontierResolved, ModelProfile, load_config
from corethread.controls import RequestControls
from corethread.errors import (
    ConfigError,
    CoreThreadError,
    ErrorCode,
    ErrorType,
    ProviderHTTPError,
    ProviderTimeout,
    ProviderUnavailable,
    error_envelope,
)
from corethread.logging_config import get_logger, setup_logging
from corethread.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    LocalConfig,
    ModelEntry,
    ModelsListResponse,
)
from corethread.obs import register_request_id_middleware
from corethread.orchestrator import Orchestrator
from corethread.providers.base import Provider
from corethread.providers.defaults import apply_profile_defaults
from corethread.providers.lmstudio import LMStudioProvider
from corethread.providers.ollama import OllamaProvider
from corethread.providers.openai import OpenAIProvider
from corethread.pubsub import TraceBus
from corethread.route_activity import (
    RouteActivityBus,
    emit_route_activity,
    route_activity_event,
    set_route_activity_bus,
)
from corethread.spa import SPAStaticFiles
from corethread.stats import StatsAggregator
from corethread.streaming import format_sse_data, format_sse_done
from corethread.transcripts import TranscriptStore, instrument_provider

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_VERSION = "1.0.0"  # in sync with pyproject.toml [project].version
_DEFAULT_CONFIG_PATH = "config.yaml"

# Verbatim per CONTEXT.md "FastAPI Skeleton & Error Envelope" decision.
# PLAN-07's tests will assert exact-string equality; Phase 5 swaps the body but
# this string defines the not-yet-wired contract for the entire Phase 1 surface.
_STUB_503_MESSAGE = "Provider integration is not yet wired in this build."

# Locked 4-axis timeout policy per CONTEXT.md (Pitfall #11 + #3 + #5).
# - connect = 5.0  : TCP handshake should complete fast on localhost / public APIs.
# - read    = 120.0: tolerates local LLM cold-start (research pitfall #3 — first
#                    Ollama generation after model load can take 13-46s on CPU).
# - write   = 10.0 : large multimodal payloads should still flush within 10s.
# - pool    = 5.0  : how long to wait for an available pooled connection.
# Phase 2+ providers inherit these via `app.state.http_client`.
_SHARED_HTTPX_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=120.0,
    write=10.0,
    pool=5.0,
)

# Pool sizing: 50 max conns is generous for a single-user local service but
# defends against socket exhaustion if a future caller fans out concurrent
# pivots. 30s keepalive matches Ollama's default `keep_alive` window.
_SHARED_HTTPX_LIMITS = httpx.Limits(
    max_connections=50,
    max_keepalive_connections=20,
    keepalive_expiry=30.0,
)


# ---------------------------------------------------------------------------
# Phase 11 / D-07 — module-level mimetypes registration. Runs ONCE at import
# time, BEFORE `app = FastAPI(...)` and BEFORE any StaticFiles mount caches
# the mime db. Defends Pitfalls 29 + 30 (.mjs / .wasm / .map MIME types).
# Inside-lifespan placement rejected (StaticFiles may already have cached the
# db); inside-class-init rejected (surprising side effect; harder to grep).
# ---------------------------------------------------------------------------
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/json", ".map")  # source maps


# ---------------------------------------------------------------------------
# Phase 9 / D-02 + D-17 — _stats_pump task body (consumer-side adapter)
# ---------------------------------------------------------------------------
#
# Subscribes to the TraceBus (replay=False — D-17 + Phase 7 D-19 — historical
# traces are NOT re-counted) and pumps each q.get() into stats.ingest(trace).
# Single bad ingest does NOT kill the pump (D-17 Pitfall #12 ethos applied
# to the consumer side): CancelledError re-raises so lifespan teardown works
# cleanly; all other exceptions log the class name only and continue.


async def _stats_pump(bus: TraceBus, stats: StatsAggregator) -> None:
    """Bus-to-aggregator bridge. Runs as an asyncio.Task created in lifespan.

    D-17 — class-name-only error logging (Pitfall #12). NEVER str(exc),
    NEVER exc.args. The bus producer is the orchestrator request path;
    a consumer failure must not propagate back through the bus.
    """
    log = get_logger("corethread.stats_pump")
    async with bus.subscribe(replay=False) as q:
        while True:
            try:
                trace = await q.get()
                stats.ingest(trace)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(
                    "stats_pump.ingest_failed",
                    exc_class=exc.__class__.__name__,
                )


async def _transcript_pump(bus: TraceBus, store: TranscriptStore) -> None:
    """Attach final RequestTrace decisions to captured request transcripts."""
    log = get_logger("corethread.transcript_pump")
    async with bus.subscribe(replay=False) as q:
        while True:
            try:
                trace = await q.get()
                store.record_trace(trace)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(
                    "transcript_pump.ingest_failed",
                    exc_class=exc.__class__.__name__,
                )


async def _controls_pump(bus: TraceBus, controls: RequestControls) -> None:
    """Bridge final request traces into quotas, cost counters, and audit logs."""
    log = get_logger("corethread.controls_pump")
    async with bus.subscribe(replay=False) as q:
        while True:
            try:
                trace = await q.get()
                await controls.record_trace(trace)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(
                    "controls_pump.record_failed",
                    exc_class=exc.__class__.__name__,
                )


def _default_base_url(provider: str) -> str:
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    if provider == "openai":
        return "https://api.openai.com/v1"
    return ""


def _api_key_env_for(profile: ModelProfile) -> str:
    if profile.api_key_env:
        return profile.api_key_env
    if profile.provider == "openrouter":
        return "OPENROUTER_API_KEY"
    return "OPENAI_API_KEY"


def _build_provider_from_profile(
    profile: ModelProfile,
    *,
    client: httpx.AsyncClient,
    constraint_prompt: str | None,
) -> Provider:
    if profile.provider in {"ollama", "lmstudio"}:
        local_kind = cast(Literal["ollama", "lmstudio"], profile.provider)
        local_cfg = LocalConfig(
            kind=local_kind,
            base_url=profile.base_url,
            model=profile.model,
            num_ctx_default=profile.num_ctx_default or 8192,
            num_ctx_overrides=profile.num_ctx_overrides,
        )
        provider: Provider
        if profile.provider == "ollama":
            provider = OllamaProvider(local_cfg, client)
        else:
            provider = LMStudioProvider(local_cfg, client)
        return apply_profile_defaults(provider, profile)

    env_name = _api_key_env_for(profile)
    env_value = os.environ.get(env_name)
    if not env_value:
        raise ConfigError(f"{profile.provider} API key env var '{env_name}' is not set.") from None
    frontier_cfg = FrontierResolved(
        api_key_env=env_name,
        base_url=profile.base_url or _default_base_url(profile.provider),
        model=profile.model,
        max_tokens=profile.max_tokens or 512,
        api_key=SecretStr(env_value),
    )
    provider = OpenAIProvider(
        frontier_cfg,
        constraint_prompt=constraint_prompt,
        client=client,
        provider_name=profile.provider,
    )
    return apply_profile_defaults(provider, profile)


# ---------------------------------------------------------------------------
# Lifespan — ordered startup: logging FIRST, then config, then shared httpx
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan — initializes logging, loads config, builds shared httpx.

    Order is load-bearing:

    1. ``setup_logging()`` — root-logger RedactingFilter installed BEFORE any
       other code runs. This means a malformed-config error path that emits log
       records (e.g., a future Pydantic validation error printed by uvicorn's
       reloader) is already scrubbed.

    2. ``load_config(path)`` — reads ``CORETHREAD_CONFIG_PATH`` (default
       ``config.yaml``). On ``ConfigError``, prints the OpenAI-shaped envelope
       to stdout and re-raises. uvicorn treats a lifespan exception as a
       startup failure and exits non-zero — closes CFG-03.

    3. ``httpx.AsyncClient`` — single shared instance with the locked 4-axis
       timeouts. Stored on ``app.state.http_client`` so Phase 2+ providers can
       inject it. ``aclose()`` runs in the finally block on shutdown — Pitfall
       #11 mitigation (Test 9 asserts ``is_closed``).
    """
    # (1) Logging FIRST — see PLAN-05 SUMMARY note: any subsequent error path
    # that emits log records is already filtered by the time we get to step 2.
    setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
    log = get_logger("corethread.lifespan")

    config_path = os.environ.get("CORETHREAD_CONFIG_PATH", _DEFAULT_CONFIG_PATH)

    # (2) Load + validate config. On failure: emit OpenAI-shape envelope to stdout
    # (operator visibility — same shape an HTTP client would see) AND re-raise so
    # uvicorn exits non-zero. CFG-03.
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        # `error_envelope()` for `CoreThreadError` uses `exc.message` (author-
        # controlled — never echoes upstream values per PLAN-04 design).
        envelope = error_envelope(exc)
        print(json.dumps(envelope), file=sys.stdout, flush=True)
        # Re-raise so uvicorn lifespan startup fails with non-zero exit.
        raise

    # Log structured "config_loaded" event — only the safe subset of config
    # fields, never the resolved SecretStr API key. The redaction filter would
    # scrub it anyway, but defense in depth (Pitfall #12).
    log.info(
        "config_loaded",
        local_kind=cfg.profile_for("local").provider,
        local_model=cfg.profile_for("local").model,
        judge_model=cfg.profile_for("judge").model,
        frontier_model=cfg.profile_for("frontier").model,
        threshold=cfg.routing.threshold,
    )

    # (3) Single shared httpx.AsyncClient — Pitfall #11. Created here, never
    # per-request. Phase 2 (Ollama), Phase 4 (LM Studio), Phase 5 (frontier)
    # all read this from `app.state.http_client`.
    client = httpx.AsyncClient(
        timeout=_SHARED_HTTPX_TIMEOUT,
        limits=_SHARED_HTTPX_LIMITS,
    )

    app.state.config = cfg
    app.state.config_path = Path(config_path)
    app.state.http_client = client
    app.state.version = _VERSION
    transcript_store = (
        TranscriptStore(maxlen=cfg.privacy.transcript_max)
        if cfg.privacy.capture_transcripts
        else None
    )
    app.state.transcript_capture_enabled = cfg.privacy.capture_transcripts
    app.state.transcript_store = transcript_store
    controls = RequestControls(cfg.controls)
    app.state.controls = controls

    # ------------------------------------------------------------------
    # (4) Provider instantiation + eager warmup.
    #
    # Model profiles are the source of truth for role dispatch. The legacy
    # local/frontier blocks remain in config.yaml for backward compatibility.
    # ------------------------------------------------------------------
    local_profile = cfg.profile_for("local")
    judge_profile = cfg.profile_for("judge")
    frontier_profile = cfg.profile_for("frontier")

    local_provider = _build_provider_from_profile(
        local_profile,
        client=client,
        constraint_prompt=None,
    )
    if cfg.role_profiles.judge == cfg.role_profiles.local:
        judge_provider = local_provider
    else:
        judge_provider = _build_provider_from_profile(
            judge_profile,
            client=client,
            constraint_prompt=None,
        )
    frontier_provider = _build_provider_from_profile(
        frontier_profile,
        client=client,
        constraint_prompt=cfg.routing.constraint_prompt,
    )
    if transcript_store is not None:
        instrument_provider(
            local_provider,
            store=transcript_store,
            role="local",
            judge_model=judge_profile.model if judge_provider is local_provider else None,
        )
        if judge_provider is not local_provider:
            instrument_provider(
                judge_provider,
                store=transcript_store,
                role="local",
                judge_model=judge_profile.model,
            )
        instrument_provider(
            frontier_provider,
            store=transcript_store,
            role="frontier",
            default_model_override=frontier_profile.model,
        )
    app.state.local_provider = local_provider
    app.state.judge_provider = judge_provider
    app.state.frontier_provider = frontier_provider
    log.info(
        "provider.instantiated",
        role="local",
        kind=local_provider.name,
        model=local_profile.model,
    )

    # Warm externally visible provider slots once. A separate judge provider is
    # allowed to fail lazily at request time, matching the old model-override path.
    log.info(
        "provider.instantiated",
        role="judge",
        kind=judge_provider.name,
        model=judge_profile.model,
    )
    log.info(
        "provider.instantiated",
        role="frontier",
        kind=frontier_provider.name,
        model=frontier_profile.model,
    )

    warmed_provider_ids: set[int] = set()
    for provider in (local_provider, frontier_provider):
        if id(provider) in warmed_provider_ids:
            continue
        warmed_provider_ids.add(id(provider))
        try:
            await provider.warmup()
        except (ProviderUnavailable, ProviderTimeout, ProviderHTTPError) as exc:
            log.warning(
                "warmup.failed",
                provider=provider.name,
                err_class=exc.__class__.__name__,
            )

    # D-24 step 5: Orchestrator construction. Keyword-only per D-02 so
    # local/frontier cannot be transposed by positional typo.
    app.state.orchestrator = Orchestrator(
        local=local_provider,
        judge_provider=judge_provider,
        frontier=frontier_provider,
        cfg=cfg,
    )
    log.info("orchestrator.instantiated")

    # ------------------------------------------------------------------
    # Phase 7 / Plan 03 (D-01 + D-02 + D-03): in-process trace bus wiring.
    #
    # The bus is a Phase 7 deliverable (Plan 07-01) that fans `RequestTrace`
    # events to UI subscribers (Phase 8 stats pump, Phase 9 SSE endpoint).
    # `obs.emit_trace()` already tees into the module-global `_TRACE_BUS`;
    # here we (a) instantiate exactly one bus per app, (b) wire the obs
    # module-global via the public setter, and (c) park the bus on
    # `app.state.trace_bus` so Phase 9's SSE handler can subscribe via
    # `request.app.state.trace_bus`.
    #
    # Ordering (load-bearing per D-01):
    #   - AFTER orchestrator construction (above) — so the bus is live for
    #     every request the orchestrator handles.
    #   - BEFORE `yield` — same reason; no request can race the wiring.
    #   - BEFORE `app.state.models_created_at = ...` (below) — keeps the
    #     v1.1 wiring grouped together at the bottom of the startup chain.
    #
    # Phase 9 will add `_stats_pump = asyncio.create_task(...)` here as well;
    # Phase 7's scope is the bus + obs tee only (D-01 trims §2.4's sketch).
    # ------------------------------------------------------------------
    bus = TraceBus()
    obs.set_trace_bus(bus)
    app.state.trace_bus = bus
    log.info("trace_bus.instantiated")

    route_activity_bus = RouteActivityBus()
    set_route_activity_bus(route_activity_bus)
    app.state.route_activity_bus = route_activity_bus
    log.info("route_activity_bus.instantiated")

    # ------------------------------------------------------------------
    # Phase 9 / D-02 + D-17 + D-18 — StatsAggregator + _stats_pump task.
    #
    # Order is load-bearing per D-02:
    #   - AFTER `app.state.trace_bus = bus` (above) — pump needs a live bus.
    #   - BEFORE `app.state.models_created_at` (below) — keeps the v1.1
    #     wiring grouped together at the bottom of startup.
    # The task name "stats_pump" surfaces in `asyncio.all_tasks()` for
    # operator debugging.
    # ------------------------------------------------------------------
    stats = StatsAggregator()
    app.state.stats = stats
    stats_task = asyncio.create_task(_stats_pump(bus, stats), name="stats_pump")
    controls_task = asyncio.create_task(_controls_pump(bus, controls), name="controls_pump")
    transcript_task = (
        asyncio.create_task(_transcript_pump(bus, transcript_store), name="transcript_pump")
        if transcript_store is not None
        else None
    )
    log.info("stats_aggregator.instantiated")
    if transcript_store is not None:
        log.info("transcript_store.instantiated", maxlen=transcript_store.maxlen)
    else:
        log.info("transcript_store.disabled")
    await asyncio.sleep(0)

    # D-09: capture the service-lifetime timestamp ONCE for use by
    # GET /v1/models. Same value across every entry; changes only on restart.
    app.state.models_created_at = int(time.time())

    # ------------------------------------------------------------------
    # Phase 11 / D-09 + D-10 + D-11 + D-12 — SPA mount registration +
    # ARC-04 mount-LAST hard assertion.
    #
    # Why mount inside lifespan startup (vs module-import time):
    #   The test suite's `app_with_config` fixture (tests/test_app_smoke.py)
    #   adds `@m.app.get(...)` test routes AFTER `importlib.reload(m)` and
    #   BEFORE TestClient enters the lifespan. If the SPA mount were
    #   registered at module-import time, those test routes would land
    #   AFTER the mount and the D-09 assertion would (correctly) fire.
    #   Mounting inside the lifespan startup body — the LAST step before
    #   `try: yield` — guarantees the mount is the last registered route
    #   regardless of test-fixture route injection.
    #
    # D-10: skip + warn when frontend/dist/ is missing. Backend-only dev
    #   (`uvicorn --reload` without `pnpm build`) uninterrupted; CI test
    #   runs that don't `pnpm build` first don't crash.
    # D-09 + D-12: hard-assert mount-LAST + reversed-scan against shadowing
    #   Mount('/'). Wrapped in `if app.state.spa_mounted:` so the dev-time
    #   skip path doesn't fire it.
    # ------------------------------------------------------------------
    if (Path("frontend/dist") / "index.html").exists():
        app.mount("/", SPAStaticFiles(directory="frontend/dist", html=True), name="ui")
        app.state.spa_mounted = True
    else:
        log.warning(
            "spa.disabled",
            reason="frontend/dist/ not found — run 'pnpm --dir frontend build' first",
        )
        app.state.spa_mounted = False

    if app.state.spa_mounted:
        assert isinstance(app.routes[-1], Mount) and app.routes[-1].name == "ui", (
            "SPA mount must be registered LAST (ARC-04)"
        )
        # D-12 reversed-scan: no Mount('/') shadowing the ui mount earlier.
        for route in reversed(app.routes[:-1]):
            assert not (isinstance(route, Mount) and route.path == "/"), (
                "Found shadowing Mount('/') before ui mount (ARC-04)"
            )

    try:
        yield
    finally:
        # Phase 9 / D-18 — teardown order is load-bearing:
        #   1. Cancel the pump FIRST so its bus.subscribe() context manager
        #      exits cleanly while the bus is still live (its finally block
        #      discards the queue from bus._subscribers).
        #   2. THEN unwire the bus (existing line below).
        #   3. THEN close httpx (existing line below).
        # contextlib.suppress per SIM105 — the cancel-and-await idiom for an
        # asyncio.Task that re-raises CancelledError on shutdown.
        stats_task.cancel()
        controls_task.cancel()
        if transcript_task is not None:
            transcript_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stats_task
        with contextlib.suppress(asyncio.CancelledError):
            await controls_task
        if transcript_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await transcript_task
        # Phase 7 / Plan 03 (D-03): unwire the module-global trace bus BEFORE
        # closing the shared httpx client. Any trace emitted during shutdown
        # (none expected — the orchestrator's path is quiesced by the time we
        # reach this finally — but defensively per D-03) becomes a no-op
        # rather than touching the bus during teardown. The lifespan's
        # `app.state.trace_bus` attribute is intentionally NOT deleted; the
        # module-global is the source of truth for the no-op contract.
        obs.set_trace_bus(None)
        set_route_activity_bus(None)
        # Always close, even on shutdown-after-exception. Pitfall #11 / Test 9.
        await client.aclose()
        log.info("shutdown_complete")


app = FastAPI(lifespan=lifespan, title="CoreThread", version=_VERSION)
# Phase 3 D-16: register X-Request-ID middleware IMMEDIATELY after app
# construction so no other future middleware can run before request_id is
# bound to structlog.contextvars (Pitfall #4 — middleware is a stack, not
# a queue; later-registered runs outer).
register_request_id_middleware(app)


# ---------------------------------------------------------------------------
# Global exception handlers — override FastAPI's default {"detail": ...} shape
# ---------------------------------------------------------------------------


# API-06: Pydantic body-validation errors must surface as the OpenAI-shape
# `{"error": {...}}` envelope, NOT FastAPI's default `{"detail": [...]}` 422
# shape. Validation errors are rendered as a generic 400 invalid_request_error
# envelope, never echoing untrusted Pydantic error loc/msg/ctx fields
# (Pitfall #12).
@app.exception_handler(RequestValidationError)
async def _request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Render Pydantic body-validation failures as OpenAI-shape 400 envelopes.

    All validation errors use 400 / `invalid_request_error` with a generic
    "request body failed validation" message. We do NOT echo Pydantic's
    loc/msg/ctx fields: they can contain user-supplied content, and
    round-tripping them would violate Pitfall #12.
    """
    # Generic body-validation failure — never echo untrusted Pydantic fields.
    get_logger("corethread.error").warning(
        "typed_error",
        type=ErrorType.INVALID_REQUEST_ERROR.value,
        code=ErrorCode.INVALID_REQUEST_ERROR.value,
        status=400,
        message="request body failed validation",
    )
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": "Request body failed validation.",
                "type": ErrorType.INVALID_REQUEST_ERROR.value,
                "code": ErrorCode.INVALID_REQUEST_ERROR.value,
            }
        },
    )


@app.exception_handler(CoreThreadError)
async def _corethread_error_handler(request: Request, exc: CoreThreadError) -> JSONResponse:
    """Handle every typed `CoreThreadError` with the trusted envelope.

    `exc.message` is always author-controlled (see `errors.py` constructors —
    `ProviderTimeout`, `ProviderHTTPError`, `ProviderUnavailable`, `JudgeParseError`,
    `ConfigError`). None of these constructors include upstream body text or
    response headers, so it is safe to pass them through `error_envelope(exc)`
    which uses `exc.message` directly.

    Logged at WARNING (not ERROR) — typed errors are expected outcomes
    (provider unavailable, judge parse failure, etc.) and ERROR level is
    reserved for the generic-exception path which represents a real bug.
    """
    get_logger("corethread.error").warning(
        "typed_error",
        type=exc.error_type.value,
        code=exc.error_code.value,
        status=exc.http_status,
        # `message` is author-controlled (no upstream text); the root
        # RedactingFilter would scrub it anyway as belt-and-suspenders.
        message=exc.message,
    )
    return JSONResponse(
        status_code=exc.http_status,
        content=error_envelope(exc),
    )


@app.exception_handler(Exception)
async def _generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Generic exception fallback — the Pitfall #12 boundary.

    CRITICAL: this handler MUST NOT include the raw exception's string
    representation, repr, or args in the response body OR the log fields.
    `httpx.HTTPStatusError` and similar exceptions stringify themselves with
    full request/response context — including the `Authorization: Bearer
    sk-...` header. The `error_envelope(exc)` helper for non-`CoreThreadError`
    exceptions returns the generic `"Internal error"` body with no upstream
    content whatsoever.

    Defense in depth: we log only `exc.__class__.__name__` (not `exc` itself).
    Even though the root RedactingFilter would scrub the resolved log record,
    a future structlog serializer change could bypass that scrubbing — so we
    never pipe untrusted exception text into structured-log values at all.

    NEVER add a kwarg to this `.error(...)` call that derives from `exc`,
    `request.url`, `request.headers`, `request.client`, or any other value
    that could carry user-supplied content. The contract enforced here is:
    only `exc.__class__.__name__` (a static Python identifier) crosses this
    boundary. `_structlog_redact_processor` only scrubs top-level `str` values
    — it does NOT recurse into nested dicts/lists — so a "helpful" debug
    field like `headers={...}` would bypass the scrubber entirely. If you
    need request context here, route it through `redact_string()` first.
    """
    get_logger("corethread.error").error(
        "unhandled_exception",
        exc_class=exc.__class__.__name__,  # class name only — never exc, str(exc), or exc.args
    )
    return JSONResponse(
        status_code=500,
        content=error_envelope(exc),  # generic envelope — never echoes exc text
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    """Service health probe. Per-provider shape locked in D-10.

    Returns::

        {
          "status": "ok" | "degraded",
          "version": "<str>",
          "providers": {
            "local": {
              "kind": "ollama",
              "state": "ready" | "warming" | "unhealthy",
              "last_error": null | "<exc_class_name>"
            }
          }
        }

    Aggregate ``status`` is ``"ok"`` iff every provider reports
    ``state == "ready"``. Phase 2 has a single local provider; Phase 4
    adds lmstudio in the "local" slot; Phase 5 adds "frontier" as a
    separate sibling provider block.

    Body intentionally omits dependency versions / internals (T-01-31).
    """
    # D-10 per-provider shape. Phase 2 has a single local provider;
    # Phase 4 adds lmstudio in the "local" slot; Phase 5 adds "frontier"
    # as a separate sibling provider block.
    # D-26/D-14: /health surfaces both local AND frontier slots. Phase 5 ships
    # OpenAIProvider which reports state="ready" whenever the client is
    # constructed successfully — aggregate flips to "ok" end-to-end as the
    # signal that the service is pivot-capable.
    local_health = await app.state.local_provider.health()
    frontier_health = await app.state.frontier_provider.health()
    providers_block: dict[str, Any] = {
        "local": dict(local_health),
        "frontier": dict(frontier_health),
    }
    # Aggregate status is "ok" iff EVERY provider reports state="ready".
    # OpenAIProvider reports state="ready" whenever the client constructs
    # successfully (D-14 — no liveness probe); the local provider reports
    # "ready" after a successful warmup ping.
    aggregate = "ok" if all(h["state"] == "ready" for h in providers_block.values()) else "degraded"
    return {
        "status": aggregate,
        "version": app.state.version,
        "providers": providers_block,
    }


def _stub_503_envelope() -> dict[str, dict[str, str]]:
    """Build the 503 not-implemented envelope for Phase 1 stub routes.

    Verbatim per CONTEXT.md "FastAPI Skeleton & Error Envelope" decision.
    Phase 5 swaps the route bodies for real handlers; the envelope shape and
    `code=not_implemented` value persist as the canonical 503 response that
    proves end-to-end envelope wiring before any provider exists.
    """
    return {
        "error": {
            "message": _STUB_503_MESSAGE,
            "type": ErrorType.PROVIDER_UNAVAILABLE.value,
            "code": ErrorCode.NOT_IMPLEMENTED.value,
        }
    }


async def _chat_stream_events(
    first_chunk: dict[str, Any],
    chunk_stream: AsyncIterator[dict[str, Any]],
) -> AsyncIterator[str]:
    """Yield OpenAI-compatible SSE frames after the route preflights chunk one."""
    yield format_sse_data(first_chunk)
    async for chunk in chunk_stream:
        yield format_sse_data(chunk)
    yield format_sse_done()


async def _empty_chat_stream_events() -> AsyncIterator[str]:
    """Fallback for an empty stream generator; OpenAI clients still expect DONE."""
    yield format_sse_done()


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """D-13: route body is the orchestrator call.

    - Pydantic validates the request body.
    - `stream=True` returns OpenAI-compatible server-sent events after the
      orchestrator has decided whether to keep local or pivot to frontier.
    - CoreThreadError subclasses bubble to `_corethread_error_handler`
      (OpenAI-shape error envelope).
    - Uncaught exceptions bubble to `_generic_error_handler` (500 envelope).
    """
    request_id = structlog.contextvars.get_contextvars().get(
        "request_id",
        "<missing>",
    )
    cfg = getattr(app.state, "config", None)
    if cfg is None:
        local_model = request.model
        judge_model = None
        frontier_model = None
    else:
        local_model = cfg.profile_for("local").model
        judge_model = cfg.profile_for("judge").model
        frontier_model = cfg.profile_for("frontier").model
    emit_route_activity(
        route_activity_event(
            request_id=request_id,
            stage="received",
            selected_local_model=local_model,
            judge_model=judge_model,
            frontier_model=frontier_model,
        )
    )
    try:
        await app.state.controls.admit(request_id=request_id, request=request)
    except CoreThreadError as exc:
        emit_route_activity(
            route_activity_event(
                request_id=request_id,
                stage="error",
                selected_local_model=local_model,
                judge_model=judge_model,
                frontier_model=frontier_model,
                error_class=exc.__class__.__name__,
            )
        )
        raise

    if request.stream is True:
        chunk_stream = app.state.orchestrator.stream(request)
        try:
            first_chunk = await anext(chunk_stream)
        except StopAsyncIteration:
            return StreamingResponse(
                _empty_chat_stream_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        return StreamingResponse(
            _chat_stream_events(first_chunk, chunk_stream),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return await app.state.orchestrator.handle(request)


@app.get("/v1/models", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
    """D-08 + D-09 + D-10: return configured local + frontier models.

    Judge model is EXCLUDED (internal control-loop detail, not routable).
    If local.model == frontier.model, dedupe to a single entry (D-08).
    `created` is the service-lifetime startup timestamp (D-09).
    `owned_by` is the `ModelEntry` class-locked Literal "corethread".
    """
    cfg = app.state.config
    created = app.state.models_created_at
    # D-08 dedupe: dict.fromkeys preserves first-seen order
    unique_ids = list(
        dict.fromkeys([cfg.profile_for("local").model, cfg.profile_for("frontier").model])
    )
    return ModelsListResponse(
        data=[ModelEntry(id=mid, created=created) for mid in unique_ids],
    )


# ---------------------------------------------------------------------------
# Phase 9 / D-04 — UI Backend Endpoints router. Registered AFTER all existing
# @app.get / @app.post decorators (chat_completions, list_models, health) so
# Phase 11's `app.mount("/", SPAStaticFiles(...))` ARC-04 mount-LAST contract
# has a clean anchor to land below. The order is reading-order hygiene only;
# FastAPI dispatch is path-match, not registration-order.
# ---------------------------------------------------------------------------
app.include_router(ui_router)


# ---------------------------------------------------------------------------
# Phase 11 / D-10 + D-11 — SPA mount registration moved INTO lifespan startup.
#
# The plan recipe (D-10 verbatim) called for the conditional `app.mount(...)`
# block here as the new LAST executable line of main.py. Bug found during
# Task 2 verification (Rule 1 - Bug): `tests/test_app_smoke.py`'s
# `app_with_config` fixture adds `@m.app.get(...)` test routes AFTER
# `importlib.reload(m)` and BEFORE TestClient enters the lifespan. With the
# mount registered at module-import time, those test routes land AFTER the
# mount and the D-09 ARC-04 assertion fires (correctly — the invariant IS
# violated for that app instance). The fix moves the mount into the lifespan
# startup body — the LAST step before `try: yield` — so it runs AFTER any
# fixture-injected routes. Semantically equivalent for production (mount
# happens at app startup); preserves the mount-LAST invariant for tests.
# `app.include_router(ui_router)` (Phase 9 D-04 anchor) stays as the LAST
# module-level executable line.
# ---------------------------------------------------------------------------
