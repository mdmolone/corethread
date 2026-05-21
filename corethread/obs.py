"""Observability module — per-request JSONL decision trace + request_id middleware.

Phase 3 / Plan 02 per 03-CONTEXT.md. Closes OBS-01 at the code layer.

This module exports the building blocks Phase 4's orchestrator and Phase 3's
Judge (Plan 03-03) compose into a single structured log line per request:

- ``RequestTrace``: TypedDict with the 13 locked fields (D-13). Static type
  only — runtime enforcement is via the Pitfall-5 ``assert`` in
  ``emit_trace`` below, not Pydantic (the TypedDict is consumed by a
  kwargs-splat into ``structlog.info``, so a Pydantic model would be cast
  to dict anyway).
- ``emit_trace(trace)``: exactly one ``structlog.get_logger("corethread.obs")
  .info("request.decision", **trace)`` call per request. Inherits Phase 1's
  full processor chain (``merge_contextvars`` → ``add_log_level`` →
  ``TimeStamper`` → ``_structlog_redact_processor`` → ``JSONRenderer``)
  — no new formatter, no sidecar file, no rotation. Operators redirect
  stdout to a file if they want persistence (``uv run uvicorn main:app 2>&1
  | tee -a corethread.jsonl``).
- ``time_block(target, key)``: async context manager that records
  ``int(elapsed_ms)`` into ``target[key]`` on exit — clean exit AND
  exception exit alike (D-15). ``time.monotonic()`` is mandatory per the
  anti-pattern list in 03-RESEARCH.md (``time.time()`` can jump backward on
  NTP adjustment).
- ``register_request_id_middleware(app)``: FastAPI ``@app.middleware("http")``
  that reads ``X-Request-ID`` or generates ``uuid4().hex[:16]`` (64-bit,
  log-ergonomic) and binds it to ``structlog.contextvars`` so every log
  line emitted in the request's async task automatically carries the
  request_id. Uses Python's per-task ``contextvars`` scope so concurrent
  requests see isolated IDs (verified in-session per 03-RESEARCH.md Example 3).

This module sits near the top of the dependency graph: it imports from
``structlog`` + stdlib + ``fastapi`` (for ``FastAPI`` / ``Request`` types
on the middleware factory). It MUST NOT import from ``judge``,
``orchestrator``, ``providers/*``, ``errors``, ``config``, or
``models``. The trace dict is built BY callers (orchestrator in Phase 4,
fake-driver test in Plan 03-05), so ``obs.py`` never peeks at config or
model types.

Phase 3 usage pattern (Plan 03-05 fake-driver test assembles this shape;
Phase 4 orchestrator produces it for real):

.. code-block:: python

    from corethread.obs import RequestTrace, emit_trace, time_block
    trace: RequestTrace = {"request_id": "...", ...}
    async with time_block(trace, "local_latency_ms"):
        local_resp = await local.chat(request)
    async with time_block(trace, "judge_latency_ms"):
        verdict = await grade(request, local_resp, provider=local, judge_model=jm)
    trace["pivoted"] = verdict.confidence_score < 0.7
    # ... frontier path populates frontier_* fields ...
    emit_trace(trace)
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal, TypedDict

import structlog
from fastapi import FastAPI, Request

# Phase 7 / Plan 03: forward-quoted to defer the import-time circularity check.
# `pubsub.py` imports `RequestTrace` from this module, so a runtime
# `from corethread.pubsub import TraceBus` here would create a circular import.
# Using `from __future__ import annotations` (line 56) means the string annotation
# below is never evaluated at runtime — type-checkers see TraceBus, the runtime
# stays decoupled.
if False:  # TYPE_CHECKING-equivalent gate — no runtime cost, no circular import
    from corethread.pubsub import TraceBus

__all__ = [
    "RequestTrace",
    "emit_trace",
    "register_request_id_middleware",
    "set_trace_bus",
    "time_block",
]


# D-14 event name — structured log consumers filter on this exact string.
# Example operator queries:
#   grep -c '"pivoted":true' corethread.jsonl
#   jq 'select(.event == "request.decision") | .confidence_score' corethread.jsonl
_TRACE_EVENT = "request.decision"

# D-16 header + entropy length (64 bits = 16 hex chars, collision-safe for a
# single-user local observability ID; shorter than full 32-hex for log-scanning
# ergonomics).
_REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_LEN = 16  # chars; corresponds to 64 bits of uuid4 entropy

# Module-level logger — name "corethread.obs" so operators can filter the
# observability subsystem with `corethread.obs.*` and distinguish it from
# provider logs (`providers.ollama`) and error logs (`corethread.error`).
_LOG = structlog.get_logger("corethread.obs")


# Phase 7 / Plan 03 (D-02 + D-03): module-global trace-bus tee. Wired by
# `corethread.main.lifespan` AFTER the orchestrator is built; torn down to
# `None` BEFORE `client.aclose()` in the same lifespan's finally block.
# v1.0 callers (no lifespan, e.g., direct unit tests of `emit_trace`) leave
# this as `None` and the broadcast in `emit_trace()` is a no-op — that is the
# explicit v1.0-compat contract documented in ARCHITECTURE.md §1.2.
_TRACE_BUS: TraceBus | None = None


def set_trace_bus(bus: TraceBus | None) -> None:
    """Wire (or unwire) the module-global trace bus.

    Called once at FastAPI lifespan startup with a live ``TraceBus`` instance
    AFTER orchestrator construction; called again in the lifespan's finally
    block with ``None`` BEFORE ``client.aclose()`` so any trace emitted during
    shutdown is a no-op rather than touching a half-closed bus (D-03).

    Tests that exercise the tee MUST call ``set_trace_bus(None)`` in a
    ``finally`` block (D-11) — even on assertion failure — to keep test
    isolation:

    .. code-block:: python

        bus = TraceBus()
        obs.set_trace_bus(bus)
        try:
            emit_trace(trace)
            ...  # assertions
        finally:
            obs.set_trace_bus(None)

    Single-process, single-FastAPI-app scope. Multi-app deployments are not
    in v1.1's scope (single-user local service per ARC-05).
    """
    global _TRACE_BUS
    _TRACE_BUS = bus


class RequestTrace(TypedDict, total=True):
    """Per-request decision trace — one JSONL line per request.

    Per CONTEXT.md D-13: ``total=True`` (the default in 3.12) — every key is
    required at the type level. Runtime enforcement is in ``emit_trace`` via
    the Pitfall-5 assertion (cheap frozenset diff).

    Fields (locked):

    - ``request_id``: str (from ``structlog.contextvars`` — but populated
      into the trace dict explicitly by the orchestrator for completeness
      in case the contextvars chain is ever bypassed).
    - ``selected_local_model``: str (``cfg.local.model`` at request time).
    - ``judge_model``: str (``cfg.judge.model``).
    - ``frontier_model``: str | None (``cfg.frontier.model`` when pivoted,
      else ``None``).
    - ``confidence_score``: float (from ``JudgeVerdict.confidence_score``
      after D-04 derivation).
    - ``pivoted``: bool (True iff orchestrator pivoted to frontier).
    - ``local_latency_ms``: int (elapsed in ``local.chat()`` — recorded by
      ``time_block``).
    - ``judge_latency_ms``: int (elapsed in ``judge.grade()``).
    - ``frontier_latency_ms``: int | None (elapsed in ``frontier.chat()``
      when pivoted, else ``None``).
    - ``input_tokens``: int (local prompt_tokens; orchestrator decides
      whether to use local or frontier counts on pivot).
    - ``output_tokens``: int (local completion_tokens, or frontier when
      pivoted).
    - ``frontier_cost_est``: float | None (Phase 5 populates via
      tokens-to-USD estimator; Phase 3 orchestrator + fake-driver write
      ``None`` on non-pivoted traces).
    - ``judge_parse_failed``: bool (True iff D-07 sentinel path was taken
      — distinguishes a low-confidence-real verdict from a judge-broke
      incident in post-hoc log analysis).
    - ``pivot_reason``: Literal["none", "low_score", "local_truncated",
      "local_error", "judge_error"] — Phase 4 D-18 fingerprint of why the
      orchestrator took its path. ``"none"`` on happy path AND on the 504/502
      re-raise paths.
    - ``local_error_class``: str | None — Phase 4 D-18 exception class name
      when local.chat raised a typed Provider error; None on happy path.

    SC#4 from ROADMAP.md enumerates the first 11 fields verbatim. The last
    two (``frontier_cost_est``, ``judge_parse_failed``) are Phase 3
    operator-useful additions agreed in CONTEXT.md D-13. Phase 4 D-18 adds
    ``pivot_reason`` + ``local_error_class`` to make SC#5's "timeout-504 vs
    unreachable-pivot distinction observable" a single-line trace fingerprint.
    """

    request_id: str
    selected_local_model: str
    judge_model: str
    frontier_model: str | None
    confidence_score: float
    pivoted: bool
    local_latency_ms: int
    judge_latency_ms: int
    frontier_latency_ms: int | None
    input_tokens: int
    output_tokens: int
    frontier_cost_est: float | None
    judge_parse_failed: bool
    pivot_reason: Literal[
        "none",
        "low_score",
        "local_truncated",
        "local_error",
        "judge_error",
    ]
    """D-18: fingerprint of WHY this request took its path. ``"none"`` on
    happy path AND on ProviderTimeout/ProviderHTTPError 504/502 re-raise paths.
    ``"low_score"`` = judge real verdict with pass=False OR score<threshold.
    ``"local_truncated"`` = local.chat returned finish_reason="length";
    judge skipped, auto-pivot.
    ``"local_error"`` = local.chat raised ProviderUnavailable; auto-pivot.
    ``"judge_error"`` = judge returned D-07 sentinel verdict; pivot."""

    local_error_class: str | None
    """D-18: exception class NAME (Pitfall #12 — class name only, NEVER str(exc))
    when local.chat raised ProviderUnavailable/ProviderTimeout/ProviderHTTPError.
    ``None`` on the happy path. Combined with ``pivot_reason`` these two fields
    form the operator jq fingerprint for PIV-05 distinction (SC#5)."""


# Pitfall-5 option B from 03-RESEARCH.md — cache the required-key set at
# import time for a zero-allocation missing-keys diff per emission.
_TRACE_REQUIRED_KEYS: frozenset[str] = frozenset(RequestTrace.__required_keys__)


def emit_trace(trace: RequestTrace) -> None:
    """Emit exactly one ``request.decision`` log line per request (D-14).

    Pitfall-5 option B: synchronously ``assert`` every required key is
    present before calling ``structlog.info`` — fails loud in dev + CI on
    Phase 4 orchestrator mistakes, NEVER emits a partial trace line.

    The kwargs-splat (``**trace``) lands in structlog's processor chain:
    ``merge_contextvars`` (picks up ``request_id`` from the middleware
    contextvar; if the caller also put it in the trace dict — recommended
    — the explicit kwarg wins via structlog's standard merge semantics),
    then ``_structlog_redact_processor`` (scrubs ``sk-*`` /
    ``Authorization: Bearer *`` in every nested string value including
    ``reasoning`` which may echo user content — Pitfall #12), then
    ``JSONRenderer`` to stdout.

    Use from Phase 4 orchestrator and Plan 03-05 fake-driver test; NOT
    called anywhere else in Phase 3.
    """
    missing = _TRACE_REQUIRED_KEYS - trace.keys()
    assert not missing, f"RequestTrace missing keys: {sorted(missing)}"
    _LOG.info(_TRACE_EVENT, **trace)
    # Phase 7 / Plan 03 — additive tee to the in-process trace bus. v1.0
    # behavior is preserved verbatim: structlog still emits exactly one JSONL
    # line above. The bus broadcast below is a no-op when ``_TRACE_BUS is
    # None`` (the v1.0-compat contract for callers without a lifespan). When
    # a bus is wired, ``publish_nowait()`` is non-blocking AND non-raising by
    # SC#1 contract — exceptions from the bus side cannot reach the
    # orchestrator's request path (Plan 01 D-08 swallow guarantee).
    if _TRACE_BUS is not None:
        _TRACE_BUS.publish_nowait(trace)


@asynccontextmanager
async def time_block(target: dict[str, Any], key: str) -> AsyncIterator[None]:
    """Record ``int(elapsed_ms)`` into ``target[key]`` on exit — D-15.

    Records on exception exit too, so partial traces preserve timing data
    when a provider call raises ``ProviderTimeout`` mid-generation. Phase 4's
    orchestrator top-level try/finally then emits the partial trace; the
    caller is responsible for populating the non-timed fields before emission.

    Uses ``time.monotonic()`` (NOT ``time.time()``) so NTP adjustments cannot
    produce backward jumps / negative elapsed values.

    ``target`` is annotated ``dict[str, Any]`` (not ``RequestTrace``) so
    callers can measure into partial builders or test scratch dicts without
    type-checker coupling; the integer assignment is structurally compatible
    with ``RequestTrace``'s int / int | None fields.
    """
    start = time.monotonic()
    try:
        yield
    finally:
        target[key] = int((time.monotonic() - start) * 1000)


def register_request_id_middleware(app: FastAPI) -> None:
    """Register the ``X-Request-ID`` middleware on ``app`` — D-16.

    MUST be called IMMEDIATELY after ``app = FastAPI(...)`` and BEFORE any
    other middleware (Pitfall #4 from 03-RESEARCH.md). FastAPI / Starlette
    middleware ordering is OUTER-LAST: a middleware registered AFTER this
    would wrap (and potentially log before) this one runs — the
    ``request_id`` wouldn't be bound yet and those log lines would lack
    ``request_id``.

    Contract:

    - Reads the ``X-Request-ID`` request header; if absent or empty,
      generates ``uuid.uuid4().hex[:16]`` (64 bits).
    - Calls ``structlog.contextvars.bind_contextvars(request_id=...)`` so
      every ``structlog.get_logger(...).<level>(...)`` call inside the
      request's async task automatically carries it via Phase 1's
      ``merge_contextvars`` processor (first in the chain — see
      ``logging_config.py`` line 222).
    - Uses ``try/finally`` to ensure ``unbind_contextvars("request_id")``
      runs even when the wrapped handler raises. Python's per-task
      contextvars scope means this unbind is defensive hygiene — a task
      that exits doesn't leak to a sibling task — but belt-and-suspenders
      matters for any future code path that reuses the task context (e.g.,
      a background task pool).

    T-03-03 (X-Request-ID spoofing): single-user local service — the
    caller IS the operator. Accepting caller-supplied IDs verbatim is
    correct for v1. If a future multi-tenant world lands, prefix supplied
    IDs with 4 hex chars of server-side entropy.
    """

    @app.middleware("http")
    async def _request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        raw = request.headers.get(_REQUEST_ID_HEADER)
        request_id = raw if raw else uuid.uuid4().hex[:_REQUEST_ID_LEN]
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            return await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
