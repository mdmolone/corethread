"""Pure-logic orchestrator — local → judge → threshold → (maybe) frontier.

Phase 4 / Plan 03 per 04-CONTEXT.md D-01 through D-06, D-19 through D-22,
D-27, D-28. Closes ORC-01 (pure-logic — no HTTP, no JSON parsing, no retries,
fake-providers drivable), ORC-02 (providers behind the Provider ABC),
PIV-04 (judge==1 + frontier<=1 invariants), PIV-05 (unreachable vs timeout
distinction observable via pivot_reason + local_error_class).

The ``Orchestrator`` class is instantiated once in FastAPI lifespan and parked
on ``app.state.orchestrator`` (Phase 5 wiring). Phase 4 Plan 03 ships the code;
the route in ``main.py`` still returns 503 per D-25 — Phase 5 wires the route.

Design invariants (D-05):

  1. Structural — ``handle()`` has EXACTLY ONE ``await judge.grade(...)`` call
     site and EXACTLY ONE ``await self._frontier.chat(...)`` call site. No
     loops, no tenacity, no retry wrapper. Physically impossible to exceed
     PIV-04.
  2. Instance-local counters (``judge_calls``, ``frontier_calls``) increment
     immediately before the respective ``await``; final ``assert`` on every
     exit path catches any invariant violation that somehow slipped through
     (belt-and-suspenders — the assertion should never fire in correct code).
  3. Unit test coverage — ``tests/test_orchestrator.py`` exercises all 6
     branches with FakeProvider.call_count / FakeJudge.call_count bounds.

Branch table (D-19):

  pivot_reason = "none"          → happy path OR ProviderTimeout 504 re-raise
                                   OR ProviderHTTPError 502 re-raise
  pivot_reason = "low_score"      → judge real verdict with pass=False OR
                                    confidence_score < threshold
  pivot_reason = "local_truncated" → local.chat returned finish_reason="length";
                                    judge SKIPPED; auto-pivot
  pivot_reason = "local_error"    → local.chat raised ProviderUnavailable;
                                    judge SKIPPED; auto-pivot per CLAUDE.md
  pivot_reason = "judge_error"    → judge returned sentinel verdict
                                    (reasoning.startswith("judge parse failure:")
                                    — Phase 3 D-07 locked contract)

This module imports only:

  - stdlib (structlog, typing)
  - judge (for grade() — monkey-patch target in tests)
  - errors (typed Provider* exceptions)
  - models (ChatCompletionRequest, ChatCompletionResponse)
  - config (AppConfig — the typed runtime config; lives in config.py per
    Phase 1 / Plan 04. The plan's prescribed `from models import AppConfig`
    is a documentation error: AppConfig is a BaseSettings subclass that
    composes the four sub-models that DO live in models.py, but AppConfig
    itself ships in config.py. Importing here is safe — config.py imports
    only `errors` and `models`, so no circular dependency arises.)
  - obs (RequestTrace, emit_trace, time_block)
  - providers.base (Provider ABC)

MUST NOT import:

  - httpx (ORC-01: no HTTP)
  - json (ORC-01: no JSON parsing)
  - tenacity (ORC-01: no retries)
  - providers.ollama / providers.lmstudio / providers.openai
    (dispatch is lifespan's job; orchestrator holds Provider instances opaque)
  - main (circular-import guard — main imports orchestrator, not vice versa)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import structlog

from corethread import judge
from corethread.config import AppConfig
from corethread.errors import (
    CoreThreadError,
    ProviderHTTPError,
    ProviderTimeout,
    ProviderUnavailable,
)
from corethread.models import ChatCompletionRequest, ChatCompletionResponse
from corethread.obs import RequestTrace, emit_trace, time_block
from corethread.providers.base import Provider
from corethread.route_activity import RouteStage, emit_route_activity, route_activity_event
from corethread.streaming import include_usage_requested, response_to_stream_chunks

__all__ = ["Orchestrator"]

_LOG = structlog.get_logger(__name__)


def _response_finished_by_length(response: ChatCompletionResponse) -> bool:
    """Return true when the local answer hit the configured token limit."""
    return any(choice.finish_reason == "length" for choice in response.choices)


class Orchestrator:
    """Compose local + judge + frontier into a single request lifecycle.

    Constructor is keyword-only (D-02) so the two Provider references
    (``local``, ``frontier``) cannot be swapped by positional typo. Takes the
    full AppConfig so downstream access sites (``self._cfg.routing.threshold``,
    ``self._cfg.judge.model``) self-document.

    Phase 4 boundary (D-27): the frontier call passes the ORIGINAL request
    unchanged. Phase 5 inserts the constraint-prompt prefix + max_tokens cap
    at that same call site without orchestrator restructuring.
    """

    def __init__(
        self,
        *,
        local: Provider,
        judge_provider: Provider | None = None,
        frontier: Provider,
        cfg: AppConfig,
    ) -> None:
        self._local = local
        self._judge = judge_provider or local
        self._frontier = frontier
        self._cfg = cfg
        self._local_model = cfg.profile_for("local").model
        judge_profile = cfg.profile_for("judge")
        self._judge_model = judge_profile.model
        self._judge_prompt = cfg.judge.prompt
        self._judge_timeout_s = (
            judge_profile.timeout_s if judge_profile.timeout_s is not None else 10.0
        )
        self._frontier_model = cfg.profile_for("frontier").model

    def _frontier_cost_estimate(self, trace: RequestTrace) -> float | None:
        pricing = self._cfg.controls.pricing.get(
            self._frontier_model
        ) or self._cfg.controls.pricing.get("*")
        if pricing is None:
            return None
        return (
            trace["input_tokens"] * pricing.input_per_1m_tokens
            + trace["output_tokens"] * pricing.output_per_1m_tokens
        ) / 1_000_000

    def _emit_stage(
        self,
        request_id: str,
        stage: RouteStage,
        *,
        pivoted: bool | None = None,
        confidence_score: float | None = None,
        error_class: str | None = None,
    ) -> None:
        emit_route_activity(
            route_activity_event(
                request_id=request_id,
                stage=stage,
                selected_local_model=self._local_model,
                judge_model=self._judge_model,
                frontier_model=self._frontier_model,
                pivoted=pivoted,
                confidence_score=confidence_score,
                error_class=error_class,
            )
        )

    async def handle(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Run local → judge → threshold → (maybe) frontier for one request.

        Emits exactly one ``"request.decision"`` trace line per invocation
        via ``emit_trace(trace)`` on every exit path — including the 504/502
        re-raise paths (D-20, D-22). Raises only on ProviderTimeout or
        ProviderHTTPError from local.chat; never on judge failure (fail-safe
        to pivot per CLAUDE.md).
        """
        # D-05 mechanism 2: instance-local counters for belt-and-suspenders
        # invariant enforcement.
        judge_calls = 0
        frontier_calls = 0

        # Read request_id from structlog contextvars (Phase 3 D-16 middleware
        # binds it). Unit tests without middleware get "<missing>" fallback;
        # tests manually bind via `structlog.contextvars.bind_contextvars(...)`
        # if they need a specific value.
        request_id = structlog.contextvars.get_contextvars().get(
            "request_id",
            "<missing>",
        )

        # D-18: trace skeleton MUST include all 15 fields with sensible
        # defaults. Pitfall P4-2: forgetting a field → emit_trace's Pitfall-5
        # assertion fires and crashes the request.
        trace: RequestTrace = {
            "request_id": request_id,
            "selected_local_model": self._local_model,
            "judge_model": self._judge_model,
            "frontier_model": None,
            "confidence_score": 0.0,
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

        # -------------------------------------------------------------------
        # Step 1: local.chat — typed-error dispatch. One of three paths:
        #   (a) success → continue to judge
        #   (b) ProviderUnavailable → auto-pivot (skip judge); judge_calls=0
        #   (c) ProviderTimeout / ProviderHTTPError → D-20/D-22 partial-trace-
        #       emit-then-reraise; judge+frontier NOT called.
        # time_block is typed dict[str, Any] (obs.py D-15 loose typing);
        # TypedDict narrowing requires # type: ignore[arg-type].
        # -------------------------------------------------------------------
        local_response: ChatCompletionResponse | None = None
        try:
            self._emit_stage(request_id, "local_started")
            async with time_block(trace, "local_latency_ms"):  # type: ignore[arg-type]
                local_response = await self._local.chat(request)
            trace["input_tokens"] = local_response.usage.prompt_tokens
            trace["output_tokens"] = local_response.usage.completion_tokens
            self._emit_stage(request_id, "local_completed", pivoted=False)
        except ProviderUnavailable as exc:
            # D-19 local_error: auto-pivot per CLAUDE.md. judge SKIPPED.
            trace["pivot_reason"] = "local_error"
            trace["local_error_class"] = exc.__class__.__name__
            trace["pivoted"] = True
            self._emit_stage(
                request_id,
                "local_error",
                pivoted=True,
                error_class=exc.__class__.__name__,
            )
            _LOG.warning(
                "orchestrator.local_unavailable_pivot",
                provider=self._local.name,
                err_class=exc.__class__.__name__,
            )
            # Fall through to frontier call below (keep this try block shallow).
        except ProviderTimeout as exc:
            # D-20: emit PARTIAL trace then re-raise → 504 to client.
            # Order is LOAD-BEARING (Pitfall P4-1): emit FIRST, then raise.
            trace["local_error_class"] = exc.__class__.__name__
            self._emit_stage(
                request_id,
                "local_error",
                pivoted=False,
                error_class=exc.__class__.__name__,
            )
            emit_trace(trace)
            raise
        except ProviderHTTPError as exc:
            # D-22: emit PARTIAL trace then re-raise → 502 to client.
            trace["local_error_class"] = exc.__class__.__name__
            self._emit_stage(
                request_id,
                "local_error",
                pivoted=False,
                error_class=exc.__class__.__name__,
            )
            emit_trace(trace)
            raise

        # -------------------------------------------------------------------
        # Step 2: judge.grade — only if local.chat SUCCEEDED.
        # D-05 structural invariant: ONE call site, no loop, no retry.
        # -------------------------------------------------------------------
        if local_response is not None:
            if _response_finished_by_length(local_response):
                trace["pivot_reason"] = "local_truncated"
                trace["pivoted"] = True
            else:
                try:
                    self._emit_stage(request_id, "judge_started")
                    async with time_block(trace, "judge_latency_ms"):  # type: ignore[arg-type]
                        judge_calls += 1
                        verdict = await judge.grade(
                            request,
                            local_response,
                            provider=self._judge,
                            judge_model=self._judge_model,
                            timeout_s=self._judge_timeout_s,
                            system_prompt=self._judge_prompt,
                        )
                except CoreThreadError as exc:
                    self._emit_stage(
                        request_id,
                        "judge_error",
                        pivoted=False,
                        error_class=exc.__class__.__name__,
                    )
                    raise
                trace["confidence_score"] = verdict.confidence_score
                # D-07 sentinel detection (Phase 3 locked contract).
                judge_parse_failed = verdict.reasoning.startswith("judge parse failure:")
                trace["judge_parse_failed"] = judge_parse_failed

                # -----------------------------------------------------------------
                # Step 3: threshold check → branch.
                # -----------------------------------------------------------------
                threshold = self._cfg.routing.threshold
                if judge_parse_failed:
                    trace["pivot_reason"] = "judge_error"
                    trace["pivoted"] = True
                elif (not verdict.pass_) or verdict.confidence_score < threshold:
                    trace["pivot_reason"] = "low_score"
                    trace["pivoted"] = True
                self._emit_stage(
                    request_id,
                    "judge_completed",
                    pivoted=trace["pivoted"],
                    confidence_score=trace["confidence_score"],
                )
            if not trace["pivoted"]:
                # Happy path — return local response.
                # D-05 mechanism 2: strict assertion on happy-path exit.
                assert judge_calls == 1, (
                    f"PIV-04 violation: judge called {judge_calls} times on happy path"
                )
                assert frontier_calls == 0, (
                    f"PIV-04 violation: frontier called {frontier_calls} times on happy path"
                )
                emit_trace(trace)
                self._emit_stage(
                    request_id,
                    "completed",
                    pivoted=False,
                    confidence_score=trace["confidence_score"],
                )
                return local_response

        # -------------------------------------------------------------------
        # Step 4: frontier.chat — reached on pivot branches (local_error,
        # local_truncated, low_score, judge_error). D-05 structural invariant:
        # ONE call site.
        # D-27: Phase 4 passes the ORIGINAL request unchanged; Phase 5 adds
        # the constraint-prompt prefix + max_tokens cap here.
        # -------------------------------------------------------------------
        trace["frontier_model"] = self._frontier_model
        try:
            self._emit_stage(
                request_id,
                "frontier_started",
                pivoted=True,
                confidence_score=trace["confidence_score"],
            )
            async with time_block(trace, "frontier_latency_ms"):  # type: ignore[arg-type]
                frontier_calls += 1
                frontier_response = await self._frontier.chat(request)
        except CoreThreadError as exc:
            self._emit_stage(
                request_id,
                "frontier_error",
                pivoted=True,
                confidence_score=trace["confidence_score"],
                error_class=exc.__class__.__name__,
            )
            emit_trace(trace)
            raise
        # D-28: on pivot, trace tokens come from FRONTIER response.
        trace["input_tokens"] = frontier_response.usage.prompt_tokens
        trace["output_tokens"] = frontier_response.usage.completion_tokens
        trace["frontier_cost_est"] = self._frontier_cost_estimate(trace)
        self._emit_stage(
            request_id,
            "frontier_completed",
            pivoted=True,
            confidence_score=trace["confidence_score"],
        )

        # D-05 mechanism 2: final invariant check on pivot path. Pitfall P4-6:
        # judge_calls is 0 on local_error/local_truncated auto-pivot, 1 on
        # low_score/judge_error.
        # `in (0, 1)` accommodates both correctly.
        assert judge_calls in (0, 1), (
            f"PIV-04 violation: judge called {judge_calls} times (expected 0 or 1)"
        )
        assert frontier_calls == 1, (
            f"PIV-04 violation: frontier called {frontier_calls} times on pivot"
        )

        emit_trace(trace)
        self._emit_stage(
            request_id,
            "completed",
            pivoted=True,
            confidence_score=trace["confidence_score"],
        )
        return frontier_response

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream the final selected answer in OpenAI chat-completion chunks.

        Local and judge stay buffered so rejected local text is never emitted.
        If the local answer passes, it is emitted as a short one-shot stream.
        If the request pivots and the frontier provider exposes ``stream_chat``,
        frontier chunks are yielded as they arrive.
        """
        judge_calls = 0
        frontier_calls = 0
        include_usage = include_usage_requested(request)
        request_id = structlog.contextvars.get_contextvars().get(
            "request_id",
            "<missing>",
        )
        trace: RequestTrace = {
            "request_id": request_id,
            "selected_local_model": self._local_model,
            "judge_model": self._judge_model,
            "frontier_model": None,
            "confidence_score": 0.0,
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

        local_response: ChatCompletionResponse | None = None
        try:
            self._emit_stage(request_id, "local_started")
            async with time_block(trace, "local_latency_ms"):  # type: ignore[arg-type]
                local_response = await self._local.chat(request)
            trace["input_tokens"] = local_response.usage.prompt_tokens
            trace["output_tokens"] = local_response.usage.completion_tokens
            self._emit_stage(request_id, "local_completed", pivoted=False)
        except ProviderUnavailable as exc:
            trace["pivot_reason"] = "local_error"
            trace["local_error_class"] = exc.__class__.__name__
            trace["pivoted"] = True
            self._emit_stage(
                request_id,
                "local_error",
                pivoted=True,
                error_class=exc.__class__.__name__,
            )
            _LOG.warning(
                "orchestrator.local_unavailable_pivot",
                provider=self._local.name,
                err_class=exc.__class__.__name__,
            )
        except ProviderTimeout as exc:
            trace["local_error_class"] = exc.__class__.__name__
            self._emit_stage(
                request_id,
                "local_error",
                pivoted=False,
                error_class=exc.__class__.__name__,
            )
            emit_trace(trace)
            raise
        except ProviderHTTPError as exc:
            trace["local_error_class"] = exc.__class__.__name__
            self._emit_stage(
                request_id,
                "local_error",
                pivoted=False,
                error_class=exc.__class__.__name__,
            )
            emit_trace(trace)
            raise

        if local_response is not None:
            if _response_finished_by_length(local_response):
                trace["pivot_reason"] = "local_truncated"
                trace["pivoted"] = True
            else:
                try:
                    self._emit_stage(request_id, "judge_started")
                    async with time_block(trace, "judge_latency_ms"):  # type: ignore[arg-type]
                        judge_calls += 1
                        verdict = await judge.grade(
                            request,
                            local_response,
                            provider=self._judge,
                            judge_model=self._judge_model,
                            timeout_s=self._judge_timeout_s,
                            system_prompt=self._judge_prompt,
                        )
                except CoreThreadError as exc:
                    self._emit_stage(
                        request_id,
                        "judge_error",
                        pivoted=False,
                        error_class=exc.__class__.__name__,
                    )
                    raise
                trace["confidence_score"] = verdict.confidence_score
                judge_parse_failed = verdict.reasoning.startswith("judge parse failure:")
                trace["judge_parse_failed"] = judge_parse_failed

                threshold = self._cfg.routing.threshold
                if judge_parse_failed:
                    trace["pivot_reason"] = "judge_error"
                    trace["pivoted"] = True
                elif (not verdict.pass_) or verdict.confidence_score < threshold:
                    trace["pivot_reason"] = "low_score"
                    trace["pivoted"] = True
                self._emit_stage(
                    request_id,
                    "judge_completed",
                    pivoted=trace["pivoted"],
                    confidence_score=trace["confidence_score"],
                )
            if not trace["pivoted"]:
                assert judge_calls == 1, (
                    f"PIV-04 violation: judge called {judge_calls} times on happy path"
                )
                assert frontier_calls == 0, (
                    f"PIV-04 violation: frontier called {frontier_calls} times on happy path"
                )
                emit_trace(trace)
                self._emit_stage(
                    request_id,
                    "completed",
                    pivoted=False,
                    confidence_score=trace["confidence_score"],
                )
                for chunk in response_to_stream_chunks(
                    local_response,
                    include_usage=include_usage,
                ):
                    yield chunk
                return

        trace["frontier_model"] = self._frontier_model
        try:
            self._emit_stage(
                request_id,
                "frontier_started",
                pivoted=True,
                confidence_score=trace["confidence_score"],
            )
            stream_chat = getattr(self._frontier, "stream_chat", None)
            if callable(stream_chat):
                frontier_response: ChatCompletionResponse | None = None

                def on_final(response: ChatCompletionResponse) -> None:
                    nonlocal frontier_response
                    frontier_response = response

                async with time_block(trace, "frontier_latency_ms"):  # type: ignore[arg-type]
                    frontier_calls += 1
                    async for chunk in stream_chat(request, on_final=on_final):
                        yield chunk

                if frontier_response is not None:
                    trace["input_tokens"] = frontier_response.usage.prompt_tokens
                    trace["output_tokens"] = frontier_response.usage.completion_tokens
                    trace["frontier_cost_est"] = self._frontier_cost_estimate(trace)
            else:
                async with time_block(trace, "frontier_latency_ms"):  # type: ignore[arg-type]
                    frontier_calls += 1
                    frontier_response = await self._frontier.chat(request)
                trace["input_tokens"] = frontier_response.usage.prompt_tokens
                trace["output_tokens"] = frontier_response.usage.completion_tokens
                trace["frontier_cost_est"] = self._frontier_cost_estimate(trace)
                for chunk in response_to_stream_chunks(
                    frontier_response,
                    include_usage=include_usage,
                ):
                    yield chunk
        except CoreThreadError as exc:
            self._emit_stage(
                request_id,
                "frontier_error",
                pivoted=True,
                confidence_score=trace["confidence_score"],
                error_class=exc.__class__.__name__,
            )
            emit_trace(trace)
            raise

        assert judge_calls in (0, 1), (
            f"PIV-04 violation: judge called {judge_calls} times (expected 0 or 1)"
        )
        assert frontier_calls == 1, (
            f"PIV-04 violation: frontier called {frontier_calls} times on pivot"
        )
        self._emit_stage(
            request_id,
            "frontier_completed",
            pivoted=True,
            confidence_score=trace["confidence_score"],
        )
        emit_trace(trace)
        self._emit_stage(
            request_id,
            "completed",
            pivoted=True,
            confidence_score=trace["confidence_score"],
        )
