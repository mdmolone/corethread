"""Runtime request controls: rate limits, quotas, cost estimates, and audit events."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

from corethread.config import ControlsConfig, ModelPricing
from corethread.errors import QuotaExceeded, RateLimitExceeded
from corethread.models import ChatCompletionRequest
from corethread.obs import RequestTrace

__all__ = ["RequestControls"]

_LOG = structlog.get_logger("corethread.controls")


class RequestControls:
    """In-memory single-process request controls.

    Enforcement is deliberately global, not per-user: CoreThread is still a
    single-user local service. Clients that send the OpenAI `user` field get
    that value reflected into audit metadata, but it is not a tenancy boundary.
    """

    def __init__(
        self,
        cfg: ControlsConfig,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._cfg = cfg
        self._clock = clock
        self._lock = asyncio.Lock()
        now = self._clock()
        self._minute_key = _minute_key(now)
        self._day_key = _day_key(now)
        self._minute_requests = 0
        self._daily_requests = 0
        self._daily_tokens = 0
        self._daily_cost_usd = 0.0

    async def admit(self, *, request_id: str, request: ChatCompletionRequest) -> None:
        """Check request controls and reserve one request slot if allowed."""
        async with self._lock:
            self._rollover_locked()

            if (
                self._cfg.requests_per_minute is not None
                and self._minute_requests >= self._cfg.requests_per_minute
            ):
                await self._audit_locked(
                    "request.rejected",
                    request_id=request_id,
                    request=request,
                    reason="requests_per_minute",
                )
                raise RateLimitExceeded("Request rate limit exceeded.")

            if (
                self._cfg.daily_request_quota is not None
                and self._daily_requests >= self._cfg.daily_request_quota
            ):
                await self._audit_locked(
                    "request.rejected",
                    request_id=request_id,
                    request=request,
                    reason="daily_request_quota",
                )
                raise QuotaExceeded("Daily request quota exceeded.")

            if (
                self._cfg.daily_token_quota is not None
                and self._daily_tokens >= self._cfg.daily_token_quota
            ):
                await self._audit_locked(
                    "request.rejected",
                    request_id=request_id,
                    request=request,
                    reason="daily_token_quota",
                )
                raise QuotaExceeded("Daily token quota exceeded.")

            if (
                self._cfg.daily_cost_quota_usd is not None
                and self._daily_cost_usd >= self._cfg.daily_cost_quota_usd
            ):
                await self._audit_locked(
                    "request.rejected",
                    request_id=request_id,
                    request=request,
                    reason="daily_cost_quota_usd",
                )
                raise QuotaExceeded("Daily cost quota exceeded.")

            self._minute_requests += 1
            self._daily_requests += 1
            await self._audit_locked("request.accepted", request_id=request_id, request=request)

    async def record_trace(self, trace: RequestTrace) -> None:
        """Record final usage/cost from the orchestrator trace bus."""
        async with self._lock:
            self._rollover_locked()
            tokens_in = trace["input_tokens"]
            tokens_out = trace["output_tokens"]
            cost = self.estimate_trace_cost(trace)
            self._daily_tokens += tokens_in + tokens_out
            self._daily_cost_usd += cost
            await self._audit_locked(
                "request.completed",
                request_id=trace["request_id"],
                trace=trace,
                cost_usd=cost,
            )

    def estimate_trace_cost(self, trace: RequestTrace) -> float:
        """Estimate USD cost for the final selected model using configured rates."""
        model = trace["frontier_model"] or trace["selected_local_model"]
        pricing = _pricing_for_model(self._cfg.pricing, model)
        if pricing is None:
            return 0.0
        return _estimate_cost(
            input_tokens=trace["input_tokens"],
            output_tokens=trace["output_tokens"],
            pricing=pricing,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return current in-memory control counters for tests and diagnostics."""
        self._rollover_locked()
        return {
            "minute_key": self._minute_key,
            "day_key": self._day_key,
            "minute_requests": self._minute_requests,
            "daily_requests": self._daily_requests,
            "daily_tokens": self._daily_tokens,
            "daily_cost_usd": self._daily_cost_usd,
        }

    def _rollover_locked(self) -> None:
        now = self._clock()
        minute_key = _minute_key(now)
        if minute_key != self._minute_key:
            self._minute_key = minute_key
            self._minute_requests = 0
        day_key = _day_key(now)
        if day_key != self._day_key:
            self._day_key = day_key
            self._daily_requests = 0
            self._daily_tokens = 0
            self._daily_cost_usd = 0.0

    async def _audit_locked(
        self,
        event: str,
        *,
        request_id: str,
        request: ChatCompletionRequest | None = None,
        trace: RequestTrace | None = None,
        reason: str | None = None,
        cost_usd: float | None = None,
    ) -> None:
        if not self._cfg.audit_enabled:
            return
        payload: dict[str, Any] = {
            "ts": int(self._clock()),
            "event": event,
            "request_id": request_id,
            "day_key": self._day_key,
        }
        if reason is not None:
            payload["reason"] = reason
        if request is not None:
            payload.update(
                {
                    "requested_model": request.model,
                    "stream": request.stream is True,
                    "user": request.user,
                }
            )
            if self._cfg.audit_include_request_body:
                payload["request"] = request.model_dump(mode="json")
        if trace is not None:
            payload.update(
                {
                    "selected_local_model": trace["selected_local_model"],
                    "judge_model": trace["judge_model"],
                    "frontier_model": trace["frontier_model"],
                    "pivoted": trace["pivoted"],
                    "pivot_reason": trace["pivot_reason"],
                    "input_tokens": trace["input_tokens"],
                    "output_tokens": trace["output_tokens"],
                    "cost_usd": cost_usd if cost_usd is not None else 0.0,
                }
            )
        try:
            _append_jsonl(Path(self._cfg.audit_path), payload)
        except OSError as exc:
            _LOG.warning("audit.write_failed", exc_class=exc.__class__.__name__)


def _minute_key(now: float) -> int:
    return int(now // 60)


def _day_key(now: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(now))


def _pricing_for_model(
    pricing: dict[str, ModelPricing],
    model: str,
) -> ModelPricing | None:
    return pricing.get(model) or pricing.get("*")


def _estimate_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    pricing: ModelPricing,
) -> float:
    return (
        input_tokens * pricing.input_per_1m_tokens + output_tokens * pricing.output_per_1m_tokens
    ) / 1_000_000


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("at", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        fh.write("\n")
