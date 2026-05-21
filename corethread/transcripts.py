"""Bounded in-memory request transcript capture for local debugging.

The normal TraceEvent/SSE surface intentionally excludes prompts and response
text. This module keeps richer request/response bodies in a separate,
last-N buffer so UI callers can opt in by request_id without changing the
lightweight trace stream.
"""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

import structlog

from corethread.models import ChatCompletionRequest, ChatCompletionResponse
from corethread.obs import RequestTrace
from corethread.providers.base import ChatOptions, Provider

__all__ = ["TranscriptStore", "instrument_provider"]

TranscriptStage = Literal["local", "judge", "frontier"]


def _response_text(response: ChatCompletionResponse) -> str:
    if not response.choices:
        return ""
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def _request_messages(request: ChatCompletionRequest) -> list[dict[str, Any]]:
    return [msg.model_dump(mode="json", exclude_none=True) for msg in request.messages]


def _current_request_id() -> str | None:
    request_id = structlog.contextvars.get_contextvars().get("request_id")
    if not isinstance(request_id, str) or request_id == "<missing>":
        return None
    return request_id


class TranscriptStore:
    """Last-N in-memory transcript buffer keyed by request_id."""

    def __init__(self, *, maxlen: int = 25) -> None:
        if maxlen < 1:
            raise ValueError("TranscriptStore maxlen must be >= 1")
        self._maxlen = maxlen
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def _ensure(self, request_id: str) -> dict[str, Any]:
        now = time.time()
        item = self._items.get(request_id)
        if item is None:
            item = {
                "request_id": request_id,
                "created_at": now,
                "updated_at": now,
                "request_messages": [],
                "local_response": None,
                "judge_response": None,
                "frontier_response": None,
                "final_response": None,
                "final_response_source": "none",
                "trace": None,
                "errors": {},
            }
        else:
            item["updated_at"] = now
            self._items.pop(request_id, None)

        self._items[request_id] = item
        while len(self._items) > self._maxlen:
            self._items.popitem(last=False)
        return item

    def record_request(
        self,
        request_id: str,
        request: ChatCompletionRequest,
    ) -> None:
        item = self._ensure(request_id)
        item["request_messages"] = _request_messages(request)

    def record_response(
        self,
        request_id: str,
        *,
        stage: TranscriptStage,
        response: ChatCompletionResponse,
    ) -> None:
        item = self._ensure(request_id)
        item[f"{stage}_response"] = _response_text(response)
        errors = item["errors"]
        if isinstance(errors, dict):
            errors.pop(stage, None)
        if stage == "local" and item["frontier_response"] is None:
            item["final_response"] = item["local_response"]
            item["final_response_source"] = "local"
        elif stage == "frontier":
            item["final_response"] = item["frontier_response"]
            item["final_response_source"] = "frontier"

    def record_error(
        self,
        request_id: str,
        *,
        stage: TranscriptStage,
        error_class: str,
    ) -> None:
        item = self._ensure(request_id)
        errors = item["errors"]
        if isinstance(errors, dict):
            errors[stage] = error_class

    def record_trace(self, trace: RequestTrace) -> None:
        item = self._ensure(trace["request_id"])
        item["trace"] = dict(trace)
        if trace["pivoted"] and item["frontier_response"] is not None:
            item["final_response"] = item["frontier_response"]
            item["final_response_source"] = "frontier"
        elif not trace["pivoted"] and item["local_response"] is not None:
            item["final_response"] = item["local_response"]
            item["final_response_source"] = "local"

    def get(self, request_id: str) -> dict[str, Any] | None:
        item = self._items.get(request_id)
        if item is None:
            return None
        return copy.deepcopy(item)

    def latest(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        items = list(reversed(self._items.values()))
        if limit is not None:
            items = items[:limit]
        return copy.deepcopy(items)


def instrument_provider(
    provider: Provider,
    *,
    store: TranscriptStore,
    role: Literal["local", "frontier"],
    judge_model: str | None = None,
    default_model_override: str | None = None,
) -> Provider:
    """Patch one provider instance to capture transcripts while preserving identity."""

    original_chat = provider.chat
    original_stream_chat = getattr(provider, "stream_chat", None)

    def stage_for(model_override: str | None) -> TranscriptStage:
        if role == "frontier":
            return "frontier"
        if judge_model is not None and model_override == judge_model:
            return "judge"
        return "local"

    async def captured_chat(
        request: ChatCompletionRequest,
        *,
        options: ChatOptions | None = None,
        model_override: str | None = None,
        timeout_s: float = 120.0,
    ) -> ChatCompletionResponse:
        request_id = _current_request_id()
        stage = stage_for(model_override)
        if request_id is not None and stage == "local":
            store.record_request(request_id, request)

        effective_model_override = model_override
        if (
            role == "frontier"
            and effective_model_override is None
            and default_model_override is not None
        ):
            effective_model_override = default_model_override

        try:
            response = await original_chat(
                request,
                options=options,
                model_override=effective_model_override,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            if request_id is not None:
                store.record_error(
                    request_id,
                    stage=stage,
                    error_class=exc.__class__.__name__,
                )
            raise

        if request_id is not None:
            store.record_response(
                request_id,
                stage=stage,
                response=response,
            )
        return response

    provider.chat = captured_chat  # type: ignore[method-assign]
    if callable(original_stream_chat):

        async def captured_stream_chat(
            request: ChatCompletionRequest,
            *,
            options: ChatOptions | None = None,
            model_override: str | None = None,
            timeout_s: float = 120.0,
            on_final: Callable[[ChatCompletionResponse], None] | None = None,
        ) -> AsyncIterator[dict[str, Any]]:
            request_id = _current_request_id()
            stage = stage_for(model_override)
            effective_model_override = model_override
            if (
                role == "frontier"
                and effective_model_override is None
                and default_model_override is not None
            ):
                effective_model_override = default_model_override

            def capture_final(response: ChatCompletionResponse) -> None:
                if request_id is not None:
                    store.record_response(
                        request_id,
                        stage=stage,
                        response=response,
                    )
                if on_final is not None:
                    on_final(response)

            try:
                async for chunk in original_stream_chat(
                    request,
                    options=options,
                    model_override=effective_model_override,
                    timeout_s=timeout_s,
                    on_final=capture_final,
                ):
                    yield chunk
            except Exception as exc:
                if request_id is not None:
                    store.record_error(
                        request_id,
                        stage=stage,
                        error_class=exc.__class__.__name__,
                    )
                raise

        provider.stream_chat = captured_stream_chat  # type: ignore[attr-defined]
    return provider
