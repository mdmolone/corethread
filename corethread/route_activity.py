"""Body-free live route activity events for the UI pipeline graphic."""

from __future__ import annotations

import asyncio
import collections
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal, TypedDict

import structlog

__all__ = [
    "RouteActivityBus",
    "RouteActivityEvent",
    "RouteStage",
    "emit_route_activity",
    "route_activity_event",
    "set_route_activity_bus",
]

RouteStage = Literal[
    "received",
    "local_started",
    "local_completed",
    "local_error",
    "judge_started",
    "judge_completed",
    "judge_error",
    "frontier_started",
    "frontier_completed",
    "frontier_error",
    "completed",
    "error",
]


class RouteActivityEvent(TypedDict, total=True):
    """Body-free event describing where one request is in the route pipeline."""

    request_id: str
    stage: RouteStage
    selected_local_model: str | None
    judge_model: str | None
    frontier_model: str | None
    pivoted: bool | None
    confidence_score: float | None
    error_class: str | None
    ts_ms: int


_LOG = structlog.get_logger("corethread.route_activity")
_MAX_QUEUE = 100
_REPLAY_LAST_N = 50
_ROUTE_ACTIVITY_BUS: RouteActivityBus | None = None


def route_activity_event(
    *,
    request_id: str,
    stage: RouteStage,
    selected_local_model: str | None,
    judge_model: str | None,
    frontier_model: str | None,
    pivoted: bool | None = None,
    confidence_score: float | None = None,
    error_class: str | None = None,
) -> RouteActivityEvent:
    return {
        "request_id": request_id,
        "stage": stage,
        "selected_local_model": selected_local_model,
        "judge_model": judge_model,
        "frontier_model": frontier_model,
        "pivoted": pivoted,
        "confidence_score": confidence_score,
        "error_class": error_class,
        "ts_ms": int(time.time() * 1000),
    }


def set_route_activity_bus(bus: RouteActivityBus | None) -> None:
    global _ROUTE_ACTIVITY_BUS
    _ROUTE_ACTIVITY_BUS = bus


def emit_route_activity(event: RouteActivityEvent) -> None:
    """Publish an activity event without ever affecting the request path."""
    if _ROUTE_ACTIVITY_BUS is None:
        return
    _ROUTE_ACTIVITY_BUS.publish_nowait(event)


class RouteActivityBus:
    """In-process pub-sub bus for body-free route activity events."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[RouteActivityEvent]] = set()
        self._replay: collections.deque[RouteActivityEvent] = collections.deque(
            maxlen=_REPLAY_LAST_N
        )
        self._lock = asyncio.Lock()

    def publish_nowait(self, event: RouteActivityEvent) -> None:
        try:
            self._replay.append(event)
            for q in list(self._subscribers):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    _LOG.warning("route_activity.subscriber_lagging", queue_size=q.qsize())
        except Exception:
            pass

    @asynccontextmanager
    async def subscribe(
        self,
        *,
        replay: bool = True,
    ) -> AsyncIterator[asyncio.Queue[RouteActivityEvent]]:
        q: asyncio.Queue[RouteActivityEvent] = asyncio.Queue(maxsize=_MAX_QUEUE)
        async with self._lock:
            if replay:
                for past in list(self._replay):
                    try:
                        q.put_nowait(past)
                    except asyncio.QueueFull:
                        break
            self._subscribers.add(q)
        try:
            yield q
        finally:
            self._subscribers.discard(q)
