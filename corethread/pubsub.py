"""In-process trace pub-sub. One TraceBus per FastAPI app. Lifecycle: instantiate in lifespan, call `obs.set_trace_bus(bus)`, park on `app.state.trace_bus`. Teardown: call `obs.set_trace_bus(None)` BEFORE the `finally`-block close. Bounded queue + drop-newest backpressure. Single-user, single-process, no external broker."""  # noqa: E501 — D-18 docstring locked verbatim

from __future__ import annotations

import asyncio
import collections
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog

from corethread.obs import RequestTrace

__all__ = ["TraceBus"]

_LOG = structlog.get_logger("corethread.pubsub")
_MAX_QUEUE = 100
_REPLAY_LAST_N = 50


class TraceBus:
    """In-process pub-sub bus for RequestTrace events.

    See module docstring for lifecycle. Producer side: `publish_nowait()`
    called from `obs.emit_trace()` in the orchestrator request path —
    MUST NOT block, MUST NOT raise. Consumer side: `async with subscribe()`
    yields a bounded asyncio.Queue seeded with up to `_REPLAY_LAST_N`
    prior traces (when replay=True).
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[RequestTrace]] = set()
        self._replay: collections.deque[RequestTrace] = collections.deque(maxlen=_REPLAY_LAST_N)
        # D-04: defends Pitfall 26 fan-out race — atomic replay-drain +
        # subscribers.add window during subscribe() handshake. Lock is
        # held for microseconds (deque iteration + set insert).
        self._lock = asyncio.Lock()

    def publish_nowait(self, trace: RequestTrace) -> None:
        """Fan out to every subscriber. Never blocks. Never raises.

        Drop policy on full queue: drop NEWEST event for THAT subscriber
        (slow subscriber's queue fills, recovery via Last-Event-ID replay
        in Phase 9). The replay deque ALWAYS receives the trace BEFORE
        the per-subscriber loop, so a dropped event still surfaces to
        a subscriber that reconnects within `_REPLAY_LAST_N` events.

        D-08: outer try/except Exception (NOT bare except — preserves
        KeyboardInterrupt / SystemExit). Belt-and-suspenders against
        a future code path adding a failure mode that escapes the
        inner per-queue try/except. This function runs inside
        `obs.emit_trace()` inside the orchestrator's request path —
        an unhandled exception here would 500 the user-facing chat
        completion (Pitfall #12 ethos).
        """
        try:
            self._replay.append(trace)
            # D-07: snapshot iteration — `list(self._subscribers)` defends
            # against `RuntimeError: Set changed size during iteration`
            # if a subscribe/unsubscribe lands on a different task during
            # publish. Combined with D-04's lock on the handshake side,
            # this is the full set-mutation defense.
            for q in list(self._subscribers):
                try:
                    q.put_nowait(trace)
                except asyncio.QueueFull:
                    # D-05: drop-newest. Operator-visible warning so a
                    # lagging SSE subscriber surfaces in the JSONL log.
                    _LOG.warning(
                        "pubsub.subscriber_lagging",
                        queue_size=q.qsize(),
                    )
        except Exception:  # D-08 swallow guarantee — deliberate broad catch
            # Belt-and-suspenders: SC#1 mandates "never raises". Any
            # failure mode that escapes the inner try/except is silently
            # absorbed so the producer (request path) is never poisoned.
            pass

    @asynccontextmanager
    async def subscribe(self, *, replay: bool = True) -> AsyncIterator[asyncio.Queue[RequestTrace]]:
        """Subscribe to the bus for the duration of the async-with block.

        Yields a bounded asyncio.Queue(maxsize=_MAX_QUEUE) seeded with
        the last `_REPLAY_LAST_N` traces (when replay=True; default).
        Phase 8's stats pump passes replay=False to avoid double-counting
        historical traces.

        D-04: replay-drain + subscribers.add are wrapped under self._lock
        so a publish_nowait racing with subscribe() cannot interleave
        mid-handshake and produce a "ghost dropped event" (Pitfall 26).

        try/finally guarantees `self._subscribers.discard(q)` runs on
        every exit path including cancellation (Pitfall 20 — SSE handler
        disconnect must release the queue).
        """
        q: asyncio.Queue[RequestTrace] = asyncio.Queue(maxsize=_MAX_QUEUE)
        async with self._lock:
            if replay:
                for past in list(self._replay):
                    try:
                        q.put_nowait(past)
                    except asyncio.QueueFull:
                        # Replay larger than queue size — should not
                        # happen given _REPLAY_LAST_N=50 < _MAX_QUEUE=100,
                        # but defensive in case constants ever drift.
                        break
            self._subscribers.add(q)
        try:
            yield q
        finally:
            self._subscribers.discard(q)
