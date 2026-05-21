"""Pure-logic in-memory stats aggregator. Stdlib-only, no I/O, no asyncio primitives in the data path. Single-process, single-user. Stats reset on every CoreThread restart — no persistence. Phase 9 owns the _stats_pump task that bridges TraceBus subscriptions into ingest()."""  # noqa: E501 — D-12 docstring locked verbatim

from __future__ import annotations

import collections
import statistics
import time
from typing import Any, Final, Literal

from corethread.obs import RequestTrace

__all__ = ["StatsAggregator"]

# D-04 / Phase 7 D-06 mirror — hardcoded sizes as module constants, NOT
# threaded through AppConfig. SC#1 locks maxlen=1000; AppConfig threading
# is a v1.1.1+ tuning concern (deferred per CONTEXT.md <deferred>).
_DEFAULT_MAXLEN: Final[int] = 1000

# STA-07 wire lock — the literal note string is the operator-visible
# "stats are session-scoped" affordance. Phase 10 UI renders it verbatim.
# Test gate is a literal-equality assertion (D-09 #6); CHANGING THIS
# STRING WOULD VIOLATE STA-07.
_NOTE: Final[str] = "Stats reset on CoreThread restart (in-memory)."

# D-01 / D-03 — window-to-seconds map for _filter_by_window.
# "all" intentionally absent — the filter short-circuits on it.
_WINDOW_SECONDS: Final[dict[str, int]] = {
    "1h": 3600,
    "24h": 86400,
    "7d": 604800,
}

# D-07 — the pivot_reason buckets, pre-seeded into snapshot()['pivot_reasons']
# so the dict ALWAYS has all keys even on quiet windows. `collections.Counter`
# alone only emits observed keys; this constant + the dict-comprehension
# rebuild in snapshot() (below) closes the "all buckets" half of D-07.
# Order matches the obs.RequestTrace.pivot_reason Literal definition.
_PIVOT_REASON_BUCKETS: Final[tuple[str, ...]] = (
    "none",
    "low_score",
    "local_truncated",
    "local_error",
    "judge_error",
)

Window = Literal["1h", "24h", "7d", "all"]
"""D-01: the time-window literal is the cross-phase contract with Phase 9.
Phase 9's GET /v1/stats endpoint validates the ?window=... query string
into this literal and passes it through to `snapshot(window=...)`.
NOT in __all__ per D-11 conservatism; Phase 9 imports it explicitly
(`from corethread.stats import Window` is the documented coupling)."""


class StatsAggregator:
    """In-memory rolling-window stats aggregator over RequestTrace events.

    See module docstring for lifecycle. Synchronous on every code path —
    SC#1 forbids asyncio primitives in the data path. Phase 9's
    `_stats_pump` task is the producer-side adapter that pumps bus events
    into `ingest()`; this class itself does not subscribe to anything.

    Snapshot shape (D-09 #1 + SC#4):
    - Empty window → 4 keys: window_started_at, window_size=0,
      max_window_size, note.
    - Populated window → 12 top-level keys: the empty-window 4 PLUS
      pivot_rate, pivot_reasons, tokens_in_total, tokens_out_total, and
      four quantile dicts (local_latency_ms, judge_latency_ms,
      frontier_latency_ms, confidence_score). `note` is always present —
      it is the STA-07 wire anchor.

    Each quantile dict is `{"p50", "p95", "max", "min"}` or `None` when
    the filtered list is empty (D-05 / D-08 None-on-empty contract).
    """

    def __init__(self, *, maxlen: int = _DEFAULT_MAXLEN) -> None:
        # D-03 — deque stores (ingest_timestamp, trace) tuples so the
        # aggregator owns its own time-of-ingestion. Zero changes to
        # RequestTrace shape (preserves ARC-01); zero changes to
        # orchestrator (preserves ARC-01); zero changes to obs.emit_trace.
        self._traces: collections.deque[tuple[float, RequestTrace]] = collections.deque(
            maxlen=maxlen
        )
        # D-02 — captured ONCE at construction; returned verbatim from
        # every snapshot() call regardless of the `window` arg. Phase 10
        # UI computes "session uptime" as Date.now()/1000 - this value.
        self._started_at: int = int(time.time())

    def ingest(self, trace: RequestTrace) -> None:
        """Record an ingest-timestamp + trace tuple. Synchronous. Cheap.

        Called by Phase 9's `_stats_pump` after `await q.get()` returns
        a trace from the bus. NO validation: RequestTrace shape is
        already enforced by `obs.emit_trace`'s Pitfall-5 frozenset diff
        on the producer side; re-validating here would be redundant.
        """
        self._traces.append((time.time(), trace))

    def snapshot(self, *, window: Window = "all") -> dict[str, Any]:
        """Compute a window-scoped snapshot. Synchronous. No I/O.

        D-01 — the `window` arg filters the deque internally before
        computing stats. Phase 9's GET /v1/stats endpoint is a thin
        wrapper that validates `?window=...` into this literal and
        passes it through; the math stays here.

        D-02 — `window_started_at` is `self._started_at` regardless of
        `window` arg. The window only affects which traces aggregate.
        """
        window_traces = _filter_by_window(self._traces, window)
        n = len(window_traces)

        # Empty-window contract (D-09 #1) — exactly 4 keys, no quantiles.
        if n == 0:
            return {
                "window_started_at": self._started_at,
                "window_size": 0,
                "max_window_size": self._traces.maxlen,
                "note": _NOTE,
            }

        # Populated-window contract (D-09 #2..#9) — full top-level keys.
        pivoted = sum(1 for t in window_traces if t["pivoted"])
        # D-07 — count observed pivot_reasons, then seed all buckets so quiet
        # windows still surface zeroes for every known reason (not just the
        # keys actually observed). Plain
        # dict[str, int] for JSON-friendliness; sum(values) == n holds.
        counts: collections.Counter[str] = collections.Counter(
            t["pivot_reason"] for t in window_traces
        )
        return {
            "window_started_at": self._started_at,
            "window_size": n,
            "max_window_size": self._traces.maxlen,
            # D-06 — denominator is the FULL window including 504/502
            # re-raise rows (pivot_reason="none" + local_error_class set).
            "pivot_rate": pivoted / n,
            "pivot_reasons": {b: counts.get(b, 0) for b in _PIVOT_REASON_BUCKETS},
            "tokens_in_total": sum(t["input_tokens"] for t in window_traces),
            "tokens_out_total": sum(t["output_tokens"] for t in window_traces),
            "local_latency_ms": _quantiles([t["local_latency_ms"] for t in window_traces]),
            "judge_latency_ms": _quantiles([t["judge_latency_ms"] for t in window_traces]),
            # D-08 — None-filter; empty-after-filter returns None for the
            # whole quantile dict (no pivots in window).
            "frontier_latency_ms": _quantiles(
                [
                    t["frontier_latency_ms"]
                    for t in window_traces
                    if t["frontier_latency_ms"] is not None
                ]
            ),
            # D-05 — exclude judge_parse_failed=True rows (the 0.0 sentinel
            # from Phase 3 D-07's fail-safe path does NOT pollute the
            # histogram). Empty-after-filter returns None.
            "confidence_score": _quantiles(
                [t["confidence_score"] for t in window_traces if t["judge_parse_failed"] is False]
            ),
            "note": _NOTE,
        }


def _filter_by_window(
    traces: collections.deque[tuple[float, RequestTrace]],
    window: Window,
) -> list[RequestTrace]:
    """D-01 + D-03 — return traces whose ingest timestamp is within the
    window cutoff. ``window == "all"`` short-circuits and returns every
    trace verbatim. Returns a list (not a generator) so callers can
    compute ``len(window_traces)`` and iterate multiple times for the
    per-quantile filter passes.
    """
    if window == "all":
        return [trace for (_ts, trace) in traces]
    cutoff = time.time() - _WINDOW_SECONDS[window]
    return [trace for (ts, trace) in traces if ts >= cutoff]


def _quantiles(values: list[float]) -> dict[str, float] | None:
    """Compute p50 / p95 / max / min for a numeric list. Returns ``None``
    on empty input (D-05 / D-08 None-on-empty contract; matches the
    ARCHITECTURE.md §2.3 sketch verbatim).

    p95 algorithm — sorted-list index ``int(len(s) * 0.95) - 1`` for
    n >= 20, else last element. The SC#4 large-sample test asserts this
    produces the expected element on a 1000-trace sample (index 949
    post-sort). Stdlib-only is non-negotiable per SC#1.
    """
    if not values:
        return None
    s = sorted(values)
    return {
        "p50": statistics.median(s),
        "p95": s[int(len(s) * 0.95) - 1] if len(s) >= 20 else s[-1],
        "max": s[-1],
        "min": s[0],
    }
