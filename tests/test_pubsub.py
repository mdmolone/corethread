# ruff: noqa: I001 — D-17: keep `TraceBus, _MAX_QUEUE, _REPLAY_LAST_N` import order
# verbatim per Plan 07-01 acceptance criteria (test-internal constants follow
# the public class name in the documented import line).
"""Phase 7 / Plan 01 — TraceBus unit tests.

Closes ROADMAP SC#4 first three named tests:

- ``test_pubsub_publishes_to_subscribers`` — fan-out happy path.
- ``test_pubsub_drops_when_full`` — drop-newest backpressure + warning log.
- ``test_pubsub_replay_seeds_new_subscriber`` — replay deque seeds new
  subscribers with the last _REPLAY_LAST_N=50 events in order.

The two integration tests that bind ``obs._TRACE_BUS`` to a real bus
(``test_emit_trace_no_op_when_bus_unset`` and
``test_emit_trace_publishes_when_bus_set``) live in
``tests/test_obs_trace.py`` per CONTEXT D-09.
"""

from __future__ import annotations

import json

import pytest

from corethread.logging_config import setup_logging
from corethread.obs import RequestTrace
from corethread.pubsub import TraceBus, _MAX_QUEUE, _REPLAY_LAST_N


def _make_trace(request_id: str) -> RequestTrace:
    """Mint a complete RequestTrace dict so callers can assert by request_id.

    Mirrors the 15-field shape locked at v1.0 in obs.RequestTrace; this is
    the ONLY trace builder pubsub tests need (we never call emit_trace from
    these tests — that's tests/test_obs_trace.py's coverage).
    """
    return {
        "request_id": request_id,
        "selected_local_model": "llama3.1:8b",
        "judge_model": "qwen2.5:7b",
        "frontier_model": None,
        "confidence_score": 0.9,
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


async def test_pubsub_publishes_to_subscribers() -> None:
    """SC#4 #1: 3 traces published, single subscriber receives all 3
    in chronological order."""
    bus = TraceBus()
    async with bus.subscribe(replay=False) as q:
        for i in range(3):
            bus.publish_nowait(_make_trace(f"req-{i:03d}"))
        received = [q.get_nowait() for _ in range(3)]

    assert [t["request_id"] for t in received] == ["req-000", "req-001", "req-002"]
    assert q.empty()


async def test_pubsub_drops_when_full(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """SC#4 #2: 105 traces published to a queue with maxsize=_MAX_QUEUE=100;
    EXACTLY 100 received; pubsub.subscriber_lagging warning emitted at least
    once for each of the 5 overflow events."""
    setup_logging()  # rebind StreamHandler to current sys.stdout for capfd

    bus = TraceBus()
    async with bus.subscribe(replay=False) as q:
        for i in range(_MAX_QUEUE + 5):
            bus.publish_nowait(_make_trace(f"req-{i:04d}"))

        drained: list[RequestTrace] = []
        while not q.empty():
            drained.append(q.get_nowait())

    # Exactly _MAX_QUEUE traces survived.
    assert len(drained) == _MAX_QUEUE, f"expected {_MAX_QUEUE} traces, got {len(drained)}"

    # Warning log emitted on overflow events (5 expected).
    out, _err = capfd.readouterr()
    warning_count = 0
    for raw_line in out.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("event") == "pubsub.subscriber_lagging":
            warning_count += 1
    assert warning_count >= 5, (
        f"expected >=5 pubsub.subscriber_lagging warnings (one per overflow), got {warning_count}"
    )


async def test_pubsub_replay_seeds_new_subscriber() -> None:
    """SC#4 #3: 60 traces published with no subscribers; new subscriber
    with replay=True receives the LAST _REPLAY_LAST_N=50 in chronological
    order (the first 10 were evicted by the deque's maxlen)."""
    bus = TraceBus()

    # Publish 60 traces to a zero-subscriber bus.
    for i in range(60):
        bus.publish_nowait(_make_trace(f"req-{i:04d}"))

    # Subscribe with replay=True and drain.
    async with bus.subscribe(replay=True) as q:
        drained = [q.get_nowait() for _ in range(_REPLAY_LAST_N)]
        assert q.empty(), "replay should yield exactly _REPLAY_LAST_N traces"

    ids = [t["request_id"] for t in drained]
    # Last 50 means req-0010..req-0059 in chronological order.
    expected_ids = [f"req-{i:04d}" for i in range(60 - _REPLAY_LAST_N, 60)]
    assert ids == expected_ids, (
        f"replay order mismatch: first={ids[:3]}, last={ids[-3:]}, "
        f"expected first={expected_ids[:3]}, last={expected_ids[-3:]}"
    )
