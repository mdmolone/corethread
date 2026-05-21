# ruff: noqa: I001 — keep the test-internal `_NOTE, _DEFAULT_MAXLEN` import
# order verbatim per acceptance-criteria grep (mirrors tests/test_pubsub.py
# I001 suppression for the same reason).
"""Phase 8 / Plan 01 — StatsAggregator unit tests.

Closes ROADMAP SC#4 (six named tests) plus three CONTEXT.md decision-coverage
tests (D-09 items 7-9):

SC#4 (D-09 #1..#6):
- ``test_empty_snapshot`` — fresh aggregator, 4-key empty contract.
- ``test_pivot_rate_math`` — pivoted/total denominator including 504 re-raise.
- ``test_quantiles_small_sample`` — n<20 fallback branch (p95 = s[-1]).
- ``test_quantiles_large_sample`` — n>=20 sorted-index branch.
- ``test_deque_cap_at_maxlen_plus_one`` — oldest eviction at 1001 ingests.
- ``test_note_literal`` — STA-07 wire string locked verbatim.

Decision coverage (D-09 #7..#9):
- ``test_snapshot_filters_by_window`` — D-01 + D-03 window filter.
- ``test_confidence_quantiles_exclude_parse_failed`` — D-05 sentinel filter.
- ``test_pivot_reasons_counter_totals_to_window_size`` — D-07 invariant.

All tests are SYNCHRONOUS — StatsAggregator is fully synchronous per SC#1.
No asyncio, no respx, no TestClient. Pure-logic assertions.
"""

from __future__ import annotations

import pytest

from corethread.obs import RequestTrace
from corethread.stats import StatsAggregator, _NOTE, _DEFAULT_MAXLEN


def _make_trace(
    request_id: str = "req-test",
    *,
    pivoted: bool = False,
    pivot_reason: str = "none",
    confidence_score: float = 0.9,
    judge_parse_failed: bool = False,
    local_latency_ms: int = 100,
    judge_latency_ms: int = 50,
    frontier_latency_ms: int | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
    local_error_class: str | None = None,
) -> RequestTrace:
    """Mint a complete 15-field RequestTrace dict; callers override per test.

    Mirrors the v1.0-locked shape in obs.RequestTrace; defaults match a
    successful local-only response (pivoted=False, pivot_reason="none",
    confidence_score=0.9, judge_parse_failed=False, frontier_latency_ms=None).
    """
    return {
        "request_id": request_id,
        "selected_local_model": "llama3.1:8b",
        "judge_model": "qwen2.5:7b",
        "frontier_model": None,
        "confidence_score": confidence_score,
        "pivoted": pivoted,
        "local_latency_ms": local_latency_ms,
        "judge_latency_ms": judge_latency_ms,
        "frontier_latency_ms": frontier_latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "frontier_cost_est": None,
        "judge_parse_failed": judge_parse_failed,
        "pivot_reason": pivot_reason,  # type: ignore[typeddict-item]
        "local_error_class": local_error_class,
    }


# ---------------------------------------------------------------------------
# SC#4 named tests — D-09 #1..#6
# ---------------------------------------------------------------------------


def test_empty_snapshot() -> None:
    """D-09 #1 / SC#4: fresh aggregator returns the 4-key empty contract.

    Locks: window_started_at present (unix int), window_size=0,
    max_window_size=_DEFAULT_MAXLEN, note=_NOTE. Asserts NO quantile keys
    and NO pivot_rate / pivot_reasons / token totals (the empty contract
    is exactly 4 keys, no more).
    """
    agg = StatsAggregator()
    snap = agg.snapshot()

    assert snap["window_size"] == 0
    assert snap["max_window_size"] == _DEFAULT_MAXLEN
    assert snap["note"] == _NOTE
    assert isinstance(snap["window_started_at"], int)

    # Exactly 4 keys — no quantiles, no pivot_rate, no token totals.
    assert set(snap.keys()) == {
        "window_started_at",
        "window_size",
        "max_window_size",
        "note",
    }


def test_pivot_rate_math() -> None:
    """D-09 #2 / SC#4 + D-06: pivoted/total denominator includes 504 re-raise rows.

    Ingests 10 traces: 3 pivoted (`pivoted=True`), 6 happy-path (`pivoted=False,
    pivot_reason="none"`), 1 504 re-raise (`pivoted=False, pivot_reason="none",
    local_error_class="ProviderTimeout"`). Per D-06 the 504 row counts as
    "not pivoted" in the denominator — the rate is 3/10, not 3/9.
    """
    agg = StatsAggregator()
    for i in range(3):
        agg.ingest(_make_trace(f"piv-{i}", pivoted=True, pivot_reason="low_score"))
    for i in range(6):
        agg.ingest(_make_trace(f"ok-{i}"))
    # 504 re-raise: NOT pivoted, but local_error_class is set.
    agg.ingest(
        _make_trace(
            "504-re-raise",
            pivoted=False,
            pivot_reason="none",
            local_error_class="ProviderTimeout",
        )
    )

    snap = agg.snapshot()
    assert snap["window_size"] == 10
    assert snap["pivot_rate"] == 0.3, (
        f"D-06: denominator must include 504 re-raise (pivoted=False); got {snap['pivot_rate']}"
    )


def test_quantiles_small_sample() -> None:
    """D-09 #3 / SC#4: n<20 falls through to s[-1] for p95.

    local_latency_ms = [10, 20, 30, 40, 50] → p50=30 (median),
    p95=50 (s[-1] — n=5 < 20), max=50, min=10.
    """
    agg = StatsAggregator()
    for v in [10, 20, 30, 40, 50]:
        agg.ingest(_make_trace(local_latency_ms=v))

    snap = agg.snapshot()
    q = snap["local_latency_ms"]
    assert q == {"p50": 30, "p95": 50, "max": 50, "min": 10}, (
        f"small-sample p95 must fall through to s[-1] when n<20; got {q}"
    )


def test_quantiles_large_sample() -> None:
    """D-09 #4 / SC#4: n>=20 sorted-index branch on a 1000-trace sample.

    local_latency_ms = range(1000) → p50=499.5 (statistics.median average
    of the two middle values), p95=949 (s[int(1000*0.95)-1] = s[949] = 949),
    max=999, min=0.
    """
    agg = StatsAggregator()
    for i in range(1000):
        agg.ingest(_make_trace(f"req-{i:04d}", local_latency_ms=i))

    snap = agg.snapshot()
    q = snap["local_latency_ms"]
    assert q is not None
    assert q["p50"] == 499.5, f"median of range(1000) is 499.5; got {q['p50']}"
    assert q["p95"] == 949, f"s[int(1000*0.95)-1] == s[949] == 949; got {q['p95']}"
    assert q["max"] == 999
    assert q["min"] == 0


def test_deque_cap_at_maxlen_plus_one() -> None:
    """D-09 #5 / SC#4: ingest 1001 traces; oldest evicted, window_size=1000.

    Ingests req-0000..req-1000 (1001 total). Asserts window_size=1000;
    asserts the deque's oldest entry is now req-0001 (req-0000 evicted).
    Locks the collections.deque(maxlen=...) automatic-eviction behavior.
    """
    agg = StatsAggregator()
    for i in range(_DEFAULT_MAXLEN + 1):
        agg.ingest(_make_trace(f"req-{i:04d}"))

    snap = agg.snapshot()
    assert snap["window_size"] == _DEFAULT_MAXLEN
    # Inspect the internal deque to verify eviction order — legitimate
    # test-internal access per D-11.
    _oldest_ts, oldest_trace = agg._traces[0]
    assert oldest_trace["request_id"] == "req-0001", (
        f"req-0000 must be evicted at maxlen+1; got oldest={oldest_trace['request_id']}"
    )


def test_note_literal() -> None:
    """D-09 #6 / SC#4 + STA-07: _NOTE constant equals the wire string AND
    snapshot() returns it verbatim on both empty and populated paths.
    """
    # Constant-equality lock — STA-07 wire string.
    assert _NOTE == "Stats reset on CoreThread restart (in-memory)."

    # Empty path returns the literal.
    agg = StatsAggregator()
    assert agg.snapshot()["note"] == _NOTE

    # Populated path also returns the literal (both branches lock to STA-07).
    agg.ingest(_make_trace())
    assert agg.snapshot()["note"] == _NOTE


# ---------------------------------------------------------------------------
# Decision-coverage tests — D-09 #7..#9
# ---------------------------------------------------------------------------


def test_snapshot_filters_by_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """D-09 #7 / D-01 + D-03: window arg filters by ingest timestamp.

    Monkeypatches `time.time` IN THE STATS MODULE so ingest timestamps are
    deterministic. Ingests 4 traces at:
      - t = NOW             (within 1h)
      - t = NOW - 30min     (within 1h)
      - t = NOW - 2h        (within 24h, NOT within 1h)
      - t = NOW - 2d        (within 7d, NOT within 24h)
    Then restores `time.time` and calls snapshot at NOW with each window arg.
    """
    from corethread import stats as stats_mod

    now_ts = 1_700_000_000  # arbitrary fixed unix timestamp
    ingest_clock = {"t": now_ts}

    def fake_time() -> float:
        return ingest_clock["t"]

    # Phase 1: monkeypatch stats.time.time to control ingest timestamps.
    monkeypatch.setattr(stats_mod.time, "time", fake_time)
    agg = StatsAggregator()
    # Aggregator construction also captures _started_at via int(time.time())
    # — it'll be now_ts under the patch. That's fine; D-02 just locks
    # "captured once at construction", not a specific value.

    ingest_clock["t"] = now_ts
    agg.ingest(_make_trace("recent"))
    ingest_clock["t"] = now_ts - 1800  # 30min ago
    agg.ingest(_make_trace("30min-ago"))
    ingest_clock["t"] = now_ts - 7200  # 2h ago
    agg.ingest(_make_trace("2h-ago"))
    ingest_clock["t"] = now_ts - 172800  # 2d ago
    agg.ingest(_make_trace("2d-ago"))

    # Phase 2: snapshot reads time.time at "now" — leave the monkeypatch
    # in place but advance the clock to now_ts.
    ingest_clock["t"] = now_ts

    assert agg.snapshot(window="1h")["window_size"] == 2, (
        "1h window must include recent + 30min-ago"
    )
    assert agg.snapshot(window="24h")["window_size"] == 3, (
        "24h window must include recent + 30min-ago + 2h-ago"
    )
    assert agg.snapshot(window="7d")["window_size"] == 4, "7d window must include all 4 traces"
    assert agg.snapshot(window="all")["window_size"] == 4, (
        "all-window must include every trace regardless of timestamp"
    )


def test_confidence_quantiles_exclude_parse_failed() -> None:
    """D-09 #8 / D-05: confidence_score quantiles EXCLUDE judge_parse_failed=True rows.

    Two scenarios:
      (a) Mixed window — 3 sentinel (judge_parse_failed=True, score=0.0)
          + 7 real (judge_parse_failed=False, score=0.85). Quantiles are
          computed over the 7 real scores only; min must be 0.85, NOT 0.0.
      (b) All-failed window — 3 traces all judge_parse_failed=True.
          `confidence_score` quantile dict is None (D-05 empty-after-filter
          contract).
    """
    # Scenario (a) — mixed window.
    agg = StatsAggregator()
    for i in range(3):
        agg.ingest(
            _make_trace(
                f"sentinel-{i}",
                confidence_score=0.0,
                judge_parse_failed=True,
                pivoted=True,
                pivot_reason="judge_error",
            )
        )
    for i in range(7):
        agg.ingest(_make_trace(f"real-{i}", confidence_score=0.85, judge_parse_failed=False))

    snap = agg.snapshot()
    cq = snap["confidence_score"]
    assert cq is not None
    assert cq["min"] == 0.85, (
        f"D-05: 0.0 sentinel must be excluded — min should be 0.85, got {cq['min']}"
    )
    assert cq["p50"] == 0.85
    # Sanity: window_size still counts ALL 10 traces (sentinel filter
    # only affects the confidence quantile list, not window membership).
    assert snap["window_size"] == 10

    # Scenario (b) — all-failed window returns None for confidence_score.
    agg2 = StatsAggregator()
    for i in range(3):
        agg2.ingest(
            _make_trace(
                f"only-sentinel-{i}",
                confidence_score=0.0,
                judge_parse_failed=True,
                pivoted=True,
                pivot_reason="judge_error",
            )
        )
    snap2 = agg2.snapshot()
    assert snap2["confidence_score"] is None, (
        "D-05 empty-after-filter: confidence_score quantile dict must be None"
    )


def test_pivot_reasons_counter_totals_to_window_size() -> None:
    """D-09 #9 / D-07: pivot_reasons dict has all buckets and totals to window_size.

    Ingests 9 traces covering all reason values:
      - 3 x "none"
      - 2 x "low_score"
      - 2 x "local_error"
      - 1 x "local_truncated"
      - 1 x "judge_error"
    Asserts snapshot['pivot_reasons'] includes every known reason AND
    sum(values) == window_size (D-07 invariant).
    """
    agg = StatsAggregator()
    for i in range(3):
        agg.ingest(_make_trace(f"none-{i}", pivoted=False, pivot_reason="none"))
    for i in range(2):
        agg.ingest(_make_trace(f"low-{i}", pivoted=True, pivot_reason="low_score"))
    for i in range(2):
        agg.ingest(_make_trace(f"loc-{i}", pivoted=True, pivot_reason="local_error"))
    agg.ingest(_make_trace("trunc-0", pivoted=True, pivot_reason="local_truncated"))
    agg.ingest(_make_trace("jud-0", pivoted=True, pivot_reason="judge_error"))

    snap = agg.snapshot()
    pr: dict[str, int] = snap["pivot_reasons"]

    # All buckets present with the expected counts.
    assert pr == {
        "none": 3,
        "low_score": 2,
        "local_truncated": 1,
        "local_error": 2,
        "judge_error": 1,
    }, f"all buckets must surface; got {pr}"
    # D-07 invariant: sum equals window_size.
    assert sum(pr.values()) == snap["window_size"] == 9

    # Sanity: it's a plain dict, not a Counter (D-07 JSON-serializability).
    assert isinstance(pr, dict)
    assert type(pr) is dict, (
        f"D-07: pivot_reasons must be a plain dict, not a Counter subclass; "
        f"got type={type(pr).__name__}"
    )


def test_pivot_reasons_quiet_window_seeds_all_buckets() -> None:
    """D-07 quiet-window guard: pivot_reasons surfaces all buckets even when
    no pivots occurred in the window.

    Closes 08-VERIFICATION.md gap (must-have #7) — the populated-window test
    above (test_pivot_reasons_counter_totals_to_window_size) only proves the
    contract when every bucket is observed; this test proves it when only
    ``"none"`` is observed (the realistic case for a healthy local-only
    operating window).

    Ingests 5 happy-path traces (pivoted=False, pivot_reason="none") and
    asserts:
      - The pivot_reasons key set is exactly the D-07 buckets.
      - Non-none values are 0.
      - The "none" bucket holds all 5 traces.
      - The D-07 sum invariant continues to hold (sum == window_size).
      - It's a plain dict (JSON-serializable; matches D-07 wire shape).
    """
    agg = StatsAggregator()
    for i in range(5):
        # _make_trace defaults already represent a happy-path trace:
        # pivoted=False, pivot_reason="none", judge_parse_failed=False,
        # frontier_latency_ms=None. No overrides needed.
        agg.ingest(_make_trace(f"happy-{i}"))

    snap = agg.snapshot()
    pr: dict[str, int] = snap["pivot_reasons"]

    # All D-07 buckets present — the gap that 08-VERIFICATION.md flagged.
    assert set(pr.keys()) == {
        "none",
        "low_score",
        "local_truncated",
        "local_error",
        "judge_error",
    }, f"D-07 quiet-window contract: all buckets must surface; got keys={set(pr.keys())}"
    # Exact value shape — none counts the 5 happy-path traces; the others are zeros.
    assert pr == {
        "none": 5,
        "low_score": 0,
        "local_truncated": 0,
        "local_error": 0,
        "judge_error": 0,
    }, f"D-07 quiet-window value shape; got {pr}"
    # D-07 sum invariant — seeded zeros contribute zero, sum still equals window_size.
    assert sum(pr.values()) == snap["window_size"] == 5

    # Plain-dict shape (D-07 JSON-serializability) — same guard as the populated test.
    assert type(pr) is dict, (
        f"D-07: pivot_reasons must be a plain dict, not a Counter subclass; "
        f"got type={type(pr).__name__}"
    )
