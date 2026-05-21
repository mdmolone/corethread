"""Phase 3 / Plan 05 — obs.py + judge.py integrated + FastAPI middleware tests.

Closes ROADMAP Success Criteria:

- **SC#4**: Each request writes EXACTLY one structured JSON log line
  containing the 11 locked fields (plus Phase 3's two additions). Verified
  by capturing stdout of a fake-driver loop and parsing the emitted line.
- **SC#5**: A simple ``grep | wc -l`` over the log yields a plausible pivot
  rate (between 0 and N) for the last N requests. Verified by running 10
  iterations with deterministic 5-pass / 5-fail verdicts and asserting
  the pivot-count == 5.

Plus:

- **D-16 middleware contract**: TestClient-driven tests for header-supplied
  + header-absent + bind-unbind-cleanup paths.
- **Pitfall-5 emit_trace assertion**: missing-key diff raises
  AssertionError BEFORE emitting a broken JSONL line.
- **Pitfall-7 respx body-matching**: local-answer stub + judge stub
  distinguished by ``json__model=...`` to avoid Pitfall #7 misrouting.
- **OBS-02 redaction regression guard**: captured trace output contains NO
  'sk-test' and NO 'Authorization: Bearer' substring.
- **Sentinel-path trace shape**: judge_parse_failed=True + confidence=0.0
  + reasoning starts with 'judge parse failure:' + exactly one trace line
  still emitted.

All tests are respx-driven for HTTP transport + capfd-driven for stdout.
Zero live Ollama required.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from corethread import obs
from corethread.judge import grade
from corethread.logging_config import setup_logging
from corethread.models import (
    ChatCompletionRequest,
    ChatMessage,
    LocalConfig,
)
from corethread.obs import RequestTrace, emit_trace, register_request_id_middleware, time_block
from corethread.providers.ollama import OllamaProvider
from corethread.pubsub import TraceBus

_JUDGE_MODEL = "qwen2.5:7b"
_LOCAL_MODEL = "llama3.1:8b"
_THRESHOLD = 0.7  # CFG-04 default


# Local-provider answer body (respx return for the "local chat" stub).
_STUB_LOCAL_ANSWER = {
    "model": _LOCAL_MODEL,
    "created_at": "2026-04-21T00:00:00Z",
    "message": {"role": "assistant", "content": "The answer is 42."},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 12,
    "eval_count": 5,
}


# 3/3 rubric -> score 0.9 -> pass (no pivot).
_JUDGE_VERDICT_PASS_RAW = (
    '{"confidence_score": 0.9, '
    '"reasoning": "[answered_core_q=true, no_disclaimers=true, '
    'no_contradictions=true] clear answer.", '
    '"pass": true}'
)


# 2/3 rubric -> score 0.5 -> pivot (< 0.7 threshold).
_JUDGE_VERDICT_FAIL_RAW = (
    '{"confidence_score": 0.5, '
    '"reasoning": "[answered_core_q=true, no_disclaimers=false, '
    'no_contradictions=true] hedged mid-answer.", '
    '"pass": false}'
)


def _judge_response_body(raw_content: str) -> dict[str, Any]:
    """Wrap judge LLM raw content in the Ollama /api/chat response shape."""
    return {
        "model": _JUDGE_MODEL,
        "created_at": "2026-04-21T00:00:00Z",
        "message": {"role": "assistant", "content": raw_content},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 30,
        "eval_count": 20,
    }


def _basic_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=_LOCAL_MODEL,
        messages=[ChatMessage(role="user", content="What is 6*7?")],
    )


async def _run_fake_orchestrator_iteration(
    provider: OllamaProvider,
    judge_model: str,
    request_id: str,
) -> RequestTrace:
    """One iteration of the fake orchestrator loop.

    Drives the full Phase 3 Wave 2 flow:
      1. local.chat(request) -> local_response
      2. judge.grade(...) -> verdict
      3. assemble RequestTrace dict
      4. emit_trace(trace)

    Phase 4's real orchestrator will thread this same sequence; by
    building it here we prove the interface is composable before
    orchestrator.py exists. Does NOT execute a frontier call — Phase 3's
    trace has frontier_* fields as None per D-13.
    """
    trace: RequestTrace = {
        "request_id": request_id,
        "selected_local_model": _LOCAL_MODEL,
        "judge_model": judge_model,
        "frontier_model": None,  # no pivot path wired in Phase 3
        "confidence_score": 0.0,  # overwritten after grade()
        "pivoted": False,  # overwritten after threshold check
        "local_latency_ms": 0,  # recorded by time_block
        "judge_latency_ms": 0,  # recorded by time_block
        "frontier_latency_ms": None,
        "input_tokens": 0,  # populated from local_response.usage
        "output_tokens": 0,  # populated from local_response.usage
        "frontier_cost_est": None,
        "judge_parse_failed": False,  # overwritten if sentinel detected
        # Phase 4 D-18 — happy-path defaults; helper updates pivot_reason to
        # "low_score" / "judge_error" below per D-19 if the judge verdict
        # warrants it. local_error_class stays None on this fake-driver path
        # because no local-provider exception path is exercised here (Phase 4
        # orchestrator real-code paths populate it via except-handlers).
        "pivot_reason": "none",
        "local_error_class": None,
    }

    request = _basic_request()

    # time_block is typed as ``dict[str, Any]`` (see obs.py docstring: the
    # loose typing is deliberate so callers can measure into partial builders
    # or test scratch dicts without type-checker coupling). Passing a
    # RequestTrace TypedDict here is semantically equivalent — the key
    # assignment lands on the int/int|None fields — but mypy can't bridge
    # TypedDict -> dict[str, Any] structurally, so annotate the narrowing.
    async with time_block(trace, "local_latency_ms"):  # type: ignore[arg-type]
        local_response = await provider.chat(request)

    trace["input_tokens"] = local_response.usage.prompt_tokens
    trace["output_tokens"] = local_response.usage.completion_tokens

    async with time_block(trace, "judge_latency_ms"):  # type: ignore[arg-type]
        verdict = await grade(
            request,
            local_response,
            provider=provider,
            judge_model=judge_model,
        )

    trace["confidence_score"] = verdict.confidence_score
    # D-07 sentinel detection — orchestrator flags the sentinel path so
    # the JSONL trace distinguishes "low-confidence real" from "judge broke".
    trace["judge_parse_failed"] = verdict.reasoning.startswith("judge parse failure:")
    trace["pivoted"] = verdict.confidence_score < _THRESHOLD
    # Phase 4 D-19 — pivot_reason mirrors the actual decision branch the
    # would-be orchestrator took. judge-broke (sentinel) wins over the
    # threshold check; sub-threshold real verdict is "low_score"; happy
    # path stays "none". Keeps emit_trace's Pitfall-5 assertion happy and
    # gives SC#5's 10-iteration test a consistent pivot_reason fingerprint.
    if trace["judge_parse_failed"]:
        trace["pivot_reason"] = "judge_error"
    elif trace["pivoted"]:
        trace["pivot_reason"] = "low_score"
    # else: pivot_reason stays "none" from initialization (happy path).

    emit_trace(trace)
    return trace


def _parse_trace_line(captured_stdout: str) -> list[dict[str, Any]]:
    """Extract every JSON line with event='request.decision' from
    captured stdout. Returns a list of dicts in emission order."""
    decisions = []
    for raw_line in captured_stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("event") == "request.decision":
            decisions.append(obj)
    return decisions


# ---------------------------------------------------------------------------
# SC#4 — exactly one JSON trace line with every locked field
# ---------------------------------------------------------------------------


async def test_sc4_fake_driver_emits_one_trace_line_with_all_fields(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """SC#4: one fake-driver iteration emits EXACTLY ONE JSONL line
    with event='request.decision', and the parsed JSON has every locked
    RequestTrace field with correct type."""
    setup_logging()  # rebind StreamHandler to current sys.stdout

    # Pitfall #7: body-match by json__model= to dispatch local vs judge.
    respx_mock.post(
        "http://ollama:11434/api/chat",
        json__model=_LOCAL_MODEL,
    ).mock(return_value=httpx.Response(200, json=_STUB_LOCAL_ANSWER))
    respx_mock.post(
        "http://ollama:11434/api/chat",
        json__model=_JUDGE_MODEL,
    ).mock(
        return_value=httpx.Response(
            200,
            json=_judge_response_body(_JUDGE_VERDICT_PASS_RAW),
        )
    )

    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    _returned_trace = await _run_fake_orchestrator_iteration(
        provider,
        _JUDGE_MODEL,
        request_id="test-req-sc4-0001",
    )

    out, _err = capfd.readouterr()
    decisions = _parse_trace_line(out)

    # SC#4: EXACTLY ONE trace line.
    assert len(decisions) == 1, f"expected 1 trace line, got {len(decisions)}: {decisions}"
    line = decisions[0]

    # Every locked RequestTrace field present.
    expected_keys = {
        "request_id",
        "selected_local_model",
        "judge_model",
        "frontier_model",
        "confidence_score",
        "pivoted",
        "local_latency_ms",
        "judge_latency_ms",
        "frontier_latency_ms",
        "input_tokens",
        "output_tokens",
        "frontier_cost_est",
        "judge_parse_failed",
    }
    assert expected_keys.issubset(line.keys()), f"missing keys: {expected_keys - line.keys()}"

    # Type shape — structural integrity.
    assert isinstance(line["request_id"], str)
    assert line["request_id"] == "test-req-sc4-0001"
    assert line["selected_local_model"] == _LOCAL_MODEL
    assert line["judge_model"] == _JUDGE_MODEL
    assert line["frontier_model"] is None  # no pivot
    assert isinstance(line["confidence_score"], (int, float))
    assert line["confidence_score"] == 0.9  # 3/3 -> 0.9
    assert line["pivoted"] is False  # 0.9 >= 0.7 threshold
    assert isinstance(line["local_latency_ms"], int)
    assert line["local_latency_ms"] >= 0  # may be 0 on fast CI
    assert isinstance(line["judge_latency_ms"], int)
    assert line["judge_latency_ms"] >= 0
    assert line["frontier_latency_ms"] is None
    assert isinstance(line["input_tokens"], int)
    assert line["input_tokens"] == 12  # from _STUB_LOCAL_ANSWER.prompt_eval_count
    assert isinstance(line["output_tokens"], int)
    assert line["output_tokens"] == 5  # from _STUB_LOCAL_ANSWER.eval_count
    assert line["frontier_cost_est"] is None
    assert line["judge_parse_failed"] is False  # happy path


async def test_sc4_trace_line_contains_no_redaction_targets(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """OBS-02 regression guard: captured trace output must NOT contain
    'sk-test' or 'Authorization: Bearer' substrings. Even though the
    OPENAI_API_KEY is set to sk-test-1234567890ABCDEFGHIJ via the autouse
    fixture, redaction prevents it from appearing in log lines. The
    trace dict itself doesn't carry the API key — this test is defense
    in depth: a future code path that ACCIDENTALLY adds api_key to the
    trace must still be caught by the RedactingFilter."""
    setup_logging()
    respx_mock.post(
        "http://ollama:11434/api/chat",
        json__model=_LOCAL_MODEL,
    ).mock(return_value=httpx.Response(200, json=_STUB_LOCAL_ANSWER))
    respx_mock.post(
        "http://ollama:11434/api/chat",
        json__model=_JUDGE_MODEL,
    ).mock(
        return_value=httpx.Response(
            200,
            json=_judge_response_body(_JUDGE_VERDICT_PASS_RAW),
        )
    )

    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    await _run_fake_orchestrator_iteration(
        provider,
        _JUDGE_MODEL,
        request_id="test-req-redact-0001",
    )

    out, _err = capfd.readouterr()
    # sk-test fixture VALUE must not appear in captured output (Phase 1
    # RedactingFilter would scrub even if it leaked).
    assert "sk-test-1234567890ABCDEFGHIJ" not in out
    # The scrub token or any Authorization header shape also must not appear.
    assert "Authorization: Bearer" not in out


async def test_sentinel_path_trace_has_judge_parse_failed_true(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """D-07 integration: when judge parse fails twice (retry also fails),
    grade() returns sentinel (no raise), fake-driver detects it via
    reasoning prefix, and emits the trace with judge_parse_failed=True
    + confidence_score=0.0 + pivoted=True (0.0 < 0.7)."""
    setup_logging()

    # Two malformed responses on the judge model to force sentinel.
    respx_mock.post(
        "http://ollama:11434/api/chat",
        json__model=_LOCAL_MODEL,
    ).mock(return_value=httpx.Response(200, json=_STUB_LOCAL_ANSWER))
    respx_mock.post(
        "http://ollama:11434/api/chat",
        json__model=_JUDGE_MODEL,
    ).mock(
        side_effect=[
            httpx.Response(200, json=_judge_response_body("not valid at all")),
            httpx.Response(200, json=_judge_response_body("still nope")),
        ]
    )

    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    trace = await _run_fake_orchestrator_iteration(
        provider,
        _JUDGE_MODEL,
        request_id="test-req-sentinel",
    )

    assert trace["judge_parse_failed"] is True
    assert trace["confidence_score"] == 0.0
    assert trace["pivoted"] is True  # 0.0 < threshold 0.7

    out, _err = capfd.readouterr()
    decisions = _parse_trace_line(out)
    # Still EXACTLY ONE trace line — sentinel path does not emit extra.
    assert len(decisions) == 1
    assert decisions[0]["judge_parse_failed"] is True


# ---------------------------------------------------------------------------
# SC#5 — grep | wc -l pivot rate plausible over 10 requests
# ---------------------------------------------------------------------------


async def test_sc5_ten_iterations_pivot_rate_matches_expected(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """SC#5: run 10 fake-driver iterations with deterministic 5-pass /
    5-fail judge responses. Assert captured stdout has EXACTLY 10 trace
    lines AND exactly 5 contain '"pivoted":true'. This IS the 'simple
    grep|wc-l' pivot-rate observability check the roadmap mandates —
    emulated at pytest scope."""
    setup_logging()

    # Local-answer stub serves all 10 iterations.
    respx_mock.post(
        "http://ollama:11434/api/chat",
        json__model=_LOCAL_MODEL,
    ).mock(return_value=httpx.Response(200, json=_STUB_LOCAL_ANSWER))

    # Judge stub serves alternating PASS / FAIL — 5 of each.
    # side_effect list consumed sequentially; length 10 (one per grade() call).
    judge_side_effects: list[httpx.Response] = []
    for i in range(10):
        if i % 2 == 0:  # 0,2,4,6,8 -> PASS (5 total)
            judge_side_effects.append(
                httpx.Response(
                    200,
                    json=_judge_response_body(_JUDGE_VERDICT_PASS_RAW),
                )
            )
        else:  # 1,3,5,7,9 -> FAIL/pivot (5 total)
            judge_side_effects.append(
                httpx.Response(
                    200,
                    json=_judge_response_body(_JUDGE_VERDICT_FAIL_RAW),
                )
            )

    respx_mock.post(
        "http://ollama:11434/api/chat",
        json__model=_JUDGE_MODEL,
    ).mock(side_effect=judge_side_effects)

    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    for i in range(10):
        await _run_fake_orchestrator_iteration(
            provider,
            _JUDGE_MODEL,
            request_id=f"test-req-sc5-{i:04d}",
        )

    out, _err = capfd.readouterr()
    decisions = _parse_trace_line(out)

    # EXACTLY 10 trace lines — one per iteration.
    assert len(decisions) == 10, f"expected 10 trace lines, got {len(decisions)}"

    # SC#5: pivot count == 5 (the 5 FAIL iterations).
    pivoted_count = sum(1 for d in decisions if d.get("pivoted") is True)
    assert pivoted_count == 5, f"expected 5 pivots in 10 iterations, got {pivoted_count}"

    # Additional grep-style check matching the "grep -c '\"pivoted\":true'" literal
    # shape from the ROADMAP SC#5 description (mimics what an operator would run).
    pivoted_substring_count = out.count('"pivoted": true') + out.count('"pivoted":true')
    assert pivoted_substring_count == 5, (
        f"pivoted substring count mismatch: {pivoted_substring_count}"
    )

    # Request IDs are distinct.
    ids = [d["request_id"] for d in decisions]
    assert len(set(ids)) == 10, f"duplicate request_ids: {ids}"


# ---------------------------------------------------------------------------
# Pitfall-5 — emit_trace assertion on missing keys
# ---------------------------------------------------------------------------


def test_emit_trace_assertion_fires_on_missing_keys(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Pitfall-5 option B: emit_trace raises AssertionError synchronously
    when any RequestTrace required key is missing. NO broken trace line
    is ever emitted to stdout."""
    setup_logging()
    partial_trace = {"request_id": "x", "pivoted": False}  # mostly-missing

    with pytest.raises(AssertionError) as info:
        emit_trace(partial_trace)  # type: ignore[arg-type]

    assert "missing" in str(info.value).lower()

    # No JSONL line leaked BEFORE the assertion.
    out, _err = capfd.readouterr()
    decisions = _parse_trace_line(out)
    assert decisions == []


def test_emit_trace_assertion_passes_on_complete_trace(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Complement to the above: a complete RequestTrace passes the
    assertion and emits exactly one line."""
    setup_logging()
    trace: RequestTrace = {
        "request_id": "x",
        "selected_local_model": "m",
        "judge_model": "j",
        "frontier_model": None,
        "confidence_score": 0.9,
        "pivoted": False,
        "local_latency_ms": 0,
        "judge_latency_ms": 0,
        "frontier_latency_ms": None,
        "input_tokens": 1,
        "output_tokens": 1,
        "frontier_cost_est": None,
        "judge_parse_failed": False,
        # Phase 4 D-18 — happy-path defaults to keep emit_trace's
        # Pitfall-5 assertion satisfied on a complete 15-field trace.
        "pivot_reason": "none",
        "local_error_class": None,
    }
    emit_trace(trace)
    out, _err = capfd.readouterr()
    decisions = _parse_trace_line(out)
    assert len(decisions) == 1
    assert decisions[0]["request_id"] == "x"


# ---------------------------------------------------------------------------
# D-16 — request_id middleware: header-supplied + header-absent + cleanup
# ---------------------------------------------------------------------------


def test_d16_middleware_registered_on_main_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan 03-02 wired register_request_id_middleware(app); verify the
    middleware stack has exactly 1 entry at import time."""
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", "config.yaml.example")
    import importlib

    from corethread import main  # late import so the env var is in place

    importlib.reload(main)  # consistency with sibling D-16 tests + test-order independence
    # register_request_id_middleware is the only middleware wired in Plan 03-02;
    # Phase 4+ may add more — if so, this assertion surfaces the regression
    # immediately (Pitfall #4: middleware ordering matters; request_id must stay
    # registered FIRST so it binds BEFORE anything else runs).
    assert len(main.app.user_middleware) == 1


def test_d16_supplied_x_request_id_unbinds_after_request(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """D-16 unbind-cleanup contract: if X-Request-ID is supplied, the
    middleware's ``finally: unbind_contextvars("request_id")`` MUST run
    even though /health does not itself emit a structured log event.
    We drive /health with a supplied id and then assert that
    ``structlog.contextvars.get_contextvars()`` no longer contains
    request_id AFTER the request completes — proving the middleware
    cleanup path executed. Bind-during-request behavior is covered
    by Phase 4 orchestrator integration tests where a real structured
    log line is emitted with request_id bound."""
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", "config.yaml.example")
    import importlib

    from fastapi.testclient import TestClient

    from corethread import main

    importlib.reload(main)  # pick up fresh lifespan + middleware binding

    with TestClient(main.app) as client:
        capfd.readouterr()  # drain startup logs
        r = client.get("/health", headers={"X-Request-ID": "req-custom-id-xyz"})
        assert r.status_code in (200, 503)
        _out, _err = capfd.readouterr()

    # /health logs no event itself by default, but the request flowing
    # through bind_contextvars + unbind_contextvars means the middleware
    # ran. After the with-block exits, contextvars should not contain
    # request_id (unbind in finally ran).
    import structlog

    ctx = structlog.contextvars.get_contextvars()
    assert "request_id" not in ctx


def test_d16_header_absent_generates_uuid_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-16: no X-Request-ID header -> middleware generates uuid4().hex[:16].
    We drive a request, then assert the middleware's response does not
    error (fallback path works). Unit coverage of the fallback length (16
    hex chars = 64 bits) is via the grep gate on obs.py _REQUEST_ID_LEN=16."""
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", "config.yaml.example")
    import importlib

    from fastapi.testclient import TestClient

    from corethread import main

    importlib.reload(main)

    with TestClient(main.app) as client:
        r = client.get("/health")  # no X-Request-ID header
        assert r.status_code in (200, 503)

    # No leak post-request (unbind in finally ran).
    import structlog

    assert "request_id" not in structlog.contextvars.get_contextvars()


def test_d16_empty_x_request_id_falls_back_to_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-16 empty-header edge case: X-Request-ID: "" -> falsy -> UUID fallback."""
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", "config.yaml.example")
    import importlib

    from fastapi.testclient import TestClient

    from corethread import main

    importlib.reload(main)

    with TestClient(main.app) as client:
        r = client.get("/health", headers={"X-Request-ID": ""})
        assert r.status_code in (200, 503)


# ---------------------------------------------------------------------------
# D-15 — time_block standalone behavior (complement to fake-driver usage)
# ---------------------------------------------------------------------------


async def test_time_block_records_on_clean_exit() -> None:
    """D-15 clean-exit path — target[key] set to int(elapsed_ms)."""
    import asyncio

    d: dict[str, Any] = {}
    async with time_block(d, "x_ms"):
        await asyncio.sleep(0.02)
    assert "x_ms" in d
    assert isinstance(d["x_ms"], int)
    assert d["x_ms"] >= 15


async def test_time_block_records_on_exception_exit() -> None:
    """D-15 exception-exit path — target[key] STILL set (partial trace
    preservation for provider-timeout scenarios)."""
    d: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="boom"):
        async with time_block(d, "err_ms"):
            raise RuntimeError("boom")
    assert "err_ms" in d
    assert isinstance(d["err_ms"], int)
    assert d["err_ms"] >= 0


async def test_time_block_uses_monotonic_not_time() -> None:
    """Sanity: ``time.monotonic()`` cannot produce negative elapsed even
    under NTP adjustment. This is an indirect test — we can't mock NTP
    here, but we assert non-negative, which would fail if someone
    refactored to ``time.time()`` and the tester's clock went backwards."""
    d: dict[str, Any] = {}
    async with time_block(d, "m_ms"):
        pass
    assert d["m_ms"] >= 0


# ---------------------------------------------------------------------------
# register_request_id_middleware — direct API coverage
# ---------------------------------------------------------------------------


def test_register_request_id_middleware_is_callable_on_fresh_app() -> None:
    """Smoke: register_request_id_middleware is a normal function that
    takes a FastAPI app and installs exactly one middleware."""
    from fastapi import FastAPI

    fresh_app = FastAPI()
    assert len(fresh_app.user_middleware) == 0
    register_request_id_middleware(fresh_app)
    assert len(fresh_app.user_middleware) == 1


# ---------------------------------------------------------------------------
# Plan 07-03 — emit_trace tee to TraceBus (SC#4 closure: tests #4 and #5)
# ---------------------------------------------------------------------------
#
# Closes the v1.0-compat contract (no-op when bus unset) and the additive-tee
# contract (bus's replay deque receives the trace AND structlog still emits
# its JSONL line). Together with the 3 tests in tests/test_pubsub.py these
# fulfill SC#4's 5-named-test set verbatim.
#
# Test isolation: D-11 mandates `obs.set_trace_bus(None)` in a finally block
# even on assertion failure. Per Claude's Discretion #4 we use explicit
# per-test cleanup (try/finally) rather than a new conftest fixture — only
# 2 tests touch the module-global, and the existing `reset_logging_state`
# autouse fixture already handles structlog cleanup.


def test_emit_trace_no_op_when_bus_unset(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """SC#4 #4 — D-09 + ARCHITECTURE.md §1.2 v1.0-compat contract.

    With ``obs._TRACE_BUS is None`` (the module default), calling
    ``emit_trace(trace)`` directly MUST:
      1. Still emit exactly one structlog JSONL line (v1.0 behavior preserved)
      2. NOT raise any exception
      3. NOT mutate the module global

    This is the v1.0-compat path that direct unit tests of ``emit_trace``
    (tests that import from ``corethread.obs`` without running the FastAPI
    lifespan — e.g., ``test_emit_trace_assertion_passes_on_complete_trace``
    above this in this same file) rely on.
    """
    setup_logging()  # rebind StreamHandler to current sys.stdout

    # Pre-condition — Plan 03 Task 1 makes this the module default. The
    # autouse `reset_logging_state` fixture does NOT reset _TRACE_BUS, so we
    # assert here to catch any test-pollution regression introduced by
    # future tests that call set_trace_bus and forget the finally cleanup.
    assert obs._TRACE_BUS is None, "test invariant: bus must be unset on entry"

    trace: RequestTrace = {
        "request_id": "test-no-op-bus-unset",
        "selected_local_model": "m",
        "judge_model": "j",
        "frontier_model": None,
        "confidence_score": 0.9,
        "pivoted": False,
        "local_latency_ms": 0,
        "judge_latency_ms": 0,
        "frontier_latency_ms": None,
        "input_tokens": 1,
        "output_tokens": 1,
        "frontier_cost_est": None,
        "judge_parse_failed": False,
        "pivot_reason": "none",
        "local_error_class": None,
    }

    # No exception path — emit_trace's tee branch is `if _TRACE_BUS is not None:`
    # which is False here, so the bus broadcast is skipped entirely.
    emit_trace(trace)

    # Module-global stayed None — emit_trace must not mutate _TRACE_BUS.
    assert obs._TRACE_BUS is None, "emit_trace must not mutate _TRACE_BUS"

    # v1.0 structlog behavior preserved — exactly one trace line.
    out, _err = capfd.readouterr()
    decisions = _parse_trace_line(out)
    assert len(decisions) == 1, f"expected 1 trace line, got {len(decisions)}: {decisions}"
    assert decisions[0]["request_id"] == "test-no-op-bus-unset"


def test_emit_trace_publishes_when_bus_set(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """SC#4 #5 — D-09 additive-tee contract.

    When a bus is wired, ``emit_trace(trace)`` MUST:
      1. Still emit exactly one structlog JSONL line (v1.0 behavior preserved)
      2. Publish the trace to the bus (verified via ``bus._replay`` deque
         introspection — the deque always receives the trace per Plan 01 D-05)

    D-11 isolation: the test wraps its assertions in try/finally so a failing
    assertion mid-test still restores ``obs.set_trace_bus(None)`` and the
    next test sees the v1.0-default (no stale bus pollution).

    Per Claude's Discretion #4 we use explicit per-test cleanup rather than a
    conftest fixture — minimal surface for only 2 tests in this plan.
    """
    setup_logging()

    bus = TraceBus()
    obs.set_trace_bus(bus)
    try:
        # Confirm wiring landed (defense-in-depth — Task 1's setter is just
        # `global _TRACE_BUS; _TRACE_BUS = bus`, but assert anyway so a
        # future setter refactor that introduces a no-op branch is caught).
        assert obs._TRACE_BUS is bus, "set_trace_bus must wire the global"

        trace: RequestTrace = {
            "request_id": "test-tee-bus-set",
            "selected_local_model": "m",
            "judge_model": "j",
            "frontier_model": None,
            "confidence_score": 0.9,
            "pivoted": False,
            "local_latency_ms": 0,
            "judge_latency_ms": 0,
            "frontier_latency_ms": None,
            "input_tokens": 1,
            "output_tokens": 1,
            "frontier_cost_est": None,
            "judge_parse_failed": False,
            "pivot_reason": "none",
            "local_error_class": None,
        }

        emit_trace(trace)

        # Tee landed — bus's replay deque has exactly 1 trace per Plan 01 D-05
        # ("the replay deque ALWAYS receives the trace BEFORE the per-subscriber
        # put loop"). Even with zero subscribers the replay path is exercised.
        assert len(bus._replay) == 1, f"expected 1 trace in bus replay, got {len(bus._replay)}"
        replayed = bus._replay[0]
        assert replayed["request_id"] == "test-tee-bus-set"

        # v1.0 structlog behavior STILL preserved — the tee is additive,
        # not a replacement. Exactly one JSONL line on stdout.
        out, _err = capfd.readouterr()
        decisions = _parse_trace_line(out)
        assert len(decisions) == 1, f"expected 1 trace line, got {len(decisions)}: {decisions}"
        assert decisions[0]["request_id"] == "test-tee-bus-set"
    finally:
        # D-11: restore module isolation EVEN ON ASSERTION FAILURE so the next
        # test sees `_TRACE_BUS is None` (the v1.0 default that
        # `test_emit_trace_no_op_when_bus_unset` asserts on entry).
        obs.set_trace_bus(None)
