"""Phase 4 / Plan 02 — orchestrator.py unit tests (RED phase of TDD).

Every test here is designed to FAIL at `import orchestrator` until Plan 03
lands orchestrator.py. When that lands, all tests turn GREEN (no production
code changes needed after Plan 03 — this file is the acceptance gate).

Closes Phase 4 ROADMAP Success Criteria:
- SC#1: every branch covered by fake providers (test_happy_path_*,
        test_low_score_pivot_*, test_judge_error_sentinel_pivot_*,
        test_local_unreachable_auto_pivot_*, test_local_timeout_504_no_pivot,
        test_local_http_error_502_no_pivot — all use FakeProvider+FakeJudge)
- SC#2: orchestrator assertion + dedicated unit test on single-call invariants
        (test_piv_04_invariants — 6 parametric branches; D-05 mechanism 3)

Closes phase requirements ORC-01 (pure-logic — no HTTP, no JSON parsing, no
retries), PIV-04 (single-call invariants), PIV-05 (unreachable vs timeout
distinction observable via pivot_reason + local_error_class).

Fakes come from tests/_fakes.py (Task 1). Judge is monkey-patched on the
orchestrator module — `orchestrator.judge.grade` substitution is the
canonical test wiring per CONTEXT.md D-30.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from corethread.errors import ProviderHTTPError, ProviderTimeout, ProviderUnavailable
from corethread.logging_config import setup_logging
from corethread.models import (
    ChatCompletionRequest,
    ChatMessage,
    JudgeVerdict,
)
from tests._fakes import (
    FakeJudge,
    FakeProvider,
    make_happy_local_response,
)

_LOCAL_MODEL = "llama3.1:8b"
_JUDGE_MODEL = "qwen2.5:7b"
_FRONTIER_MODEL = "gpt-4o"
_THRESHOLD = 0.7


def _basic_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=_LOCAL_MODEL,
        messages=[ChatMessage(role="user", content="What is 6*7?")],
    )


def _verdict_pass() -> JudgeVerdict:
    """3/3 rubric → score 0.9 → pass=True; happy path."""
    return JudgeVerdict(
        pass_=True,
        confidence_score=0.9,
        reasoning=(
            "[answered_core_q=true, no_disclaimers=true, no_contradictions=true] clean answer."
        ),
    )


def _verdict_low_score() -> JudgeVerdict:
    """2/3 rubric → score 0.5 → pass=False; low_score pivot."""
    return JudgeVerdict(
        pass_=False,
        confidence_score=0.5,
        reasoning=(
            "[answered_core_q=true, no_disclaimers=false, "
            "no_contradictions=true] hedged mid-answer."
        ),
    )


def _verdict_sentinel() -> JudgeVerdict:
    """D-07 sentinel — reasoning.startswith('judge parse failure:')."""
    return JudgeVerdict(
        pass_=False,
        confidence_score=0.0,
        reasoning="judge parse failure: ValidationError",
    )


def _make_app_config() -> Any:
    """Build a minimal AppConfig for Orchestrator construction.

    Tests don't care about config values beyond local.model / judge.model /
    frontier.model / routing.threshold — use the valid_yaml_path fixture or
    construct AppConfig directly via load_config. The shape is opaque to
    orchestrator tests; the orchestrator only reads these four attributes.

    Note: the `set_test_api_key` autouse fixture in tests/conftest.py sets
    OPENAI_API_KEY before each test, so config.yaml.example's
    `api_key_env: OPENAI_API_KEY` resolves cleanly inside test bodies.
    """
    from corethread.config import load_config

    # Reuse config.yaml.example for a valid, phase-consistent AppConfig.
    return load_config(Path("config.yaml.example"))


def _parse_trace_lines(captured_stdout: str) -> list[dict[str, Any]]:
    """Return all JSON lines whose event == 'request.decision'."""
    out: list[dict[str, Any]] = []
    for line in captured_stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("event") == "request.decision":
            out.append(obj)
    return out


# ---------------------------------------------------------------------------
# Happy path — local returns, judge passes, no pivot (D-19 pivot_reason="none")
# ---------------------------------------------------------------------------


async def test_happy_path_returns_local_response(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Happy path: local.chat→judge.grade(pass=True,score=0.9)→return local.
    fake_local.call_count==1, fake_judge.call_count==1, fake_frontier==0.
    Trace: pivot_reason='none', pivoted=False, local_error_class=None.
    """
    setup_logging()
    local_resp = make_happy_local_response(model=_LOCAL_MODEL)
    fake_local = FakeProvider(name="local", response=local_resp)
    fake_frontier = FakeProvider(
        name="frontier", response=make_happy_local_response(model=_FRONTIER_MODEL)
    )
    fake_judge = FakeJudge(verdicts=[_verdict_pass()])

    from corethread import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.judge, "grade", fake_judge)

    orch = orch_mod.Orchestrator(
        local=fake_local,
        frontier=fake_frontier,
        cfg=_make_app_config(),
    )
    response = await orch.handle(_basic_request())

    # Contract: local response passes through untouched on happy path
    assert response is local_resp
    assert fake_local.call_count == 1
    assert fake_judge.call_count == 1
    assert fake_frontier.call_count == 0

    out, _err = capfd.readouterr()
    decisions = _parse_trace_lines(out)
    assert len(decisions) == 1
    line = decisions[0]
    assert line["pivot_reason"] == "none"
    assert line["local_error_class"] is None
    assert line["pivoted"] is False
    assert line["judge_parse_failed"] is False
    assert line["confidence_score"] == 0.9


async def test_judge_profile_timeout_passes_to_grade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Judge model profile timeout controls judge.grade's per-call timeout."""
    local_resp = make_happy_local_response(model=_LOCAL_MODEL)
    fake_local = FakeProvider(name="local", response=local_resp)
    fake_frontier = FakeProvider(
        name="frontier", response=make_happy_local_response(model=_FRONTIER_MODEL)
    )
    fake_judge = FakeJudge(verdicts=[_verdict_pass()])
    cfg = _make_app_config()
    cfg.profile_for("judge").timeout_s = 42.0

    from corethread import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.judge, "grade", fake_judge)

    orch = orch_mod.Orchestrator(
        local=fake_local,
        frontier=fake_frontier,
        cfg=cfg,
    )

    await orch.handle(_basic_request())

    assert fake_judge.calls[0][4] == 42.0


async def test_judge_prompt_passes_to_grade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level judge.prompt controls the prompt passed to judge.grade."""
    local_resp = make_happy_local_response(model=_LOCAL_MODEL)
    fake_local = FakeProvider(name="local", response=local_resp)
    fake_frontier = FakeProvider(
        name="frontier", response=make_happy_local_response(model=_FRONTIER_MODEL)
    )
    fake_judge = FakeJudge(verdicts=[_verdict_pass()])
    cfg = _make_app_config()
    cfg.judge.prompt = "Custom orchestrator judge prompt."

    from corethread import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.judge, "grade", fake_judge)

    orch = orch_mod.Orchestrator(
        local=fake_local,
        frontier=fake_frontier,
        cfg=cfg,
    )

    await orch.handle(_basic_request())

    assert fake_judge.calls[0][5] == "Custom orchestrator judge prompt."


# ---------------------------------------------------------------------------
# Low-score pivot — judge pass=False OR score<threshold (pivot_reason="low_score")
# ---------------------------------------------------------------------------


async def test_low_score_pivot_calls_frontier_with_original_request(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """D-19 low_score branch + D-27 original-request invariant.

    Asserts:
    - fake_frontier.call_count == 1
    - frontier received the ORIGINAL request object (identity eq per D-27)
    - pivot_reason='low_score', pivoted=True, judge_parse_failed=False
    - Input/output tokens in trace come from FRONTIER response (D-28)
    """
    setup_logging()
    local_resp = make_happy_local_response(
        model=_LOCAL_MODEL,
        prompt_tokens=10,
        completion_tokens=3,
    )
    frontier_resp = make_happy_local_response(
        model=_FRONTIER_MODEL,
        prompt_tokens=25,
        completion_tokens=40,
    )
    fake_local = FakeProvider(name="local", response=local_resp)
    fake_frontier = FakeProvider(name="frontier", response=frontier_resp)
    fake_judge = FakeJudge(verdicts=[_verdict_low_score()])

    from corethread import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.judge, "grade", fake_judge)

    orch = orch_mod.Orchestrator(
        local=fake_local,
        frontier=fake_frontier,
        cfg=_make_app_config(),
    )
    req = _basic_request()
    response = await orch.handle(req)

    assert response is frontier_resp
    assert fake_local.call_count == 1
    assert fake_judge.call_count == 1
    assert fake_frontier.call_count == 1
    # D-27: frontier received the UNCHANGED request object (Phase 4 does NOT
    # apply constraint prefix / max_tokens cap — that's Phase 5).
    frontier_req_arg, _opts, _override, _timeout = fake_frontier.calls[0]
    assert frontier_req_arg is req

    out, _err = capfd.readouterr()
    decisions = _parse_trace_lines(out)
    assert len(decisions) == 1
    line = decisions[0]
    assert line["pivot_reason"] == "low_score"
    assert line["pivoted"] is True
    assert line["judge_parse_failed"] is False
    assert line["local_error_class"] is None
    # D-28: on pivot, trace tokens are from FRONTIER response, not local.
    assert line["input_tokens"] == 25
    assert line["output_tokens"] == 40


async def test_local_length_finish_reason_skips_judge_and_pivots(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A local response capped by max_tokens is incomplete by construction.

    The router should pivot deterministically instead of asking the judge to
    notice truncation from text alone.
    """
    setup_logging()
    local_resp = make_happy_local_response(
        model=_LOCAL_MODEL,
        content="This answer ends because the local token budget ran out",
        prompt_tokens=10,
        completion_tokens=220,
        finish_reason="length",
    )
    frontier_resp = make_happy_local_response(
        model=_FRONTIER_MODEL,
        content="complete frontier answer",
        prompt_tokens=25,
        completion_tokens=40,
    )
    fake_local = FakeProvider(name="local", response=local_resp)
    fake_frontier = FakeProvider(name="frontier", response=frontier_resp)
    fake_judge = FakeJudge(verdicts=[])

    from corethread import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.judge, "grade", fake_judge)

    orch = orch_mod.Orchestrator(
        local=fake_local,
        frontier=fake_frontier,
        cfg=_make_app_config(),
    )
    response = await orch.handle(_basic_request())

    assert response is frontier_resp
    assert fake_local.call_count == 1
    assert fake_judge.call_count == 0
    assert fake_frontier.call_count == 1

    out, _err = capfd.readouterr()
    decisions = _parse_trace_lines(out)
    assert len(decisions) == 1
    line = decisions[0]
    assert line["pivot_reason"] == "local_truncated"
    assert line["pivoted"] is True
    assert line["confidence_score"] == 0.0
    assert line["judge_latency_ms"] == 0
    assert line["judge_parse_failed"] is False
    assert line["local_error_class"] is None
    assert line["input_tokens"] == 25
    assert line["output_tokens"] == 40


async def test_stream_local_length_finish_reason_skips_judge_and_pivots(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Streaming path applies the same deterministic truncation pivot."""
    setup_logging()
    local_resp = make_happy_local_response(
        model=_LOCAL_MODEL,
        content="truncated local answer",
        finish_reason="length",
    )
    frontier_resp = make_happy_local_response(
        model=_FRONTIER_MODEL,
        content="complete frontier stream answer",
        prompt_tokens=25,
        completion_tokens=40,
    )
    fake_local = FakeProvider(name="local", response=local_resp)
    fake_frontier = FakeProvider(name="frontier", response=frontier_resp)
    fake_judge = FakeJudge(verdicts=[])

    from corethread import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.judge, "grade", fake_judge)

    orch = orch_mod.Orchestrator(
        local=fake_local,
        frontier=fake_frontier,
        cfg=_make_app_config(),
    )
    chunks = [chunk async for chunk in orch.stream(_basic_request())]

    assert fake_local.call_count == 1
    assert fake_judge.call_count == 0
    assert fake_frontier.call_count == 1
    assert any(
        chunk["choices"][0].get("delta", {}).get("content") == "complete frontier stream answer"
        for chunk in chunks
        if chunk.get("choices")
    )

    out, _err = capfd.readouterr()
    decisions = _parse_trace_lines(out)
    assert len(decisions) == 1
    assert decisions[0]["pivot_reason"] == "local_truncated"
    assert decisions[0]["pivoted"] is True
    assert decisions[0]["judge_latency_ms"] == 0


# ---------------------------------------------------------------------------
# Judge-error sentinel pivot — reasoning.startswith('judge parse failure:')
# (pivot_reason="judge_error"; judge_parse_failed=True)
# ---------------------------------------------------------------------------


async def test_judge_error_sentinel_pivot(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """D-19 judge_error branch: sentinel verdict triggers pivot with
    judge_parse_failed=True + pivot_reason='judge_error'.
    """
    setup_logging()
    local_resp = make_happy_local_response(model=_LOCAL_MODEL)
    frontier_resp = make_happy_local_response(model=_FRONTIER_MODEL)
    fake_local = FakeProvider(name="local", response=local_resp)
    fake_frontier = FakeProvider(name="frontier", response=frontier_resp)
    fake_judge = FakeJudge(verdicts=[_verdict_sentinel()])

    from corethread import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.judge, "grade", fake_judge)

    orch = orch_mod.Orchestrator(
        local=fake_local,
        frontier=fake_frontier,
        cfg=_make_app_config(),
    )
    response = await orch.handle(_basic_request())

    assert response is frontier_resp
    assert fake_judge.call_count == 1
    assert fake_frontier.call_count == 1

    out, _err = capfd.readouterr()
    decisions = _parse_trace_lines(out)
    assert len(decisions) == 1
    line = decisions[0]
    assert line["pivot_reason"] == "judge_error"
    assert line["judge_parse_failed"] is True
    assert line["pivoted"] is True


# ---------------------------------------------------------------------------
# Local-unreachable auto-pivot — ProviderUnavailable skips judge
# (pivot_reason="local_error", local_error_class="ProviderUnavailable")
# ---------------------------------------------------------------------------


async def test_local_unreachable_auto_pivot(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """D-19 local_error branch + CLAUDE.md 'local unreachable → auto-pivot'
    policy. Judge is SKIPPED entirely (fake_judge.call_count == 0).
    """
    setup_logging()
    frontier_resp = make_happy_local_response(model=_FRONTIER_MODEL)
    fake_local = FakeProvider(
        name="local",
        raise_on_chat=ProviderUnavailable("fake", "connection refused"),
    )
    fake_frontier = FakeProvider(name="frontier", response=frontier_resp)
    fake_judge = FakeJudge(verdicts=[])  # should never be called

    from corethread import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.judge, "grade", fake_judge)

    orch = orch_mod.Orchestrator(
        local=fake_local,
        frontier=fake_frontier,
        cfg=_make_app_config(),
    )
    response = await orch.handle(_basic_request())

    assert response is frontier_resp
    assert fake_local.call_count == 1  # the failing attempt
    assert fake_judge.call_count == 0  # D-19 auto-pivot skips judge
    assert fake_frontier.call_count == 1

    out, _err = capfd.readouterr()
    decisions = _parse_trace_lines(out)
    assert len(decisions) == 1
    line = decisions[0]
    assert line["pivot_reason"] == "local_error"
    assert line["local_error_class"] == "ProviderUnavailable"
    assert line["pivoted"] is True


# ---------------------------------------------------------------------------
# Local-timeout 504 no-pivot — ProviderTimeout re-raises; trace emits first
# (pivot_reason="none", local_error_class="ProviderTimeout", pivoted=False)
# ---------------------------------------------------------------------------


async def test_local_timeout_504_no_pivot(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """D-20: ProviderTimeout path — orchestrator emits PARTIAL trace and
    re-raises. Judge + frontier NOT called. 504 renders to client via
    FastAPI (tested at main.py layer in Plan 05 — this test only
    verifies orchestrator emits-then-raises).

    SC#5 fingerprint: pivot_reason='none' + local_error_class='ProviderTimeout'
    distinguishes this from the pivot paths in a single jq query.
    """
    setup_logging()
    fake_local = FakeProvider(
        name="local",
        raise_on_chat=ProviderTimeout("fake", 120.0),
    )
    fake_frontier = FakeProvider(name="frontier", response=make_happy_local_response())
    fake_judge = FakeJudge(verdicts=[])

    from corethread import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.judge, "grade", fake_judge)

    orch = orch_mod.Orchestrator(
        local=fake_local,
        frontier=fake_frontier,
        cfg=_make_app_config(),
    )

    with pytest.raises(ProviderTimeout):
        await orch.handle(_basic_request())

    assert fake_local.call_count == 1
    assert fake_judge.call_count == 0  # D-20: no judge
    assert fake_frontier.call_count == 0  # D-20: no pivot

    out, _err = capfd.readouterr()
    decisions = _parse_trace_lines(out)
    assert len(decisions) == 1, (
        f"expected 1 trace line on 504 path (D-20 emit-then-reraise), "
        f"got {len(decisions)}: {decisions}"
    )
    line = decisions[0]
    assert line["pivot_reason"] == "none"
    assert line["local_error_class"] == "ProviderTimeout"
    assert line["pivoted"] is False


# ---------------------------------------------------------------------------
# Local 4xx/5xx 502 no-pivot — ProviderHTTPError re-raises; trace emits first
# ---------------------------------------------------------------------------


async def test_local_http_error_502_no_pivot(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """D-22: ProviderHTTPError bubbles out; trace emits partial trace with
    pivot_reason='none', local_error_class='ProviderHTTPError', pivoted=False.
    """
    setup_logging()
    fake_local = FakeProvider(
        name="local",
        raise_on_chat=ProviderHTTPError("fake", 502, "upstream bad gateway"),
    )
    fake_frontier = FakeProvider(name="frontier", response=make_happy_local_response())
    fake_judge = FakeJudge(verdicts=[])

    from corethread import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.judge, "grade", fake_judge)

    orch = orch_mod.Orchestrator(
        local=fake_local,
        frontier=fake_frontier,
        cfg=_make_app_config(),
    )

    with pytest.raises(ProviderHTTPError):
        await orch.handle(_basic_request())

    assert fake_judge.call_count == 0
    assert fake_frontier.call_count == 0

    out, _err = capfd.readouterr()
    decisions = _parse_trace_lines(out)
    assert len(decisions) == 1
    line = decisions[0]
    assert line["pivot_reason"] == "none"
    assert line["local_error_class"] == "ProviderHTTPError"
    assert line["pivoted"] is False


# ---------------------------------------------------------------------------
# PIV-04 invariants — D-05 mechanism 3: parametric over ALL 6 branches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "branch,local_cfg,judge_verdicts,frontier_cfg,expect_raises,"
    "expect_local_calls,expect_judge_calls_range,expect_frontier_calls",
    [
        pytest.param(
            "happy",
            {"response": "happy_local"},
            [_verdict_pass()],
            {"response": "happy_frontier"},
            None,
            1,
            (1, 1),
            0,
            id="happy_local_pass",
        ),
        pytest.param(
            "low_score",
            {"response": "happy_local"},
            [_verdict_low_score()],
            {"response": "happy_frontier"},
            None,
            1,
            (1, 1),
            1,
            id="low_score_pivot",
        ),
        pytest.param(
            "local_truncated",
            {"response": "happy_local", "finish_reason": "length"},
            [],
            {"response": "happy_frontier"},
            None,
            1,
            (0, 0),
            1,
            id="local_truncated_auto_pivot",
        ),
        pytest.param(
            "judge_error",
            {"response": "happy_local"},
            [_verdict_sentinel()],
            {"response": "happy_frontier"},
            None,
            1,
            (1, 1),
            1,
            id="judge_error_sentinel_pivot",
        ),
        pytest.param(
            "local_error",
            {"raise_on_chat": ProviderUnavailable("fake", "refused")},
            [],  # judge never called
            {"response": "happy_frontier"},
            None,
            1,
            (0, 0),
            1,
            id="local_unreachable_auto_pivot",
        ),
        pytest.param(
            "local_timeout",
            {"raise_on_chat": ProviderTimeout("fake", 120.0)},
            [],
            {"response": "happy_frontier"},
            ProviderTimeout,
            1,
            (0, 0),
            0,
            id="local_timeout_504_no_pivot",
        ),
        pytest.param(
            "local_http_error",
            {"raise_on_chat": ProviderHTTPError("fake", 502, "boom")},
            [],
            {"response": "happy_frontier"},
            ProviderHTTPError,
            1,
            (0, 0),
            0,
            id="local_http_error_502_no_pivot",
        ),
    ],
)
async def test_piv_04_invariants(
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
    local_cfg: dict[str, Any],
    judge_verdicts: list[JudgeVerdict],
    frontier_cfg: dict[str, Any],
    expect_raises: type[Exception] | None,
    expect_local_calls: int,
    expect_judge_calls_range: tuple[int, int],
    expect_frontier_calls: int,
) -> None:
    """D-05 mechanism 3: PIV-04 single-call invariants verified for all 6
    orchestrator branches parametrically.

    Asserts:
    - fake_local.call_count == expect_local_calls (always 1 — counts the
      attempt even when it raises)
    - fake_judge.call_count in [lo, hi] from expect_judge_calls_range
      (0 on local_error/local_timeout/local_http_error; 1 on happy/
      low_score/judge_error)
    - fake_frontier.call_count == expect_frontier_calls (<=1 per PIV-04)
    """
    setup_logging()
    # Build local provider
    if "response" in local_cfg:
        fake_local = FakeProvider(
            name="local",
            response=make_happy_local_response(
                model=_LOCAL_MODEL,
                finish_reason=local_cfg.get("finish_reason", "stop"),
            ),
        )
    else:
        fake_local = FakeProvider(name="local", raise_on_chat=local_cfg["raise_on_chat"])
    # Build frontier provider
    fake_frontier = FakeProvider(
        name="frontier",
        response=make_happy_local_response(
            model=_FRONTIER_MODEL,
            prompt_tokens=25,
            completion_tokens=40,
        ),
    )
    # Build judge
    fake_judge = FakeJudge(verdicts=list(judge_verdicts))

    from corethread import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.judge, "grade", fake_judge)

    orch = orch_mod.Orchestrator(
        local=fake_local,
        frontier=fake_frontier,
        cfg=_make_app_config(),
    )

    if expect_raises is not None:
        with pytest.raises(expect_raises):
            await orch.handle(_basic_request())
    else:
        await orch.handle(_basic_request())

    # PIV-04 invariant checks (D-05 mechanism 3)
    assert fake_local.call_count == expect_local_calls, (
        f"PIV-04 [{branch}]: expected local.call_count=={expect_local_calls}, "
        f"got {fake_local.call_count}"
    )
    lo, hi = expect_judge_calls_range
    assert lo <= fake_judge.call_count <= hi, (
        f"PIV-04 [{branch}]: expected judge.call_count in [{lo},{hi}], got {fake_judge.call_count}"
    )
    assert fake_frontier.call_count == expect_frontier_calls, (
        f"PIV-04 [{branch}]: expected frontier.call_count=={expect_frontier_calls}, "
        f"got {fake_frontier.call_count}"
    )
    # Upper-bound invariant ALWAYS holds regardless of branch
    assert fake_frontier.call_count <= 1, (
        f"PIV-04 violation [{branch}]: frontier called {fake_frontier.call_count} times"
    )


# ---------------------------------------------------------------------------
# Orchestrator construction — D-02 keyword-only ctor, D-01 class shape
# ---------------------------------------------------------------------------


async def test_orchestrator_constructor_is_keyword_only() -> None:
    """D-02 ctor signature: local/frontier/cfg are keyword-only (the '*,'
    prefix). Positional construction must fail.
    """
    from corethread import orchestrator as orch_mod

    sig = inspect.signature(orch_mod.Orchestrator.__init__)
    params = sig.parameters
    # self + local + frontier + cfg
    assert "local" in params
    assert "frontier" in params
    assert "cfg" in params
    for name in ("local", "frontier", "cfg"):
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"D-02: {name} must be KEYWORD_ONLY"
        )


async def test_orchestrator_handle_signature() -> None:
    """D-03 public method surface: Orchestrator has exactly one public
    async method `handle` taking a ChatCompletionRequest and returning a
    ChatCompletionResponse.
    """
    from corethread import orchestrator as orch_mod

    assert hasattr(orch_mod, "Orchestrator")
    assert inspect.iscoroutinefunction(orch_mod.Orchestrator.handle)


# ---------------------------------------------------------------------------
# Phase 5 / Plan 05-06 — D-04 / D-16 architectural guard
#
# Phase 5 moves the constraint-prompt prepend + max_tokens clamp transform
# INSIDE OpenAIProvider.chat (D-04) — a deliberate divergence from Phase 4's
# D-27 plan that would have put the transform in the orchestrator. This guard
# enforces the new architecture at the orchestrator layer: on pivot the
# orchestrator passes the ORIGINAL, UNMODIFIED request object to
# frontier.chat(...). Without this test, a future developer could accidentally
# reinstate the Phase 4 D-27 pattern (`pivoted_request = request.with_system_prefix(...)`)
# and the existing 14 Phase 4 orchestrator tests would still pass — the
# regression would only surface at the OpenAIProvider test layer (which is
# further removed from the architectural site). This test fails FIRST.
# ---------------------------------------------------------------------------


async def test_orchestrator_passes_original_request_to_frontier_on_pivot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-04 / D-16 architectural guard: on pivot, the orchestrator calls
    ``frontier.chat(request)`` with the ORIGINAL, UNMODIFIED request object.

    The constraint-prompt prepend (D-05) and max_tokens clamp (D-06) are
    Phase 5 OpenAIProvider internals — they happen INSIDE
    ``OpenAIProvider.chat()``, not inside ``Orchestrator.handle()``. If a
    future developer accidentally reinstates Phase 4 D-27's orchestrator-
    level ``pivoted_request = request.with_system_prefix(...)`` pattern,
    this test fails BEFORE the Phase 5 test suite catches it at the
    provider layer.

    Guard mechanism: ``FakeProvider.calls`` records the exact ``request``
    positional arg per invocation (tests/_fakes.py:81). We assert IDENTITY
    on the first (and only) frontier call — same Python object, no
    prepended system message, no clamped max_tokens.

    Also reaffirms PIV-04 single-call invariant at the orchestrator layer
    (Phase 4 D-05): ``fake_frontier.call_count == 1`` on pivot.
    """
    # Build an ORIGINAL request with an intentionally unclamped max_tokens
    # (9999) and a single user message. If any transform leaked into the
    # orchestrator, at least one of these fields would change.
    original_request = ChatCompletionRequest(
        model=_LOCAL_MODEL,
        messages=[ChatMessage(role="user", content="What is 6*7?")],
        max_tokens=9999,
    )

    fake_local = FakeProvider(
        name="fake-local",
        response=make_happy_local_response(content="local-answer"),
    )
    fake_frontier = FakeProvider(
        name="fake-frontier",
        response=make_happy_local_response(content="frontier-answer"),
    )
    # Low-score verdict → confidence_score=0.5 < threshold=0.7 → pivot branch.
    fake_judge = FakeJudge(verdicts=[_verdict_low_score()])
    monkeypatch.setattr("corethread.orchestrator.judge.grade", fake_judge)

    from corethread.orchestrator import Orchestrator

    orch = Orchestrator(
        local=fake_local,
        frontier=fake_frontier,
        cfg=_make_app_config(),
    )

    response = await orch.handle(original_request)

    # Pivot happened: frontier was called exactly once (PIV-04 single-call).
    assert fake_frontier.call_count == 1, (
        f"D-16 violation: expected exactly 1 frontier call on pivot, got {fake_frontier.call_count}"
    )

    # Recover the request positional arg from the recorded tuple.
    # tests/_fakes.py:81 records (request, options, model_override, timeout_s).
    passed_request, _options, _model_override, _timeout_s = fake_frontier.calls[0]

    # D-04 IDENTITY guard: SAME Python object (is, not ==).
    # If the orchestrator built a pivoted_request via model_copy(...) or
    # with_system_prefix(...), this would be a DIFFERENT object and assert
    # would fire BEFORE the content-level guards below.
    assert passed_request is original_request, (
        "D-04 violation: orchestrator passed a DIFFERENT request object to "
        "frontier.chat() — the transform must live INSIDE OpenAIProvider, "
        "not inside Orchestrator.handle()."
    )

    # D-05 content guard (defense-in-depth): messages list was NOT prepended
    # with a constraint system message. If a future refactor replaced the
    # identity-preserving call with a structurally-identical model_copy, the
    # `is` check above would fire, but this additional assertion makes the
    # intent explicit and catches the case where someone builds a new request
    # with IDENTICAL messages but still "transforms" via some other field.
    assert passed_request.messages == original_request.messages, (
        "D-05 violation: orchestrator modified messages before pivot — "
        "constraint prepend must happen in OpenAIProvider."
    )
    assert len(passed_request.messages) == 1, (
        f"D-05 violation: expected 1 message in original request, got "
        f"{len(passed_request.messages)} — implies a system msg was prepended"
    )
    assert passed_request.messages[0].role == "user", (
        "D-05 violation: messages[0].role changed from 'user' to "
        f"'{passed_request.messages[0].role}' — orchestrator-level prepend"
    )

    # D-06 content guard: max_tokens was NOT clamped from 9999 down to
    # cfg.frontier.max_tokens (which is 512 per config.yaml.example). The
    # clamp is an OpenAIProvider internal — never the orchestrator's job.
    assert passed_request.max_tokens == 9999, (
        f"D-06 violation: max_tokens was clamped to "
        f"{passed_request.max_tokens} before reaching frontier.chat() — "
        "the clamp must happen INSIDE OpenAIProvider, not in the orchestrator."
    )

    # Sanity: orchestrator returned the frontier response (pivot actually
    # executed the full branch, not a silent fall-through).
    assert response.choices[0].message.content == "frontier-answer"
