"""Phase 3 / Plan 04 — judge.py unit + behavioral tests.

Closes ROADMAP Success Criteria:
- **SC#1**: Judge invoked exactly once with temperature=0 + format=schema;
  parsed into locked JudgeVerdict (extras rejected).
- **SC#2**: 7 adversarial inputs — 4 recoverable first-pass, 3 unrecoverable
  → sentinel on retry-also-fails. Client request NEVER 500s.
- **SC#3**: Judge model independently configured (model_override kwarg
  threads judge_model to outbound /api/chat body.model).

Plus regression guards for:
- **Pitfall #2** (populate_by_name): JudgeVerdict(pass_=...) construction works.
- **Pitfall #3** (schema alias drift): model_json_schema() default emits "pass".
- **D-11** call-count invariant: 1 call happy, 2 calls retry, never more.
- **D-07** sentinel discipline: grade() returns (not raises) on retry-fail.
- **D-08** transport propagation: ProviderHTTPError / ProviderUnavailable
  from provider.chat bubble out of grade() unchanged.

All tests are respx-driven. Zero live Ollama required.
"""

from __future__ import annotations

import inspect
import json

import httpx
import pydantic
import pytest
import respx

from corethread.errors import ProviderHTTPError, ProviderUnavailable
from corethread.judge import (
    _RETRY_CORRECTION,
    _SCORE_TABLE,
    _build_judge_messages,
    _derive_score,
    _extract_json,
    _parse_verdict,
    _sentinel_verdict,
    grade,
)
from corethread.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    JudgeVerdict,
    LocalConfig,
    to_openai_chat_completion,
)
from corethread.providers.ollama import OllamaProvider

# Phase 2 fixtures `cfg_ollama`, `ollama_http_client`, `STUB_OLLAMA_RESPONSE`
# are provided by tests/conftest.py. STUB_OLLAMA_RESPONSE is a module-level
# constant imported below (not a pytest fixture — just a dict).
from tests.conftest import STUB_OLLAMA_RESPONSE

# ---------------------------------------------------------------------------
# Module-level test helpers (NOT in conftest — per-plan isolation)
# ---------------------------------------------------------------------------


def _basic_request(question: str = "What is the capital of France?") -> ChatCompletionRequest:
    """Build a minimal ChatCompletionRequest with a single user message."""
    return ChatCompletionRequest(
        model="llama3.1:8b",
        messages=[ChatMessage(role="user", content=question)],
    )


def _local_response_with_answer(
    answer: str = "The capital of France is Paris.",
) -> ChatCompletionResponse:
    """Build a local-provider response that the judge will grade.

    Uses to_openai_chat_completion (the canonical adapter) so the shape
    matches what OllamaProvider.chat actually returns.
    """
    return to_openai_chat_completion(
        model="llama3.1:8b",
        message_content=answer,
        prompt_tokens=10,
        completion_tokens=8,
        finish_reason="stop",
    )


def _judge_llm_response(raw_content: str) -> dict[str, object]:
    """Build a variant of STUB_OLLAMA_RESPONSE with the judge's raw text
    as message.content. Used to stub respx returns for judge calls."""
    body = dict(STUB_OLLAMA_RESPONSE)  # shallow copy
    body["message"] = {"role": "assistant", "content": raw_content}
    body["model"] = "qwen2.5:7b"  # the judge_model tests use
    return body


# Happy-path judge output: 3/3 rubric, pass=true.
_HAPPY_3_3 = (
    '{"confidence_score": 0.9, '
    '"reasoning": "[answered_core_q=true, no_disclaimers=true, '
    'no_contradictions=true] answers cleanly.", '
    '"pass": true}'
)


# 2/3 rubric (will be DERIVED as confidence_score=0.5, pass=false)
_MIXED_2_3 = (
    '{"confidence_score": 0.5, '
    '"reasoning": "[answered_core_q=true, no_disclaimers=true, '
    'no_contradictions=false] contradicts itself partway.", '
    '"pass": false}'
)


# Judge model used across tests
_JUDGE_MODEL = "qwen2.5:7b"


# ---------------------------------------------------------------------------
# _extract_json — D-05 three-stage extractor
# ---------------------------------------------------------------------------


def test_extract_json_plain_json_passes_through() -> None:
    raw = '{"confidence_score": 0.9, "reasoning": "ok", "pass": true}'
    out = _extract_json(raw)
    assert json.loads(out) == {"confidence_score": 0.9, "reasoning": "ok", "pass": True}


def test_extract_json_strips_triple_backtick_fence() -> None:
    raw = '```json\n{"confidence_score": 0.9, "reasoning": "ok", "pass": true}\n```'
    out = _extract_json(raw)
    assert json.loads(out)["pass"] is True


def test_extract_json_strips_bare_triple_backtick_fence() -> None:
    raw = '```\n{"confidence_score": 0.2, "reasoning": "r", "pass": false}\n```'
    out = _extract_json(raw)
    assert json.loads(out)["pass"] is False


def test_extract_json_handles_trailing_prose() -> None:
    raw = '{"confidence_score": 0.9, "reasoning": "r", "pass": true}\n\nHope this helps!'
    out = _extract_json(raw)
    assert json.loads(out)["pass"] is True


def test_extract_json_handles_leading_prose() -> None:
    raw = 'Here is my verdict:\n{"confidence_score": 0.9, "reasoning": "r", "pass": true}'
    out = _extract_json(raw)
    assert json.loads(out)["pass"] is True


def test_extract_json_handles_nested_braces_in_reasoning() -> None:
    """Balanced-brace scanner MUST correctly track depth when reasoning
    contains literal `{...}` characters (regex cannot do this per
    Pitfall #6)."""
    raw = (
        '{"confidence_score": 0.5, '
        '"reasoning": "saw nested {object} in output - partial match", '
        '"pass": false}'
    )
    out = _extract_json(raw)
    parsed = json.loads(out)
    assert parsed["pass"] is False
    assert "{object}" in parsed["reasoning"]


def test_extract_json_handles_bom_prefix() -> None:
    """BOM-prefixed input — the pre-{ prefix is discarded by find('{')."""
    raw = '﻿{"confidence_score": 0.9, "reasoning": "r", "pass": true}'
    out = _extract_json(raw)
    assert json.loads(out)["pass"] is True


def test_extract_json_handles_escaped_quotes_in_strings() -> None:
    """Balanced-brace scanner must NOT descend into strings — escape-aware."""
    raw = '{"confidence_score": 0.9, "reasoning": "He said \\"hi\\"", "pass": true}'
    out = _extract_json(raw)
    assert json.loads(out)["reasoning"] == 'He said "hi"'


def test_extract_json_truncated_falls_through_to_regex_fallback() -> None:
    """Truncated JSON — balanced-brace scanner fails mid-string; regex
    fallback grabs first-{ to last-} substring. Result may be malformed
    (json.loads would reject it) but extractor returns something for retry."""
    raw = '{"confidence_score": 0.9, "reasoning": "ok", "pa'
    out = _extract_json(raw)
    # No closing brace at all in this case — fallback returns raw[start:].strip()
    # which starts with the leading `{"`. The important property is that
    # _extract_json does NOT raise and returns a string that will fail
    # json.loads (driving grade() into the retry path).
    assert out.startswith('{"')


def test_extract_json_plain_english_no_braces() -> None:
    """Plain English with no JSON at all — returns stripped original."""
    raw = "I think this looks good, about 80% confident."
    out = _extract_json(raw)
    assert "{" not in out


# ---------------------------------------------------------------------------
# _derive_score — D-03 / D-04 rubric-to-score overwrite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rubric_prefix,expected_score,expected_pass",
    [
        (
            "[answered_core_q=true, no_disclaimers=true, no_contradictions=true]",
            0.9,
            True,
        ),
        (
            "[answered_core_q=true, no_disclaimers=true, no_contradictions=false]",
            0.5,
            False,
        ),
        (
            "[answered_core_q=true, no_disclaimers=false, no_contradictions=false]",
            0.2,
            False,
        ),
        (
            "[answered_core_q=false, no_disclaimers=false, no_contradictions=false]",
            0.0,
            False,
        ),
        # Permutations — position of the `true` should not matter
        (
            "[answered_core_q=false, no_disclaimers=true, no_contradictions=true]",
            0.5,
            False,
        ),
        (
            "[answered_core_q=false, no_disclaimers=false, no_contradictions=true]",
            0.2,
            False,
        ),
    ],
)
def test_derive_score_table(
    rubric_prefix: str,
    expected_score: float,
    expected_pass: bool,
) -> None:
    """D-04: score table is locked. pass iff all three rubric booleans true.

    The LLM may have emitted raw pass=true + confidence_score=1.0 — these
    MUST be overwritten by the rubric-derived values. This is the T-03-01
    tier-2 STRUCTURAL defense: even a jailbroken judge cannot produce a
    passing verdict unless it also emits all three rubric booleans as true.
    """
    v = JudgeVerdict(
        pass_=True,
        confidence_score=1.0,
        reasoning=f"{rubric_prefix} explanation here.",
    )
    d = _derive_score(v)
    assert d.confidence_score == expected_score
    assert d.pass_ is expected_pass


def test_derive_score_handles_missing_rubric_prefix() -> None:
    """Judge misbehaved — rubric prefix absent. Falls back to 0/3."""
    v = JudgeVerdict(
        pass_=True,
        confidence_score=0.95,
        reasoning="This answer is fine.",  # no rubric prefix
    )
    d = _derive_score(v)
    assert d.confidence_score == 0.0
    assert d.pass_ is False


def test_derive_score_case_insensitive_rubric() -> None:
    """IGNORECASE flag on _RUBRIC_RE — TRUE / True / true all count."""
    v = JudgeVerdict(
        pass_=True,
        confidence_score=1.0,
        reasoning=(
            "[answered_core_q=TRUE, no_disclaimers=True, no_contradictions=true] mixed case"
        ),
    )
    d = _derive_score(v)
    assert d.confidence_score == 0.9
    assert d.pass_ is True


def test_derive_score_raw_judge_json_can_veto_all_true_rubric() -> None:
    """A lower raw judge score/pass=false should still pivot conservatively."""
    v = JudgeVerdict(
        pass_=False,
        confidence_score=0.65,
        reasoning=(
            "[answered_core_q=true, no_disclaimers=true, no_contradictions=true] "
            "The answer starts correctly but is abruptly cut off."
        ),
    )
    d = _derive_score(v)
    assert d.confidence_score == 0.65
    assert d.pass_ is False


# ---------------------------------------------------------------------------
# _sentinel_verdict — D-07 fail-safe construction
# ---------------------------------------------------------------------------


def test_sentinel_verdict_shape() -> None:
    v = _sentinel_verdict("JSONDecodeError")
    assert v.pass_ is False
    assert v.confidence_score == 0.0
    assert v.reasoning == "judge parse failure: JSONDecodeError"


def test_sentinel_verdict_populate_by_name_regression() -> None:
    """Pitfall #2 regression guard: JudgeVerdict(pass_=False, ...) must
    work because models.py declares populate_by_name=True. If that flag
    is ever dropped in a refactor, this test fails loud.
    """
    v = JudgeVerdict(
        pass_=False,
        confidence_score=0.0,
        reasoning="pitfall-2 regression guard",
    )
    assert v.pass_ is False


# ---------------------------------------------------------------------------
# JudgeVerdict.model_json_schema — Pitfall #3 regression guard
# ---------------------------------------------------------------------------


def test_judge_verdict_schema_uses_pass_alias_with_no_args() -> None:
    """Pitfall #3: calling .model_json_schema() with no args (default
    by_alias=True) MUST emit "pass" (not "pass_"). If a future call adds
    by_alias=False, Ollama schema matching would break silently."""
    schema = JudgeVerdict.model_json_schema()
    assert "pass" in schema["properties"]
    assert "pass_" not in schema["properties"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"confidence_score", "reasoning", "pass"}


# ---------------------------------------------------------------------------
# _build_judge_messages — D-02 context + T-03-01 delimiters
# ---------------------------------------------------------------------------


def test_build_judge_messages_uses_last_user_only() -> None:
    """D-02: only the LAST user message + the local answer appear in the
    judge's context — not the full chat history."""
    req = ChatCompletionRequest(
        model="llama3.1:8b",
        messages=[
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="What is 2+2?"),
            ChatMessage(role="assistant", content="4."),
            ChatMessage(role="user", content="Now what is the capital of France?"),
        ],
    )
    local_resp = _local_response_with_answer("Paris.")
    msgs = _build_judge_messages(req, local_resp)

    # Exactly 2 messages: system (judge role) + user (question + answer).
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"

    # ONLY the last user turn appears; earlier turns are NOT in the judge user msg.
    user_content = msgs[1].content
    assert isinstance(user_content, str)
    assert "capital of France" in user_content
    assert "What is 2+2" not in user_content  # earlier user turn excluded
    assert "Paris" in user_content  # local answer is in the ANSWER block


def test_build_judge_messages_includes_delimiter_framing() -> None:
    """T-03-01: delimiter framing wraps user content as DATA, not instructions."""
    req = _basic_request("Question text?")
    local_resp = _local_response_with_answer("Answer text.")
    msgs = _build_judge_messages(req, local_resp)
    user_content = msgs[1].content
    assert isinstance(user_content, str)
    assert "---BEGIN QUESTION---" in user_content
    assert "---END QUESTION---" in user_content
    assert "---BEGIN ANSWER---" in user_content
    assert "---END ANSWER---" in user_content


def test_build_judge_messages_system_has_negative_bias_phrase() -> None:
    """T-03-01 tier-3 (prompt-layer): system message contains the
    negative-bias counter-prompt."""
    req = _basic_request()
    local_resp = _local_response_with_answer()
    msgs = _build_judge_messages(req, local_resp)
    system_content = msgs[0].content
    assert isinstance(system_content, str)
    assert "err on the side of flagging" in system_content.lower()


def test_build_judge_messages_uses_configured_system_prompt() -> None:
    req = _basic_request()
    local_resp = _local_response_with_answer()
    msgs = _build_judge_messages(
        req,
        local_resp,
        system_prompt="Custom judge rubric prompt.",
    )

    assert msgs[0].content == "Custom judge rubric prompt."


# ---------------------------------------------------------------------------
# grade() — happy path (D-10 + D-11 + SC#1 + SC#3)
# ---------------------------------------------------------------------------


async def test_grade_happy_path_calls_provider_exactly_once(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-11: happy path — provider.chat called exactly ONCE. SC#1."""
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=_judge_llm_response(_HAPPY_3_3))
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True  # skip lazy warmup

    verdict = await grade(
        _basic_request(),
        _local_response_with_answer(),
        provider=provider,
        judge_model=_JUDGE_MODEL,
    )

    # D-11: exactly one call.
    assert route.call_count == 1
    # D-04 derived score (3/3 → 0.9) overwrites LLM's raw value.
    assert verdict.confidence_score == 0.9
    assert verdict.pass_ is True


async def test_grade_forwards_temperature_0_and_format_schema(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """SC#1: JDG-02 temperature=0 + Ollama format=JudgeVerdict.model_json_schema().
    Inspects the outbound body of the judge's /api/chat POST."""
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=_judge_llm_response(_HAPPY_3_3))
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    await grade(
        _basic_request(),
        _local_response_with_answer(),
        provider=provider,
        judge_model=_JUDGE_MODEL,
    )

    body = json.loads(route.calls[0].request.content.decode("utf-8"))
    # D-03 temperature=0.0 (ChatOptions temperature forwarded by adapter)
    assert body["options"]["temperature"] == 0.0
    # D-03 top_p=1.0 forwarded
    assert body["options"].get("top_p") == 1.0
    # Pitfall #3: format is JudgeVerdict.model_json_schema() shape
    fmt = body["format"]
    assert fmt["additionalProperties"] is False
    assert set(fmt["required"]) == {"confidence_score", "reasoning", "pass"}
    assert "pass" in fmt["properties"]
    assert "pass_" not in fmt["properties"]


async def test_grade_forwards_model_override_for_judge_dispatch(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """SC#3 + D-09 / JDG-06: judge_model threads through model_override;
    outbound /api/chat body.model == judge_model (NOT cfg.local.model)."""
    respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=_judge_llm_response(_HAPPY_3_3))
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    await grade(
        _basic_request(),
        _local_response_with_answer(),
        provider=provider,
        judge_model="mistral:7b-instruct",  # different from cfg_ollama.model (llama3.1:8b)
    )

    body = json.loads(respx_mock.calls[0].request.content.decode("utf-8"))
    assert body["model"] == "mistral:7b-instruct"


async def test_grade_forwards_configured_system_prompt(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json=_judge_llm_response(_HAPPY_3_3))
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    await grade(
        _basic_request(),
        _local_response_with_answer(),
        provider=provider,
        judge_model=_JUDGE_MODEL,
        system_prompt="Custom judge rubric prompt.",
    )

    body = json.loads(respx_mock.calls[0].request.content.decode("utf-8"))
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "Custom judge rubric prompt."


# ---------------------------------------------------------------------------
# grade() — retry path (D-06 + D-11)
# ---------------------------------------------------------------------------


async def test_grade_retry_path_calls_provider_exactly_twice(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-06 + D-11 retry path: first call returns malformed → parse fails →
    retry call returns valid JSON → parse succeeds. provider.chat called
    EXACTLY TWICE."""
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        side_effect=[
            httpx.Response(200, json=_judge_llm_response("not valid json at all")),
            httpx.Response(200, json=_judge_llm_response(_HAPPY_3_3)),
        ]
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    verdict = await grade(
        _basic_request(),
        _local_response_with_answer(),
        provider=provider,
        judge_model=_JUDGE_MODEL,
    )

    # D-11: exactly two calls (one more, not N).
    assert route.call_count == 2
    # Retry recovered with 3/3 rubric.
    assert verdict.pass_ is True
    assert verdict.confidence_score == 0.9
    # NOT the sentinel path.
    assert not verdict.reasoning.startswith("judge parse failure:")


async def test_grade_retry_appends_correction_system_message(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-06: the retry request contains the ORIGINAL messages + one
    appended system message with _RETRY_CORRECTION verbatim."""
    respx_mock.post("http://ollama:11434/api/chat").mock(
        side_effect=[
            httpx.Response(200, json=_judge_llm_response("garbage")),
            httpx.Response(200, json=_judge_llm_response(_HAPPY_3_3)),
        ]
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    await grade(
        _basic_request(),
        _local_response_with_answer(),
        provider=provider,
        judge_model=_JUDGE_MODEL,
    )

    # Inspect the two call bodies.
    first_body = json.loads(respx_mock.calls[0].request.content.decode("utf-8"))
    retry_body = json.loads(respx_mock.calls[1].request.content.decode("utf-8"))
    retry_messages = retry_body["messages"]
    # Retry has EXACTLY one more message than the first call.
    assert len(retry_messages) == len(first_body["messages"]) + 1
    # The last message is a system message with _RETRY_CORRECTION.
    assert retry_messages[-1]["role"] == "system"
    assert retry_messages[-1]["content"] == _RETRY_CORRECTION


# ---------------------------------------------------------------------------
# grade() — D-07 sentinel fail-safe
# ---------------------------------------------------------------------------


async def test_grade_both_calls_fail_returns_sentinel_verdict(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-07: both first and retry return malformed JSON → sentinel verdict.
    NEVER raises. pass_=False, confidence_score=0.0, reasoning starts with
    'judge parse failure:'."""
    route = respx_mock.post("http://ollama:11434/api/chat").mock(
        side_effect=[
            httpx.Response(200, json=_judge_llm_response("totally malformed")),
            httpx.Response(200, json=_judge_llm_response("still malformed")),
        ]
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    verdict = await grade(
        _basic_request(),
        _local_response_with_answer(),
        provider=provider,
        judge_model=_JUDGE_MODEL,
    )

    assert route.call_count == 2
    assert verdict.pass_ is False
    assert verdict.confidence_score == 0.0
    assert verdict.reasoning.startswith("judge parse failure:")
    # err_class NAME ONLY, not str(exc). Regression guard for Pitfall #12.
    assert (
        "JSONDecodeError" in verdict.reasoning
        or "ValidationError" in verdict.reasoning
        or "ValueError" in verdict.reasoning
    )


async def test_grade_never_raises_judge_parse_error(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-07 discipline: even on retry-also-fails, grade returns (not
    raises). Client requests should never 500 over a judge parse failure."""
    respx_mock.post("http://ollama:11434/api/chat").mock(
        side_effect=[
            httpx.Response(200, json=_judge_llm_response("nope")),
            httpx.Response(200, json=_judge_llm_response("still nope")),
        ]
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    # Must not raise — if it does, the test fails.
    verdict = await grade(
        _basic_request(),
        _local_response_with_answer(),
        provider=provider,
        judge_model=_JUDGE_MODEL,
    )
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.reasoning.startswith("judge parse failure:")


# ---------------------------------------------------------------------------
# grade() — D-08 transport propagation
# ---------------------------------------------------------------------------


async def test_grade_propagates_provider_http_error(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-08: ProviderHTTPError from adapter (upstream 5xx) propagates out of
    grade() — NOT caught, NOT converted to sentinel."""
    respx_mock.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(503, text="service unavailable")
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    with pytest.raises(ProviderHTTPError):
        await grade(
            _basic_request(),
            _local_response_with_answer(),
            provider=provider,
            judge_model=_JUDGE_MODEL,
        )


async def test_grade_propagates_provider_unavailable(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
) -> None:
    """D-08: ProviderUnavailable (connection refused) propagates — the
    orchestrator auto-pivots at this class per CLAUDE.md."""
    respx_mock.post("http://ollama:11434/api/chat").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    with pytest.raises(ProviderUnavailable):
        await grade(
            _basic_request(),
            _local_response_with_answer(),
            provider=provider,
            judge_model=_JUDGE_MODEL,
        )


# ---------------------------------------------------------------------------
# grade() — SC#2 parametrized adversarial-input sweep (7 cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expect_sentinel",
    [
        # (1) fenced JSON — first-pass recoverable
        (
            f"```json\n{_HAPPY_3_3}\n```",
            False,
        ),
        # (2) trailing prose — first-pass recoverable
        (
            f"{_HAPPY_3_3}\n\nHope this helps!",
            False,
        ),
        # (3) nested braces in reasoning — first-pass recoverable
        (
            '{"confidence_score": 0.9, "reasoning": '
            '"[answered_core_q=true, no_disclaimers=true, '
            'no_contradictions=true] contains literal {nested} braces.", '
            '"pass": true}',
            False,
        ),
        # (4) BOM-prefixed JSON — first-pass recoverable
        (
            "﻿" + _HAPPY_3_3,
            False,
        ),
        # (5) wrong key names — first call fails, retry ALSO fails → sentinel
        (
            '{"score": 0.9, "reason": "oops", "passed": true}',
            True,
        ),
        # (6) truncated JSON — first call fails, retry ALSO fails → sentinel
        (
            '{"confidence_score": 0.9, "reasoning": "ok", "pa',
            True,
        ),
        # (7) plain English prose — first call fails, retry ALSO fails → sentinel
        (
            "I think this looks good, about 80% confident.",
            True,
        ),
    ],
)
async def test_grade_adversarial_inputs_sc2(
    respx_mock: respx.MockRouter,
    cfg_ollama: LocalConfig,
    ollama_http_client: httpx.AsyncClient,
    raw: str,
    expect_sentinel: bool,
) -> None:
    """SC#2: each adversarial input either recovers first-pass OR falls
    through retry to sentinel. Client request NEVER 500s — grade returns
    a valid JudgeVerdict in every case.

    For cases 5/6/7 we stub BOTH first and retry with the same malformed
    input (same side_effect list with two entries) to drive the sentinel
    path deterministically. For cases 1-4 one stub (return_value) suffices —
    only one provider.chat call happens.
    """
    if expect_sentinel:
        # Two same-malformed responses to drive sentinel (both parse-fail).
        respx_mock.post("http://ollama:11434/api/chat").mock(
            side_effect=[
                httpx.Response(200, json=_judge_llm_response(raw)),
                httpx.Response(200, json=_judge_llm_response(raw)),
            ]
        )
    else:
        # One recoverable response (no retry needed).
        respx_mock.post("http://ollama:11434/api/chat").mock(
            return_value=httpx.Response(200, json=_judge_llm_response(raw))
        )

    provider = OllamaProvider(cfg_ollama, ollama_http_client)
    provider._warmed = True

    verdict = await grade(
        _basic_request(),
        _local_response_with_answer(),
        provider=provider,
        judge_model=_JUDGE_MODEL,
    )

    # NEVER raised — grade ALWAYS returns a JudgeVerdict.
    assert isinstance(verdict, JudgeVerdict)

    if expect_sentinel:
        assert verdict.reasoning.startswith("judge parse failure:"), (
            f"expected sentinel for {raw!r}, got reasoning={verdict.reasoning!r}"
        )
        assert verdict.pass_ is False
        assert verdict.confidence_score == 0.0
    else:
        # Recoverable: rubric prefix present → 3/3 → 0.9, pass=True.
        # (All recoverable cases in this parametrization carry the 3/3 rubric.)
        assert verdict.pass_ is True
        assert verdict.confidence_score == 0.9


# ---------------------------------------------------------------------------
# grade() — D-10 signature contract
# ---------------------------------------------------------------------------


def test_grade_signature_contract() -> None:
    """D-10 locked signature: keyword-only provider + judge_model +
    timeout_s (default 10.0) + configurable system_prompt."""
    sig = inspect.signature(grade)
    params = sig.parameters
    assert list(params.keys()) == [
        "request",
        "local_response",
        "provider",
        "judge_model",
        "timeout_s",
        "system_prompt",
    ]
    assert params["provider"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["judge_model"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["timeout_s"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["system_prompt"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["timeout_s"].default == 10.0


# ---------------------------------------------------------------------------
# _SCORE_TABLE literal-value regression
# ---------------------------------------------------------------------------


def test_score_table_is_locked_values() -> None:
    """D-04 regression guard: if someone retunes _SCORE_TABLE without
    also updating cfg.routing.threshold, the bimodal distribution breaks.
    Lock the literal values."""
    assert _SCORE_TABLE == {3: 0.9, 2: 0.5, 1: 0.2, 0: 0.0}


# ---------------------------------------------------------------------------
# _parse_verdict — raises on malformed input
# ---------------------------------------------------------------------------


def test_parse_verdict_raises_on_wrong_keys() -> None:
    """extra='forbid' on JudgeVerdict — wrong key names raise ValidationError."""
    raw = '{"score": 0.9, "reason": "oops", "passed": true}'
    with pytest.raises((pydantic.ValidationError, json.JSONDecodeError, ValueError)):
        _parse_verdict(raw)


def test_parse_verdict_raises_on_plain_english() -> None:
    """Plain English with no JSON at all — extract returns stripped text;
    pydantic-core's JSON parser rejects it."""
    raw = "I think this looks good, about 80% confident."
    with pytest.raises((pydantic.ValidationError, json.JSONDecodeError, ValueError)):
        _parse_verdict(raw)
