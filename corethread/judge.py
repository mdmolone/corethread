"""Judge module — strict-JSON LLM-as-judge with one-retry + fail-safe sentinel.

Phase 3 / Plan 03 per 03-CONTEXT.md. Closes JDG-01..JDG-06.

This module is the confidence-check in the core value proposition: after
every local inference, Phase 4's orchestrator calls ``grade(...)`` to
evaluate the local answer against a binary 3-item rubric
(``answered_core_q``, ``no_disclaimers``, ``no_contradictions``) per D-01.
The rubric booleans are emitted by the Judge LLM as a structured prefix
inside ``JudgeVerdict.reasoning`` (D-03). After parse, ``_derive_score``
overwrites ``confidence_score`` and ``pass_`` deterministically from the
parsed rubric booleans (D-04: 3:0.9, 2:0.5, 1:0.2, 0:0.0; pass iff all
three True). The LLM's raw ``confidence_score`` / ``pass`` fields from
strict-JSON output are DISCARDED — the judge's job is answering three
yes/no questions reliably, not calibrating a float (Pitfall #7 mitigation
baked into the scoring layer).

Parse is defensive per D-05: triple-backtick fence strip → balanced-brace
scan (handles nested ``{...}`` in reasoning, which regex cannot reliably
per Pitfall #6) → first-``{``/last-``}`` regex fallback on truncation.
Pydantic ``JudgeVerdict.model_validate_json(...)`` does the final
validation with ``extra="forbid"`` (JDG-03). On parse failure, ONE retry
is sent with the ORIGINAL ``[system, user]`` messages plus one appended
system message (``_RETRY_CORRECTION``) telling the LLM to output ONLY the
JSON object — no markdown, no prose (D-06). On retry-also-fails, a
sentinel verdict with failing pass + zero score + a ``"judge parse
failure: <err_class>"`` reasoning string is returned, NEVER raised
(D-07). The orchestrator's threshold check
(``confidence_score < 0.7``) naturally pivots sentinel verdicts to the
frontier without special-case code.

Transport errors from the provider (``ProviderUnavailable``,
``ProviderTimeout``, ``ProviderHTTPError``) PROPAGATE unchanged out of
``grade()`` (D-08). The orchestrator (Phase 4) distinguishes
``ProviderUnavailable`` (auto-pivot, skip judge) from ``ProviderTimeout``
(504 to client, do NOT pivot) per CLAUDE.md — ``judge.py`` only
transforms parse-outcomes; transport policy is upstream.

This module may import from ``models`` (JudgeVerdict + request/response
shapes), ``providers.base`` (Provider + ChatOptions types), stdlib (re,
json — though json is consumed transitively via Pydantic's
``model_validate_json``), and Pydantic (ValidationError catch). It MUST
NOT import from ``errors`` (D-07 — ``JudgeParseError`` exists but is
never raised here), ``main``, ``config``, ``orchestrator``, ``obs`` (peer
of judge.py — the orchestrator composes them, not each other), or any
concrete provider (``providers.ollama``, ``providers.lmstudio``,
``providers.openai``) — dispatch is through the ABC per D-10/D-11.
"""

from __future__ import annotations

import json
import re

import pydantic
import structlog

from corethread.models import (
    DEFAULT_JUDGE_PROMPT,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    JudgeVerdict,
)
from corethread.providers.base import ChatOptions, Provider

__all__ = ["DEFAULT_JUDGE_PROMPT", "grade"]


# Module-level logger — logger name becomes "judge" so operators can filter
# the judge subsystem separately from providers.* and corethread.obs. Inherits
# the root RedactingFilter + structlog processor chain installed by
# `logging_config.py` (Pitfall #12 defense: nested-dict-recursing redaction
# processor scrubs sk-* / Authorization: Bearer * in any reasoning echo).
_LOG = structlog.get_logger(__name__)


# D-06 retry correction message — exact string; locks the reprompt shape.
# Do NOT paraphrase: the Pitfall #6 mitigation relies on a short,
# unambiguous instruction delta between call 1 and call 2 so the LLM
# sees a NEW constraint rather than a rephrased existing one.
_RETRY_CORRECTION = (
    "Your previous response was not valid JSON. Output ONLY the JSON object"
    " matching the schema — no markdown fences, no prose before or after."
)


# D-05 extractor: triple-backtick fence strip precedes balanced-brace scan.
# Matches ```json\n...\n``` and ```\n...\n``` with the fenced body in
# group(1). DOTALL so `.` matches newlines; IGNORECASE so `json`/`JSON`
# language markers both match.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


# D-03 rubric-prefix regex: matches the structured prefix in reasoning
# content `"[answered_core_q=true, no_disclaimers=true, no_contradictions=false]
# <one-sentence>"`. The three capture groups are the three booleans in
# order. IGNORECASE allows True/TRUE/true equally (judge LLMs are
# inconsistent). Whitespace is tolerated around the `=` and `,` to absorb
# formatting drift.
_RUBRIC_RE = re.compile(
    r"\[\s*answered_core_q\s*=\s*(true|false)\s*,"
    r"\s*no_disclaimers\s*=\s*(true|false)\s*,"
    r"\s*no_contradictions\s*=\s*(true|false)\s*\]",
    re.IGNORECASE,
)


# D-04 locked score mapping — DO NOT RETUNE without changing
# cfg.routing.threshold to match (0.7 default produces clean bimodal
# distribution: 0.9 pass / ≤0.5 pivot).
_SCORE_TABLE: dict[int, float] = {3: 0.9, 2: 0.5, 1: 0.2, 0: 0.0}


# D-02 + T-03-01 delimiter framing: user content is wrapped between these
# markers so the judge system message can tell the LLM to treat delimited
# content as DATA, not instructions.
_Q_BEGIN = "---BEGIN QUESTION---"
_Q_END = "---END QUESTION---"
_A_BEGIN = "---BEGIN ANSWER---"
_A_END = "---END ANSWER---"


def _build_judge_messages(
    request: ChatCompletionRequest,
    local_response: ChatCompletionResponse,
    *,
    system_prompt: str = DEFAULT_JUDGE_PROMPT,
) -> list[ChatMessage]:
    """Compose the judge's [system, user] message pair — D-02 locked shape.

    Context window is the LAST user message + the local answer ONLY; NOT
    the full chat history (Pitfall #4 defense against judge-ctx truncation
    + Pitfall #7 mitigation against history-length self-preference drift).

    System message: role definition + negative-bias counter-prompt ("err
    on the side of flagging") + rubric schema + output format contract.
    The delimiter framing (T-03-01) instructs the judge to treat embedded
    QUESTION / ANSWER content as data, not instructions — prompt-injection
    mitigation is reinforced structurally by ``_derive_score`` overwriting
    the LLM's raw pass/score fields (so a jailbroken LLM that emits
    ``pass=true`` still cannot produce a pass verdict unless it ALSO
    emits the rubric booleans as all-true in the reasoning prefix).
    """
    # Find the last USER message. User's request.messages can legally
    # contain system messages (OpenAI role), assistant turns (multi-turn
    # history), and tool messages; we want the last USER turn specifically
    # because that is the question being graded.
    last_user_content: str = ""
    for msg in reversed(request.messages):
        if msg.role == "user" and isinstance(msg.content, str):
            last_user_content = msg.content
            break
        if msg.role == "user" and isinstance(msg.content, list):
            # Multimodal content list — flatten to string representation.
            # v1 does not grade image/audio parts individually; we fall
            # back to a repr-ish string so the rubric can still apply.
            last_user_content = str(msg.content)
            break

    # Local answer: choices[0].message.content. Phase 2's
    # to_openai_chat_completion always populates at least one choice;
    # OllamaChatResponse guarantees a non-None content string (empty
    # string is legal for 1-token warmup but will never happen on a real
    # chat turn — if it does, the rubric will correctly fail
    # answered_core_q).
    choice_0_msg = local_response.choices[0].message
    if isinstance(choice_0_msg.content, str):
        local_answer = choice_0_msg.content
    else:
        # Assistant multimodal content is rare but possible; stringify.
        local_answer = str(choice_0_msg.content or "")

    user_content = (
        f"{_Q_BEGIN}\n{last_user_content}\n{_Q_END}\n\n{_A_BEGIN}\n{local_answer}\n{_A_END}"
    )

    return [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_content),
    ]


def _extract_json(raw: str) -> str:
    """Defensive extract-then-parse per D-05.

    Sequence:
      (1) Triple-backtick fence strip (``_FENCE_RE``).
      (2) Balanced-brace scan from first ``{`` to its matching ``}``
          (handles nested braces in reasoning — regex cannot reliably per
          Pitfall #6 guidance).
      (3) On unbalanced (truncated) input, fall back to first-``{`` to
          last-``}`` substring; ``json.loads`` will reject the malformed
          result and grade() will retry.

    Pure function — no HTTP, no provider reference; unit-testable with
    the 7 adversarial inputs from 03-RESEARCH.md Example 2. Claude's
    discretion (03-CONTEXT.md): zero-width / BOM stripping is SKIPPED
    here — respx fixtures don't reveal models that emit them; revisit if
    live-UAT flags an unrecoverable-JSON with hidden chars.
    """
    # (1) Triple-backtick fence strip
    fence = _FENCE_RE.search(raw)
    if fence:
        raw = fence.group(1)

    # (2) Balanced-brace scan
    start = raw.find("{")
    if start < 0:
        return raw.strip()  # no brace — Pydantic will reject
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]

    # (3) Unbalanced — regex fallback
    last = raw.rfind("}")
    if last > start:
        return raw[start : last + 1]
    return raw[start:].strip()


def _response_content_as_str(response: ChatCompletionResponse) -> str:
    """Extract ``choices[0].message.content`` as a plain str for parsing.

    ``ChatMessage.content`` is typed ``str | list[dict[str, Any]] | None``
    (multimodal forward-compat per models.py). The judge response is
    always plain text because the judge prompt demands strict JSON —
    but the type system does not know that, so we normalize here. A
    ``None`` or unexpected list becomes ``""`` which will fail extraction
    / validation and route through the retry → sentinel path per D-06/D-07.
    """
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    # List/multimodal — judge response should never be this shape. Stringify
    # so _extract_json sees SOMETHING; Pydantic validation will reject it.
    return str(content)


def _parse_verdict(raw_content: str) -> JudgeVerdict:
    """Extract → Pydantic validate. Raises on any failure.

    Raises:
        json.JSONDecodeError: balanced-brace scanner produced malformed
            JSON (typically on truncation + fallback path).
        pydantic.ValidationError: JSON parsed but failed JudgeVerdict's
            ``extra="forbid"`` / field-type / range constraints.
        ValueError: catch-all for any other parse failure.

    ``JudgeVerdict``'s ``hide_input_in_errors=True`` flag means the raw
    judge text WILL NOT appear in any ``ValidationError.errors()[i]["input"]``
    — so this function's callers may log ``field_paths`` derived from
    ``exc.errors()`` without leaking user content (Pitfall #12).
    """
    extracted = _extract_json(raw_content)
    # model_validate_json is the Pydantic v2 idiom; uses Pydantic's
    # internal JSON parser (pydantic-core), NOT json.loads — slightly
    # faster and integrates with validator + coercer chain directly.
    return JudgeVerdict.model_validate_json(extracted)


def _derive_score(verdict: JudgeVerdict) -> JudgeVerdict:
    """D-03/D-04: parse rubric booleans from ``verdict.reasoning``;
    derive conservative ``confidence_score`` + ``pass_`` values.

    Raw ``confidence_score`` / ``pass`` values are allowed to lower a verdict,
    but never to raise one. The rubric-boolean derivation remains the security
    property (T-03-01 tier-2
    defense: even a jailbroken judge emitting ``pass=true`` cannot pass
    unless it ALSO emits all three rubric booleans as true). Raw judge JSON
    can still veto: ``pass=false`` fails, and the raw ``confidence_score`` is
    used only as a lower bound via ``min(raw, rubric_score)``. It can lower
    confidence, never inflate it.

    If the rubric prefix cannot be parsed (judge ignored the format
    instruction despite strict-JSON mode), treat as 0/3 (judge
    misbehaved, pivot). No retry here — the JSON parse already succeeded;
    the LLM just produced reasoning in a non-prefixed format. Score of
    0.0 and pass_=False will trigger the orchestrator pivot naturally.
    """
    m = _RUBRIC_RE.search(verdict.reasoning)
    if not m:
        # Rubric prefix absent — judge misbehaved. Produce 0/3.
        return verdict.model_copy(
            update={
                "confidence_score": 0.0,
                "pass_": False,
            }
        )
    trues = sum(1 for g in m.groups() if g.lower() == "true")
    confidence_score = min(_SCORE_TABLE[trues], verdict.confidence_score)
    return verdict.model_copy(
        update={
            "confidence_score": confidence_score,
            "pass_": (trues == 3 and verdict.pass_),
        }
    )


def _sentinel_verdict(err_class_name: str) -> JudgeVerdict:
    """D-07 fail-safe sentinel. ``err_class_name`` is EXCEPTION CLASS NAME
    ONLY — NEVER ``str(exc)`` or ``repr(exc)`` per Pitfall #12 / T-02-01
    discipline (same pattern as ``providers/ollama.py`` _last_warmup_error).

    Constructed via ``JudgeVerdict(pass_=...)`` using the
    ``populate_by_name=True`` flag from the model's ``ConfigDict``. If a
    future refactor drops that flag, this call fails at runtime — Pitfall
    #2 regression surfaces immediately.
    """
    return JudgeVerdict(
        pass_=False,
        confidence_score=0.0,
        reasoning=f"judge parse failure: {err_class_name}",
    )


async def grade(
    request: ChatCompletionRequest,
    local_response: ChatCompletionResponse,
    *,
    provider: Provider,
    judge_model: str,
    timeout_s: float = 10.0,
    system_prompt: str = DEFAULT_JUDGE_PROMPT,
) -> JudgeVerdict:
    """Grade the local answer. Returns a ``JudgeVerdict`` — NEVER raises
    for parse failures (D-07 sentinel). Transport errors propagate (D-08).

    Structural call-count invariant (D-11):
        - Happy path: ``provider.chat`` called exactly ONCE.
        - Retry path: ``provider.chat`` called exactly TWICE.
        - Never three or more. No tenacity, no N-attempts loop.

    Parameters (all keyword-only except ``request`` and ``local_response``):
        request: the original client request (used to find the last
            user message for D-02 context composition).
        local_response: the local provider's answer (used as the ANSWER
            block in the judge's user message).
        provider: the ``Provider`` instance to dispatch against. The
            judge REUSES the local provider (Phase 4 orchestrator passes
            the same ``OllamaProvider`` instance that produced
            ``local_response``) — shared ``httpx.AsyncClient`` per
            Pitfall #11. Dispatch to a different backend model is via
            ``model_override`` (D-09 from 03-01), not a different
            provider instance.
        judge_model: the backend model name the judge calls. Typically
            ``cfg.judge.model``. Passed verbatim to the ``model_override``
            kwarg on every ``provider.chat(...)`` call in this function.
        timeout_s: per-call timeout. Default 10.0s per D-10. Worst-case
            grade latency is bounded at ``2 * timeout_s = 20s`` by D-06.
        system_prompt: configurable judge system prompt. It must keep the
            locked JSON shape and rubric prefix if you want the parser/scorer
            to work.

    Raises:
        ProviderUnavailable, ProviderTimeout, ProviderHTTPError: transport
            errors propagate unchanged (D-08). Orchestrator (Phase 4)
            handles policy per error class.

    Does NOT raise for parse failures — returns ``_sentinel_verdict`` on
    retry-also-fails. Does NOT catch ``asyncio.CancelledError``.
    """
    # D-02: judge context = last user message + local answer.
    messages = _build_judge_messages(
        request,
        local_response,
        system_prompt=system_prompt,
    )

    # D-03 + JDG-02 + Pitfall #3: JudgeVerdict.model_json_schema() with NO
    # args — default by_alias=True emits "pass" (not "pass_"), matches
    # what the LLM's JSON output produces. JDG-02: temperature=0.0 for
    # deterministic verdicts; top_p=1.0 is the redundant-but-explicit
    # canonical decoding parameter pair.
    options: ChatOptions = {
        "format": JudgeVerdict.model_json_schema(),
        "temperature": 0.0,
        "top_p": 1.0,
    }

    # D-11 first call. Request.model echoes the judge model so D-14
    # response echo in OllamaProvider produces resp.model=judge_model
    # (consistent with the orchestrator-level trace, which uses
    # judge_model for the judge_latency attribution). Transport errors
    # (ProviderUnavailable, ProviderTimeout, ProviderHTTPError) are NOT
    # caught here per D-08 — they propagate unchanged out of grade().
    # The ABSENCE of any `except Provider*` clause below is the
    # structural enforcement of that invariant (see grep gates in
    # acceptance criteria).
    judge_req = ChatCompletionRequest(model=judge_model, messages=messages)
    raw = await provider.chat(
        judge_req,
        options=options,
        model_override=judge_model,
        timeout_s=timeout_s,
    )

    try:
        return _derive_score(_parse_verdict(_response_content_as_str(raw)))
    except (json.JSONDecodeError, pydantic.ValidationError, ValueError) as first_err:
        # D-06 retry path begins. Log the failure with FIELD PATHS ONLY
        # (never the raw content — JudgeVerdict.hide_input_in_errors=True
        # means exc.errors()[i]["input"] is scrubbed; we also avoid
        # logging exc args directly per Pitfall #12).
        field_paths = _field_paths_or_empty(first_err)
        _LOG.warning(
            "judge.parse_failed",
            attempt=1,
            err_class=first_err.__class__.__name__,
            field_paths=field_paths,
        )

    # D-06 retry: ORIGINAL [system, user] messages + one appended system
    # message with the correction instruction. Same options, same
    # timeout_s, same model_override.
    retry_messages = [
        *messages,
        ChatMessage(role="system", content=_RETRY_CORRECTION),
    ]
    retry_req = ChatCompletionRequest(model=judge_model, messages=retry_messages)
    raw2 = await provider.chat(
        retry_req,
        options=options,
        model_override=judge_model,
        timeout_s=timeout_s,
    )

    try:
        return _derive_score(_parse_verdict(_response_content_as_str(raw2)))
    except (json.JSONDecodeError, pydantic.ValidationError, ValueError) as second_err:
        # D-07 sentinel: NEVER raise. Return a verdict that will naturally
        # pivot via the orchestrator's threshold check (0.0 < 0.7).
        field_paths = _field_paths_or_empty(second_err)
        _LOG.warning(
            "judge.parse_failed",
            attempt=2,
            err_class=second_err.__class__.__name__,
            field_paths=field_paths,
        )
        return _sentinel_verdict(second_err.__class__.__name__)


def _field_paths_or_empty(exc: Exception) -> str:
    """Extract field-path fragments from a Pydantic ``ValidationError``
    for logging. For non-ValidationError exceptions returns empty string
    — we never log ``str(exc)`` per Pitfall #12.

    ``JudgeVerdict.hide_input_in_errors=True`` guarantees
    ``exc.errors()[i]["input"]`` is scrubbed, but this helper does NOT
    log the ``input`` subfield anyway — only ``loc`` (structural field
    paths), which are safe per the Phase 2 Pitfall-E precision pattern
    (loc values are field names, not user content).
    """
    if isinstance(exc, pydantic.ValidationError):
        return ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
    return ""
