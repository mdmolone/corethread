"""Tests for models.py — covers Phase 1 Success Criterion #1 (response round-trip).

Locks:
- ChatCompletionResponse round-trip preserves every required OpenAI field (id,
  created [int], object, choices[].finish_reason, usage [all three counts]).
- to_openai_chat_completion synthesizes id / created / object / finish_reason /
  usage exactly per CONTEXT.md "Response-envelope adapter is bulletproof".
- JudgeVerdict is strict: extra="forbid", confidence_score in [0, 1], non-empty
  reasoning, JSON-side `pass` aliases to Python attribute `pass_`.
- ChatCompletionRequest is permissive (extra="allow") for forward-compat.
- with_system_prefix returns a NEW request — original is untouched (T-01-12).
"""

from __future__ import annotations

import re
import time

import pytest
from pydantic import ValidationError

from corethread.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    JudgeVerdict,
    to_openai_chat_completion,
)

# ---------------------------------------------------------------------------
# Success Criterion #1 — round-trip
# ---------------------------------------------------------------------------


def test_response_round_trip() -> None:
    """SC#1: hand-crafted OpenAI body round-trips with every required field populated."""
    hand_crafted = {
        "id": "chatcmpl-abcdef1234567890abcdef1234567890",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    parsed = ChatCompletionResponse.model_validate(hand_crafted)
    dumped = parsed.model_dump(mode="json")

    assert dumped["id"] == hand_crafted["id"]
    assert isinstance(dumped["created"], int)
    assert dumped["object"] == "chat.completion"
    assert dumped["model"] == "gpt-4o"
    assert dumped["choices"][0]["finish_reason"] == "stop"  # never null
    assert dumped["choices"][0]["message"]["role"] == "assistant"
    assert dumped["choices"][0]["message"]["content"] == "Hi there."
    assert dumped["usage"]["prompt_tokens"] == 10
    assert dumped["usage"]["completion_tokens"] == 5
    assert dumped["usage"]["total_tokens"] == 15


def test_response_extra_allow_keeps_provider_specific_fields() -> None:
    """ConfigDict(extra='allow') passes through provider-specific extras like
    Ollama's `system_fingerprint` so adapters that preserve them survive a
    round-trip."""
    body = {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "system_fingerprint": "fp_ollama",
    }
    r = ChatCompletionResponse.model_validate(body)
    assert r.system_fingerprint == "fp_ollama"


# ---------------------------------------------------------------------------
# to_openai_chat_completion adapter — Pitfall #1 / #2
# ---------------------------------------------------------------------------


def test_to_openai_chat_completion_synthesizes_all_fields() -> None:
    """to_openai_chat_completion() populates every Pitfall #1 required field."""
    r = to_openai_chat_completion(
        model="gpt-4o",
        message_content="hi",
        prompt_tokens=10,
        completion_tokens=20,
        finish_reason=None,  # adapter must default to "stop", never leave null
    )
    # id has the synthesized chatcmpl-<32-hex> shape
    assert re.match(r"^chatcmpl-[0-9a-f]{32}$", r.id), r.id
    # created is a recent unix timestamp
    assert isinstance(r.created, int)
    assert abs(r.created - int(time.time())) < 60
    # object is the OpenAI literal
    assert r.object == "chat.completion"
    # finish_reason for None gets mapped via OLLAMA_FINISH_MAP -> "stop"
    assert r.choices[0].finish_reason == "stop"
    # message content survives
    assert r.choices[0].message.role == "assistant"
    assert r.choices[0].message.content == "hi"
    # usage is fully populated and totals correctly
    assert r.usage.prompt_tokens == 10
    assert r.usage.completion_tokens == 20
    assert r.usage.total_tokens == 30


def test_to_openai_chat_completion_maps_unknown_finish_reason_to_stop() -> None:
    """An unknown upstream finish_reason value is mapped to 'stop' (Pitfall #2)."""
    r = to_openai_chat_completion(
        model="m",
        message_content="x",
        prompt_tokens=0,
        completion_tokens=0,
        finish_reason="some-future-reason-not-in-the-enum",
    )
    assert r.choices[0].finish_reason == "stop"


# ---------------------------------------------------------------------------
# JudgeVerdict — strict-JSON contract
# ---------------------------------------------------------------------------


def test_judge_verdict_extra_forbid_rejects_unknown_keys() -> None:
    """extra='forbid' on JudgeVerdict — drift in judge output must fail loudly."""
    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate(
            {
                "confidence_score": 0.5,
                "reasoning": "ok",
                "pass": True,
                "extra_unexpected_field": "no",
            }
        )


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_judge_verdict_score_bounds(score: float) -> None:
    """confidence_score must be in [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate({"confidence_score": score, "reasoning": "x", "pass": True})


def test_judge_verdict_empty_reasoning_rejected() -> None:
    """reasoning has min_length=1; empty string rejected."""
    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate({"confidence_score": 0.5, "reasoning": "", "pass": True})


def test_judge_verdict_pass_alias() -> None:
    """JSON key `pass` aliases to Python attribute `pass_`."""
    v = JudgeVerdict.model_validate({"confidence_score": 0.8, "reasoning": "ok", "pass": True})
    assert v.pass_ is True
    # by_alias dump round-trips with the JSON key, not the Python keyword-safe one.
    dumped = v.model_dump(by_alias=True)
    assert "pass" in dumped
    assert "pass_" not in dumped
    assert dumped["pass"] is True


def test_judge_verdict_score_string_coercion() -> None:
    """Judge LLMs sometimes return stringified scores like '0.82' — accepted."""
    v = JudgeVerdict.model_validate({"confidence_score": "0.82", "reasoning": "ok", "pass": True})
    assert v.confidence_score == pytest.approx(0.82)


# ---------------------------------------------------------------------------
# ChatCompletionRequest — extra='allow' + with_system_prefix
# ---------------------------------------------------------------------------


def test_request_extra_allow_forwards_unknown_fields() -> None:
    """Forward-compat: a future OpenAI field like reasoning_effort passes through."""
    req = ChatCompletionRequest.model_validate(
        {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
        }
    )
    assert req.model == "gpt-5"
    assert req.messages[0].role == "user"
    # extra='allow' surfaces unknown fields on the model_dump
    assert req.model_dump().get("reasoning_effort") == "high"


def test_with_system_prefix_returns_new_instance_unchanged_original() -> None:
    """with_system_prefix returns NEW request; original unchanged (T-01-12)."""
    req = ChatCompletionRequest(model="x", messages=[ChatMessage(role="user", content="hi")])
    pivoted = req.with_system_prefix("Be concise.")
    # New instance has system message at index 0, original user at index 1
    assert pivoted is not req
    assert len(pivoted.messages) == 2
    assert pivoted.messages[0].role == "system"
    assert pivoted.messages[0].content == "Be concise."
    assert pivoted.messages[1].role == "user"
    assert pivoted.messages[1].content == "hi"
    # Original is untouched — T-01-12
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
