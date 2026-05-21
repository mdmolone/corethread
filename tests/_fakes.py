"""Shared test scaffolding for orchestrator + future Phase 5 integration tests.

Phase 4 / Plan 02 per 04-CONTEXT.md D-30. The underscore-prefixed filename
signals "test-only, not test-collected" to pytest — verified by Plan 02
acceptance: `pytest --collect-only tests/_fakes.py` yields zero items.

Single-configurable-class pattern (D-30): one FakeProvider with behavior
kwargs (response / raise_on_chat) beats a family of subclasses
(HappyFakeProvider / UnavailableFakeProvider / TimeoutFakeProvider) — matches
Phase 2/3 respx idioms and avoids module-namespace pollution.

Imported by:
- tests/test_orchestrator.py (Plan 02 RED phase → Plan 03 GREEN)
- Phase 5 integration tests (future)

NOT imported by tests/test_obs_trace.py — Phase 3 Plan 05's fake-driver uses
OllamaProvider with respx directly, and that pattern is preserved verbatim.
"""

from __future__ import annotations

from typing import Any

from corethread.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    JudgeVerdict,
    to_openai_chat_completion,
)
from corethread.providers.base import ChatOptions, Provider, ProviderHealth

__all__ = [
    "FakeJudge",
    "FakeProvider",
    "make_happy_local_response",
]


class FakeProvider(Provider):
    """Configurable fake Provider for orchestrator unit tests.

    Direct Provider ABC subclass — NOT a subclass of any concrete adapter
    (OllamaProvider / LMStudioProvider). Tests assert on the ABC surface only.

    Behavior flags (init kwargs, all keyword-only):
        name: class-instance attribute set on `self.name` (default "fake")
        response: ChatCompletionResponse returned from chat() on success path
        raise_on_chat: exception instance to raise from chat() — if non-None
                       takes precedence over `response` (call_count still
                       increments BEFORE the raise, so counters reflect
                       attempted calls)

    Call tracking:
        call_count: int incremented on every chat() entry
        calls: list[tuple[req, options, model_override, timeout_s]] recorded
               per call (for structural invariants — e.g., D-27 asserts the
               frontier call passes the UNCHANGED request object).
    """

    def __init__(
        self,
        *,
        name: str = "fake",
        response: ChatCompletionResponse | None = None,
        raise_on_chat: Exception | None = None,
    ) -> None:
        self.name = name
        self._response = response
        self._raise_on_chat = raise_on_chat
        self.call_count: int = 0
        self.calls: list[tuple[Any, ...]] = []

    async def chat(
        self,
        request: ChatCompletionRequest,
        *,
        options: ChatOptions | None = None,
        model_override: str | None = None,
        timeout_s: float = 120.0,
    ) -> ChatCompletionResponse:
        self.call_count += 1
        self.calls.append((request, options, model_override, timeout_s))
        if self._raise_on_chat is not None:
            raise self._raise_on_chat
        if self._response is None:
            raise RuntimeError("FakeProvider: no response configured and no raise_on_chat set")
        return self._response

    async def health(self, *, timeout_s: float = 2.0) -> ProviderHealth:
        return ProviderHealth(kind=self.name, state="ready", last_error=None)

    async def warmup(self, *, timeout_s: float = 300.0) -> None:
        return None


class FakeJudge:
    """Callable stand-in for `judge.grade(...)`. Tracks call count + records
    args; pops one verdict from `verdicts` per call.

    Wiring in tests: monkeypatch `orchestrator.judge.grade` with a FakeJudge
    instance BEFORE constructing the Orchestrator — the instance is async-
    callable via `__call__`.

    The D-07 sentinel path is NOT internal to FakeJudge — tests configure it
    by passing a pre-built JudgeVerdict whose reasoning starts with
    `"judge parse failure: "` (which IS the contract per Phase 3 locked
    _sentinel_verdict f-string prefix).
    """

    def __init__(
        self,
        *,
        verdicts: list[JudgeVerdict] | None = None,
        raise_on_call: Exception | None = None,
    ) -> None:
        self._verdicts: list[JudgeVerdict] = list(verdicts) if verdicts else []
        self._raise_on_call = raise_on_call
        self.call_count: int = 0
        self.calls: list[tuple[Any, ...]] = []

    async def __call__(
        self,
        request: ChatCompletionRequest,
        local_response: ChatCompletionResponse,
        *,
        provider: Provider,
        judge_model: str,
        timeout_s: float = 10.0,
        system_prompt: str = "",
    ) -> JudgeVerdict:
        self.call_count += 1
        self.calls.append(
            (request, local_response, provider, judge_model, timeout_s, system_prompt)
        )
        if self._raise_on_call is not None:
            raise self._raise_on_call
        if not self._verdicts:
            raise RuntimeError("FakeJudge: no verdict configured")
        return self._verdicts.pop(0)


def make_happy_local_response(
    *,
    model: str = "llama3.1:8b",
    content: str = "42",
    prompt_tokens: int = 10,
    completion_tokens: int = 3,
    finish_reason: str | None = "stop",
    system_fingerprint: str = "fp_corethread_fake",
) -> ChatCompletionResponse:
    """Build a plausible ChatCompletionResponse via the Phase 2 canonical helper.

    Use as the local/frontier success response in orchestrator tests.
    """
    return to_openai_chat_completion(
        model=model,
        message_content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        system_fingerprint=system_fingerprint,
    )
