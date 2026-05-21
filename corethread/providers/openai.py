"""OpenAI frontier `/v1/chat/completions` adapter — the pivot-to-frontier
transformer. Implements the `Provider` ABC from `providers/base.py`.
Phase 5 / Plan 05-03 per 05-CONTEXT.md.

This module is the THIRD concrete `Provider` implementation — the CoreThread
pivot layer. v1 has exactly one call site: the orchestrator's pivot branch
passes the ORIGINAL request through, and this adapter is solely responsible
for transforming it into a constrained frontier call.

Locked user decisions this file implements (per 05-CONTEXT.md):
- **D-01** transport is `openai.AsyncOpenAI` (the official SDK), NOT raw
  httpx. Deliberate divergence from Phase 2/4's raw-httpx pattern — the SDK
  provides typed exception hierarchy, built-in exponential-backoff retry on
  408/429/5xx, and tracks OpenAI wire-format changes automatically. The
  SDK's typed `openai.types.ChatCompletion` serves the schema-drift-detector
  role that Phase 2/4 adapters delegate to an adapter-local Pydantic model
  — so NO adapter-local intermediate Pydantic model exists here (D-15).
- **D-02** shared `httpx.AsyncClient` injected via ctor + passed to
  `openai.AsyncOpenAI(http_client=...)` — preserves the single-AsyncClient
  invariant across the process (Pitfall #11). `max_retries=2` SDK default
  (tests pass `max_retries=0` to disable retries during error-mapping
  assertions). NO tenacity layer on top — layering tenacity over SDK
  retries would multiply attempts (2x2 = 4 real calls worst case) and is
  explicitly rejected. `api_key` read ONCE via `.get_secret_value()` in
  ctor; plaintext lives only inside the SDK client.
- **D-03** SDK exception → CoreThread typed error mapping with
  `raise ... from None` on every row (Pitfall #12 `__cause__`-leak defense).
  Divergent row: `AuthenticationError` → `ConfigError` (fail-fast
  misconfiguration, NOT `ProviderHTTPError`).
- **D-04** transform-in-provider architecture — the orchestrator passes
  the ORIGINAL request unmodified; THIS module owns the constraint prepend
  + max_tokens clamp. Deliberate reinterpretation of Phase 4 D-27. The
  orchestrator stays `await self._frontier.chat(request)` with zero lines
  changed (D-16).
- **D-05** constraint prompt is PREPENDED as a new system message at
  index 0 of `request.messages` — do NOT merge with existing system
  messages; existing messages shift right by one position. Implemented
  via `request.model_copy(update={...})` which preserves `extra="allow"`
  fields for forward-compat per API-01.
- **D-06** `max_tokens` clamp: ALWAYS clamp, unconditionally. cfg_max
  is a HARD CEILING — user can go below it, user cannot go above. Insurance
  policy against a config accident compounding with a user max_tokens to
  burn tokens on one call.
- **D-07** `OpenAIProvider.__init__` is keyword-only past `cfg`:
  `constraint_prompt`, `client`, `max_retries` are all KEYWORD_ONLY.
  `cfg` is TYPED against `FrontierResolved` (config.py:86-106) — NOT
  `FrontierConfig` — because by the time this ctor runs, `load_config`
  has resolved `api_key_env` → `api_key: SecretStr`. Typing against
  `FrontierResolved` keeps mypy clean AND documents the contract that
  api_key resolution has already happened upstream.
- **D-14** `warmup()` is a guaranteed no-op returning None; `health()`
  always returns `state='ready'` (no liveness-probe of api.openai.com —
  would be slow AND billable). `_warmup_state` retained for Phase 4 D-26
  aggregate-logic compat but never transitions away from `"ready"` in v1.
- **D-15** response construction terminates through the
  `to_openai_chat_completion` helper only — Phase 2 D-16 / Phase 4 D-09
  grep-gate invariant extended to the frontier adapter. The SDK's typed
  `ChatCompletion` is 1:1 OpenAI-shape but we still route through the
  helper so we mint our OWN `id = chatcmpl-{uuid4().hex}` (making
  CoreThread responses indistinguishable from OpenAI at the client
  boundary AND letting observability correlate id ↔ request_id) and
  synthesize `created = int(time.time())` at proxy-return time.

DIVERGENCES from Phase 2/4 adapters (surfaced here for code review):
- SDK transport (openai.AsyncOpenAI), NOT raw httpx.
- NO adapter-local intermediate Pydantic model (SDK's typed
  `openai.types.ChatCompletion` is the schema contract).
- `warmup()` is a guaranteed no-op — frontier APIs have no cold-start;
  a `models.retrieve(...)` probe would be both slow and billable, and
  may fail for keys without list-models permission.
- `health()` always returns `state='ready'` — no liveness probe.

`asyncio.CancelledError` is NEVER caught anywhere in this module — the
cancellation exception propagates so FastAPI can cancel the task cleanly
when a client disconnects (Provider ABC contract).

This module may import from `models.py`, `errors.py`, `config.py` (for
`FrontierResolved`), `providers/base.py`, and stdlib/httpx/openai/structlog.
It MUST NOT import from `main`, `orchestrator`, or `judge`. The shared
`httpx.AsyncClient` arrives via `__init__` dependency injection — this
file NEVER constructs its own httpx client (Pitfall #11).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

import httpx
import openai
import structlog

from corethread.config import FrontierResolved
from corethread.errors import (
    ConfigError,
    ProviderHTTPError,
    ProviderTimeout,
    ProviderUnavailable,
)
from corethread.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    to_openai_chat_completion,
)
from corethread.providers.base import ChatOptions, Provider, ProviderHealth

__all__ = ["OpenAIProvider"]

# Module-level logger — logger name becomes "providers.openai" so operators
# can filter the entire provider layer with `providers.*`. Inherits the root
# RedactingFilter + structlog processor chain installed by `logging_config.py`.
_LOG = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure helper — unit-testable without SDK / HTTP mocking.
# ---------------------------------------------------------------------------


def _include_usage_requested(request: ChatCompletionRequest) -> bool:
    """Return true when the OpenAI stream_options.include_usage flag is set."""
    stream_options = getattr(request, "stream_options", None)
    if stream_options is None and request.model_extra:
        stream_options = request.model_extra.get("stream_options")
    return isinstance(stream_options, dict) and stream_options.get("include_usage") is True


def _build_openai_transform_payload(
    request: ChatCompletionRequest,
    *,
    constraint_prompt: str,
    capped_tokens: int,
) -> ChatCompletionRequest:
    """D-04/D-05: prepend constraint as a new system message at index 0;
    D-06: set max_tokens to the pre-computed capped value.

    Pure helper — unit-testable without the SDK client (no HTTP, no IO).
    Returns a NEW `ChatCompletionRequest` via `model_copy(update=...)`,
    which preserves `extra="allow"` fields (tools, tool_choice,
    response_format, seed, logprobs, etc.) for API-01 forward-compat per
    05-CONTEXT.md D-05. The caller's original request is UNMODIFIED —
    concurrent orchestrator pivots sharing the same provider instance
    cannot tamper with each other's message lists (T-01-12 analog).
    """
    constraint_msg = ChatMessage(role="system", content=constraint_prompt)
    new_messages = [constraint_msg, *request.messages]
    return request.model_copy(update={"messages": new_messages, "max_tokens": capped_tokens})


# ---------------------------------------------------------------------------
# OpenAIProvider — the concrete adapter.
# ---------------------------------------------------------------------------


class OpenAIProvider(Provider):
    """OpenAI frontier `/v1/chat/completions` adapter — the pivot-to-frontier
    transformer. Implements the Provider ABC.

    Injected shared `httpx.AsyncClient` (NEVER constructs its own — Pitfall
    #11 defense) is passed to `openai.AsyncOpenAI(http_client=...)` so the
    single-AsyncClient-per-process invariant holds across Ollama/LM Studio/
    OpenAI providers.

    The name reads "generic OpenAI client" but behaves as the CoreThread
    pivot layer per 05-CONTEXT.md D-04: v1 has exactly one call site (the
    orchestrator's pivot branch). The orchestrator passes the ORIGINAL
    request; THIS class owns (a) prepending the constraint prompt as a new
    system message at index 0, and (b) flooring max_tokens to cfg.max_tokens
    unconditionally. If v2 adds a direct-frontier route that needs a raw
    OpenAI client, split into `OpenAIProvider` + `OpenAIFrontierPivotProvider`
    at that time.

    `warmup()` is a guaranteed no-op (D-14) — frontier APIs have no
    cold-start. `health()` always returns `state="ready"` — we don't
    liveness-probe api.openai.com at /health time (would be slow AND
    billable). `_warmup_state` field retained for Phase 4 D-26 aggregate-
    logic compat but never transitions away from `"ready"` in v1.

    Task cancellation is NEVER caught (`asyncio.CancelledError` propagates
    so FastAPI can cancel cleanly when a client disconnects — ABC contract).
    """

    name: str = "openai"  # class-level per ABC contract

    def __init__(
        self,
        cfg: FrontierResolved,
        *,
        constraint_prompt: str | None,
        client: httpx.AsyncClient,
        max_retries: int = 2,
        provider_name: str = "openai",
    ) -> None:
        """Store config + construct the openai SDK client with the shared
        httpx client injected.

        `cfg` is typed against `FrontierResolved` (NOT `FrontierConfig`)
        because `api_key: SecretStr` MUST be present by the time we reach
        this ctor — `load_config` has resolved it from
        `os.environ[api_key_env]` upstream (config.py:86-106).

        `max_retries` defaults to 2 per D-02 (the openai SDK's built-in
        exponential-backoff retry count on 408/429/5xx). Tests pass
        `max_retries=0` to disable retries during error-mapping assertions.

        `api_key` is read once via `.get_secret_value()` here — the
        `SecretStr` wrapper stays in `cfg` and the resolved plaintext lives
        only inside the SDK client. OBS-02 redaction filter guards against
        accidental leakage through log frames.
        """
        self.name = provider_name
        self._cfg = cfg
        self._constraint_prompt = constraint_prompt
        # D-02: construct the openai.AsyncOpenAI SDK client with:
        #   - api_key read via SecretStr.get_secret_value() (ONCE, here)
        #   - base_url from cfg (api.openai.com/v1 or a compat endpoint)
        #   - locked 4-axis timeouts matching Phase 2 D-06
        #     (connect=5, read=120, write=10, pool=5)
        #   - max_retries=2 (SDK default exponential backoff; no tenacity atop)
        #   - http_client=<shared> (Pitfall #11: one AsyncClient per process)
        self._client: openai.AsyncOpenAI = openai.AsyncOpenAI(
            api_key=cfg.api_key.get_secret_value(),
            base_url=cfg.base_url,
            timeout=httpx.Timeout(
                connect=5.0,
                read=120.0,
                write=10.0,
                pool=5.0,
            ),
            max_retries=max_retries,
            http_client=client,
        )
        # D-14: frontier has no cold-start. _warmup_state retained for
        # Phase 4 D-26 aggregate-logic compat; starts (and stays) "ready".
        self._warmup_state: Literal["ready", "warming", "unhealthy"] = "ready"
        self._last_warmup_error: str | None = None

        _LOG.info(
            "provider.instantiated",
            kind=self.name,
            model=cfg.model,
            max_retries=max_retries,
        )

    async def chat(
        self,
        request: ChatCompletionRequest,
        *,
        options: ChatOptions | None = None,
        model_override: str | None = None,
        timeout_s: float = 120.0,
    ) -> ChatCompletionResponse:
        """Pivot chat handler — applies constraint_prompt + max_tokens clamp,
        dispatches through the openai SDK, returns an OpenAI-shape response
        synthesized through the canonical response-envelope helper.

        D-04: transform-in-provider architecture. The caller (orchestrator)
        passes the ORIGINAL request; this method builds the pivoted request
        INSIDE before dispatch. The original request is NOT mutated.

        D-05: constraint_prompt is prepended as a NEW system message at
        index 0. Existing system messages shift right by one (NOT merged).

        D-06: max_tokens is ALWAYS clamped to cfg.max_tokens as a hard
        ceiling. User-set values below the cap pass through; values above
        the cap (or unset) are floored to cfg.

        D-15: response terminates through the response-envelope helper —
        we mint our own `id = chatcmpl-{uuid4().hex}` and synthesize
        `created = int(time.time())` at proxy-return time so responses are
        indistinguishable from OpenAI at the client boundary.

        Edge case: `model_dump(exclude_none=True)` strips fields explicitly
        set to None. Notably `temperature=0` is sent (0 is not None) but a
        user who EXPLICITLY sets any field to None will see it dropped —
        OpenAI's default will apply. This is acceptable given OpenAI's
        defaults are well-documented.

        `asyncio.CancelledError` is NEVER caught — Provider ABC contract.

        Raises (all via `from None` per Pitfall #12):
        - `ProviderUnavailable` — `openai.APIConnectionError` (frontier
          unreachable). Orchestrator does NOT further pivot in v1.
        - `ProviderTimeout` — `openai.APITimeoutError` (mid-generation
          timeout after SDK retries). FastAPI returns 504.
        - `ConfigError` — `openai.AuthenticationError` (401). Runtime 401
          means the configured key is invalid — fail loud like CFG-03;
          NOT caught by lifespan's degraded-start wrapper.
        - `ProviderHTTPError` — every other SDK exception
          (PermissionDeniedError / RateLimitError / BadRequestError /
          InternalServerError / catch-all APIError).
        """
        # D-06: max_tokens clamp — unconditional, HARD CEILING at cfg.
        # If user set max_tokens, min(user, cfg); if user unset, cfg.
        capped_tokens = (
            min(request.max_tokens, self._cfg.max_tokens)
            if request.max_tokens is not None
            else self._cfg.max_tokens
        )

        if self._constraint_prompt:
            # D-04/D-05: transform happens INSIDE the provider (not the
            # orchestrator). Uses model_copy so extra="allow" fields are
            # preserved for API-01 forward-compat; original request untouched.
            outbound = _build_openai_transform_payload(
                request,
                constraint_prompt=self._constraint_prompt,
                capped_tokens=capped_tokens,
            )
        else:
            outbound = request.model_copy(update={"max_tokens": capped_tokens})

        # Build SDK kwargs. `model_override` is accepted per ABC (Phase 3
        # D-09 / JDG-06) but the orchestrator never passes it on pivot; use
        # outbound.model as source unless explicitly overridden.
        sdk_model = model_override if model_override is not None else outbound.model
        sdk_kwargs: dict[str, Any] = outbound.model_dump(
            exclude_none=True,
            exclude={"stream"},  # D-11: stream rejected upstream; never forward
        )
        sdk_kwargs["model"] = sdk_model
        if options:
            for key in ("temperature", "top_p", "seed", "stop"):
                if key in options and key not in sdk_kwargs:
                    sdk_kwargs[key] = options[key]  # type: ignore[literal-required]
            if "num_predict" in options and "max_tokens" not in sdk_kwargs:
                sdk_kwargs["max_tokens"] = options["num_predict"]
            if "format" in options and "response_format" not in sdk_kwargs:
                fmt = options["format"]
                if isinstance(fmt, dict):
                    sdk_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "corethread_response",
                            "schema": fmt,
                            "strict": True,
                        },
                    }
                elif fmt == "json":
                    sdk_kwargs["response_format"] = {"type": "json_object"}

        _LOG.info(
            "frontier.transform.applied",
            kind=self.name,
            capped_tokens=capped_tokens,
            injected_constraint=bool(self._constraint_prompt),
            # Do NOT log the constraint_prompt text itself — it may contain
            # product IP (Claude's Discretion privacy note in 05-CONTEXT.md).
        )

        # D-03: SDK exception → CoreThread typed error mapping. Order:
        # timeout → connect → auth (ConfigError, NOT ProviderHTTPError)
        # → permission → rate → bad-request → 5xx → catch-all APIError.
        #
        # CRITICAL MRO DISCIPLINE (Phase 2 Research-Delta-3 analog for the
        # openai SDK): in openai 2.x, `APITimeoutError` IS-A `APIConnectionError`
        # (see MRO chain: APITimeoutError → APIConnectionError → APIError →
        # OpenAIError → Exception). Therefore `except APITimeoutError` MUST
        # come BEFORE `except APIConnectionError` or first-matching-wins
        # would misroute every timeout to ProviderUnavailable. This is the
        # same "specific-before-general" ordering discipline Phase 2/4
        # adopted for httpx.ConnectTimeout vs httpx.TimeoutException.
        #
        # Within the APIStatusError family (Permission/RateLimit/BadRequest/
        # InternalServer), ordering is less load-bearing because they are
        # siblings in the MRO, but codify the order for intent.
        try:
            sdk_response = await self._client.chat.completions.create(**sdk_kwargs)
        except openai.APITimeoutError:
            # MUST precede APIConnectionError (APITimeoutError IS-A APIConnectionError)
            raise ProviderTimeout("frontier", timeout_s) from None
        except openai.APIConnectionError:
            raise ProviderUnavailable(
                "frontier",
                "openai connection error",
            ) from None
        except openai.AuthenticationError:
            # D-03 divergent row: runtime 401 → ConfigError (fail-fast,
            # NOT caught by lifespan's degraded-start wrapper).
            raise ConfigError(
                f"{self.name} API authentication failed: check the env var "
                f"{self._cfg.api_key_env}. Make sure it contains a valid key "
                "for that provider."
            ) from None
        except openai.PermissionDeniedError:
            raise ProviderHTTPError(
                "frontier",
                403,
                "permission denied",
            ) from None
        except openai.RateLimitError:
            # SDK already retried max_retries=2 with backoff before raising;
            # further retries would worsen the rate limit.
            raise ProviderHTTPError("frontier", 429, "rate limited") from None
        except openai.BadRequestError:
            raise ProviderHTTPError("frontier", 400, "bad request") from None
        except openai.InternalServerError as exc:
            status = getattr(exc, "status_code", 500) or 500
            raise ProviderHTTPError(
                "frontier",
                status,
                "upstream server error",
            ) from None
        except openai.APIError as exc:
            # Catch-all: forward-compat for new SDK exception classes.
            status = getattr(exc, "status_code", 500) or 500
            raise ProviderHTTPError("frontier", status, "api error") from None
        # asyncio.CancelledError intentionally NOT caught — ABC contract.

        # D-15: response construction through to_openai_chat_completion ONLY.
        # We mint our own id (chatcmpl-{uuid4().hex}) so CoreThread responses
        # are indistinguishable from OpenAI at the client boundary; `created`
        # is synthesized at proxy-return time. `system_fingerprint` is
        # synthesized (NOT passed through) per the D-09 identity-masking
        # spirit — clients see a CoreThread-branded opaque string rather
        # than the upstream frontier fingerprint.
        choice = sdk_response.choices[0]
        message_content = choice.message.content or ""
        finish_reason = choice.finish_reason

        _LOG.info(
            "frontier.response.received",
            kind=self.name,
            tokens_in=sdk_response.usage.prompt_tokens if sdk_response.usage else 0,
            tokens_out=sdk_response.usage.completion_tokens if sdk_response.usage else 0,
        )

        prompt_tokens = sdk_response.usage.prompt_tokens if sdk_response.usage else 0
        completion_tokens = sdk_response.usage.completion_tokens if sdk_response.usage else 0

        return to_openai_chat_completion(
            model=sdk_response.model,
            message_content=message_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            system_fingerprint=f"fp_corethread_{sdk_response.model[:10]}",
        )

    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        *,
        options: ChatOptions | None = None,
        model_override: str | None = None,
        timeout_s: float = 120.0,
        on_final: Callable[[ChatCompletionResponse], None] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming frontier chat handler.

        Uses the same transform policy as ``chat()`` but calls the OpenAI SDK
        with ``stream=True``. Chunks are yielded in OpenAI's native
        ``chat.completion.chunk`` shape; the accumulated final response is
        passed to ``on_final`` for trace/token accounting.
        """
        capped_tokens = (
            min(request.max_tokens, self._cfg.max_tokens)
            if request.max_tokens is not None
            else self._cfg.max_tokens
        )

        if self._constraint_prompt:
            outbound = _build_openai_transform_payload(
                request,
                constraint_prompt=self._constraint_prompt,
                capped_tokens=capped_tokens,
            )
        else:
            outbound = request.model_copy(update={"max_tokens": capped_tokens})

        sdk_model = model_override if model_override is not None else outbound.model
        sdk_kwargs: dict[str, Any] = outbound.model_dump(
            exclude_none=True,
            exclude={"stream"},
        )
        sdk_kwargs["model"] = sdk_model
        sdk_kwargs["stream"] = True
        if options:
            for key in ("temperature", "top_p", "seed", "stop"):
                if key in options and key not in sdk_kwargs:
                    sdk_kwargs[key] = options[key]  # type: ignore[literal-required]
            if "num_predict" in options and "max_tokens" not in sdk_kwargs:
                sdk_kwargs["max_tokens"] = options["num_predict"]
            if "format" in options and "response_format" not in sdk_kwargs:
                fmt = options["format"]
                if isinstance(fmt, dict):
                    sdk_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "corethread_response",
                            "schema": fmt,
                            "strict": True,
                        },
                    }
                elif fmt == "json":
                    sdk_kwargs["response_format"] = {"type": "json_object"}

        _LOG.info(
            "frontier.transform.applied",
            kind=self.name,
            capped_tokens=capped_tokens,
            injected_constraint=bool(self._constraint_prompt),
            stream=True,
        )

        content_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason: str | None = "stop"
        response_model = sdk_model
        include_usage = _include_usage_requested(request)
        usage_chunk_seen = False
        stream_id: str | None = None
        stream_created: int | None = None
        stream_system_fingerprint: str | None = None

        try:
            sdk_stream = await self._client.chat.completions.create(**sdk_kwargs)
            async for chunk in sdk_stream:
                response_model = getattr(chunk, "model", response_model) or response_model
                payload = chunk.model_dump(mode="json", exclude_none=True)
                if stream_id is None and isinstance(payload.get("id"), str):
                    stream_id = payload["id"]
                if stream_created is None and isinstance(payload.get("created"), int):
                    stream_created = payload["created"]
                if stream_system_fingerprint is None and isinstance(
                    payload.get("system_fingerprint"), str
                ):
                    stream_system_fingerprint = payload["system_fingerprint"]
                if payload.get("usage") is not None:
                    usage_chunk_seen = True

                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    prompt_tokens = getattr(chunk_usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(chunk_usage, "completion_tokens", 0) or 0

                for choice in getattr(chunk, "choices", []) or []:
                    if getattr(choice, "index", 0) != 0:
                        continue
                    delta = getattr(choice, "delta", None)
                    delta_content = getattr(delta, "content", None)
                    if isinstance(delta_content, str):
                        content_parts.append(delta_content)
                    choice_finish = getattr(choice, "finish_reason", None)
                    if choice_finish is not None:
                        finish_reason = str(choice_finish)

                yield payload
        except openai.APITimeoutError:
            raise ProviderTimeout("frontier", timeout_s) from None
        except openai.APIConnectionError:
            raise ProviderUnavailable(
                "frontier",
                "openai connection error",
            ) from None
        except openai.AuthenticationError:
            raise ConfigError(
                f"{self.name} API authentication failed: check the env var "
                f"{self._cfg.api_key_env}. Make sure it contains a valid key "
                "for that provider."
            ) from None
        except openai.PermissionDeniedError:
            raise ProviderHTTPError(
                "frontier",
                403,
                "permission denied",
            ) from None
        except openai.RateLimitError:
            raise ProviderHTTPError("frontier", 429, "rate limited") from None
        except openai.BadRequestError:
            raise ProviderHTTPError("frontier", 400, "bad request") from None
        except openai.InternalServerError as exc:
            status = getattr(exc, "status_code", 500) or 500
            raise ProviderHTTPError(
                "frontier",
                status,
                "upstream server error",
            ) from None
        except openai.APIError as exc:
            status = getattr(exc, "status_code", 500) or 500
            raise ProviderHTTPError("frontier", status, "api error") from None

        response = to_openai_chat_completion(
            model=response_model,
            message_content="".join(content_parts),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            system_fingerprint=f"fp_corethread_{response_model[:10]}",
        )
        if on_final is not None:
            on_final(response)
        if include_usage and not usage_chunk_seen:
            usage_payload: dict[str, Any] = {
                "id": stream_id or response.id,
                "object": "chat.completion.chunk",
                "created": stream_created or response.created,
                "model": response_model,
                "choices": [],
                "usage": response.usage.model_dump(mode="json"),
            }
            if stream_system_fingerprint is not None:
                usage_payload["system_fingerprint"] = stream_system_fingerprint
            yield usage_payload

    async def health(self, *, timeout_s: float = 2.0) -> ProviderHealth:
        """D-14: always returns ready. Never liveness-probes api.openai.com
        (would be slow AND billable). `_warmup_state` retained for Phase 4
        D-26 aggregate-logic compat.

        `timeout_s` accepted for ABC compatibility; a pure dict-build has
        no wall-clock to bound.
        """
        return ProviderHealth(
            kind=self.name,
            state=self._warmup_state,  # always "ready" in v1
            last_error=self._last_warmup_error,
        )

    async def warmup(self, *, timeout_s: float = 300.0) -> None:
        """D-14: guaranteed no-op. No `openai.models.retrieve(...)` probe
        (would be slow AND billable; may fail for keys without list-models
        permission). Frontier is considered ready whenever the SDK client
        constructs successfully.

        `timeout_s` accepted for ABC compatibility; the no-op path has no
        wall-clock to bound.
        """
        return None
