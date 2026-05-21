"""Abstract Provider interface — the contract every adapter in `providers/`
implements. Locked in Phase 2 / Plan 01.

This module establishes the single async surface that the orchestrator (Phase
4) dispatches against and that every concrete backend adapter populates: local
Ollama (Phase 2, PLAN-02), local LM Studio (Phase 4), and the OpenAI frontier
(Phase 5). Pinning the ABC one file + one review ahead of the first
implementation is what lets Phase 4's orchestrator dispatch on `cfg.local.kind`
with confidence that every concrete Provider honours the same `chat` /
`health` / `warmup` contract. Closes LOC-05.

Imported by:
- `providers/ollama.py` (Phase 2, PLAN-02) — first concrete implementation.
- `providers/lmstudio.py` (Phase 4) — same ABC, asymmetric `/v1/chat/completions` transport.
- `providers/openai.py` (Phase 5) — same ABC, no-op `warmup()`.
- `orchestrator.py` (Phase 4) — type-hints `Provider` on its constructor
  dependencies so Python's structural subtyping catches missing methods at
  author time.

This module sits near the bottom of the import graph. It MAY import from
`models.py` (for the OpenAI-shape request/response types) and stdlib. It MUST
NOT import from `errors`, `config`, `main`, `logging_config`, or any concrete
provider submodule (`ollama`, `lmstudio`, `openai`). That decoupling is what
makes the provider interface mockable in unit tests (`test_providers_base.py`)
without pulling in FastAPI or httpx, and it's what lets the concrete Ollama
adapter in PLAN-02 import `Provider` without accidentally taking a circular
dependency back on itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, TypedDict

from corethread.models import ChatCompletionRequest, ChatCompletionResponse

__all__ = [
    "ChatOptions",
    "Provider",
    "ProviderHealth",
]


# ---------------------------------------------------------------------------
# TypedDict: opaque kwargs passed through Provider.chat(..., options=...).
# ---------------------------------------------------------------------------


class ChatOptions(TypedDict, total=False):
    """Opaque keyword-argument dict passed to `Provider.chat(..., options=...)`.

    Per CONTEXT.md D-05: `TypedDict(total=False)` is the PEP-692-endorsed idiom
    for opaque kwarg shapes — every key is optional, misspelled keys are caught
    by mypy at author time without the runtime overhead of a Pydantic model.
    Each concrete Provider forwards only the keys it recognizes; unknown keys
    are silently ignored by the adapter (PEP-692 does not provide runtime
    rejection of extra keys, which is accepted per threat T-02-03).

    Keys (all optional):
    - `num_ctx`: explicit Ollama context window (Pitfall #4 defense — without
      this, Ollama silently truncates to 4096 tokens). Phase 2 `OllamaProvider`
      resolves the value via D-17 exact-match lookup before calling `chat`.
    - `keep_alive`: int seconds or Go-duration string (e.g. `"24h"`); `-1`
      means "load the model indefinitely until Ollama restart". Phase 2
      `OllamaProvider` IGNORES this key and always sends `-1` per D-13.
    - `format`: `"json"` or a JSON Schema dict; Phase 3's Judge sets this to
      `JudgeVerdict.model_json_schema()` to force strict-JSON decode.
    - `temperature`, `top_p`, `seed`, `num_predict`, `stop`: standard
      generation parameters; the concrete adapter maps them to its native wire
      shape (Ollama nests them under `options`; LM Studio / OpenAI pass them
      at the top of the body).
    """

    num_ctx: int
    keep_alive: int | str
    format: dict | str
    temperature: float
    top_p: float
    seed: int
    num_predict: int
    stop: list[str]


# ---------------------------------------------------------------------------
# TypedDict: return value of Provider.health().
# ---------------------------------------------------------------------------


class ProviderHealth(TypedDict):
    """Return value of `Provider.health()` — serializes straight into
    `/health` JSON via `dict(provider_health)`.

    Per CONTEXT.md D-11: a richer shape than `bool` so `/health` can distinguish
    `"warming"` (boot-time warmup in flight) from `"ready"` and `"unhealthy"`
    (warmup failed OR explicit probe failed). Total-True (the default) because
    every key must be populated on return — callers serialize directly without
    a `model_dump()` step.

    `last_error` is ALWAYS the exception class name
    (`exc.__class__.__name__`) — NEVER `str(exc)` or `repr(exc)` — per Pitfall
    #12 / T-02-01. The field value is surfaced verbatim in `/health` JSON; an
    implementation that stuffs `str(exc)` in there would leak upstream response
    bodies including `Authorization: Bearer sk-...` headers that
    `httpx.HTTPStatusError.__repr__` embeds. This mitigation is enforced at
    each concrete adapter's assignment site (PLAN-02 for Ollama); the ABC
    establishes the contract.

    Keys (all required):
    - `kind`: the concrete provider's `name` (e.g. `"ollama"`, `"lmstudio"`,
      `"openai"`).
    - `state`: `"ready"` (warmed and recent probe OK), `"warming"` (warmup
      in flight), or `"unhealthy"` (warmup failed or probe failed).
    - `last_error`: `None` when healthy; otherwise the exception class name
      of the most recent failure.
    """

    kind: str
    state: Literal["ready", "warming", "unhealthy"]
    last_error: str | None


# ---------------------------------------------------------------------------
# ABC: the Provider surface every concrete adapter implements.
# ---------------------------------------------------------------------------


class Provider(ABC):
    """Abstract adapter interface. Subclasses implement transport to a specific
    backend (Ollama, LM Studio, OpenAI frontier, ...) and return results shaped
    per the OpenAI `/v1/chat/completions` contract.

    Subclasses MUST:
    - Set `name: str` as a class attribute (e.g. `name = "ollama"`). The value
      is surfaced in `ProviderHealth.kind` and in structured log events.
    - Receive a shared `httpx.AsyncClient` via `__init__` (Pitfall #11
      defense: never construct a per-request client).
    - Construct their response by calling
      `models.to_openai_chat_completion(...)` (D-16); never construct
      `ChatCompletionResponse` by hand — the synthesizer is the single place
      that populates every required OpenAI field (Pitfall #1 / #2).
    - Wrap upstream exceptions in the typed `CoreThreadError` subclasses from
      `errors.py` using `from None` (Pitfall #12 defense against
      `__cause__`-chain secret leakage via `httpx.HTTPStatusError.__repr__`).

    `asyncio.CancelledError` MUST NOT be caught anywhere in the implementation
    — it must propagate so FastAPI can cancel the task cleanly when a client
    disconnects.

    The ABC itself declares no `__init__`; subclasses define their own
    constructor signature so the surface stays sharp and mockable.
    """

    name: str  # concrete provider sets e.g. `name = "ollama"`

    @abstractmethod
    async def chat(
        self,
        request: ChatCompletionRequest,
        *,
        options: ChatOptions | None = None,
        model_override: str | None = None,
        timeout_s: float = 120.0,
    ) -> ChatCompletionResponse:
        """Issue a chat completion against the backend and return an
        OpenAI-shaped response.

        `timeout_s` defaults to 120.0s per CONTEXT.md D-06. The concrete
        adapter enforces this as the read-axis timeout on its httpx call; the
        client-level `httpx.Timeout(connect=5, read=120, write=10, pool=5)`
        from `main.py` lifespan is the absolute ceiling.

        `options` is the opaque `ChatOptions` TypedDict — the concrete adapter
        forwards only the keys it recognizes.

        ``model_override`` (NEW, Phase 3 D-09 / JDG-06): when non-None, the
        concrete adapter MUST dispatch the request against this backend
        model name instead of its configured default. Enables the Judge
        (Phase 3 ``judge.grade``) to reuse the same Provider instance +
        shared ``httpx.AsyncClient`` while grading with ``cfg.judge.model``
        rather than ``cfg.local.model``. The adapter resolves any
        model-specific settings (e.g. Ollama's ``num_ctx_overrides``
        exact-match lookup per 02-D-17) against the RESOLVED backend model.
        ``None`` preserves the Phase 2 default (``self._cfg.model``).

        Raises:
            ProviderUnavailable: TCP-level failure (connection refused, DNS
                fail, connect timeout). The orchestrator (Phase 4)
                auto-pivots on this class.
            ProviderTimeout: generation exceeded `timeout_s` (read, write, or
                pool timeout). The orchestrator does NOT pivot — the FastAPI
                exception handler returns 504 to the client.
            ProviderHTTPError: 4xx/5xx from upstream, OR Pydantic
                `ValidationError` on response-shape drift (surfaced as
                `status_code=200`).
        """

    @abstractmethod
    async def health(self, *, timeout_s: float = 2.0) -> ProviderHealth:
        """Return the provider's current readiness state.

        Default timeout 2.0s per CONTEXT.md D-06. This method MAY be a pure
        state read (Phase 2 `OllamaProvider` is) or MAY issue a probe request.
        Either way it MUST return within `timeout_s`.

        Does NOT raise on upstream unreachability — an unhealthy provider
        returns `ProviderHealth(kind=..., state="unhealthy", last_error=...)`
        so `GET /health` can serve a `"degraded"` status rather than 500ing
        the health route.
        """

    @abstractmethod
    async def warmup(self, *, timeout_s: float = 300.0) -> None:
        """Prepare the backend for fast first-request serving.

        Default timeout 300.0s per CONTEXT.md D-06 — cold-start of a 70B model
        on modest hardware can legitimately take several minutes, so the
        warmup axis is deliberately generous relative to the 120.0s chat
        default.

        Concrete behavior varies by provider:

        - Phase 2 `OllamaProvider`: POST `/api/chat` with a 1-token generation
          to load the model into VRAM (D-12); includes `keep_alive: -1` so it
          stays loaded (D-13).
        - Phase 4 `LMStudioProvider`: validate the configured model against
          `GET /v1/models` (defeats LM Studio bug #619 where a single loaded
          model ignores the `model` field).
        - Phase 5 `OpenAIProvider`: no-op — frontier APIs have no cold-start.

        Raises on failure; the FastAPI lifespan catches-and-continues to boot
        degraded (D-08), while the `chat()` lazy-warmup path re-raises to the
        caller so the request receives a typed `ProviderUnavailable` /
        `ProviderTimeout`.
        """
