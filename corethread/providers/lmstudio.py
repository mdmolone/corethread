"""Native LM Studio `/v1/chat/completions` adapter. Implements the `Provider`
ABC from `providers/base.py`. Phase 4 / Plan 04-04 per 04-CONTEXT.md.

This module is the SECOND concrete `Provider` implementation — file-for-file
mirror of `providers/ollama.py` with the deltas listed below. Validating the
Provider ABC against TWO concrete implementations before Phase 5's frontier
adapter lands is the point of doing LM Studio in Phase 4.

Locked user decisions this file implements (per 04-CONTEXT.md):
- **D-07** raw httpx against the shared `app.state.http_client` (NOT
  `openai.AsyncOpenAI`) — matches the Phase 2 OllamaProvider pattern, reuses
  the locked 4-axis timeouts, keeps `openai` dep out of this file.
- **D-08** endpoint paths use `cfg.base_url` AS-IS (operator's base_url already
  includes `/v1`) and append `/chat/completions` and `/models` directly. Both
  use `.rstrip('/')` to defeat Pitfall P4-4 double-`/v1/` (operator may or
  may not include a trailing slash).
- **D-09** adapter-local `LMStudioChatResponse(BaseModel)` with
  `ConfigDict(extra="forbid")` — schema drift surfaces immediately. Response
  construction routes through `to_openai_chat_completion(...)` only.
- **D-10** `num_ctx_default` / `num_ctx_overrides` are SILENTLY IGNORED — LM
  Studio has no native equivalent. One-time startup WARN log on instantiation
  if these fields are non-default so operators migrating from Ollama notice
  the lost setting.
- **D-11** httpx → typed exception mapping with CRITICAL specific-before-general
  `except`-clause ordering (Phase 2 Research-Delta-3): connect-level failures
  precede the read/write/pool timeout-trio because `ConnectTimeout` IS-A
  `TimeoutException` in httpx 0.28.1's MRO and first-matching-wins would
  misroute it.
- **D-12** lazy double-checked warmup with `asyncio.Lock` + `_warmed` +
  `_warmup_state` triplet — IDENTICAL to Phase 2 D-09. N concurrent
  first-requests trigger exactly one warmup call.
- **D-13** chat payload is OpenAI-shape ONLY — NO `keep_alive`, NO `options`
  block, NO `num_ctx`. Standard top-level fields (model, messages,
  temperature, max_tokens, top_p, stop, stream=False).
- **D-14** TWO-step `warmup()`: GET `/models` FIRST to validate the configured
  model is loaded (defeats LM Studio bug #619), then 1-token POST
  `/chat/completions` for cold-start canary.
- **D-15** error-class distinction:
  - `GET /models` 200 + model NOT in list → `ConfigError` (NOT caught by
    lifespan; uvicorn exits non-zero — fail-fast LOC-03)
  - connect refused / connect timeout → `ProviderUnavailable` (caught by
    lifespan per Phase 2 D-08; service boots degraded)
  - read/write/pool timeout → `ProviderTimeout`
  - 4xx/5xx → `ProviderHTTPError`
  - 200 + malformed JSON / missing `data` key → `ProviderHTTPError(200, ...)`

Limitations (Pitfall P4-7): assumes single LM Studio instance at
`cfg.base_url`. Distributed / load-balanced LM Studio setups (where
GET /v1/models hits one instance and POST /chat/completions hits another) can
defeat the D-14 model validation. Out of scope for v1 single-user local
service per PROJECT.md — file an issue if you hit this.

This module may import from `models.py`, `errors.py`, `providers/base.py`, and
stdlib/httpx/pydantic/structlog. It MUST NOT import from `main`, `config`, or
`orchestrator`/`judge`. The shared `httpx.AsyncClient` arrives via `__init__`
dependency injection — this file NEVER constructs its own client (Pitfall #11).
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import httpx
import pydantic
import structlog
from pydantic import BaseModel, ConfigDict

from corethread.errors import (
    ConfigError,
    ProviderHTTPError,
    ProviderTimeout,
    ProviderUnavailable,
)
from corethread.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    LocalConfig,
    to_openai_chat_completion,
)
from corethread.providers.base import ChatOptions, Provider, ProviderHealth

__all__ = [
    "LMStudioChatResponse",
    "LMStudioProvider",
]

# Module-level logger — logger name becomes "providers.lmstudio" so operators
# can filter the entire provider layer with `providers.*`. Inherits the root
# RedactingFilter + structlog processor chain installed by `logging_config.py`.
_LOG = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Adapter-local Pydantic models (per D-09 — NOT in models.py).
# ---------------------------------------------------------------------------


class _LMStudioMessage(BaseModel):
    """Inner `message` block of an LM Studio /v1/chat/completions response.

    `ConfigDict(extra="forbid")` per D-09 — surfaces schema drift instead of
    silently dropping new fields. `content: str` (no Field min_length) because
    empty-string content is legal for max_tokens=1 warmup responses.

    Recent LM Studio builds add OpenAI-compatible extension fields such as
    reasoning_content and tool_calls. They are accepted here but intentionally
    not forwarded into CoreThread's normalized response.
    """

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class _LMStudioChoice(BaseModel):
    """One element of LMStudioChatResponse.choices."""

    model_config = ConfigDict(extra="forbid")

    index: int
    message: _LMStudioMessage
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None  # LM Studio returns null by default


class _LMStudioUsage(BaseModel):
    """LM Studio usage block. All three counts are conventionally present
    but Pitfall P4-8 documents edge cases where LM Studio omits the entire
    `usage` block — handled at the parent model level by making `usage`
    Optional."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: dict[str, Any] | None = None


class LMStudioChatResponse(BaseModel):
    """Adapter-local Pydantic validator for /v1/chat/completions stream=false bodies.

    Per 04-CONTEXT.md D-09: lives in providers/lmstudio.py, NOT in models.py —
    it is an LM-Studio-wire-shape detail, not a CoreThread contract. Strict
    extras-forbid is intentional: when LM Studio adds a new field we want a
    loud ValidationError (routed to ProviderHTTPError(200, <field_paths>) per
    D-15) so schema drift is caught immediately rather than silently passing
    unknown values downstream.

    Per Pitfall P4-8 (04-RESEARCH.md): some LM Studio configurations / older
    versions return responses WITHOUT the `usage` block. Marking `usage` as
    Optional + defaulting prompt_tokens / completion_tokens to 0 in the
    to_openai_chat_completion call mirrors the Ollama adapter's defensive
    pattern (Phase 2 ollama.py line 368: `prompt_tokens=parsed.prompt_eval_count or 0`).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    object: str  # literal "chat.completion"
    created: int  # unix int (UNLIKE Ollama's ISO string)
    model: str
    choices: list[_LMStudioChoice]
    usage: _LMStudioUsage | None = None  # Pitfall P4-8: Optional
    system_fingerprint: str | None = None
    stats: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Pure helper — unit-testable without httpx mocking.
# ---------------------------------------------------------------------------


def _build_lmstudio_payload(
    request: ChatCompletionRequest,
    *,
    model_override: str,
    options: ChatOptions | None = None,
) -> dict[str, Any]:
    """Compose the /v1/chat/completions POST body. Pure function —
    unit-testable without mocking httpx.

    Per D-13: LM Studio is OpenAI-compat — no nested `options` block, no
    keep-alive directive (Ollama-specific), no `num_ctx` (LM Studio context
    window is configured in the LM Studio UI per D-10).

    Always sets:
    - model = model_override
    - messages = [m.model_dump(exclude_none=True) for m in request.messages]
    - stream = False (v1 never streams)

    From `options` kwarg (if present): forwards `temperature`, `top_p`, `seed`,
    `stop` at the TOP LEVEL (OpenAI shape, NOT nested under options like
    Ollama). IGNORES `keep_alive` and `num_ctx` (D-10 / D-13 — LM Studio has
    no equivalents) and `format` (LM Studio's strict-JSON path uses a
    different mechanism than Ollama's format= kwarg; deferred to Phase 5/6
    if needed). Uses request-level `max_tokens` if present, else does not
    set it (LM Studio's default applies).

    NOTE: per CONTEXT.md D-13 the adapter MUST NOT send fields LM Studio
    doesn't consume. Bypassing the `options.format` passthrough that
    OllamaProvider has is intentional — Phase 4 only needs OpenAI-shape
    chat-completions; Phase 5 strict-JSON support (if any) lands separately.
    """
    payload: dict[str, Any] = {
        "model": model_override,
        "messages": [m.model_dump(exclude_none=True) for m in request.messages],
        "stream": False,
    }
    # Top-level forwarding of standard OpenAI generation params from
    # request body (NOT from options kwarg — request fields take precedence).
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    if request.stop is not None:
        payload["stop"] = request.stop
    # Forward from options kwarg (lower precedence than request fields above).
    if options:
        for key in ("temperature", "top_p", "seed", "stop"):
            if key in options and key not in payload:
                payload[key] = options[key]  # type: ignore[literal-required]
    return payload


# ---------------------------------------------------------------------------
# LMStudioProvider — the concrete adapter.
# ---------------------------------------------------------------------------


class LMStudioProvider(Provider):
    """LM Studio /v1/chat/completions adapter. Implements the Provider ABC.

    Injected shared httpx.AsyncClient (NEVER constructs its own — Pitfall #11
    defense). Lazy double-checked warmup (D-12 — same pattern as Phase 2's
    OllamaProvider D-09); every request body is OpenAI-shape (D-13 — NO
    keep-alive directive, NO options block); response goes through
    `to_openai_chat_completion(...)` only (D-09 closing sentence).

    Startup model validation (D-14 + D-15): warmup() does GET {base_url}/models
    FIRST and asserts `cfg.model` is in the returned list. On model-not-in-list,
    raises ConfigError (NOT caught by lifespan per Phase 2 D-08 + D-15) so
    uvicorn exits non-zero — defeats LM Studio bug #619 (single-loaded-model
    silently ignores `model` field in request body).

    Limitations (Pitfall P4-7): assumes single LM Studio instance at
    `cfg.base_url`. Distributed / load-balanced LM Studio setups (where
    GET /v1/models hits one instance and POST /chat/completions hits another)
    can defeat the D-14 validation. Out of scope for v1 single-user local
    service per PROJECT.md.

    Task cancellation is NEVER caught (`asyncio.CancelledError` propagates
    so FastAPI can cancel cleanly when a client disconnects — ABC contract).
    """

    name = "lmstudio"  # class-level per ABC contract

    def __init__(
        self,
        cfg: LocalConfig,
        http_client: httpx.AsyncClient,
    ) -> None:
        """Store the config + injected shared httpx client.

        Emits a one-time WARN log if `num_ctx_default` is non-default (8192)
        OR `num_ctx_overrides` is non-empty — those fields are Ollama-only and
        LM Studio silently ignores them per D-10. The warning helps operators
        who migrate config.yaml from Ollama to LM Studio notice the lost
        setting (LM Studio context window is configured in the LM Studio UI).
        """
        self._cfg = cfg
        self._client = http_client  # D-injected — NEVER constructed locally
        # D-12 warmup coordination state — instance-scoped, event-loop-bound
        # at lifespan time. Initial state is "unhealthy" so /health reports
        # "degraded" until the first warmup success flips it to "ready".
        self._warmed: bool = False
        self._warmup_lock: asyncio.Lock = asyncio.Lock()
        self._warmup_state: Literal["ready", "warming", "unhealthy"] = "unhealthy"
        self._last_warmup_error: str | None = None

        # D-10: one-time startup WARN if Ollama-only num_ctx fields are non-default.
        # Default num_ctx_default per LocalConfig (models.py:375) is 8192;
        # default num_ctx_overrides is {} (empty dict). Operators migrating from
        # Ollama may have non-default values that LM Studio will silently ignore.
        if cfg.num_ctx_default != 8192 or cfg.num_ctx_overrides:
            _LOG.warning(
                "lmstudio.num_ctx.ignored",
                provider="lmstudio",
                num_ctx_default=cfg.num_ctx_default,
                overrides_count=len(cfg.num_ctx_overrides),
                note=(
                    "LM Studio has no native num_ctx equivalent — context "
                    "window is configured in the LM Studio UI. The configured "
                    "values will be silently ignored."
                ),
            )

    async def chat(
        self,
        request: ChatCompletionRequest,
        *,
        options: ChatOptions | None = None,
        model_override: str | None = None,
        timeout_s: float = 120.0,
    ) -> ChatCompletionResponse:
        """Issue a chat completion against LM Studio and return an OpenAI-shape
        response.

        On first call (or after a failed warmup), lazily runs `warmup()` under
        `self._warmup_lock` so N concurrent first-requests trigger exactly one
        warmup call (D-12 double-checked lock — IDENTICAL to Phase 2 D-09).

        Raises (all via `from None` per Pitfall #12):
        - `ProviderUnavailable` — TCP-level failure (connect refused / connect
          timeout). Orchestrator auto-pivots on this class (CLAUDE.md).
        - `ProviderTimeout` — read/write/pool timeout during generation.
          Orchestrator does NOT pivot (CLAUDE.md "distinguish timeout from
          quality failure"); FastAPI returns 504.
        - `ProviderHTTPError` — upstream 4xx/5xx, OR Pydantic ValidationError
          on response shape drift (status_code=200).
        - `ConfigError` — possible from `warmup()` if model not in /v1/models
          list (D-15); propagates to lifespan which does NOT catch it.
        """
        # D-12 lazy warmup, double-checked under lock. Fast path is a plain
        # attribute read — no lock acquisition after first success.
        if not self._warmed:
            async with self._warmup_lock:
                if not self._warmed:
                    await self.warmup()  # raises on failure; _warmed stays False
                    # IMPORTANT (Phase 2 Pitfall G): set _warmed=True ONLY after
                    # warmup returns cleanly. NEVER in a finally — a failed
                    # warmup must be retried by the next request.
                    self._warmed = True

        # D-09 model_override support per Phase 3 ABC change.
        backend_model = model_override or self._cfg.model
        payload = _build_lmstudio_payload(
            request,
            model_override=backend_model,
            options=options,
        )

        # D-11 error mapping with CRITICAL specific-before-general except
        # ordering (Phase 2 Research-Delta-3 + Pitfall A). ConnectTimeout IS-A
        # TimeoutException in httpx 0.28.1, so the timeout-trio catch MUST
        # come AFTER both Connect* clauses.
        try:
            response = await self._client.post(
                f"{self._cfg.base_url.rstrip('/')}/chat/completions",
                json=payload,
                timeout=timeout_s,
            )
            response.raise_for_status()
            parsed = LMStudioChatResponse.model_validate(response.json())
        except httpx.ConnectError:
            raise ProviderUnavailable("lmstudio", "connection refused") from None
        except httpx.ConnectTimeout:
            raise ProviderUnavailable("lmstudio", "connect timeout") from None
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):
            raise ProviderTimeout("lmstudio", timeout_s) from None
        except (
            httpx.NetworkError,
            httpx.ProtocolError,
            httpx.ProxyError,
            httpx.UnsupportedProtocol,
        ) as exc:
            raise ProviderUnavailable("lmstudio", exc.__class__.__name__) from None
        except httpx.HTTPStatusError as exc:
            raise ProviderHTTPError(
                "lmstudio",
                exc.response.status_code,
                exc.response.text,
            ) from None
        except pydantic.ValidationError as exc:
            # Pitfall E: NEVER log raw body / exc.errors()[i]["input"]. Field
            # paths only — joined structural identifiers, no user content.
            field_paths = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
            _LOG.warning(
                "lmstudio.schema_drift",
                provider="lmstudio",
                field_paths=field_paths,
            )
            raise ProviderHTTPError("lmstudio", 200, field_paths) from None
        except ValueError:
            raise ProviderHTTPError("lmstudio", 200, "invalid JSON response") from None
        # asyncio.CancelledError intentionally NOT caught — let it propagate.

        # D-09 closing sentence: response construction ONLY via the canonical
        # to_openai_chat_completion helper. Pitfall P4-8 defensive default:
        # usage may be missing on some LM Studio responses.
        prompt_tokens = parsed.usage.prompt_tokens if parsed.usage else 0
        completion_tokens = parsed.usage.completion_tokens if parsed.usage else 0
        finish_reason = parsed.choices[0].finish_reason if parsed.choices else "stop"
        message_content = parsed.choices[0].message.content if parsed.choices else ""

        return to_openai_chat_completion(
            model=request.model,  # D-14 echo
            message_content=message_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            system_fingerprint=f"fp_corethread_{request.model[:10]}",  # D-15
        )

    async def health(self, *, timeout_s: float = 2.0) -> ProviderHealth:
        """Pure state read in Phase 4 — no HTTP probe. /health endpoint
        can call this frequently without rate-limiting LM Studio.

        `timeout_s` is accepted for ABC compatibility; a pure dict-build has
        no wall-clock to bound.
        """
        return ProviderHealth(
            kind="lmstudio",
            state=self._warmup_state,
            last_error=self._last_warmup_error,
        )

    async def warmup(self, *, timeout_s: float = 300.0) -> None:
        """D-14 + D-15: TWO-step warmup.

        Step 1: GET {base_url}/models. Parse response. Assert cfg.model in
        [m["id"] for m in response["data"]]. On model-not-in-list, raise
        ConfigError (D-15 — NOT caught by lifespan; uvicorn exits non-zero).

        Step 2: POST /chat/completions with max_tokens=1, messages=[{role:user,
        content:hi}], stream=False. Mirrors Phase 2 D-12 — proves the full
        chat code path is live (cold-start canary).

        Error class distinction per D-15:
        - GET /v1/models 200 + model NOT in list → ConfigError (fail-fast)
        - GET /v1/models connect refused / DNS / connect timeout → ProviderUnavailable
        - GET /v1/models read timeout → ProviderTimeout
        - GET /v1/models 4xx/5xx → ProviderHTTPError
        - GET /v1/models 200 + malformed JSON / missing data key → ProviderHTTPError(200, ...)

        State transitions: _warmup_state starts "unhealthy" from __init__.
        On method entry → "warming". On success → "ready". On any failure →
        "unhealthy" + _last_warmup_error = exc.__class__.__name__ (Pitfall #12 —
        class name only, NEVER str(exc)/repr(exc)).
        """
        self._warmup_state = "warming"
        models_url = f"{self._cfg.base_url.rstrip('/')}/models"
        chat_url = f"{self._cfg.base_url.rstrip('/')}/chat/completions"

        # ---- D-14 step 1: GET /v1/models ----
        try:
            r = await self._client.get(models_url, timeout=timeout_s)
            r.raise_for_status()
            body = r.json()
        except httpx.ConnectError:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = "ConnectError"
            _LOG.warning("warmup.failed", provider="lmstudio", err_class="ConnectError")
            raise ProviderUnavailable("lmstudio", "connection refused") from None
        except httpx.ConnectTimeout:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = "ConnectTimeout"
            _LOG.warning("warmup.failed", provider="lmstudio", err_class="ConnectTimeout")
            raise ProviderUnavailable("lmstudio", "connect timeout") from None
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = exc.__class__.__name__
            _LOG.warning(
                "warmup.failed",
                provider="lmstudio",
                err_class=exc.__class__.__name__,
            )
            raise ProviderTimeout("lmstudio", timeout_s) from None
        except (
            httpx.NetworkError,
            httpx.ProtocolError,
            httpx.ProxyError,
            httpx.UnsupportedProtocol,
        ) as exc:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = exc.__class__.__name__
            _LOG.warning(
                "warmup.failed",
                provider="lmstudio",
                err_class=exc.__class__.__name__,
            )
            raise ProviderUnavailable("lmstudio", exc.__class__.__name__) from None
        except httpx.HTTPStatusError as exc:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = "HTTPStatusError"
            _LOG.warning("warmup.failed", provider="lmstudio", err_class="HTTPStatusError")
            raise ProviderHTTPError(
                "lmstudio",
                exc.response.status_code,
                exc.response.text,
            ) from None
        except ValueError as exc:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = exc.__class__.__name__
            _LOG.warning(
                "warmup.failed",
                provider="lmstudio",
                err_class=exc.__class__.__name__,
            )
            raise ProviderHTTPError("lmstudio", 200, "invalid JSON response") from None

        # D-14 step 2: validate response shape — D-15 malformed → ProviderHTTPError(200)
        if not isinstance(body, dict) or "data" not in body or not isinstance(body["data"], list):
            self._warmup_state = "unhealthy"
            self._last_warmup_error = "MalformedModelsResponse"
            _LOG.warning(
                "warmup.failed",
                provider="lmstudio",
                err_class="MalformedModelsResponse",
            )
            raise ProviderHTTPError(
                "lmstudio",
                200,
                "GET /models returned unexpected shape (missing 'data' list)",
            )

        # D-14 step 3: model-in-list check → D-15 ConfigError on mismatch
        available_ids = [
            m.get("id")
            for m in body["data"]
            if isinstance(m, dict) and isinstance(m.get("id"), str)
        ]
        if self._cfg.model not in available_ids:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = "ConfigError"
            _LOG.warning(
                "warmup.failed",
                provider="lmstudio",
                err_class="ConfigError",
                configured_model=self._cfg.model,
                available_count=len(available_ids),
            )
            # ConfigError is NOT caught by lifespan per D-15 — kills startup.
            raise ConfigError(
                f"LM Studio model '{self._cfg.model}' not found in /v1/models list: {available_ids}"
            )

        _LOG.info(
            "lmstudio.model.validated",
            provider="lmstudio",
            model=self._cfg.model,
            available=len(available_ids),
        )

        # ---- D-14 step 4: 1-token POST /chat/completions warmup ----
        warmup_payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "stream": False,
        }
        try:
            r2 = await self._client.post(chat_url, json=warmup_payload, timeout=timeout_s)
            r2.raise_for_status()
            LMStudioChatResponse.model_validate(r2.json())  # exercise validator
        except httpx.ConnectError:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = "ConnectError"
            _LOG.warning("warmup.failed", provider="lmstudio", err_class="ConnectError")
            raise ProviderUnavailable("lmstudio", "connection refused") from None
        except httpx.ConnectTimeout:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = "ConnectTimeout"
            _LOG.warning("warmup.failed", provider="lmstudio", err_class="ConnectTimeout")
            raise ProviderUnavailable("lmstudio", "connect timeout") from None
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = exc.__class__.__name__
            _LOG.warning(
                "warmup.failed",
                provider="lmstudio",
                err_class=exc.__class__.__name__,
            )
            raise ProviderTimeout("lmstudio", timeout_s) from None
        except (
            httpx.NetworkError,
            httpx.ProtocolError,
            httpx.ProxyError,
            httpx.UnsupportedProtocol,
        ) as exc:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = exc.__class__.__name__
            _LOG.warning(
                "warmup.failed",
                provider="lmstudio",
                err_class=exc.__class__.__name__,
            )
            raise ProviderUnavailable("lmstudio", exc.__class__.__name__) from None
        except httpx.HTTPStatusError as exc:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = "HTTPStatusError"
            _LOG.warning("warmup.failed", provider="lmstudio", err_class="HTTPStatusError")
            raise ProviderHTTPError(
                "lmstudio",
                exc.response.status_code,
                exc.response.text,
            ) from None
        except pydantic.ValidationError as exc:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = "ValidationError"
            field_paths = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
            _LOG.warning(
                "warmup.failed",
                provider="lmstudio",
                err_class="ValidationError",
                field_paths=field_paths,
            )
            raise ProviderHTTPError("lmstudio", 200, field_paths) from None
        except ValueError as exc:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = exc.__class__.__name__
            _LOG.warning(
                "warmup.failed",
                provider="lmstudio",
                err_class=exc.__class__.__name__,
            )
            raise ProviderHTTPError("lmstudio", 200, "invalid JSON response") from None

        # Success path — only reached when both warmup steps complete cleanly.
        self._warmup_state = "ready"
        self._last_warmup_error = None
        _LOG.info(
            "warmup.ok",
            provider="lmstudio",
            model=self._cfg.model,
        )
