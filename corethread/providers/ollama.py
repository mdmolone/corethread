"""Native Ollama `/api/chat` adapter. Implements the `Provider` ABC from
`providers/base.py`. Phase 2 / Plan 02 per 02-CONTEXT.md.

This module is the first concrete `Provider` implementation — the adapter
Phase 4's orchestrator and Phase 3's Judge dispatch against for every
local-path request. It owns all Ollama-specific transport, wire-shape, and
warmup detail; later phases (LM Studio PLAN-04, Frontier PLAN-05, Orchestrator
Phase 4, Judge Phase 3) consume it only through the ABC surface and never
touch Ollama-level concerns.

Locked user decisions this file implements (per 02-CONTEXT.md):
- **D-01** native `/api/chat` transport (NOT the `/v1/chat/completions` shim,
  NOT the `ollama` Python client) — raw httpx against the shared
  `app.state.http_client`.
- **D-02** adapter-local `OllamaChatResponse(BaseModel)` with strict-forbid
  extra handling — surfaces Ollama schema drift immediately.
- **D-03** `format` passthrough as an opaque key (Phase 3 Judge populates it
  with a JSON Schema; Phase 2 does not interpret it).
- **D-09** lazy warmup on first `chat()` request, guarded by `asyncio.Lock`
  double-check — 10 concurrent first-requests trigger exactly one warmup.
- **D-12** warmup payload shape: 1-token generation against `/api/chat` with
  a single user message, `num_predict=1`, explicit `num_ctx`, indefinite
  `keep_alive`, non-streaming.
- **D-13** indefinite `keep_alive` in EVERY chat AND warmup request body
  (never relying on the `OLLAMA_KEEP_ALIVE` env var — the weakest override).
- **D-14** `response.model` echoes the CLIENT's requested model
  (`request.model`), NOT the backend model (`cfg.local.model`).
- **D-15** `system_fingerprint` is a deterministic prefix string derived
  from the first 10 chars of the echoed/requested model name.
- **D-16** response construction ONLY via `to_openai_chat_completion(...)`
  — NEVER the response model constructor by hand. Grep-gated.
- **D-17** `num_ctx` lookup is exact-match against the backend model name
  with default fallback. No prefix / family matching.
- **D-18** No tiktoken. Post-response heuristic warning emitted when the
  char/4 estimate exceeds `num_ctx` and `prompt_eval_count` is materially
  lower than the estimate.
- **D-19** httpx → typed exception mapping with CRITICAL specific-before-general
  `except`-clause ordering (Research-Delta-3): connect-level failures precede
  read/write/pool timeout catches because `ConnectTimeout` IS-A
  `TimeoutException` in httpx 0.28.1's MRO and first-matching-wins would
  misroute it to `ProviderTimeout`.

Research deltas incorporated:
- **Delta 1** — `done_reason` is `Optional[str]` with `None` default
  (conditionally present in Ollama responses).
- **Delta 2** — DO NOT forward Ollama's `created_at` (ISO-8601 string) to
  the OpenAI-shape helper; the helper synthesizes `created=int(time.time())`
  itself.
- **Delta 3** — the connect-timeout catch MUST come before any clause
  that catches the parent `httpx.TimeoutException` family.

This module may import from `models.py`, `errors.py`, `providers/base.py`,
and stdlib/httpx/pydantic/structlog. It MUST NOT import from `main`,
`config`, or `orchestrator`/`judge` (neither exists yet). The shared
`httpx.AsyncClient` arrives via `__init__` dependency injection — this file
NEVER constructs its own client (Pitfall #11).
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import httpx
import pydantic
import structlog
from pydantic import BaseModel, ConfigDict

from corethread.errors import ProviderHTTPError, ProviderTimeout, ProviderUnavailable
from corethread.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    LocalConfig,
    to_openai_chat_completion,
)
from corethread.providers.base import ChatOptions, Provider, ProviderHealth

__all__ = [
    "OllamaChatResponse",
    "OllamaProvider",
]

# Module-level logger — logger name becomes "providers.ollama" so operators
# can filter the entire provider layer with `providers.*`. Inherits the root
# RedactingFilter + structlog processor chain installed by `logging_config.py`.
_LOG = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Adapter-local Pydantic models (per D-02 — NOT in models.py).
# ---------------------------------------------------------------------------


class _OllamaMessage(BaseModel):
    """Inner `message` block of an Ollama `/api/chat` response.

    Strict-forbid extras per D-02 — surfaces schema drift instead of silently
    dropping new fields. `content: str` (no `Field(min_length=1)`) because
    empty-string content is legal for 1-token warmup responses
    (Pitfall H in 02-RESEARCH.md).
    """

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class OllamaChatResponse(BaseModel):
    """Adapter-local Pydantic validator for `/api/chat` stream=false bodies.

    Per 02-CONTEXT.md D-02: lives in `providers/ollama.py`, NOT in `models.py`
    — it is an Ollama-wire-shape detail, not a CoreThread contract. Strict
    extras-forbid is intentional: when Ollama adds a new field we want a
    loud `ValidationError` (routed to a provider HTTP error with synthetic
    status 200 per D-19) so schema drift is caught immediately rather than
    silently returning 0 prompt tokens for months.

    Per 02-RESEARCH.md Delta-1: `done_reason` is CONDITIONALLY present in
    Ollama responses (docs example omits it for one body, includes it for
    another). Typed optional for both shapes. The downstream
    `to_openai_chat_completion` call maps `None` via `OLLAMA_FINISH_MAP[None]
    = "stop"` so `None` round-trips safely.

    Per 02-RESEARCH.md Delta-2: `created_at` is an ISO-8601 string — NOT a
    unix int. The adapter MUST NOT pass it to the OpenAI-shape helper,
    which synthesizes `created=int(time.time())` itself. We validate it
    so schema drift on this field also surfaces.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    created_at: str  # ISO-8601; discarded by adapter (D-14 / Delta-2)
    message: _OllamaMessage
    done: bool
    done_reason: str | None = None  # Delta-1: conditionally present
    total_duration: int | None = None  # discarded
    load_duration: int | None = None  # discarded
    prompt_eval_count: int | None = None  # -> usage.prompt_tokens
    prompt_eval_duration: int | None = None
    eval_count: int | None = None  # -> usage.completion_tokens
    eval_duration: int | None = None


# ---------------------------------------------------------------------------
# Pure helper — unit-testable without httpx mocking.
# ---------------------------------------------------------------------------


def _build_ollama_payload(
    request: ChatCompletionRequest,
    *,
    model_override: str,
    num_ctx: int,
    options: ChatOptions | None = None,
) -> dict[str, Any]:
    """Compose the `/api/chat` POST body. Pure function — unit-testable
    without mocking httpx.

    Always sets:
    - `model` = `model_override` (D-14: the backend model; the client-model
      echo is handled at the RESPONSE layer, NOT the request layer — the
      request to Ollama uses the backend model so Ollama knows which weights
      to run).
    - `options.num_ctx` = `num_ctx` (D-17: resolved via
      `OllamaProvider._resolve_num_ctx(backend_model)` BEFORE this is called).
    - `keep_alive` = `-1` (D-13: hardcoded in every request body, never from
      env var).
    - `stream` = `False` (v1 never streams; Phase 5 rejects client
      `stream=true` with 400 before it reaches here).

    From `options` kwarg (if present): forwards `temperature`, `top_p`, `seed`,
    `num_predict`, `stop` into the nested `options` dict; forwards `format`
    opaquely at the top level of the body (D-03). IGNORES
    `options["keep_alive"]` and `options["num_ctx"]` — our server-side
    resolution is the source of truth per D-13 / D-17.
    """
    ollama_options: dict[str, Any] = {"num_ctx": num_ctx}
    if options:
        for key in ("temperature", "top_p", "seed", "num_predict", "stop"):
            if key in options:
                ollama_options[key] = options[key]  # type: ignore[literal-required]

    payload: dict[str, Any] = {
        "model": model_override,
        "messages": [m.model_dump(exclude_none=True) for m in request.messages],
        "options": ollama_options,
        "keep_alive": -1,  # D-13 — hardcoded literal; NEVER from env
        "stream": False,  # v1 never streams
    }
    if options and "format" in options:
        payload["format"] = options["format"]  # D-03 opaque passthrough
    return payload


# ---------------------------------------------------------------------------
# OllamaProvider — the concrete adapter.
# ---------------------------------------------------------------------------


class OllamaProvider(Provider):
    """Ollama native `/api/chat` adapter. Implements the `Provider` ABC.

    Injected shared `httpx.AsyncClient` (never constructs its own — Pitfall #11
    defense). Lazy double-checked warmup (D-09); every request body carries
    explicit `options.num_ctx` (D-17) and `keep_alive: -1` (D-13); response
    goes through `to_openai_chat_completion(...)` only (D-16).

    Task cancellation is never caught anywhere in this class — `asyncio`'s
    cancel exception propagates so FastAPI can cancel cleanly when a client
    disconnects.
    """

    name = "ollama"  # class-level per ABC contract

    def __init__(
        self,
        cfg: LocalConfig,
        http_client: httpx.AsyncClient,
    ) -> None:
        """Store the config + injected shared httpx client.

        `http_client` is the singleton built in `main.py` lifespan; this
        constructor MUST NEVER construct a new `httpx` AsyncClient — Pitfall #11.
        """
        self._cfg = cfg
        self._client = http_client  # D-injected — never constructed locally
        # D-09 warmup coordination state — instance-scoped, event-loop-bound
        # at lifespan time. Initial state is "unhealthy" so /health reports
        # "degraded" until the first warmup success flips it to "ready".
        self._warmed: bool = False
        self._warmup_lock: asyncio.Lock = asyncio.Lock()
        self._warmup_state: Literal["ready", "warming", "unhealthy"] = "unhealthy"
        self._last_warmup_error: str | None = None

    def _resolve_num_ctx(self, model: str) -> int:
        """D-17: exact-match lookup against the model actually sent to Ollama,
        with default fallback. No prefix / family matching — explicit beats
        clever; users wanting family defaults duplicate the entry.
        """
        return self._cfg.num_ctx_overrides.get(model, self._cfg.num_ctx_default)

    async def chat(
        self,
        request: ChatCompletionRequest,
        *,
        options: ChatOptions | None = None,
        model_override: str | None = None,
        timeout_s: float = 120.0,
    ) -> ChatCompletionResponse:
        """Issue a chat completion against Ollama and return an OpenAI-shaped
        response.

        When ``model_override`` is non-None, the payload dispatches to that
        backend model instead of ``self._cfg.model`` (D-09 — enables the
        Phase 3 Judge to reuse this provider with ``cfg.judge.model``).

        On first call (or after a failed warmup), lazily runs `warmup()` under
        `self._warmup_lock` so N concurrent first-requests trigger exactly one
        warmup call (D-09 double-checked lock).

        Raises (all via `from None` to suppress `__cause__` chain per Pitfall
        #12):
        - `ProviderUnavailable` — TCP-level failure (connect refused, connect
          timeout).
        - `ProviderTimeout` — read/write/pool timeout during generation.
        - `ProviderHTTPError` — upstream 4xx/5xx, OR Pydantic
          `ValidationError` on response-shape drift (status_code=200).
        """
        # D-09 lazy warmup, double-checked under lock. The fast path is a
        # plain attribute read — no lock acquisition after first success.
        if not self._warmed:
            async with self._warmup_lock:
                if not self._warmed:
                    await self.warmup()  # raises on failure; _warmed stays False
                    # IMPORTANT (Pitfall G): set _warmed=True ONLY after warmup
                    # returns cleanly. NEVER set it in a `finally` — a failed
                    # warmup must be retried by the next request.
                    self._warmed = True

        # D-09 / JDG-06: allow the caller (judge.grade, Phase 3) to override
        # the backend dispatch model without touching request.model (which
        # D-14 echoes back on the response). num_ctx resolves against the
        # RESOLVED backend_model so D-17 exact-match still works for the
        # judge-model row in cfg.local.num_ctx_overrides.
        backend_model = model_override or self._cfg.model
        num_ctx = self._resolve_num_ctx(backend_model)
        payload = _build_ollama_payload(
            request,
            model_override=backend_model,
            num_ctx=num_ctx,
            options=options,
        )

        # D-19 error mapping with CRITICAL specific-before-general except
        # ordering (Research-Delta-3 + Pitfall A). `ConnectTimeout` IS-A
        # `TimeoutException` in httpx 0.28.1, so catching `TimeoutException`
        # first would misroute it to `ProviderTimeout`. Both `Connect*`
        # clauses precede the timeout-trio clause.
        try:
            response = await self._client.post(
                f"{self._cfg.base_url}/api/chat",
                json=payload,
                timeout=timeout_s,  # overrides read axis only (D-06)
            )
            response.raise_for_status()
            parsed = OllamaChatResponse.model_validate(response.json())
        except httpx.ConnectError:
            raise ProviderUnavailable("ollama", "connection refused") from None
        except httpx.ConnectTimeout:
            raise ProviderUnavailable("ollama", "connect timeout") from None
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):
            raise ProviderTimeout("ollama", timeout_s) from None
        except (
            httpx.NetworkError,
            httpx.ProtocolError,
            httpx.ProxyError,
            httpx.UnsupportedProtocol,
        ) as exc:
            raise ProviderUnavailable("ollama", exc.__class__.__name__) from None
        except httpx.HTTPStatusError as exc:
            # `exc.response.text` is captured on the ProviderHTTPError
            # instance for the redacting logger to scrub before any log
            # emission — and the FastAPI global handler emits ONLY the
            # generic envelope (main.py _corethread_error_handler discipline),
            # never the body, to the client.
            raise ProviderHTTPError(
                "ollama",
                exc.response.status_code,
                exc.response.text,
            ) from None
        except pydantic.ValidationError as exc:
            # Pitfall E: NEVER log raw body or the input subfield of errors.
            # Field paths only — a joined structural identifier like
            # `"message.content, prompt_eval_count"` with no user content.
            field_paths = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
            _LOG.warning(
                "ollama.schema_drift",
                provider="ollama",
                field_paths=field_paths,
            )
            raise ProviderHTTPError("ollama", 200, field_paths) from None
        except ValueError:
            raise ProviderHTTPError("ollama", 200, "invalid JSON response") from None
        # `asyncio.CancelledError` intentionally NOT caught — let it propagate
        # so FastAPI can cancel the task cleanly (Research Anti-Patterns).

        # D-18 post-response heuristic truncation warning.
        # Rough char/4 estimator — NOT a correctness check, just a signal to
        # operators that `num_ctx_default` may be too low for their workload.
        # Content may be str, list[dict] (multimodal parts), or None; in the
        # list case `len()` returns element count which is fine for a rough
        # heuristic — real fix is right-sizing num_ctx, not a perfect tokenizer.
        estimated = sum(len(m.content or "") for m in request.messages) // 4
        if (
            parsed.prompt_eval_count
            and estimated > num_ctx
            and parsed.prompt_eval_count < estimated * 0.9
        ):
            _LOG.warning(
                "prompt.ctx.suspect_truncation",
                provider="ollama",
                model=backend_model,
                estimated_tokens=estimated,
                prompt_eval_count=parsed.prompt_eval_count,
                num_ctx=num_ctx,
            )

        # D-16 canonical response construction — the ONLY response path.
        # D-14 echo: response.model is request.model, NOT backend_model.
        # D-15 fingerprint: derived from request.model (truncated to 10 chars).
        # Delta-2: do NOT pass created_at — helper synthesizes `created` itself.
        return to_openai_chat_completion(
            model=request.model,  # D-14
            message_content=parsed.message.content,
            prompt_tokens=parsed.prompt_eval_count or 0,
            completion_tokens=parsed.eval_count or 0,
            # OLLAMA_FINISH_MAP handles None -> "stop"
            finish_reason=parsed.done_reason,
            # D-15 fingerprint derived from request.model (10-char truncation)
            system_fingerprint=f"fp_corethread_{request.model[:10]}",
        )

    async def health(self, *, timeout_s: float = 2.0) -> ProviderHealth:
        """Pure state read in Phase 2 — no HTTP probe. `/health` endpoint
        can call this frequently without rate-limiting the upstream.

        `timeout_s` is accepted for ABC compatibility; a pure dict-build has
        no wall-clock to bound.
        """
        return ProviderHealth(
            kind="ollama",
            state=self._warmup_state,
            last_error=self._last_warmup_error,
        )

    async def warmup(self, *, timeout_s: float = 300.0) -> None:
        """D-12 warmup: POST `/api/chat` with a 1-token generation to load
        the model into VRAM.

        Exercises the exact code path a real request hits (chat-template
        application, explicit `num_ctx`, `keep_alive: -1` re-affirmation) so
        a warmup success is genuine proof the first real request won't
        cold-start (Pitfall #3 canary).

        Raises on failure. Callers:
        - FastAPI lifespan (eager warmup; PLAN-03) catches
          `ProviderUnavailable` / `ProviderTimeout` and boots degraded per D-08.
        - Lazy warmup path in `chat()` re-raises so the originating request
          gets a typed `ProviderUnavailable` / `ProviderTimeout`.

        State transitions: `_warmup_state` starts `"unhealthy"` from
        `__init__`. On method entry -> `"warming"`. On success -> `"ready"`.
        On failure -> `"unhealthy"` (and `_last_warmup_error` is set to
        `exc.__class__.__name__` — Pitfall #12 / T-02-01 discipline: class
        name only, NEVER `str(exc)` or `repr(exc)`).
        """
        self._warmup_state = "warming"
        num_ctx = self._resolve_num_ctx(self._cfg.model)
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_predict": 1, "num_ctx": num_ctx},  # D-12
            "keep_alive": -1,  # D-13
            "stream": False,
        }
        try:
            response = await self._client.post(
                f"{self._cfg.base_url}/api/chat",
                json=payload,
                timeout=timeout_s,
            )
            response.raise_for_status()
            OllamaChatResponse.model_validate(response.json())  # exercise validator
        except httpx.ConnectError:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = "ConnectError"
            _LOG.warning("warmup.failed", provider="ollama", err_class="ConnectError")
            raise ProviderUnavailable("ollama", "connection refused") from None
        except httpx.ConnectTimeout:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = "ConnectTimeout"
            _LOG.warning("warmup.failed", provider="ollama", err_class="ConnectTimeout")
            raise ProviderUnavailable("ollama", "connect timeout") from None
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = exc.__class__.__name__
            _LOG.warning(
                "warmup.failed",
                provider="ollama",
                err_class=exc.__class__.__name__,
            )
            raise ProviderTimeout("ollama", timeout_s) from None
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
                provider="ollama",
                err_class=exc.__class__.__name__,
            )
            raise ProviderUnavailable("ollama", exc.__class__.__name__) from None
        except httpx.HTTPStatusError as exc:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = "HTTPStatusError"
            _LOG.warning("warmup.failed", provider="ollama", err_class="HTTPStatusError")
            raise ProviderHTTPError(
                "ollama",
                exc.response.status_code,
                exc.response.text,
            ) from None
        except pydantic.ValidationError as exc:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = "ValidationError"
            field_paths = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
            _LOG.warning(
                "warmup.failed",
                provider="ollama",
                err_class="ValidationError",
                field_paths=field_paths,
            )
            raise ProviderHTTPError("ollama", 200, field_paths) from None
        except ValueError as exc:
            self._warmup_state = "unhealthy"
            self._last_warmup_error = exc.__class__.__name__
            _LOG.warning(
                "warmup.failed",
                provider="ollama",
                err_class=exc.__class__.__name__,
            )
            raise ProviderHTTPError("ollama", 200, "invalid JSON response") from None

        # Success path — only reached when every step above returned cleanly.
        self._warmup_state = "ready"
        self._last_warmup_error = None
        _LOG.info(
            "warmup.ok",
            provider="ollama",
            model=self._cfg.model,
            num_ctx=num_ctx,
        )
