"""Typed exception hierarchy + closed OpenAI-shaped error envelope taxonomy.

Phase 1 / Plan 02 — locks the `type` / `code` surface the FastAPI global exception
handler returns. The seven `ErrorType` members and eight `ErrorCode` members are
**closed**: adding an 8th `ErrorType` member or a 9th `ErrorCode` member is a code
change, not a config change. This is the explicit countermeasure to envelope drift
across later phases (Pitfall #1) and secret leakage via raw exception text
(Pitfall #12).

Imported by:
- `config.py` (PLAN-04) — raises `ConfigError` on fail-fast startup.
- `main.py` (PLAN-06) — `error_envelope()` builds the response body for the
  global exception handler; typed exceptions let the handler pattern-match to
  HTTP status codes without string-sniffing.

This module imports only stdlib. It MUST NOT import any provider, orchestrator,
or judge module — that boundary is enforced by Phase 1's success criteria.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

__all__ = [
    "ErrorType",
    "ErrorCode",
    "CoreThreadError",
    "ProviderUnavailable",
    "ProviderTimeout",
    "ProviderHTTPError",
    "JudgeParseError",
    "ConfigError",
    "QuotaExceeded",
    "RateLimitExceeded",
    "error_envelope",
]


class ErrorType(str, Enum):
    """Closed enum locking the OpenAI-shaped error envelope's `type` field.

    Mirrors OpenAI's error type strings where they exist. Per CONTEXT.md
    (FastAPI Skeleton & Error Envelope decision), this set is locked in
    Phase 1 so later phases cannot drift it.
    """

    INTERNAL_ERROR = "internal_error"
    INVALID_REQUEST_ERROR = "invalid_request_error"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_TIMEOUT = "provider_timeout"
    JUDGE_PARSE_ERROR = "judge_parse_error"
    STREAMING_UNSUPPORTED = "streaming_unsupported"
    NOT_IMPLEMENTED = "not_implemented"
    RATE_LIMIT_ERROR = "rate_limit_error"


class ErrorCode(str, Enum):
    """Closed enum locking the OpenAI-shaped error envelope's `code` field.

    Same seven values as `ErrorType` plus `CONFIG_ERROR` for fail-fast
    startup paths in `config.py` (PLAN-04). Per-error codes are allowed to
    be more specific than `ErrorType` per CONTEXT.md.
    """

    INTERNAL_ERROR = "internal_error"
    INVALID_REQUEST_ERROR = "invalid_request_error"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_TIMEOUT = "provider_timeout"
    JUDGE_PARSE_ERROR = "judge_parse_error"
    STREAMING_UNSUPPORTED = "streaming_unsupported"
    NOT_IMPLEMENTED = "not_implemented"
    CONFIG_ERROR = "config_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    QUOTA_EXCEEDED = "quota_exceeded"


class CoreThreadError(Exception):
    """Base class for every exception CoreThread itself raises.

    Class-level `error_type`, `error_code`, and `http_status` defaults let the
    FastAPI exception handler pattern-match `except CoreThreadError as exc`
    without string-sniffing. Subclasses override by re-declaring the class
    attributes at class scope.
    """

    error_type: ErrorType = ErrorType.INTERNAL_ERROR
    error_code: ErrorCode = ErrorCode.INTERNAL_ERROR
    http_status: int = 500

    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(message, *args)
        self.message = message


class ProviderUnavailable(CoreThreadError):
    """Local or frontier provider connection refused / DNS fail / not reachable.

    Distinct from `ProviderTimeout`: this means the provider socket never
    opened (CLAUDE.md: "local unreachable / connection refused -> auto-pivot").
    """

    error_type = ErrorType.PROVIDER_UNAVAILABLE
    error_code = ErrorCode.PROVIDER_UNAVAILABLE
    http_status = 503

    def __init__(self, provider_name: str, reason: str = "") -> None:
        super().__init__(
            f"Provider '{provider_name}' is unavailable"
            + (f": {reason}" if reason else "")
        )
        self.provider_name = provider_name


class ProviderTimeout(CoreThreadError):
    """Local or frontier provider exceeded its configured timeout.

    Distinct from `ProviderUnavailable`: the connection succeeded but the
    read/write/connect deadline elapsed (CLAUDE.md: "local read-timeout
    mid-generation -> surface 504, do NOT pivot"). This protects against
    cold-start burn-through.
    """

    error_type = ErrorType.PROVIDER_TIMEOUT
    error_code = ErrorCode.PROVIDER_TIMEOUT
    http_status = 504

    def __init__(self, provider_name: str, timeout_s: float) -> None:
        super().__init__(f"Provider '{provider_name}' timed out after {timeout_s}s")
        self.provider_name = provider_name
        self.timeout_s = timeout_s


class ProviderHTTPError(CoreThreadError):
    """Provider returned a non-2xx HTTP response.

    Pitfall #12 countermeasure: the upstream `body` is captured on the
    instance for log redaction (PLAN-05's filter scrubs it before write) but
    the user-facing `message` is generic (`"Provider 'X' returned HTTP N"`)
    and contains no upstream text. This defeats `httpx.HTTPStatusError`'s
    `__repr__` which would dump the `Authorization: Bearer sk-...` header.
    """

    error_type = ErrorType.INTERNAL_ERROR
    error_code = ErrorCode.INTERNAL_ERROR
    http_status = 502

    def __init__(self, provider_name: str, status_code: int, body: str = "") -> None:
        # Generic user-facing message — body MUST NOT echo to client (Pitfall #12).
        super().__init__(f"Provider '{provider_name}' returned HTTP {status_code}")
        self.provider_name = provider_name
        self.status_code = status_code
        self.body = body  # for internal logging only; PLAN-05 redaction filter scrubs


class JudgeParseError(CoreThreadError):
    """Judge output failed Pydantic validation after retry.

    Phase 3 (Judge) raises this when the judge model returns malformed JSON
    even after the one-retry-with-error pass. CLAUDE.md fail-safe rule says
    the orchestrator should treat this as a failed local answer and pivot
    rather than surface a 500; defining the exception in Phase 1 lets the
    type live in the surface from day one.
    """

    error_type = ErrorType.JUDGE_PARSE_ERROR
    error_code = ErrorCode.JUDGE_PARSE_ERROR
    http_status = 500


class ConfigError(CoreThreadError):
    """Config file missing, malformed, or referenced env var not set.

    Used by PLAN-04 (`config.py`) and PLAN-06 (`main.py` lifespan) for
    fail-fast startup. `error_type` stays `INTERNAL_ERROR` (config errors
    are server-side) but `error_code` is the more specific `CONFIG_ERROR`
    so operators can distinguish them in logs.
    """

    error_type = ErrorType.INTERNAL_ERROR
    error_code = ErrorCode.CONFIG_ERROR
    http_status = 500


class RateLimitExceeded(CoreThreadError):
    """Configured request rate limit rejected the request before routing."""

    error_type = ErrorType.RATE_LIMIT_ERROR
    error_code = ErrorCode.RATE_LIMIT_EXCEEDED
    http_status = 429


class QuotaExceeded(CoreThreadError):
    """Configured daily request/token/cost quota rejected the request."""

    error_type = ErrorType.RATE_LIMIT_ERROR
    error_code = ErrorCode.QUOTA_EXCEEDED
    http_status = 429


def error_envelope(
    exc: BaseException,
    *,
    message_override: str | None = None,
) -> dict[str, dict[str, str]]:
    """Build the OpenAI-shaped error envelope from any exception.

    Returns EXACTLY `{"error": {"message": <str>, "type": <str>, "code": <str>}}` —
    one top-level key, three inner keys, no extras.

    Pitfall #12: NEVER include the raw exception text (`str(exc)`, `repr(exc)`,
    `exc.args`) from a non-`CoreThreadError` — that's how `Authorization: Bearer
    sk-...` headers and `SecretStr` inputs leak into error responses via
    `httpx.HTTPStatusError.__repr__`. For unknown exceptions we return a
    generic `"Internal error"` envelope with no upstream content whatsoever.

    `CoreThreadError` instances are trusted: their `message` is constructed by
    the exception class itself (e.g. `f"Provider '{name}' returned HTTP {n}"`)
    and never includes upstream body text — see `ProviderHTTPError.__init__`.
    """
    if isinstance(exc, CoreThreadError):
        return {
            "error": {
                "message": message_override or exc.message,
                "type": exc.error_type.value,
                "code": exc.error_code.value,
            }
        }
    # Unknown exception — generic envelope. NEVER read exc.args / str(exc) / repr(exc).
    return {
        "error": {
            "message": message_override or "Internal error",
            "type": ErrorType.INTERNAL_ERROR.value,
            "code": ErrorCode.INTERNAL_ERROR.value,
        }
    }
