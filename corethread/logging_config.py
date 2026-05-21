"""Structured JSON logging + root-level API-key redaction filter.

This module is the load-bearing safety net for OBS-02 (CLAUDE.md Conventions;
Pitfall #7 / #12): every log record — regardless of which library emits it —
passes through `RedactingFilter`, which strips `sk-*` API keys and
`Authorization: Bearer *` values before any formatter sees them.

Public surface:
    setup_logging(level="INFO") -> None
        Idempotent. Installs RedactingFilter on the ROOT stdlib logger + a
        single stdout StreamHandler. Configures structlog to render JSON to
        stdout via the same stdlib handler chain so structlog kwargs are
        redacted by the `_structlog_redact_processor` *and* the root filter.

    get_logger(name=None) -> structlog.stdlib.BoundLogger
        Returns a configured structlog logger bound to `name`. Call
        `setup_logging()` first (PLAN-06's lifespan does this before
        load_config so even config errors are filtered).

    RedactingFilter : logging.Filter
        Subclass attached to the ROOT logger (and the stdout handler, defense
        in depth). Mutates `record.msg` and `record.args` so redaction is
        visible in every formatter — JSON, human, whatever.

    redact_string(s) -> str
        Public helper for callers that want to scrub a string manually before
        logging (e.g., PLAN-06's FastAPI exception handler on an already-
        formatted traceback).

Regex patterns are module-level constants so tests can assert exact shape.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any

import structlog

# ---------------------------------------------------------------------------
# Redaction patterns
# ---------------------------------------------------------------------------
#
# OpenAI-style key prefix is `sk-` followed by >= 20 chars from [A-Za-z0-9_-].
# Project keys (`sk-proj-...`) and admin keys (`sk-admin-...`) share the same
# surface. The 20-char floor avoids false positives on harmless identifiers
# like `sk-data` or `sk-id` while staying well below real key length (~48-56).
_SK_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{20,}")

# Bearer token in an Authorization header. `re.IGNORECASE` because some
# libraries lowercase headers (e.g., `authorization: bearer ...`). Group 1
# captures the prefix so the substitution preserves it verbatim.
_BEARER_PATTERN = re.compile(r"(Authorization:\s*Bearer\s+)[A-Za-z0-9_\-\.]+", re.IGNORECASE)

_SK_REPLACEMENT = "sk-***REDACTED***"
_BEARER_REPLACEMENT = r"\1***REDACTED***"


def _redact(text: str) -> str:
    """Apply both redaction patterns to a string.

    Order matters: bearer first (captures and preserves the prefix), then the
    pure-substring `sk-*` pattern. If `sk-*` ran first it would still be
    correct (replacement string starts with literal `sk-***REDACTED***` which
    the bearer regex does not match), but doing bearer first is cleaner.
    """
    text = _BEARER_PATTERN.sub(_BEARER_REPLACEMENT, text)
    text = _SK_PATTERN.sub(_SK_REPLACEMENT, text)
    return text


def redact_string(s: str) -> str:
    """Public helper — redact a string using the same patterns.

    Exposed for callers (e.g., FastAPI exception handler in PLAN-06) that want
    to scrub an already-formatted traceback or error message before passing it
    to a logger.
    """
    return _redact(s)


# ---------------------------------------------------------------------------
# RedactingFilter — attached to the ROOT logger
# ---------------------------------------------------------------------------


class RedactingFilter(logging.Filter):
    """Strip API-key patterns from every log record.

    Attached to the ROOT logger so library logs (uvicorn, fastapi, httpx,
    ollama, openai) cannot bypass it — this is the OBS-02 guarantee.

    Mutates `record.msg` and `record.args` BEFORE any formatter sees them so
    the redaction is visible in JSON formatters and human formatters alike.
    Always returns True (we scrub; we never drop records).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Redact the format string itself
        if isinstance(record.msg, str):
            record.msg = _redact(record.msg)
        # Redact each arg (positional tuple OR mapping) if it is a string
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: (_redact(v) if isinstance(v, str) else v) for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(_redact(a) if isinstance(a, str) else a for a in record.args)
        # Also scrub a pre-resolved `message` attr if an upstream filter
        # already called record.getMessage() and cached it.
        if hasattr(record, "message") and isinstance(record.message, str):
            record.message = _redact(record.message)
        return True  # always pass through


# ---------------------------------------------------------------------------
# structlog processor — redact event_dict values before JSON serialization
# ---------------------------------------------------------------------------


def _redact_nested(value: Any) -> Any:
    """Recursively redact strings inside nested `dict | list | tuple` containers.

    Defense in depth (REVIEW WR-01): the structlog processor used to scrub
    only top-level `str` values, so a future caller that passed a `dict` /
    `list` / nested kwarg (e.g., `logger.info("evt", headers={...})`) would
    bypass the scrubber entirely. We now walk the structure. Non-container
    non-string values (ints, bools, None, custom objects) pass through
    unchanged — keeps the processor cheap on the hot path.
    """
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, dict):
        return {k: _redact_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_nested(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_nested(v) for v in value)
    return value


def _structlog_redact_processor(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> Mapping[str, Any]:
    """structlog processor that redacts every string value in the event dict.

    Must run BEFORE `JSONRenderer` in the processor chain — we want to scrub
    the structured values while the dict is still a dict, not after it has
    been serialized to a single JSON string.

    Recurses into nested dict/list/tuple values so callers that pass a
    structured kwarg (e.g., `logger.info("evt", headers={...})`) cannot
    bypass redaction by nesting (REVIEW WR-01 hardening).
    """
    for k, v in list(event_dict.items()):
        if isinstance(v, str):
            event_dict[k] = _redact(v)
        elif isinstance(v, (dict, list, tuple)):
            event_dict[k] = _redact_nested(v)
    return event_dict


# ---------------------------------------------------------------------------
# setup_logging / get_logger — public entry points
# ---------------------------------------------------------------------------

_SETUP_DONE = False  # idempotency guard — see Test 7 / T-01-25


def setup_logging(level: str = "INFO") -> None:
    """Install RedactingFilter on the root logger + configure structlog.

    Idempotent: safe to call multiple times (e.g., test setup + app lifespan).
    Should be called BEFORE `load_config()` so even malformed-config error
    paths are filtered.

    Args:
        level: stdlib logging level name (``"DEBUG"``, ``"INFO"``, ...).
               Defaults to ``"INFO"``. Re-applied on every call so callers
               can change level without re-attaching handlers.
    """
    global _SETUP_DONE

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    # Always (re-)set the level, even on repeat calls.
    root.setLevel(numeric_level)

    if _SETUP_DONE:
        return

    # Single stdout handler. The formatter is `%(message)s` because structlog
    # renders the JSON payload itself; stdlib callers that log via
    # `logging.getLogger(...).info(...)` get their (already-redacted) message
    # passed through as-is.
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))

    # Defense in depth: attach RedactingFilter to BOTH the handler AND the
    # root logger. Root catches records before they reach a handler; handler
    # catches records that arrive via a non-standard path (e.g., callers that
    # invoke `handler.handle(record)` directly).
    redactor = RedactingFilter()
    handler.addFilter(redactor)
    root.addHandler(handler)
    root.addFilter(redactor)

    # Configure structlog: redaction processor runs BEFORE JSONRenderer so
    # `logger.info("hi", key="sk-...")` has the `key` value scrubbed while the
    # event is still a dict. `LoggerFactory()` routes structlog events through
    # the stdlib logger chain, where the root RedactingFilter will scrub them
    # a second time (harmless — idempotent on already-redacted text).
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _structlog_redact_processor,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _SETUP_DONE = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to `name`.

    Call `setup_logging()` first (PLAN-06's lifespan handles this). Safe to
    call before configuration, but you will get structlog's default chain
    until setup runs.
    """
    return structlog.get_logger(name)
