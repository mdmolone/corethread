"""Tests for logging_config.py — covers OBS-02 (root-logger redaction filter).

Locks:
- The pure ``_redact()`` helper scrubs both `sk-*` (>=20 chars) and `Authorization:
  Bearer ...` patterns.
- ``RedactingFilter`` is attached to the ROOT stdlib logger so library logs
  (httpx, uvicorn, openai, ollama) are scrubbed regardless of which logger emits.
- structlog's `_structlog_redact_processor` scrubs `event_dict` string values
  BEFORE JSONRenderer serializes them — `get_logger().info("evt", key="sk-...")`
  produces JSON with the value masked, not the literal key.
- ``setup_logging()`` is idempotent — calling it twice does not duplicate handlers.
- Non-secret strings pass through unchanged (no false positives).

Capture strategy:
    The stdout StreamHandler installed by ``setup_logging`` binds to
    ``sys.stdout`` at handler creation time. Swapping ``sys.stdout`` AFTER setup
    does NOT redirect that handler. So the helper here looks up the existing
    StreamHandler on the root logger and temporarily swaps its ``.stream``
    attribute — the only stable way to capture formatted output.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable

import pytest

from corethread.logging_config import _redact, get_logger, setup_logging


@pytest.fixture(autouse=True)
def _setup_per_test() -> None:
    """Each test starts from a fresh setup. The autouse ``reset_logging_state``
    fixture in conftest.py clears the root logger first; here we re-install
    setup_logging cleanly bound to the current sys.stdout."""
    setup_logging("INFO")


def _capture_handler_output(fn: Callable[[], None]) -> str:
    """Capture the stdout StreamHandler's output by temporarily swapping its stream.

    The root logger's StreamHandler captured ``sys.stdout`` at construction; we
    cannot redirect it by reassigning ``sys.stdout``. Swapping ``handler.stream``
    is the robust fix and works for both stdlib logging AND structlog (since
    structlog routes through the stdlib chain via LoggerFactory()).
    """
    buf = io.StringIO()
    root = logging.getLogger()
    handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
    saved = [(h, h.stream) for h in handlers]
    for h in handlers:
        h.stream = buf
    try:
        fn()
        for h in handlers:
            h.flush()
    finally:
        for h, s in saved:
            h.stream = s
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Pure _redact() — patterns work in isolation
# ---------------------------------------------------------------------------


def test_pure_redact_function() -> None:
    """The pure helper redacts both pattern shapes."""
    out = _redact("sk-1234567890ABCDEFGHIJ leaked")
    assert "sk-1234567890ABCDEFGHIJ" not in out
    assert "***REDACTED***" in out

    out2 = _redact("Authorization: Bearer sk-foo-bar-12345-extra-chars")
    assert "sk-foo-bar" not in out2
    assert "***REDACTED***" in out2


def test_pure_redact_short_sk_not_matched() -> None:
    """sk- with <20 chars after is NOT matched (avoids false positives like sk-id)."""
    out = _redact("sk-id is short")
    assert out == "sk-id is short"  # unchanged


# ---------------------------------------------------------------------------
# OBS-02 — RedactingFilter on root logger catches library logs
# ---------------------------------------------------------------------------


def test_sk_redaction_in_library_logger() -> None:
    """OBS-02: filter on root catches LIBRARY logs (httpx), not just CoreThread."""
    out = _capture_handler_output(
        lambda: logging.getLogger("httpx").warning("config: sk-libleak-1234567890ABCDEF leaked")
    )
    assert "sk-libleak-1234567890ABCDEF" not in out, f"LEAK: {out}"
    assert "***REDACTED***" in out


def test_bearer_redaction() -> None:
    """OBS-02: Authorization: Bearer * substring is redacted."""
    out = _capture_handler_output(
        lambda: logging.getLogger("uvicorn").error(
            "Authorization: Bearer sk-bearer-test-1234567890"
        )
    )
    assert "sk-bearer-test-1234567890" not in out, f"LEAK: {out}"
    assert "***REDACTED***" in out


def test_redaction_in_args() -> None:
    """%-format args are redacted before the formatter sees them."""
    out = _capture_handler_output(
        lambda: logging.getLogger("app").info("key=%s", "sk-argleak-1234567890ABCDEF")
    )
    assert "sk-argleak-1234567890ABCDEF" not in out, f"LEAK in args: {out}"
    assert "***REDACTED***" in out


# ---------------------------------------------------------------------------
# structlog event-dict redaction — _structlog_redact_processor
# ---------------------------------------------------------------------------


def test_structlog_event_dict_redaction() -> None:
    """structlog kwargs are scrubbed before JSONRenderer serializes the event."""
    out = _capture_handler_output(
        lambda: get_logger("test").info("evt", key="sk-stleak-1234567890ABCDEF", other="harmless")
    )
    assert "sk-stleak-1234567890ABCDEF" not in out, f"LEAK in structlog: {out}"
    assert "***REDACTED***" in out
    # Non-secret kwarg passes through
    assert "other" in out
    assert "harmless" in out


# ---------------------------------------------------------------------------
# Idempotence + false-positive guard
# ---------------------------------------------------------------------------


def test_setup_logging_idempotent() -> None:
    """Calling setup_logging twice does not duplicate the production StreamHandler.

    Filters out pytest's own ``LogCaptureHandler`` (a StreamHandler subclass that
    pytest injects on the root logger to power the ``caplog`` fixture); we count
    only handlers whose exact class is ``logging.StreamHandler`` — that's the
    one ``setup_logging`` installs.
    """
    # autouse fixture already called setup_logging; calling more times must be a no-op
    setup_logging()
    setup_logging()
    handlers = [h for h in logging.getLogger().handlers if type(h) is logging.StreamHandler]
    assert len(handlers) == 1, f"expected 1 stdout handler, got {len(handlers)}: {handlers}"


def test_non_secret_passes_unchanged() -> None:
    """A normal log line containing no secrets passes through unchanged."""
    out = _capture_handler_output(lambda: logging.getLogger("app").info("normal log line"))
    assert "normal log line" in out
    assert "***REDACTED***" not in out
