"""Phase 9 / SC#4 — corethread/cli.py CORETHREAD_WORKERS fail-fast guard tests.

Closes the SC#4 binding contract:
- workers=2 raises SystemExit with 'single-worker' substring
- non-integer CORETHREAD_WORKERS raises SystemExit echoing the value
- default invocation calls uvicorn.run(workers=1, host='127.0.0.1', ...)

The 'single-worker' literal in the error message is the D-03 verbatim
contract; do NOT change it. The host='127.0.0.1' invariant is the
SEC-01 carry-forward (no regression of v1.0 cli.py:25 behavior).
"""

from __future__ import annotations

import runpy
from typing import Any

import pytest

from corethread import cli


def test_cli_refuses_workers_above_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SC#4 — CORETHREAD_WORKERS=2 raises SystemExit with the locked substrings."""
    monkeypatch.setenv("CORETHREAD_WORKERS", "2")

    # Belt-and-suspenders: even if the guard somehow doesn't fire, intercept
    # uvicorn.run so the test never actually starts a server.
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: None)

    with pytest.raises(SystemExit, match=r"single-worker") as excinfo:
        cli.main()

    # D-03 anchors: the verbatim message contains 'single-worker' AND '[ARC-05]'.
    assert "single-worker" in str(excinfo.value), str(excinfo.value)
    assert "[ARC-05]" in str(excinfo.value), str(excinfo.value)


def test_cli_rejects_non_integer_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CORETHREAD_WORKERS='lots' raises SystemExit echoing the rejected value.

    Mirrors the existing CORETHREAD_PORT pattern (cli.py lines 30-34) where
    non-integer values raise SystemExit with the value in repr form.
    """
    monkeypatch.setenv("CORETHREAD_WORKERS", "lots")
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: None)

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    msg = str(excinfo.value)
    assert "CORETHREAD_WORKERS" in msg, msg
    assert "lots" in msg, msg
    assert "must be an integer" in msg, msg


def test_cli_default_invocation_has_workers_one_and_localhost_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default invocation: uvicorn.run called with workers=1 + host='127.0.0.1'.

    SEC-01 carry-forward (no regression of v1.0 cli.py:25 default bind) +
    D-03 defense-in-depth (workers=1 explicit kwarg). Uses monkeypatch to
    intercept uvicorn.run and capture its kwargs.
    """
    # Ensure none of the relevant env vars are set (they default).
    monkeypatch.delenv("CORETHREAD_WORKERS", raising=False)
    monkeypatch.delenv("CORETHREAD_HOST", raising=False)
    monkeypatch.delenv("CORETHREAD_PORT", raising=False)
    monkeypatch.delenv("CORETHREAD_RELOAD", raising=False)

    captured: dict[str, Any] = {}

    def _capture(*args: Any, **kwargs: Any) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr("uvicorn.run", _capture)

    cli.main()

    # uvicorn.run("corethread.main:app", host=..., port=..., reload=..., workers=1)
    assert captured["args"] == ("corethread.main:app",), captured["args"]
    kw = captured["kwargs"]
    assert kw.get("workers") == 1, f"D-03 defense-in-depth failed: workers={kw.get('workers')!r}"
    assert kw.get("host") == "127.0.0.1", f"SEC-01 regression: host={kw.get('host')!r}"
    assert kw.get("port") == 8000, f"port default drifted: port={kw.get('port')!r}"
    assert kw.get("reload") is False or kw.get("reload") == "", (
        f"reload default drifted: reload={kw.get('reload')!r}"
    )


def test_cli_module_mode_invokes_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """`python -m corethread.cli` starts through the same guarded main path."""
    monkeypatch.delenv("CORETHREAD_WORKERS", raising=False)
    monkeypatch.delenv("CORETHREAD_HOST", raising=False)
    monkeypatch.delenv("CORETHREAD_PORT", raising=False)
    monkeypatch.delenv("CORETHREAD_RELOAD", raising=False)

    captured: dict[str, Any] = {}

    def _capture(*args: Any, **kwargs: Any) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr("uvicorn.run", _capture)

    with pytest.warns(RuntimeWarning, match="corethread.cli"):
        runpy.run_module("corethread.cli", run_name="__main__")

    assert captured["args"] == ("corethread.main:app",)
    assert captured["kwargs"]["workers"] == 1
