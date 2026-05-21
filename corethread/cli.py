"""CoreThread CLI entry point - Phase 6 / D-15 / D-16.

Thin shim: reads CORETHREAD_HOST / CORETHREAD_PORT / CORETHREAD_RELOAD from
os.environ, prints a one-line startup banner to stdout, and hands off to
``uvicorn.run("corethread.main:app", ...)``. No argparse, no click, no request
handling. ``CORETHREAD_CONFIG_PATH`` is owned by :mod:`corethread.config`
(D-18, Phase 1 lock); this CLI does NOT read it.

Env-var contract (D-16 + Phase 9 D-03):
    CORETHREAD_HOST    default "127.0.0.1"
    CORETHREAD_PORT    default 8000 (parsed as int; invalid -> SystemExit)
    CORETHREAD_RELOAD  default "" (any truthy string enables --reload)
    CORETHREAD_WORKERS default 1 (parsed as int; >1 -> SystemExit; ARC-05 lock)

Phase 9 / D-03 adds the CORETHREAD_WORKERS fail-fast guard (ARC-05 / Pitfall 25).
Closes PKG-02 (``uv run corethread`` works end-to-end).
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Entry point for ``[project.scripts] corethread = 'corethread.cli:main'``."""
    host = os.environ.get("CORETHREAD_HOST", "127.0.0.1")
    port_raw = os.environ.get("CORETHREAD_PORT", "8000")
    try:
        port = int(port_raw)
    except ValueError as exc:
        # Match Phase 1 CFG-03 fail-fast ethos: SystemExit with a clear message
        # so uvicorn never loads with a bogus port.
        raise SystemExit(f"CORETHREAD_PORT must be an integer, got: {port_raw!r}") from exc
    reload = bool(os.environ.get("CORETHREAD_RELOAD", ""))

    # Phase 9 / D-03 — multi-worker fragmentation defense (Pitfall 25 / ARC-05).
    # Pattern mirrors CORETHREAD_PORT verbatim (lines above): int-parse with
    # try/except SystemExit, then a > 1 guard. Multi-worker uvicorn would
    # fragment the in-process TraceBus (Phase 7) and StatsAggregator (Phase 8)
    # across worker processes — each worker has its own module-global bus
    # and its own app.state.stats, so a round-robin balancer would route
    # SSE subscribers to a worker that sees only a subset of traces.
    workers_raw = os.environ.get("CORETHREAD_WORKERS", "1")
    try:
        workers = int(workers_raw)
    except ValueError as exc:
        raise SystemExit(f"CORETHREAD_WORKERS must be an integer, got: {workers_raw!r}") from exc
    if workers > 1:
        raise SystemExit(
            "CoreThread requires single-worker mode — multi-worker breaks the trace stream "
            "and stats aggregation [ARC-05]."
        )

    # stdout banner - human-facing hello, NOT a structured-log decision event
    # (per Claude Discretion in 06-CONTEXT.md).
    print(f"CoreThread starting on {host}:{port} (reload={reload})")

    # Defense-in-depth (D-03): even if a future env-var bypass lands the workers=1
    # kwarg here ensures uvicorn never spawns more than one worker for this app.
    uvicorn.run("corethread.main:app", host=host, port=port, reload=reload, workers=1)


if __name__ == "__main__":
    main()
