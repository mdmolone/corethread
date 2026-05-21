"""SEC-04 / SC#2 — ARC-04 mount-LAST invariant pin test.

This test pins Pitfall 17 against future regression. If a future PR
re-orders `corethread/main.py` such that `app.mount("/", SPAStaticFiles(...))`
shadows `/v1/chat/completions`, this test fails: the SPA's catch-all
serves index.html on the API path instead of the JSON 503 envelope.

Fixture monkeypatches `app.state.orchestrator` to raise
`ProviderUnavailable` on every call. The global typed-error handler in
`corethread/main.py:_corethread_error_handler` catches the exception and
returns a 503 with an OpenAI-shape JSON envelope. The SPA mount MUST NOT
intercept this — that's the ARC-04 invariant Phase 11 / Plan 02 enforces
at startup (lifespan assertion D-09 + D-12) and that this test enforces
at request time.

Test count delta: +1 (was 310, now 311).
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from corethread.errors import ProviderUnavailable


@pytest.fixture
def test_client_with_503_orchestrator(
    valid_yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Boot main.app with a stub orchestrator that always raises ProviderUnavailable.

    Mirrors `tests/conftest.py::app_with_fake_orchestrator` (lines 456-509):
    real lifespan runs (constructs real providers, sets `app.state.config`,
    wires the trace bus + stats pump, runs the Phase 11 D-09 ARC-04 mount-LAST
    assertion), then the test injects the stub orchestrator inside the with-block.
    """
    monkeypatch.setenv("CORETHREAD_CONFIG_PATH", str(valid_yaml_path))
    from corethread import main as m

    importlib.reload(m)

    class _Raising503Orchestrator:
        async def handle(self, request: object) -> object:
            raise ProviderUnavailable("test-stub", "503 simulated for SEC-04 pin test")

    with TestClient(m.app, raise_server_exceptions=False) as client:
        m.app.state.orchestrator = _Raising503Orchestrator()
        yield client


def test_spa_does_not_eat_api(test_client_with_503_orchestrator: TestClient) -> None:
    """ARC-04: API errors MUST return JSON, never the SPA's index.html.

    Pins SEC-04 against future regressions where someone re-orders main.py
    and the SPA mount accidentally swallows API responses (Pitfall 17).
    """
    response = test_client_with_503_orchestrator.post(
        "/v1/chat/completions",
        json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "test"}],
        },
    )

    # ProviderUnavailable maps to 503 via the typed-error handler in main.py.
    assert response.status_code == 503, (
        f"SEC-04 expected 503 from forced ProviderUnavailable, got {response.status_code}. "
        f"Body (first 200 chars): {response.text[:200]!r}"
    )

    # The CRITICAL assertion: API responses MUST be JSON, never HTML.
    # If this fails, the SPA mount is eating the API path (Pitfall 17 / ARC-04 violation).
    assert response.headers["content-type"].startswith("application/json"), (
        f"SEC-04 violation (Pitfall 17 / ARC-04 broken): "
        f"API returned content-type {response.headers['content-type']!r} "
        f"(expected application/json — SPA mount is shadowing /v1/chat/completions)"
    )

    # Defense in depth: even if Content-Type were somehow correct, body must not be HTML.
    assert "<html" not in response.text.lower(), (
        f"SEC-04 violation: API response body contains '<html' substring "
        f"(SPA index.html is leaking into API responses). "
        f"Body (first 500 chars): {response.text[:500]!r}"
    )

    # Defense in depth #2: the OpenAI-shape error envelope is JSON-parseable
    # and contains the expected error.type discriminator.
    body = response.json()
    assert "error" in body, f"SEC-04 envelope check: 'error' key absent in body {body!r}"
    assert body["error"]["type"] == "provider_unavailable", (
        f"SEC-04 envelope check: expected error.type=='provider_unavailable', "
        f"got {body['error'].get('type')!r}"
    )
