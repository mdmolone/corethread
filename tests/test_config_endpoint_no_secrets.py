"""SC#2 named test — Pitfall 19 / SEC-03 / CFG-02 secret-leak defense.

Loads a config with OPENAI_API_KEY=sk-marker-must-not-leak, hits /v1/config,
asserts the literal string is absent from the entire response body
(response.text + response.json() recursive walk) and that Cache-Control:
no-cache is present.

The literal `sk-marker-must-not-leak` is the SEC-03 binding contract —
do NOT change it. CONTEXT.md D-19 #2 names this verbatim.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from corethread.api_ui import router as ui_router
from tests.conftest import SEC_03_MARKER_KEY


def _walk(obj: Any) -> list[str]:
    """Recursively flatten dict/list/scalar into a list of stringified leaves."""
    out: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_walk(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_walk(v))
    else:
        out.append(str(obj))
    return out


def test_config_endpoint_no_secrets(cfg_with_marker_key: tuple[Any, Any]) -> None:
    """SC#2 — the SEC-03 binding contract."""
    _yaml_path, cfg = cfg_with_marker_key

    app = FastAPI()
    app.state.config = cfg
    app.include_router(ui_router)

    with TestClient(app) as client:
        resp = client.get("/v1/config")
        assert resp.status_code == 200, resp.text

        # 1. Literal must not appear in the raw response text
        assert SEC_03_MARKER_KEY not in resp.text, (
            "SEC-03 violation: marker key appeared in /v1/config response.text"
        )

        # 2. Literal must not appear in any leaf of the recursive .json() walk
        body = resp.json()
        for leaf in _walk(body):
            assert SEC_03_MARKER_KEY not in leaf, (
                f"SEC-03 violation: marker key appeared in body leaf {leaf!r}"
            )

        # 3. Cache-Control: no-cache header present (D-09)
        assert resp.headers.get("Cache-Control") == "no-cache", (
            f"D-09 violation: Cache-Control header missing/wrong: "
            f"{resp.headers.get('Cache-Control')!r}"
        )

        # 4. Defense in depth — the body MUST contain api_key_env (proves
        # the field IS surfaced; ensures the literal-absent check above
        # is not vacuously passing on an empty body).
        assert "api_key_env" in resp.text
        assert body["frontier"]["api_key_env"] == "OPENAI_API_KEY"
