"""Unit tests for corethread.spa.SPAStaticFiles (Phase 11 / Plan 02 / D-20).

Class-direct tests — no FastAPI app boot. Uses tmp_path to build a fake dist
tree so the test is fast and dist-independent (D-20). Mirrors
tests/test_pubsub.py's class-direct, no-app-boot pattern.

Pins SC#1 (ARC-04 — SPA mount serves index.html on extension-less 404 +
correct Cache-Control headers). Test count delta: +4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from starlette.exceptions import HTTPException

from corethread.spa import SPAStaticFiles


@pytest.fixture
def fake_dist(tmp_path: Path) -> Path:
    """Build a stub frontend/dist/ tree for class-direct testing (D-20).

    Layout:
        tmp_path/
          index.html
          assets/
            foo-abc123.js
          favicon.ico
    """
    (tmp_path / "index.html").write_text(
        '<!doctype html><html><body><div id="root"></div></body></html>'
    )
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "foo-abc123.js").write_text("console.log('ok')")
    (tmp_path / "favicon.ico").write_text("")
    return tmp_path


def _scope(path: str) -> dict[str, Any]:
    """Minimal ASGI scope dict for SPAStaticFiles.get_response(path, scope)."""
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "headers": [],
        "scheme": "http",
        "path": "/" + path,
        "raw_path": ("/" + path).encode("ascii"),
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 50000),
        "root_path": "",
    }


@pytest.mark.asyncio
async def test_404_with_extension_returns_real_404(fake_dist: Path) -> None:
    """D-05: /missing.png (extension) returns a real 404 — NOT index.html.

    Defends against curl/clients that parse Content-Type being fed HTML on
    a path they expected to be image/png.
    """
    spa = SPAStaticFiles(directory=str(fake_dist), html=True)
    with pytest.raises(HTTPException) as excinfo:
        await spa.get_response("missing.png", _scope("missing.png"))
    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_404_without_extension_returns_index_html_with_no_cache(
    fake_dist: Path,
) -> None:
    """D-05 + D-06: /dashboard (extension-less) → index.html + Cache-Control: no-cache.

    The SPA deep-link convention. A future v1.2 may add `/dashboard` as a
    client-side route; Phase 11's mount must serve index.html so React Router
    or equivalent can take over.
    """
    spa = SPAStaticFiles(directory=str(fake_dist), html=True)
    response = await spa.get_response("dashboard", _scope("dashboard"))
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-cache"


@pytest.mark.asyncio
async def test_index_html_hit_sets_no_cache(fake_dist: Path) -> None:
    """D-06: index.html hit → Cache-Control: no-cache.

    Deploys instantly invalidate the entry document so a fresh SPA bundle is
    fetched on the next page load. Mirrors api_ui.py's /v1/config Cache-Control
    pattern (Phase 9 D-09).
    """
    spa = SPAStaticFiles(directory=str(fake_dist), html=True)
    response = await spa.get_response("index.html", _scope("index.html"))
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-cache"


@pytest.mark.asyncio
async def test_root_path_hit_sets_no_cache(fake_dist: Path) -> None:
    """Rule 1 regression test (found by Plan 11-04 E2E test).

    When mounted at "/", a `GET /` request reaches StaticFiles with the
    mounted path stripped, which Starlette normalizes to `"."`. The
    `html=True` mode then auto-appends index.html and serves it. Plan 11-04's
    E2E test discovered that the original `path in ("", "/", "index.html")`
    set didn't include `"."`, so `GET /` through the real mount slipped
    through without the no-cache header.

    This test pins both the "" (direct call) AND "." (mount-normalized) cases.
    """
    spa = SPAStaticFiles(directory=str(fake_dist), html=True)

    # Empty path — direct-call shape (older Starlette behavior or html=False).
    response_empty = await spa.get_response("", _scope(""))
    assert response_empty.status_code == 200
    assert response_empty.headers["Cache-Control"] == "no-cache"

    # "." path — mount-normalized shape (Starlette current behavior at root).
    response_dot = await spa.get_response(".", _scope(""))
    assert response_dot.status_code == 200
    assert response_dot.headers["Cache-Control"] == "no-cache"


@pytest.mark.asyncio
async def test_hashed_asset_hit_sets_immutable(fake_dist: Path) -> None:
    """D-06: assets/foo-abc123.js (Vite hashed-output convention) → immutable.

    Hashed bundle filenames are content-addressed; the cache is forever-safe.
    `public, max-age=31536000, immutable` is the canonical "cache forever" header.
    """
    spa = SPAStaticFiles(directory=str(fake_dist), html=True)
    response = await spa.get_response("assets/foo-abc123.js", _scope("assets/foo-abc123.js"))
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "public, max-age=31536000, immutable"
