"""SPAStaticFiles — extension-aware 404→index.html rewrite + Cache-Control headers for the bundled SPA at frontend/dist/. Subclass of fastapi.staticfiles.StaticFiles. Mounted LAST in main.py per ARC-04. Single-class module — one concern per file (matches pubsub.py / stats.py / api_ui.py / cli.py)."""  # noqa: E501 — Phase 11 D-08 docstring locked verbatim

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

__all__ = ["SPAStaticFiles"]

_LOG = structlog.get_logger("corethread.spa")


class SPAStaticFiles(StaticFiles):
    """Subclass of StaticFiles for SPA serving.

    On 404 with extension-less path → re-serve index.html with Cache-Control: no-cache.
    On hit at index.html (or root /) → set Cache-Control: no-cache.
    On hit at assets/* (Vite hashed-output convention) → Cache-Control:
    public, max-age=31536000, immutable.
    On 404 with file extension (e.g., /missing.png) → re-raise the original 404
    so curl/clients that parse Content-Type aren't confused by HTML on /some.txt.
    """

    async def get_response(self, path: str, scope: Any) -> Response:  # type: ignore[override]
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            # D-05: extension-aware 404→index.html — only paths WITHOUT a file extension
            # get the SPA deep-link fallback. Paths with extensions (/missing.png,
            # /foo.css) re-raise so curl/clients aren't fed HTML on a content-type
            # they expected to be image/png or text/css.
            if exc.status_code == 404 and "." not in path.rsplit("/", 1)[-1]:
                fallback = FileResponse(Path(self.directory) / "index.html")  # type: ignore[arg-type]
                fallback.headers["Cache-Control"] = "no-cache"
                return fallback
            raise
        # D-06: Cache-Control on hits.
        # index.html (or root "/") → no-cache so deploys instantly invalidate.
        # assets/* (Vite hashed-output convention) → immutable; safe forever.
        # Anything else (favicon.ico, robots.txt) → default StaticFiles headers.
        #
        # Path normalization (Rule 1 fixes found by Plan 11-04 E2E test):
        #
        # 1. Root path is `"."` — when mounted at "/", a `GET /` request reaches
        #    StaticFiles with the mounted path stripped, which Starlette normalizes
        #    to `"."` (the `html=True` mode then auto-appends index.html and serves
        #    it). The original equality check `path in ("", "/", "index.html")`
        #    did not include `"."`, so `GET /` slipped through without no-cache.
        #
        # 2. On Windows, Starlette's `os.path.join(directory, path)` produces
        #    backslash separators (e.g., `assets\\index-abc.js`), so the original
        #    `path.startswith("assets/")` check missed legitimate asset hits.
        #    Normalize backslash to forward slash before the prefix check so the
        #    immutable cache header applies on every platform.
        normalized = path.replace("\\", "/")
        if normalized in ("", ".", "/", "index.html"):
            response.headers["Cache-Control"] = "no-cache"
        elif normalized.startswith("assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response
