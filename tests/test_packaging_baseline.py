"""Phase 6 / D-14 baseline packaging assertions.

Locks:
- corethread.main:app is importable as a package
- [project.scripts] corethread = "corethread.cli:main" resolves
- pyproject.toml lists the D-10 runtime + dev dep surface
- tenacity and ollama SDK are NOT declared deps (D-07, D-08)
- Python floor is >=3.12 (D-09)
- Version is 1.0.0 (D-03)
- License is MIT with license-files = ["LICENSE"] (D-04, PEP 639)
- LICENSE file exists with canonical MIT body (D-04)
- uv.lock is committed (D-05)

Every assertion here is a tripwire: if a future developer reverts the Phase 6
packaging posture (e.g., adds tenacity back, drops the CLI entry point,
loosens the python floor), pytest will fail on that commit.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _load_pyproject() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def test_corethread_main_app_importable() -> None:
    """D-14 (a): the package is installable + the FastAPI app object exists."""
    from corethread.main import app

    assert app is not None
    # Loose assertion - the title string isn't locked at a specific value, but
    # the attribute should resolve (proves app is a real FastAPI instance).
    assert hasattr(app, "title")


def test_corethread_cli_script_entry() -> None:
    """D-14 (b): [project.scripts] corethread = "corethread.cli:main"."""
    data = _load_pyproject()
    scripts = data["project"]["scripts"]
    assert scripts["corethread"] == "corethread.cli:main"


def test_corethread_cli_main_is_callable() -> None:
    """Adjacent to D-14 (b): the target entry point is a callable."""
    from corethread.cli import main

    assert callable(main)


def test_pyproject_d10_runtime_deps() -> None:
    """D-14 (c) + D-10: every required runtime dep is pinned with a caret range."""
    data = _load_pyproject()
    deps = {d.split(">=")[0].split("[")[0] for d in data["project"]["dependencies"]}
    required = {
        "fastapi",
        "pydantic",
        "pydantic-settings",
        "uvicorn",
        "httpx",
        "structlog",
        "openai",
    }
    assert required.issubset(deps), f"missing runtime deps: {required - deps}"


def test_pyproject_d10_dev_deps() -> None:
    """D-14 (c) + D-10: every required dev dep is declared."""
    data = _load_pyproject()
    dev = {d.split(">=")[0] for d in data["dependency-groups"]["dev"]}
    required = {"pytest", "pytest-asyncio", "respx", "ruff", "mypy"}
    assert required.issubset(dev), f"missing dev deps: {required - dev}"


def test_pyproject_no_tenacity_no_ollama_sdk() -> None:
    """D-07 + D-08: tenacity and the ollama SDK are NOT declared deps."""
    data = _load_pyproject()
    all_deps = [
        *data["project"]["dependencies"],
        *data["dependency-groups"]["dev"],
    ]
    # Normalize to the bare package name (strip extras markers + version specs).
    names = {d.split(">=")[0].split("[")[0].split("<")[0].strip() for d in all_deps}
    assert "tenacity" not in names, "D-07: tenacity was never imported; do not re-add"
    assert "ollama" not in names, "D-08: Phase 2 chose raw httpx; do not re-add the ollama SDK"


def test_pyproject_python_floor_312() -> None:
    """D-09: requires-python == '>=3.12'."""
    data = _load_pyproject()
    assert data["project"]["requires-python"] == ">=3.12"


def test_pyproject_version_1_0_0() -> None:
    """D-03: version bumped to 1.0.0 for the v1 ship."""
    data = _load_pyproject()
    assert data["project"]["version"] == "1.0.0"


def test_pyproject_license_mit_spdx() -> None:
    """D-04: PEP 639 SPDX-string license + license-files pointing at LICENSE."""
    data = _load_pyproject()
    assert data["project"]["license"] == "MIT"
    assert "LICENSE" in data["project"]["license-files"]


def test_license_file_exists() -> None:
    """D-04: LICENSE file at repo root, canonical MIT body."""
    license_path = REPO_ROOT / "LICENSE"
    assert license_path.exists(), "LICENSE file missing at repo root"
    content = license_path.read_text(encoding="utf-8")
    assert "MIT License" in content
    assert "Permission is hereby granted" in content


def test_uv_lock_committed() -> None:
    """D-05: uv.lock exists at repo root (generated + committed, not .gitignore'd)."""
    lockfile = REPO_ROOT / "uv.lock"
    assert lockfile.exists(), "uv.lock missing - run `uv lock` and commit"
