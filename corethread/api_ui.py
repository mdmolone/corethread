"""UI Backend Endpoints router. Owns three /v1/* endpoints + the documentation-only /v1/_traces_event_schema stub. SCHEMA LOCKS: (a) ConfigView has extra='forbid' and exposes api_key_env: str only — NEVER api_key, NEVER api_key_resolved (Pitfall 19 / SEC-03 / CFG-02). (b) TraceEvent has extra='forbid' and contains NO prompt / message / content / body / request_body / response_body fields (Pitfall 18 / TRC-09). (c) StatsSnapshot variants have extra='forbid'; the snapshot dict from corethread.stats never contains body fields by Phase 8 D-07's _PIVOT_REASON_BUCKETS contract. Single-process, single-user (ARC-05 enforced by cli.py guard)."""  # noqa: E501 — D-22 docstring locked verbatim

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Literal

import httpx
import structlog
import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ConfigDict, Field

from corethread.config import AppConfig, ConfigError, load_config
from corethread.route_activity import RouteStage
from corethread.stats import Window

__all__ = ["router"]

_LOG = structlog.get_logger("corethread.api_ui")
_RESTART_TASKS: set[asyncio.Task[None]] = set()


# ---------------------------------------------------------------------------
# ConfigView family — D-06 (lives here, NOT in models.py per ARC-01)
# All sub-views are extra='forbid' (D-22 schema lock).
# ---------------------------------------------------------------------------


class LocalView(BaseModel):
    """Mirror of LocalConfig (corethread/models.py) — no secrets, plain copy."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["ollama", "lmstudio"]
    base_url: str
    model: str
    num_ctx_default: int
    num_ctx_overrides: dict[str, int]


class JudgeView(BaseModel):
    """Mirror of JudgeConfigModel. Contains no secrets."""

    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str


class FrontierView(BaseModel):
    """Frontier block exposed in /v1/config — D-07 / Pitfall 19 LOCKED.

    Fields are EXACTLY (model, api_key_env, max_tokens). NEVER add api_key.
    NEVER add api_key_resolved. NEVER add base_url (D-07 omits internal-only
    endpoint URL). Pitfall 19 + CFG-02 + SEC-03 anchor here.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    api_key_env: str  # CFG-02 / Pitfall 19: NEVER add cfg.frontier.api_key here
    max_tokens: int  # CFG-03: cost-lever surface


class RoutingView(BaseModel):
    """Mirror of RoutingConfig — threshold + constraint_prompt (CFG-03 cost lever)."""

    model_config = ConfigDict(extra="forbid")

    threshold: float
    constraint_prompt: str


class ModelProfileView(BaseModel):
    """Reusable model/provider profile surfaced for the config editor."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama", "lmstudio", "openai", "openrouter"]
    model: str
    base_url: str
    api_key_env: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout_s: float | None = None
    num_ctx_default: int | None = None
    num_ctx_overrides: dict[str, int] = Field(default_factory=dict)


class RoleProfilesView(BaseModel):
    """Active model profile ids for CoreThread roles."""

    model_config = ConfigDict(extra="forbid")

    local: str
    judge: str
    frontier: str


class UiView(BaseModel):
    """Persisted UI preferences."""

    model_config = ConfigDict(extra="forbid")

    theme: Literal["system", "light", "dark"] = "system"


class PrivacyView(BaseModel):
    """Privacy-sensitive local debugging settings."""

    model_config = ConfigDict(extra="forbid")

    capture_transcripts: bool = False
    transcript_max: int = Field(default=25, ge=1)


class ModelPricingView(BaseModel):
    """Public config view of per-model billing rates."""

    model_config = ConfigDict(extra="forbid")

    input_per_1m_tokens: float = Field(default=0.0, ge=0.0)
    output_per_1m_tokens: float = Field(default=0.0, ge=0.0)


class ControlsView(BaseModel):
    """Rate limits, quotas, billing estimates, and body-free audit logging."""

    model_config = ConfigDict(extra="forbid")

    requests_per_minute: int | None = Field(default=None, ge=1)
    daily_request_quota: int | None = Field(default=None, ge=1)
    daily_token_quota: int | None = Field(default=None, ge=1)
    daily_cost_quota_usd: float | None = Field(default=None, ge=0.0)
    pricing: dict[str, ModelPricingView] = Field(default_factory=dict)
    audit_enabled: bool = False
    audit_path: str = "outputs/audit.jsonl"
    audit_include_request_body: bool = False


# ---------------------------------------------------------------------------
# TraceEvent — D-14 / TRC-09 schema lock + Pitfall 18 hard lock
# ---------------------------------------------------------------------------
#
# Mirrors the 15 RequestTrace fields VERBATIM. extra='forbid' rejects any
# future field addition at validation time. NEVER add prompt / message /
# content / body / request_body / response_body — D-22 module docstring
# binds this and the SC#1 named test (tests/test_sse_no_prompt_leak.py)
# is the binding contract.
#
# Phase 9 publishes this schema via /v1/_traces_event_schema for Phase 10
# `pnpm gen:types` consumption (Pitfall 27 — Pydantic↔TS drift defense).


class TraceEvent(BaseModel):
    """SSE event payload for /v1/traces/stream — TRC-09 schema lock.

    15 fields verbatim from corethread.obs.RequestTrace. extra='forbid' is
    the structural Pitfall 18 defense. The model is also the OpenAPI
    publication target via /v1/_traces_event_schema (D-14).
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    selected_local_model: str
    judge_model: str
    frontier_model: str | None
    confidence_score: float
    pivoted: bool
    local_latency_ms: int
    judge_latency_ms: int
    frontier_latency_ms: int | None
    input_tokens: int
    output_tokens: int
    frontier_cost_est: float | None
    judge_parse_failed: bool
    pivot_reason: Literal[
        "none",
        "low_score",
        "local_truncated",
        "local_error",
        "judge_error",
    ]
    local_error_class: str | None


class RouteActivityEventView(BaseModel):
    """Body-free SSE payload for the live route graphic."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    stage: RouteStage
    selected_local_model: str | None
    judge_model: str | None
    frontier_model: str | None
    pivoted: bool | None
    confidence_score: float | None
    error_class: str | None
    ts_ms: int


# ---------------------------------------------------------------------------
# Stats response shapes — D-15 discriminated union with extra='forbid'
# ---------------------------------------------------------------------------
#
# The stats.snapshot() return is two-shape: empty-window has 4 keys; populated
# has 12. Pydantic 2's discriminated-union projection (response_model=A | B)
# emits both schemas to OpenAPI; Phase 10 type-gen narrows by window_size.
#
# Each quantile dict from stats.snapshot() is `{p50, p95, max, min}` or None
# (D-05 / D-08 None-on-empty contract from Phase 8). _Quantiles mirrors the
# populated dict shape; StatsSnapshotPopulated types each quantile field as
# `_Quantiles | None` so the None case projects cleanly.


class _Quantiles(BaseModel):
    """Mirrors stats._quantiles() return dict — 4-field quantile descriptor."""

    model_config = ConfigDict(extra="forbid")

    p50: float
    p95: float
    max: float
    min: float


class StatsSnapshotEmpty(BaseModel):
    """Empty-window snapshot — 4 keys (D-15 / Phase 8 D-09 #1).

    Returned when window_size == 0 (no traces in the requested window).
    Discriminator: `window_size: Literal[0]` for clean Phase 10 type-gen
    narrowing.
    """

    model_config = ConfigDict(extra="forbid")

    window_started_at: int
    window_size: Literal[0]
    max_window_size: int
    note: str


class StatsSnapshotPopulated(BaseModel):
    """Populated-window snapshot — 12 keys (D-15 / Phase 8 D-09 #2).

    pivot_reasons is always seeded with every Phase 8 D-07 bucket
    (_PIVOT_REASON_BUCKETS constant). Each quantile dict can be None on
    empty-after-filter (e.g., no pivots in window → frontier_latency_ms is None; all traces have
    judge_parse_failed=True → confidence_score is None).
    """

    model_config = ConfigDict(extra="forbid")

    window_started_at: int
    window_size: int
    max_window_size: int
    pivot_rate: float
    pivot_reasons: dict[str, int]  # always seeded per Phase 8 D-07 _PIVOT_REASON_BUCKETS
    tokens_in_total: int
    tokens_out_total: int
    local_latency_ms: _Quantiles | None
    judge_latency_ms: _Quantiles | None
    frontier_latency_ms: _Quantiles | None
    confidence_score: _Quantiles | None
    note: str


class ConfigView(BaseModel):
    """Top-level /v1/config response — extra='forbid' (D-22 lock).

    Mirrors AppConfig.{local, judge, frontier, routing} via the hand-rolled
    to_config_view(cfg) builder (D-08 allowlist semantics — a future
    AppConfig field defaults to NOT EXPOSED unless someone explicitly adds
    it to a sub-view here).
    """

    model_config = ConfigDict(extra="forbid")

    local: LocalView
    judge: JudgeView
    frontier: FrontierView
    routing: RoutingView
    model_profiles: dict[str, ModelProfileView]
    role_profiles: RoleProfilesView
    ui: UiView
    privacy: PrivacyView
    controls: ControlsView


class ConfigProbeResult(BaseModel):
    """Sanitized provider probe result for the config editor."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    provider: Literal["ollama", "lmstudio", "openai", "openrouter"]
    models: list[str] = Field(default_factory=list)
    selected_model_available: bool | None = None
    error_class: str | None = None
    message: str


class AdminRestartResponse(BaseModel):
    """Response from the local-only admin restart endpoint."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["restarting"]
    message: str


class TraceTranscriptView(BaseModel):
    """Prompt/response transcript for a retained trace.

    Unlike TraceEvent, this model intentionally carries request/response text.
    It is returned only from explicit transcript endpoints, never from SSE or
    stats.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    created_at: float
    updated_at: float
    request_messages: list[dict[str, Any]]
    local_response: str | None
    judge_response: str | None
    frontier_response: str | None
    final_response: str | None
    final_response_source: Literal["local", "frontier", "none"]
    trace: TraceEvent | None
    errors: dict[str, str]


class TraceTranscriptListView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    maxlen: int
    count: int
    items: list[TraceTranscriptView]


def to_config_view(cfg: AppConfig) -> ConfigView:
    """Hand-rolled, explicit field-by-field copy from AppConfig to ConfigView.

    D-08 allowlist semantics: NOT cfg.model_dump(exclude={"frontier": {"api_key"}}).
    Reviewer reading this function sees every field that gets surfaced;
    a future AppConfig field defaults to NOT EXPOSED unless someone
    explicitly adds it to a sub-view + this builder (Pitfall 19 defense).
    """
    return ConfigView(
        local=LocalView(
            kind=cfg.local.kind,
            base_url=cfg.local.base_url,
            model=cfg.local.model,
            num_ctx_default=cfg.local.num_ctx_default,
            num_ctx_overrides=cfg.local.num_ctx_overrides,
        ),
        judge=JudgeView(
            model=cfg.judge.model,
            prompt=cfg.judge.prompt,
        ),
        frontier=FrontierView(
            # CFG-02 / Pitfall 19: NEVER add cfg.frontier.api_key here
            model=cfg.frontier.model,
            api_key_env=cfg.frontier.api_key_env,
            max_tokens=cfg.frontier.max_tokens,
        ),
        routing=RoutingView(
            threshold=cfg.routing.threshold,
            constraint_prompt=cfg.routing.constraint_prompt,
        ),
        model_profiles={
            profile_id: ModelProfileView.model_validate(profile.model_dump())
            for profile_id, profile in cfg.model_profiles.items()
        },
        role_profiles=RoleProfilesView.model_validate(cfg.role_profiles.model_dump()),
        ui=UiView.model_validate(cfg.ui.model_dump()),
        privacy=PrivacyView.model_validate(cfg.privacy.model_dump()),
        controls=ControlsView.model_validate(cfg.controls.model_dump()),
    )


def _default_base_url(provider: str) -> str:
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    if provider == "openai":
        return "https://api.openai.com/v1"
    return ""


def _default_api_key_env(provider: str) -> str:
    return "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"


def _profile_for_role(
    view: ConfigView,
    role: Literal["local", "judge", "frontier"],
) -> ModelProfileView:
    profile_id = getattr(view.role_profiles, role)
    profile = view.model_profiles.get(profile_id)
    if profile is None:
        raise HTTPException(
            status_code=400,
            detail=f"role_profiles.{role} references unknown profile '{profile_id}'.",
        )
    return profile


def _config_yaml_payload(view: ConfigView) -> dict[str, Any]:
    """Build the YAML payload without resolved secrets."""
    data = view.model_dump(mode="json", exclude_none=True)

    local_profile = _profile_for_role(view, "local")
    if local_profile.provider in {"ollama", "lmstudio"}:
        data["local"] = {
            "kind": local_profile.provider,
            "base_url": local_profile.base_url,
            "model": local_profile.model,
            "num_ctx_default": local_profile.num_ctx_default or view.local.num_ctx_default,
            "num_ctx_overrides": local_profile.num_ctx_overrides,
        }

    judge_profile = _profile_for_role(view, "judge")
    data["judge"] = {
        "model": judge_profile.model,
        "prompt": view.judge.prompt,
    }

    frontier_profile = _profile_for_role(view, "frontier")
    if frontier_profile.provider in {"openai", "openrouter"}:
        data["frontier"] = {
            "api_key_env": frontier_profile.api_key_env
            or _default_api_key_env(frontier_profile.provider),
            "base_url": frontier_profile.base_url or _default_base_url(frontier_profile.provider),
            "model": frontier_profile.model,
            "max_tokens": frontier_profile.max_tokens or view.frontier.max_tokens,
        }
    return data


def _probe_timeout(profile: ModelProfileView) -> float:
    """Keep UI probes responsive even if chat generation timeouts are huge."""
    if profile.timeout_s is None:
        return 10.0
    return max(1.0, min(profile.timeout_s, 30.0))


def _models_url(profile: ModelProfileView) -> str:
    base_url = profile.base_url.rstrip("/")
    if profile.provider == "ollama":
        return f"{base_url}/api/tags"
    return f"{base_url}/models"


def _models_headers(profile: ModelProfileView) -> dict[str, str]:
    if profile.provider not in {"openai", "openrouter"}:
        return {}
    env_name = profile.api_key_env or _default_api_key_env(profile.provider)
    value = os.environ.get(env_name)
    if not value:
        raise KeyError(env_name)
    return {"Authorization": f"Bearer {value}"}


def _extract_model_ids(provider: str, body: Any) -> list[str]:
    if provider == "ollama":
        entries = body.get("models") if isinstance(body, dict) else None
        if not isinstance(entries, list):
            raise ValueError("MalformedModelsResponse")
        ids = [
            entry.get("name") or entry.get("model") for entry in entries if isinstance(entry, dict)
        ]
    else:
        entries = body.get("data") if isinstance(body, dict) else None
        if not isinstance(entries, list):
            raise ValueError("MalformedModelsResponse")
        ids = [entry.get("id") for entry in entries if isinstance(entry, dict)]
    return sorted({model_id for model_id in ids if isinstance(model_id, str) and model_id})


def _probe_success(profile: ModelProfileView, models: list[str]) -> ConfigProbeResult:
    selected_available = profile.model in models if profile.model else None
    if models:
        message = f"Connected; found {len(models)} model{'s' if len(models) != 1 else ''}."
    else:
        message = "Connected, but the provider returned no models."
    return ConfigProbeResult(
        ok=True,
        provider=profile.provider,
        models=models,
        selected_model_available=selected_available,
        message=message,
    )


def _probe_error(
    profile: ModelProfileView,
    *,
    error_class: str,
    message: str,
) -> ConfigProbeResult:
    return ConfigProbeResult(
        ok=False,
        provider=profile.provider,
        models=[],
        selected_model_available=None,
        error_class=error_class,
        message=message,
    )


def _is_loopback_request(request: Request) -> bool:
    if request.client is None:
        return False
    host = request.client.host
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


async def _terminate_after_response(delay_s: float = 0.5) -> None:
    await asyncio.sleep(delay_s)
    os.kill(os.getpid(), signal.SIGTERM)


def _helper_creation_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {
            "creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        }
    return {"start_new_session": True}


def _schedule_default_restart(request: Request) -> None:
    port = request.url.port or int(os.environ.get("CORETHREAD_PORT", "8000"))
    env = os.environ.copy()
    env.setdefault("CORETHREAD_HOST", "127.0.0.1")
    env.setdefault("CORETHREAD_PORT", str(port))
    cwd = str(Path.cwd())
    helper_script = f"""
import datetime
import os
import socket
import subprocess
import sys
import tempfile
import time

host = os.environ.get("CORETHREAD_HOST", "127.0.0.1")
port = int(os.environ.get("CORETHREAD_PORT", "{port}"))
cwd = {cwd!r}
deadline = time.time() + 15
while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        try:
            sock.connect((host, port))
        except OSError:
            break
    time.sleep(0.25)

stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
stdout_path = os.path.join(tempfile.gettempdir(), f"corethread-restart-{{stamp}}.out.log")
stderr_path = os.path.join(tempfile.gettempdir(), f"corethread-restart-{{stamp}}.err.log")
kwargs = {{}}
if os.name == "nt":
    kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        0,
    )
else:
    kwargs["start_new_session"] = True
with open(stdout_path, "ab") as stdout, open(stderr_path, "ab") as stderr:
    subprocess.Popen(
        [sys.executable, "-m", "corethread.cli"],
        cwd=cwd,
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        **kwargs,
    )
"""
    subprocess.Popen(
        [sys.executable, "-c", helper_script],
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **_helper_creation_kwargs(),
    )
    task = asyncio.create_task(_terminate_after_response())
    _RESTART_TASKS.add(task)
    task.add_done_callback(_RESTART_TASKS.discard)


# ---------------------------------------------------------------------------
# APIRouter — D-01 (one router for all three endpoints), D-05 (full paths)
# ---------------------------------------------------------------------------

router = APIRouter()


def _route_activity_event_id(event: RouteActivityEventView) -> str:
    return f"{event.request_id}:{event.ts_ms}:{event.stage}"


@router.get("/v1/config", response_model=ConfigView)
async def get_config(request: Request) -> JSONResponse:
    """Return the redacted ConfigView. D-09: Cache-Control: no-cache.

    Config is restart-only mutable in v1.1; an operator hot-edits config.yaml
    and restarts → the browser MUST re-fetch. Hand-rolled JSONResponse so we
    can attach the no-cache header alongside the validated body.
    """
    cfg: AppConfig = request.app.state.config
    view = to_config_view(cfg)
    return JSONResponse(
        content=view.model_dump(),
        headers={"Cache-Control": "no-cache"},
    )


@router.put("/v1/config", response_model=ConfigView)
async def put_config(request: Request, view: ConfigView) -> JSONResponse:
    """Persist config.yaml without writing resolved API keys."""
    config_path = Path(getattr(request.app.state, "config_path", "config.yaml"))
    tmp_path = config_path.with_name(f".{config_path.name}.tmp")
    payload = _config_yaml_payload(view)

    try:
        tmp_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        cfg = load_config(tmp_path)
        tmp_path.replace(config_path)
    except ConfigError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=exc.message) from None
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        _LOG.warning("config.write_failed", exc_class=exc.__class__.__name__)
        raise HTTPException(
            status_code=500,
            detail="Unable to write config.yaml.",
        ) from None

    request.app.state.config = cfg
    saved_view = to_config_view(cfg)
    return JSONResponse(
        content=saved_view.model_dump(),
        headers={
            "Cache-Control": "no-cache",
            "X-CoreThread-Restart-Required": "true",
        },
    )


@router.post(
    "/v1/admin/restart",
    response_model=AdminRestartResponse,
    status_code=202,
)
async def restart_corethread(request: Request) -> JSONResponse:
    """Local-only admin endpoint used by the config UI after restart-required saves."""
    if not _is_loopback_request(request):
        raise HTTPException(status_code=403, detail="Restart is only available locally.")

    restart_callback = getattr(request.app.state, "restart_callback", None)
    if restart_callback is None:
        _schedule_default_restart(request)
    else:
        result = restart_callback(request)
        if asyncio.iscoroutine(result):
            await result

    body = AdminRestartResponse(
        status="restarting",
        message="CoreThread restart has been scheduled.",
    )
    return JSONResponse(
        status_code=202,
        content=body.model_dump(),
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/v1/config/probe", response_model=ConfigProbeResult)
async def probe_config_profile(
    request: Request,
    profile: ModelProfileView,
) -> JSONResponse:
    """Probe a single model profile and return sanitized model-list results."""
    client: httpx.AsyncClient = request.app.state.http_client
    timeout_s = _probe_timeout(profile)
    url = _models_url(profile)
    try:
        headers = _models_headers(profile)
        response = await client.get(url, headers=headers, timeout=timeout_s)
        response.raise_for_status()
        models = _extract_model_ids(profile.provider, response.json())
        result = _probe_success(profile, models)
    except KeyError as exc:
        env_name = str(exc).strip("'")
        result = _probe_error(
            profile,
            error_class="MissingApiKeyEnv",
            message=f"Environment variable {env_name} is not set.",
        )
    except httpx.ConnectError:
        result = _probe_error(
            profile,
            error_class="ConnectError",
            message="Could not connect to the configured base URL.",
        )
    except httpx.ConnectTimeout:
        result = _probe_error(
            profile,
            error_class="ConnectTimeout",
            message="Timed out while connecting to the configured base URL.",
        )
    except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
        result = _probe_error(
            profile,
            error_class=exc.__class__.__name__,
            message=f"Timed out while reading models after {timeout_s:g}s.",
        )
    except httpx.HTTPStatusError as exc:
        result = _probe_error(
            profile,
            error_class="HTTPStatusError",
            message=f"Provider returned HTTP {exc.response.status_code}.",
        )
    except (json.JSONDecodeError, ValueError):
        result = _probe_error(
            profile,
            error_class="MalformedModelsResponse",
            message="Connected, but the provider returned an unexpected models response.",
        )

    return JSONResponse(
        content=result.model_dump(),
        headers={"Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------
# /v1/traces/stream — SSE endpoint with Last-Event-ID replay
# D-10 (request_id drain) + D-11 (unknown ID full replay) + D-13 (single event)
# Pitfall 20 (subscriber leak — asynccontextmanager + is_disconnected poll)
# Pitfall 21 (heartbeat — fastapi.sse._PING_INTERVAL=15s; ping=15)
# ---------------------------------------------------------------------------
#
# FastAPI 0.136 SSE pattern: decorate handler with response_class=EventSourceResponse
# and define it as an async generator yielding ServerSentEvent. The keep-alive
# heartbeat (ping=15) is module-level via fastapi.sse._PING_INTERVAL = 15.0
# applied automatically by the FastAPI routing layer (see fastapi/routing.py
# _keepalive_inserter using anyio.fail_after(_PING_INTERVAL)).


@router.get("/v1/route/stream", response_class=EventSourceResponse)
async def route_activity_stream(request: Request):
    """SSE stream of body-free route activity events for the live UI graphic."""
    bus = request.app.state.route_activity_bus
    last_event_id = request.headers.get("Last-Event-ID")

    async with bus.subscribe(replay=True) as q:
        if last_event_id:
            while not q.empty():
                raw_event = q.get_nowait()
                view = RouteActivityEventView.model_validate(raw_event)
                if _route_activity_event_id(view) == last_event_id:
                    break

        while True:
            if await request.is_disconnected():
                return
            try:
                raw_event = await asyncio.wait_for(q.get(), timeout=1.0)
            except TimeoutError:
                continue
            event_payload = RouteActivityEventView.model_validate(raw_event)
            yield ServerSentEvent(
                event="route",
                id=_route_activity_event_id(event_payload),
                raw_data=json.dumps(event_payload.model_dump()),
            )


@router.get(
    "/v1/_route_activity_event_schema",
    response_model=RouteActivityEventView,
    include_in_schema=True,
    summary=(
        "OpenAPI schema for SSE route activity events. Returns 410 - events "
        "are streamed via /v1/route/stream."
    ),
)
async def _route_activity_event_schema() -> RouteActivityEventView:
    raise HTTPException(
        status_code=410,
        detail=(
            "SSE event payloads are streamed via /v1/route/stream. "
            "This endpoint exists only to publish the RouteActivityEventView "
            "schema in OpenAPI."
        ),
    )


@router.get("/v1/traces/stream", response_class=EventSourceResponse)
async def traces_stream(request: Request):
    """SSE stream of RequestTrace events. Native fastapi.sse, ping=15 (D-13).

    D-10 drain-then-stream: if Last-Event-ID is present, drain the replay-
    seeded queue UNTIL we see a trace with matching request_id, then enter
    live mode (the next q.get() yields the first NEW event after disconnect).
    D-11: if no match is found (unknown ID), the drain loop completes and
    we fall through to live mode — frontend dedupes by request_id (Phase 10).

    Pitfall 20: request.is_disconnected() polled once per asyncio.wait_for
    timeout (1.0s). Phase 7 D-04's asynccontextmanager try/finally
    guarantees bus.subscribe()'s queue is unregistered on every exit path.

    Pitfall 18: TraceEvent.model_validate(t).model_dump_json() is the
    runtime drift catch — extra='forbid' rejects any future RequestTrace
    field that a hypothetical (forbidden) obs.py mutation might add.
    """
    bus = request.app.state.trace_bus
    last_event_id = request.headers.get("Last-Event-ID")

    async with bus.subscribe(replay=True) as q:
        # D-10 / D-11: drain replay buffer until we find a match. If
        # last_event_id is None or no match is found, the drain falls
        # through naturally and the next q.get() returns the live tail.
        if last_event_id:
            while not q.empty():
                t = q.get_nowait()
                if t["request_id"] == last_event_id:
                    # Found the disconnect point; the next q.get()
                    # yields the first NEW event the subscriber missed.
                    break

        # Live-mode loop. Pitfall 20 disconnect poll runs once per
        # 1-second wait_for timeout — bounded subscriber lifetime.
        while True:
            if await request.is_disconnected():
                return
            try:
                t = await asyncio.wait_for(q.get(), timeout=1.0)
            except TimeoutError:
                # Quiet bus tick — loop back to is_disconnected poll.
                continue
            # Pitfall 18 runtime defense: validate against TraceEvent
            # schema (extra='forbid'). If a future code path ever
            # smuggles a body field into the trace dict, this raises
            # at runtime BEFORE the data: line is emitted.
            event_payload = TraceEvent.model_validate(t).model_dump()
            yield ServerSentEvent(
                event="trace",
                id=t["request_id"],
                raw_data=json.dumps(event_payload),
            )


# ---------------------------------------------------------------------------
# /v1/traces/{request_id}/transcript and /v1/transcripts
# Explicit debug surfaces carrying retained prompt/response text.
# ---------------------------------------------------------------------------


def _get_transcript_store(request: Request) -> Any:
    if getattr(request.app.state, "transcript_capture_enabled", False) is not True:
        raise HTTPException(
            status_code=403,
            detail=(
                "Transcript capture is disabled. Set privacy.capture_transcripts=true "
                "in config.yaml and restart CoreThread to retain prompt/response bodies."
            ),
        )
    store = getattr(request.app.state, "transcript_store", None)
    if store is None:
        raise HTTPException(
            status_code=404,
            detail="Transcript capture is not available for this app instance.",
        )
    return store


@router.get(
    "/v1/traces/{request_id}/transcript",
    response_model=TraceTranscriptView,
)
async def get_trace_transcript(request: Request, request_id: str) -> JSONResponse:
    store = _get_transcript_store(request)
    item = store.get(request_id)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail="Transcript was not found. It may have been evicted.",
        )
    view = TraceTranscriptView.model_validate(item)
    return JSONResponse(
        content=view.model_dump(),
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/v1/transcripts", response_model=TraceTranscriptListView)
async def list_trace_transcripts(
    request: Request,
    limit: int = Query(default=25, ge=1, le=100),
) -> JSONResponse:
    store = _get_transcript_store(request)
    items = [TraceTranscriptView.model_validate(item) for item in store.latest(limit=limit)]
    view = TraceTranscriptListView(
        maxlen=store.maxlen,
        count=len(items),
        items=items,
    )
    return JSONResponse(
        content=view.model_dump(),
        headers={"Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------
# /v1/_traces_event_schema — D-14 OpenAPI schema publication stub
# ---------------------------------------------------------------------------
#
# FastAPI's openapi() introspects the response_model to publish the
# TraceEvent schema. The endpoint itself returns 410 — events are streamed
# via /v1/traces/stream, not pulled from this endpoint. Phase 10's
# `pnpm gen:types` consumes the published schema (Pitfall 27 defense).


@router.get(
    "/v1/_traces_event_schema",
    response_model=TraceEvent,
    include_in_schema=True,
    summary=(
        "OpenAPI schema for SSE trace events. Returns 410 — events are "
        "streamed via /v1/traces/stream."
    ),
)
async def _traces_event_schema() -> TraceEvent:  # pragma: no cover — never returns
    raise HTTPException(
        status_code=410,
        detail=(
            "SSE event payloads are streamed via /v1/traces/stream. "
            "This endpoint exists only to publish the TraceEvent schema in OpenAPI."
        ),
    )


# ---------------------------------------------------------------------------
# /v1/stats — D-15 (two-shape union) + D-16 (Window Literal validation) + D-09
# ---------------------------------------------------------------------------


@router.get(
    "/v1/stats",
    response_model=StatsSnapshotEmpty | StatsSnapshotPopulated,
)
async def get_stats(
    request: Request,
    window: Window = "all",
) -> JSONResponse:
    """Return a window-scoped StatsAggregator snapshot.

    D-16: `window: Window = "all"` — FastAPI auto-validates the query string
    against the Literal["1h","24h","7d","all"] (imported from
    corethread.stats). Invalid values raise RequestValidationError, which
    routes through main.py:359's existing handler → OpenAI-shape 400
    envelope. Zero new error-path code.

    D-09: Cache-Control: no-cache so a hot snapshot reflects the
    latest aggregator state on every UI re-fetch.
    """
    stats = request.app.state.stats
    snapshot = stats.snapshot(window=window)
    return JSONResponse(
        content=snapshot,
        headers={"Cache-Control": "no-cache"},
    )
