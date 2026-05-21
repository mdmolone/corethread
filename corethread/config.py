"""YAML + env-var loader for CoreThread's runtime config.

Phase 1 / Plan 04 — composes the four `*Config` sub-models from `models.py`
into a single typed `AppConfig(BaseSettings)`, and provides `load_config(path)`:
the public entry point PLAN-06's FastAPI lifespan calls once at startup.

Closes 6 of the 11 phase requirements:

- **CFG-01:** Single `config.yaml` is the source of truth.
- **CFG-02:** Frontier API key resolved from env-var named by `frontier.api_key_env`
  (default `OPENAI_API_KEY`), wrapped in `pydantic.SecretStr`. The literal value
  never appears in YAML on disk. **Defended at the loader layer** via an explicit
  ``if "api_key" in frontier_block`` guard that raises BEFORE env-var injection
  (NOT relied on schema-level `extra="forbid"`, since `FrontierResolved`
  legitimately adds the field after injection).
- **CFG-03:** Fail-fast on missing file, malformed YAML, missing required fields,
  or an unset env var — every failure mode wraps in a typed `ConfigError`.
- **CFG-04:** Confidence threshold defaults to 0.7 (via `RoutingConfig`), overridable
  in YAML.
- **CFG-05:** Constraint prompt configurable via `routing.constraint_prompt`.
- **OBS-03:** `SecretStr` + `hide_input_in_errors=True` on both `FrontierResolved`
  and `AppConfig` so Pydantic `ValidationError` paths never echo the literal key
  (Pitfall #7 / Pydantic #8303).

Threat-model surface (see `01-PLAN-04-config.md` <threat_model>):

- **T-01-14** (RCE via the deprecated unsafe yaml loader): MITIGATED — uses
  ``yaml.safe_load`` exclusively; the unsafe loader name does not appear anywhere
  in this file (acceptance criterion: ``grep yaml<dot>load<paren>`` returns 0).
- **T-01-15** (SecretStr leak via ValidationError): MITIGATED — ``hide_input_in_errors=True``
  set twice (FrontierResolved + AppConfig) AND the loader's ``except ValidationError``
  branch only surfaces field paths, never values.
- **T-01-16** (cause-chain leaks YAML line containing inlined key): MITIGATED —
  every ``raise ConfigError(...)`` ends in ``from None`` to suppress the cause chain.
- **T-01-17** (repr of cfg echoes SecretStr): MITIGATED — ``SecretStr.__repr__`` returns
  ``**********``; verified by tests 6 and 7.
- **T-01-18** (developer inlines ``api_key`` in YAML): MITIGATED — loader-layer guard
  ``if "api_key" in frontier_block`` raises BEFORE env-var injection. Schema-level
  ``extra="forbid"`` cannot defend CFG-02 alone because ``FrontierResolved`` legitimately
  adds the ``api_key`` field after injection — so the loader is the correct trust boundary.
- **T-01-19** (loader emits the post-injection raw dict to stdout): MITIGATED — this
  module contains zero stdlib-stream output calls. PLAN-05 adds the redaction filter
  as defense-in-depth.

This module imports only stdlib (`os`, `pathlib`, `yaml`), Pydantic types
(`SecretStr`, `ValidationError`), pydantic-settings (`BaseSettings`,
`SettingsConfigDict`), and the typed `ConfigError` + four sub-models from sibling
CoreThread modules. It MUST NOT import from `main`, `providers`, `orchestrator`,
or `judge` — that boundary is enforced by Phase 1's success criteria.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from corethread.errors import ConfigError
from corethread.models import (
    FrontierConfig,
    JudgeConfigModel,
    LocalConfig,
    RoutingConfig,
)

__all__ = [
    "AppConfig",
    "ConfigError",  # re-export for callers: `from config import ConfigError`
    "ControlsConfig",
    "FrontierResolved",
    "ModelPricing",
    "ModelProfile",
    "PrivacyConfig",
    "RoleProfiles",
    "UiConfig",
    "load_config",
]


# ---------------------------------------------------------------------------
# FrontierResolved — FrontierConfig + the SecretStr `api_key` resolved by the
# loader from `os.environ[api_key_env]`. Two-stage shape because the YAML carries
# `api_key_env` (an env-var NAME, not the value) and we resolve the env var into
# a `SecretStr` field that all downstream code consumes.
# ---------------------------------------------------------------------------


class FrontierResolved(FrontierConfig):
    """`FrontierConfig` + the resolved `SecretStr` API key (CFG-02 + OBS-03).

    `extra="forbid"` rejects any additional unknown fields on the frontier block.
    `hide_input_in_errors=True` is the load-bearing OBS-03 mitigation: when
    Pydantic raises a `ValidationError` on a SecretStr-typed field, the offending
    input value (the resolved API key) MUST NOT appear in `ValidationError.errors()`
    or `str(err)`. Without this flag, a downstream type mismatch could leak the
    key into a logged exception (Pitfall #7 / Pydantic #8303).

    Note: the CFG-02 defense for inlined-in-YAML `api_key` lives at the LOADER
    LAYER (`load_config`), NOT here — because by the time `FrontierResolved` is
    being validated, `load_config` has already injected the resolved key into
    the dict, so `api_key` is legitimately present. See T-01-18.
    """

    model_config = ConfigDict(
        extra="forbid",
        hide_input_in_errors=True,  # OBS-03 / Pitfall #7 — never echo SecretStr inputs
    )
    api_key: SecretStr  # resolved from os.environ[api_key_env] by load_config


ProviderKind = Literal["ollama", "lmstudio", "openai", "openrouter"]
ThemeMode = Literal["system", "light", "dark"]


class ModelProfile(BaseModel):
    """Reusable model/provider profile assignable to local, judge, or frontier."""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderKind
    model: str
    base_url: str
    api_key_env: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    timeout_s: float | None = Field(default=None, gt=0)
    num_ctx_default: int | None = Field(default=None, ge=1)
    num_ctx_overrides: dict[str, int] = Field(default_factory=dict)


class RoleProfiles(BaseModel):
    """Active profile ids for each CoreThread role."""

    model_config = ConfigDict(extra="forbid")

    local: str = "local-default"
    judge: str = "judge-default"
    frontier: str = "frontier-default"


class UiConfig(BaseModel):
    """User interface preferences persisted in config.yaml."""

    model_config = ConfigDict(extra="forbid")

    theme: ThemeMode = "system"


class PrivacyConfig(BaseModel):
    """Privacy-sensitive local debugging settings."""

    model_config = ConfigDict(extra="forbid")

    capture_transcripts: bool = False
    transcript_max: int = Field(default=25, ge=1)


class ModelPricing(BaseModel):
    """USD pricing for a model, expressed per one million tokens."""

    model_config = ConfigDict(extra="forbid")

    input_per_1m_tokens: float = Field(default=0.0, ge=0.0)
    output_per_1m_tokens: float = Field(default=0.0, ge=0.0)


class ControlsConfig(BaseModel):
    """Single-user local request controls and body-free audit logging."""

    model_config = ConfigDict(extra="forbid")

    requests_per_minute: int | None = Field(default=None, ge=1)
    daily_request_quota: int | None = Field(default=None, ge=1)
    daily_token_quota: int | None = Field(default=None, ge=1)
    daily_cost_quota_usd: float | None = Field(default=None, ge=0.0)
    pricing: dict[str, ModelPricing] = Field(default_factory=dict)
    audit_enabled: bool = False
    audit_path: str = "outputs/audit.jsonl"
    audit_include_request_body: bool = False


def _legacy_local_profile(local: LocalConfig) -> ModelProfile:
    return ModelProfile(
        provider=local.kind,
        base_url=local.base_url,
        model=local.model,
        num_ctx_default=local.num_ctx_default,
        num_ctx_overrides=dict(local.num_ctx_overrides),
    )


def _legacy_frontier_profile(frontier: FrontierConfig) -> ModelProfile:
    provider: ProviderKind = "openrouter" if "openrouter.ai" in frontier.base_url else "openai"
    return ModelProfile(
        provider=provider,
        base_url=frontier.base_url,
        model=frontier.model,
        api_key_env=frontier.api_key_env,
        max_tokens=frontier.max_tokens,
    )


def _normalise_profile_blocks(raw: dict[str, Any]) -> None:
    """Populate profile blocks from legacy local/judge/frontier config when absent."""
    profiles = raw.get("model_profiles")
    if profiles is None:
        local_block = raw.get("local") or {}
        judge_block = raw.get("judge") or {}
        frontier_block = raw.get("frontier") or {}
        if isinstance(local_block, dict) and isinstance(frontier_block, dict):
            frontier_base_url = str(frontier_block.get("base_url", "https://api.openai.com/v1"))
            profiles = {
                "local-default": {
                    "provider": local_block.get("kind", "lmstudio"),
                    "base_url": local_block.get("base_url", ""),
                    "model": local_block.get("model", ""),
                    "num_ctx_default": local_block.get("num_ctx_default", 8192),
                    "num_ctx_overrides": local_block.get("num_ctx_overrides", {}),
                },
                "judge-default": {
                    "provider": local_block.get("kind", "lmstudio"),
                    "base_url": local_block.get("base_url", ""),
                    "model": judge_block.get("model", local_block.get("model", "")),
                    "num_ctx_default": local_block.get("num_ctx_default", 8192),
                    "num_ctx_overrides": local_block.get("num_ctx_overrides", {}),
                },
                "frontier-default": {
                    "provider": (
                        "openrouter" if "openrouter.ai" in frontier_base_url else "openai"
                    ),
                    "base_url": frontier_base_url,
                    "model": frontier_block.get("model", ""),
                    "api_key_env": frontier_block.get("api_key_env", "OPENAI_API_KEY"),
                    "max_tokens": frontier_block.get("max_tokens", 512),
                },
            }
    if profiles is not None:
        raw["model_profiles"] = profiles
    raw.setdefault(
        "role_profiles",
        {
            "local": "local-default",
            "judge": "judge-default",
            "frontier": "frontier-default",
        },
    )
    raw.setdefault("ui", {"theme": "system"})
    raw.setdefault("privacy", {"capture_transcripts": False, "transcript_max": 25})
    raw.setdefault("controls", {})

    role_profiles = raw.get("role_profiles")
    model_profiles = raw.get("model_profiles")
    if isinstance(role_profiles, dict) and isinstance(model_profiles, dict):
        frontier_profile_id = role_profiles.get("frontier")
        frontier_profile = model_profiles.get(frontier_profile_id)
        if isinstance(frontier_profile, dict) and frontier_profile.get("provider") in {
            "openai",
            "openrouter",
        }:
            frontier_block = dict(raw.get("frontier") or {})
            frontier_block.update(
                {
                    "api_key_env": frontier_profile.get("api_key_env")
                    or frontier_block.get("api_key_env", "OPENAI_API_KEY"),
                    "base_url": frontier_profile.get("base_url")
                    or frontier_block.get("base_url", "https://api.openai.com/v1"),
                    "model": frontier_profile.get("model") or frontier_block.get("model", ""),
                    "max_tokens": frontier_profile.get("max_tokens")
                    or frontier_block.get("max_tokens", 512),
                }
            )
            raw["frontier"] = frontier_block


# ---------------------------------------------------------------------------
# AppConfig — top-level typed runtime config. Composes the four sub-models
# from models.py (per ORC-03: all Pydantic models in one file; pydantic-settings
# boilerplate stays out of models.py).
# ---------------------------------------------------------------------------


class AppConfig(BaseSettings):
    """Top-level runtime config. CONTEXT.md: nested-by-domain YAML.

    Loaded once at startup by `load_config()`. PLAN-06's FastAPI lifespan calls
    `load_config(path)` and binds the resulting `AppConfig` into the application
    state; per-request handlers read from it but never reload it.

    Inherits from `BaseSettings` (not `BaseModel`) so:

    1. `hide_input_in_errors=True` is honored by `SettingsConfigDict` exactly the
       same way as on `BaseModel.model_config` — defending OBS-03 at the top
       level too (defense in depth: the inner `FrontierResolved` already carries
       the same flag, but a future field-type mismatch elsewhere in the model
       could trigger ValidationError from the AppConfig level).
    2. Future env-var overlays (e.g. `CORETHREAD_LOCAL__BASE_URL=...`) are
       possible without retrofitting the class hierarchy.

    `extra="forbid"` rejects unknown top-level keys in YAML — typos like
    ``rotuing:`` (instead of ``routing:``) fail loudly at load time rather
    than being silently dropped. Per CONTEXT.md "all OpenAI shapes use
    extra=allow; AppConfig + every sub-model uses extra=forbid".
    """

    model_config = SettingsConfigDict(
        extra="forbid",
        hide_input_in_errors=True,  # OBS-03 — defense in depth atop FrontierResolved's flag
        # `env_file` and `env_prefix` are intentionally NOT set (T-01-20):
        # the only env var this loader reads is the explicit
        # `os.environ.get(frontier.api_key_env)` lookup in `load_config`.
        # Bounded surface; no auto-loading of `.env` files in cwd.
    )

    local: LocalConfig
    judge: JudgeConfigModel
    frontier: FrontierResolved
    routing: RoutingConfig = RoutingConfig()  # CFG-04 default; RoutingConfig has its own defaults
    model_profiles: dict[str, ModelProfile] = Field(default_factory=dict)
    role_profiles: RoleProfiles = Field(default_factory=RoleProfiles)
    ui: UiConfig = Field(default_factory=UiConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    controls: ControlsConfig = Field(default_factory=ControlsConfig)

    def profile_for(self, role: Literal["local", "judge", "frontier"]) -> ModelProfile:
        profile_id = getattr(self.role_profiles, role)
        profile = self.model_profiles.get(profile_id)
        if profile is not None:
            return profile
        if role == "local":
            return _legacy_local_profile(self.local)
        if role == "judge":
            base = _legacy_local_profile(self.local)
            return base.model_copy(update={"model": self.judge.model})
        return _legacy_frontier_profile(self.frontier)


# ---------------------------------------------------------------------------
# load_config — public entry point.
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> AppConfig:
    """Load and validate ``config.yaml``. Fail fast on every problem (CFG-03).

    Resolves ``frontier.api_key_env -> os.environ[that_name] -> SecretStr`` (CFG-02).
    The literal API key value never appears in YAML on disk; this loader is the
    sole boundary that reads the env var, wraps it in `SecretStr`, and injects
    it into the validated `AppConfig`. From this point on, downstream code only
    sees `cfg.frontier.api_key` as a `SecretStr` whose `.__repr__()` returns
    ``"**********"`` (Pitfall #12).

    Every failure mode raises a typed `ConfigError` (NOT a raw `yaml.YAMLError`,
    `OSError`, or `pydantic.ValidationError`). Every `raise ConfigError(...)`
    uses ``from None`` to suppress the `__cause__` chain — without this, a
    downstream traceback formatter could echo the original `yaml.YAMLError`'s
    repr, which on a developer-inlined key would include the offending YAML line
    (defense in depth atop the loader-layer guard; T-01-16).

    Failure modes:

    1. File missing -> `ConfigError("Config file not found: <path>")`
    2. YAML parse error -> `ConfigError("Malformed YAML in <path>: <yaml-error-msg>")`
    3. Cannot read file (OSError) -> `ConfigError("Cannot read config file <path>: ...")`
    4. Root not a mapping -> `ConfigError("Config root must be a mapping; got <type>")`
    5. Inlined `frontier.api_key` in YAML -> `ConfigError(...)` naming `api_key_env`
       (CFG-02 defensive guard at loader layer; T-01-18)
    6. Env var named by `api_key_env` is unset -> `ConfigError(...)` naming the env var
    7. Pydantic validation error on the assembled dict -> `ConfigError("Invalid config
       in <path>: failed fields = [...]")` — surfaces field paths only, never values
       (T-01-15)

    Args:
        path: Filesystem path to ``config.yaml``. Accepts `str` or `pathlib.Path`.

    Returns:
        A fully-validated `AppConfig` instance with the resolved `SecretStr`
        API key in `.frontier.api_key`.

    Raises:
        ConfigError: For any of the seven failure modes above.
    """
    p = Path(path)

    # --- (1) CFG-03: file missing -----------------------------------------
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}") from None

    # --- (2/3) CFG-01: YAML parse via yaml.safe_load (NEVER yaml.load —
    # T-01-14 / Pitfall: arbitrary Python obj construction via !!python/object).
    # We parse to a raw dict first, then validate, because we must resolve
    # `frontier.api_key_env -> SecretStr` in the dict BEFORE handing it to
    # `AppConfig.model_validate(...)`.
    try:
        with p.open("rt", encoding="utf-8") as fh:
            raw: Any = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        # `from None` (T-01-16): suppress the __cause__ chain so the original
        # YAMLError's repr — which can include the offending line content —
        # cannot leak into a downstream traceback formatter.
        raise ConfigError(f"Malformed YAML in {p}: {e}") from None
    except OSError as e:
        raise ConfigError(f"Cannot read config file {p}: {e}") from None

    # --- (4) Root must be a mapping ---------------------------------------
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping; got {type(raw).__name__}") from None

    _normalise_profile_blocks(raw)

    # --- (5) CFG-02 defensive guard: reject inlined api_key BEFORE env-var
    # injection. This is the linchpin of CFG-02 at the loader layer. Schema-level
    # `extra="forbid"` on `FrontierResolved` cannot defend CFG-02 alone because
    # `FrontierResolved` legitimately adds the `api_key` field — but only AFTER
    # we inject it below. So the developer-inlined value must be caught HERE,
    # before injection runs. T-01-18.
    frontier_block: dict[str, Any] = raw.get("frontier") or {}
    if not isinstance(frontier_block, dict):
        raise ConfigError(
            f"Config 'frontier' must be a mapping; got {type(frontier_block).__name__}"
        ) from None

    if "api_key" in frontier_block:
        # IMPORTANT: error message MUST NOT echo `frontier_block['api_key']` —
        # that would be the very leak we are trying to prevent. We only mention
        # the field name and the correct pattern. Test 12 verifies the literal
        # value is absent from `str(e)`.
        raise ConfigError(
            "frontier.api_key MUST NOT be inlined in YAML. "
            "Use frontier.api_key_env to name an env var instead [CFG-02]."
        ) from None

    # --- (6) CFG-02: resolve frontier.api_key_env -> SecretStr ------------
    env_name = frontier_block.get("api_key_env", "OPENAI_API_KEY")
    env_value = os.environ.get(env_name)
    if not env_value:
        raise ConfigError(
            f"Frontier API key env var '{env_name}' is not set. "
            f"Set it in your shell (e.g. `export {env_name}=sk-...`) before starting."
        ) from None

    # Inject the resolved key into the parsed dict so AppConfig validation sees
    # it. `frontier_block` is the same dict referenced by `raw["frontier"]`
    # (when it existed) — but if `raw["frontier"]` was missing/None we replaced
    # it with `{}` above and never wrote the new dict back, so be explicit:
    frontier_block["api_key"] = env_value
    raw["frontier"] = frontier_block

    # IMPORTANT: from this point on, `raw` contains the literal API key in memory.
    # Do NOT log `raw`. Do NOT include `raw` in any exception message. T-01-19.

    # --- (7) Construct typed AppConfig — ConfigError on any validation issue
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as e:
        # T-01-15: extract only field paths from the ValidationError. Do NOT
        # include `str(e)` wholesale because `hide_input_in_errors=True` is
        # best-effort across Pydantic versions and the safest defense is to
        # never include the raw error text in our own message at all.
        # `from None` suppresses the __cause__ chain that would otherwise
        # echo the original ValidationError's repr (which on some versions
        # may include input values despite the flag).
        field_paths = ", ".join(".".join(str(part) for part in err["loc"]) for err in e.errors())
        raise ConfigError(f"Invalid config in {p}: failed fields = [{field_paths}]") from None
