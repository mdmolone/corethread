"""Tests for config.py — covers Phase 1 Success Criteria #2 (CFG-03) + #5 (OBS-03).

Locks:
- Happy path: valid YAML + OPENAI_API_KEY env var resolves into SecretStr.
- SecretStr leak protection: literal value never appears in repr() or model_dump().
- ValidationError path does NOT echo the SecretStr value (Pitfall #7).
- Fail-fast on: missing file, malformed YAML, unset env var (CFG-03).
- threshold + constraint_prompt are configurable (CFG-04 / CFG-05).
- Inlined `frontier.api_key:` in YAML is REJECTED at the loader layer (CFG-02
  defensive guard, T-01-18) — and the error message itself does not echo the
  inlined value.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corethread.config import load_config
from corethread.errors import ConfigError
from corethread.models import DEFAULT_JUDGE_PROMPT

# Single source of truth for the autouse fixture's test key (REVIEW WR-02).
# >=20 chars after `sk-` so it would be redacted by `_SK_PATTERN` if it ever
# leaked into a log path — gives the leak-detection assertions two independent
# signals (literal-substring absence + regex-redaction coverage) on the same value.
from tests.conftest import TEST_OPENAI_API_KEY

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# SC#5 — happy path + SecretStr leak protection (OBS-03)
# ---------------------------------------------------------------------------


def test_load_config_happy_path(valid_yaml_path: Path) -> None:
    """SC#5: valid YAML + env var resolves; key usable via .get_secret_value()."""
    cfg = load_config(valid_yaml_path)
    # The autouse `set_test_api_key` fixture sets OPENAI_API_KEY to
    # `TEST_OPENAI_API_KEY` (>=20 chars after `sk-`, REVIEW WR-02).
    assert cfg.frontier.api_key.get_secret_value() == TEST_OPENAI_API_KEY
    assert cfg.local.kind == "ollama"
    assert cfg.local.model == "llama3.1:8b"
    assert cfg.judge.model == "qwen2.5:7b"
    assert cfg.judge.prompt == DEFAULT_JUDGE_PROMPT
    assert cfg.frontier.model == "gpt-4o"
    assert cfg.frontier.max_tokens == 512
    assert cfg.privacy.capture_transcripts is False
    assert cfg.privacy.transcript_max == 25
    assert cfg.controls.requests_per_minute is None
    assert cfg.controls.daily_request_quota is None
    assert cfg.controls.audit_enabled is False


@pytest.mark.parametrize(
    "relative_path",
    [
        "config.yaml.example",
        "examples/config.ollama-openai.yaml",
        "examples/config.lmstudio-openai.yaml",
        "examples/config.lmstudio-openai-judge.yaml",
        "examples/config.openrouter.yaml",
    ],
)
def test_public_example_configs_load(
    relative_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", TEST_OPENAI_API_KEY)
    monkeypatch.setenv("OPENROUTER_API_KEY", TEST_OPENAI_API_KEY)

    cfg = load_config(ROOT / relative_path)

    assert cfg.model_profiles
    assert cfg.profile_for("local").model
    assert cfg.profile_for("judge").model
    assert cfg.profile_for("frontier").model
    assert cfg.privacy.capture_transcripts is False


def test_load_config_accepts_str_path(valid_yaml_path: Path) -> None:
    """load_config signature is `path: str | Path` — both work."""
    cfg = load_config(str(valid_yaml_path))
    assert cfg.local.kind == "ollama"


def test_secret_not_in_repr_or_model_dump(valid_yaml_path: Path) -> None:
    """SC#5: SecretStr value never appears in repr() or model_dump() string forms.

    Uses ``TEST_OPENAI_API_KEY`` (>=20 chars after ``sk-``) so this assertion
    exercises BOTH the SecretStr ``__repr__`` mask AND the production
    ``_SK_PATTERN`` redaction filter on the same string — two independent leak
    signals from one assertion (REVIEW WR-02).
    """
    cfg = load_config(valid_yaml_path)
    assert TEST_OPENAI_API_KEY not in repr(cfg), f"LEAK in repr: {cfg!r}"
    dumped = cfg.model_dump()
    assert TEST_OPENAI_API_KEY not in str(dumped), f"LEAK in model_dump: {dumped}"
    # Defense in depth: also check the JSON serialization
    dumped_json = cfg.model_dump_json()
    assert TEST_OPENAI_API_KEY not in dumped_json, f"LEAK in model_dump_json: {dumped_json}"


def test_validation_error_does_not_echo_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OBS-03 + Pitfall #7: ValidationError on a sibling field MUST NOT echo the
    SecretStr key value into the ConfigError message."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leak-9999-secret-test-token-xyz")
    bad = (
        "local: {kind: ollama, base_url: http://x, model: m}\n"
        "judge: {model: j}\n"
        "frontier:\n"
        "  api_key_env: OPENAI_API_KEY\n"
        "  base_url: https://api.openai.com/v1\n"
        "  model: gpt-4o\n"
        "  max_tokens: not_a_number\n"  # type error triggers ValidationError
        "routing: {threshold: 0.7, constraint_prompt: x}\n"
    )
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    with pytest.raises(ConfigError) as exc_info:
        load_config(p)
    assert "sk-leak-9999" not in str(exc_info.value), f"LEAK in ConfigError: {exc_info.value}"


# ---------------------------------------------------------------------------
# SC#2 — CFG-03 fail-fast paths
# ---------------------------------------------------------------------------


def test_missing_file_fails_fast() -> None:
    """SC#2 (a): missing config raises ConfigError naming the path."""
    with pytest.raises(ConfigError) as exc_info:
        load_config("/definitely/nonexistent/path/x.yaml")
    assert "not found" in str(exc_info.value).lower()


def test_malformed_yaml_fails_fast(tmp_path: Path) -> None:
    """SC#2 (b): YAML parse error wraps to ConfigError, NOT a raw yaml.YAMLError."""
    p = tmp_path / "bad.yaml"
    p.write_text("this: is: not: valid: yaml: [unclosed")
    with pytest.raises(ConfigError) as exc_info:
        load_config(p)
    assert "malformed" in str(exc_info.value).lower() or "yaml" in str(exc_info.value).lower()


def test_root_must_be_mapping(tmp_path: Path) -> None:
    """A YAML doc whose root is a scalar/list (not a mapping) is rejected."""
    p = tmp_path / "list.yaml"
    p.write_text("- one\n- two\n")
    with pytest.raises(ConfigError) as exc_info:
        load_config(p)
    assert "mapping" in str(exc_info.value).lower()


def test_missing_env_var_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SC#2 (c): unset env var raises ConfigError naming the env var."""
    yml = (
        "local: {kind: ollama, base_url: http://x, model: m}\n"
        "judge: {model: j}\n"
        "frontier:\n"
        "  api_key_env: CORETHREAD_DEFINITELY_UNSET_77777\n"
        "  base_url: https://api.openai.com/v1\n"
        "  model: gpt-4o\n"
        "  max_tokens: 100\n"
        "routing: {threshold: 0.7, constraint_prompt: x}\n"
    )
    p = tmp_path / "envless.yaml"
    p.write_text(yml)
    monkeypatch.delenv("CORETHREAD_DEFINITELY_UNSET_77777", raising=False)
    with pytest.raises(ConfigError) as exc_info:
        load_config(p)
    # Error message names the env var so operators know what to set.
    assert "CORETHREAD_DEFINITELY_UNSET_77777" in str(exc_info.value)


# ---------------------------------------------------------------------------
# CFG-04 / CFG-05 — threshold + constraint_prompt are configurable
# ---------------------------------------------------------------------------


def test_threshold_default_and_override(tmp_path: Path, valid_yaml_path: Path) -> None:
    """CFG-04: threshold defaults to 0.7 and is overridable in YAML."""
    cfg = load_config(valid_yaml_path)
    assert cfg.routing.threshold == 0.7
    # Override
    p = tmp_path / "override.yaml"
    p.write_text(valid_yaml_path.read_text().replace("threshold: 0.7", "threshold: 0.85"))
    cfg2 = load_config(p)
    assert cfg2.routing.threshold == 0.85


def test_constraint_prompt_configurable(tmp_path: Path, valid_yaml_path: Path) -> None:
    """CFG-05: routing.constraint_prompt is configurable via YAML."""
    p = tmp_path / "cp.yaml"
    p.write_text(valid_yaml_path.read_text().replace("Be concise.", "Custom prompt text."))
    cfg = load_config(p)
    assert cfg.routing.constraint_prompt == "Custom prompt text."


def test_judge_prompt_configurable(tmp_path: Path, valid_yaml_path: Path) -> None:
    """judge.prompt is configurable via YAML while keeping a default."""
    p = tmp_path / "judge-prompt.yaml"
    p.write_text(
        valid_yaml_path.read_text().replace(
            "judge:\n  model: qwen2.5:7b\n",
            "judge:\n  model: qwen2.5:7b\n  prompt: Custom judge prompt.\n",
        ),
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.judge.prompt == "Custom judge prompt."


# ---------------------------------------------------------------------------
# CFG-02 — inlined api_key in YAML is rejected (loader-layer guard, T-01-18)
# ---------------------------------------------------------------------------


def test_api_key_field_in_yaml_rejected(tmp_path: Path) -> None:
    """CFG-02: load_config rejects an inlined `frontier.api_key` field BEFORE
    env-var injection (loader-layer guard from PLAN-04). The api_key_env pattern
    is the only permitted path to bind the key."""
    yml = (
        "local: {kind: ollama, base_url: http://x, model: m}\n"
        "judge: {model: j}\n"
        # api_key MUST be rejected even when api_key_env is also present
        "frontier:\n"
        "  api_key_env: OPENAI_API_KEY\n"
        "  api_key: sk-inline-leak-1234567890ABCDEF\n"
        "  base_url: https://api.openai.com/v1\n"
        "  model: gpt-4o\n"
        "  max_tokens: 100\n"
        "routing: {threshold: 0.7, constraint_prompt: x}\n"
    )
    p = tmp_path / "inline.yaml"
    p.write_text(yml)
    with pytest.raises(ConfigError) as exc_info:
        load_config(p)
    # Error message names the field and points to the correct pattern
    msg = str(exc_info.value)
    assert "frontier.api_key" in msg, msg
    assert "api_key_env" in msg, msg
    # Defense: the loader's error message itself does NOT echo the leaked key
    assert "sk-inline-leak" not in msg, f"LEAK in ConfigError msg: {msg}"
