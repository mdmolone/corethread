from __future__ import annotations

from pathlib import Path

from scripts import public_audit
from scripts.create_public_export import excluded

ROOT = Path(__file__).resolve().parents[1]


def test_powershell_api_key_helper_honors_requested_name() -> None:
    script = (ROOT / "scripts" / "set-openai-api-key.ps1").read_text(encoding="utf-8")

    assert "$env:OPENAI_API_KEY =" not in script
    assert '[Environment]::SetEnvironmentVariable($Name, $apiKey, "User")' in script
    assert '[Environment]::SetEnvironmentVariable($Name, $apiKey, "Process")' in script
    assert 'Set-Item -Path "Env:$Name" -Value $apiKey' in script


def test_public_audit_allows_api_key_env_configuration() -> None:
    inline_key_rule = next(rule for rule in public_audit.RULES if rule.name == "inline_api_key")

    assert inline_key_rule.pattern.search("api_key_env: OPENAI_API_KEY") is None
    assert inline_key_rule.pattern.search("api_key: sk-test-real-looking-value") is not None
    assert public_audit.is_forbidden_path("config.yaml")
    assert not public_audit.is_forbidden_path("config.yaml.example")


def test_public_export_excludes_private_and_local_artifacts() -> None:
    assert excluded(".planning/ROADMAP.md")
    assert excluded(".claude/settings.json")
    assert excluded(".codex-run/session.json")
    assert excluded("outputs/corethread.jsonl")
    assert excluded("config.yaml")
    assert excluded("config.local.yaml")
    assert excluded("CLAUDE.md")
    assert not excluded("config.yaml.example")
    assert not excluded("README.md")
