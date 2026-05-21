"""Public-release audit for CoreThread.

Scans tracked files for secrets, personal paths, local hostnames, and files that
should not appear in a clean public repository. Use ``--history`` only on the
clean public export branch/repo, because the private development history
intentionally contains internal planning material.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]


RULES = [
    Rule("openai_key", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    Rule("bearer_token", re.compile(r"Authorization:\s*Bearer\s+\S+", re.IGNORECASE)),
    Rule("inline_api_key", re.compile(r"\bapi[_-]?key\s*:\s*['\"]?[^'\"\s]+", re.IGNORECASE)),
    Rule("personal_windows_path", re.compile(r"C:\\Users\\[^\\\s]+", re.IGNORECASE)),
    Rule("personal_unix_path", re.compile(r"/c/Users/[^/\s]+", re.IGNORECASE)),
    Rule("private_lmstudio_host", re.compile(r"\bnucllm\.mole\.com\b", re.IGNORECASE)),
]

FORBIDDEN_TRACKED_PATHS = (
    ".planning/",
    ".claude/",
    ".codex-run/",
    "outputs/",
    "config.yaml",
)

SYNTHETIC_SECRET_ALLOWLIST = re.compile(
    r"sk-(test|leak|marker|inline|stub|libleak|bearer|argleak|stleak|foo|your|replace|1234567890)",
    re.IGNORECASE,
)


def git_lines(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return [line for line in result.stdout.splitlines() if line]


def tracked_files() -> list[str]:
    return git_lines("ls-files")


def is_allowed(rule_name: str, text: str) -> bool:
    if (
        rule_name in {"openai_key", "inline_api_key", "bearer_token"}
        and SYNTHETIC_SECRET_ALLOWLIST.search(text) is not None
    ):
        return True
    if rule_name == "bearer_token":
        lower = text.lower()
        return any(
            marker in lower
            for marker in (
                "authorization: bearer *",
                "authorization: bearer ...",
                "authorization: bearer <something>",
                "authorization: bearer {_fake_bearer_key}",
                "authorization: bearer sk-...",
                "authorization: bearer sk-|",
            )
        )
    if rule_name == "inline_api_key":
        return "api_key: SecretStr" in text or "frontier.api_key:" in text
    if rule_name in {"personal_windows_path", "personal_unix_path"}:
        return "Rule(" in text and "Users" in text
    return False


def is_forbidden_path(rel: str) -> bool:
    normalized = rel.replace("\\", "/")
    for path in FORBIDDEN_TRACKED_PATHS:
        if path.endswith("/"):
            if normalized.startswith(path):
                return True
        elif normalized == path:
            return True
    return False


def scan_tree() -> list[str]:
    findings: list[str] = []
    for rel in tracked_files():
        if is_forbidden_path(rel):
            findings.append(f"forbidden tracked path: {rel}")
            continue

        path = ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            findings.append(f"cannot read {rel}: {exc.__class__.__name__}")
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            for rule in RULES:
                if rule.pattern.search(line) and not is_allowed(rule.name, line):
                    findings.append(f"{rel}:{line_no}: {rule.name}: {line.strip()[:180]}")
    return findings


def scan_history() -> list[str]:
    findings: list[str] = []
    commits = git_lines("rev-list", "--all")
    for commit in commits:
        paths = git_lines("ls-tree", "-r", "--name-only", commit)
        for rel in paths:
            if is_forbidden_path(rel):
                findings.append(f"history:forbidden tracked path: {commit}:{rel}")
    for rule in RULES:
        pattern = rule.pattern.pattern
        result = subprocess.run(
            ["git", "grep", "-n", "-I", "-E", pattern, *commits, "--"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in result.stdout.splitlines():
            if not is_allowed(rule.name, line):
                findings.append(f"history:{rule.name}: {line[:220]}")
    return findings


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="store_true", help="also scan all reachable commits")
    args = parser.parse_args()

    findings = scan_tree()
    if args.history:
        findings.extend(scan_history())

    if findings:
        print("Public audit failed:")
        for finding in findings:
            print(f"  - {finding}")
        return 1

    print("Public audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
