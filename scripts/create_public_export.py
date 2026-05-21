"""Create a curated source tree for a clean public CoreThread repository.

This copies tracked and non-ignored source files into a new directory while excluding private
planning material, local configs, run artifacts, and caches. It does not create
or push a GitHub repository; use the output directory as the input for an orphan
public-history commit.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_PATTERNS = (
    ".planning/**",
    ".claude/**",
    ".codex-run/**",
    "outputs/**",
    "config.yaml",
    "config.*.yaml",
    "CLAUDE.md",
    "*.log",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.webp",
    "*.pptx",
    "**/__pycache__/**",
    "**/*.pyc",
    "frontend/node_modules/**",
    "frontend/dist/**",
    "frontend/*.tsbuildinfo",
)


def git_source_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def excluded(rel: str) -> bool:
    normalized = rel.replace("\\", "/")
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in EXCLUDE_PATTERNS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="empty directory to receive the public tree")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        print(f"Output directory is not empty: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for rel in git_source_files():
        if excluded(rel):
            skipped += 1
            continue
        src = ROOT / rel
        dst = output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"Copied {copied} source files to {output}")
    print(f"Skipped {skipped} private or generated files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
