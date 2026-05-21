# Contributing

Thanks for taking a look at CoreThread.

## Development Setup

```bash
uv sync
pnpm --dir frontend install --frozen-lockfile
cp config.yaml.example config.yaml
```

Set the API key environment variable required by your chosen example config.
Never commit local `config.yaml`, logs, outputs, screenshots, or generated run
artifacts.

## Quality Gates

Run the full gate before opening a pull request:

```bash
uv run ruff check
uv run ruff format --check .
uv run mypy .
uv run pytest -q
pnpm --dir frontend lint
pnpm --dir frontend test
pnpm --dir frontend build
```

## Pull Request Scope

Keep changes focused. CoreThread is intentionally narrow: local-first chat
completion routing with judge-based frontier fallback. Broader API surfaces such
as embeddings, images, audio, hosted auth, and multi-tenant deployment need their
own design discussion before implementation.

## Security Hygiene

Before publishing release branches or public-ready pull requests, run:

```bash
uv run python scripts/public_audit.py
```

Run `scripts/public_audit.py --history` only on the clean public export history.
The private development archive intentionally contains internal planning material.
