# Public Release Process

The private development repository should remain an internal archive. Do not make
that history public directly because it contains tracked internal planning files.

## Clean History Strategy

Use one of these approaches:

1. Rename the existing private repository to `corethread-private`, then create a
   fresh public `corethread` repository from a curated source tree.
2. Create a new public repository with an orphan first commit from a clean export.

The public history should start at the curated tree. It should not include the
private planning history.

To create a curated local export directory:

```bash
uv run python scripts/create_public_export.py ../corethread-public-export
```

Then initialize the public repository from that output directory.

## Excluded From Public Tree And History

- `.planning/`
- `.claude/`
- `.codex-run/`
- `outputs/`
- `config.yaml`
- local config variants
- caches and bytecode
- logs
- local screenshots
- generated presentations
- private run artifacts

`CLAUDE.md` is also private-process material and should be excluded from the
public export.

## Audit Commands

Run on the curated public tree:

```bash
uv run python scripts/public_audit.py
```

Run on the clean public repository after creating the orphan history:

```bash
uv run python scripts/public_audit.py --history
```

Also run the normal gates:

```bash
uv run ruff check
uv run ruff format --check .
uv run mypy .
uv run pytest -q
pnpm --dir frontend lint
pnpm --dir frontend test
pnpm --dir frontend build
```

## Final Checklist

- Repository visibility is public only after the clean-history export is created.
- `.planning/`, `.claude/`, `.codex-run/`, `outputs/`, and `config.yaml` are absent
  from tracked files.
- `scripts/public_audit.py --history` passes on the public repo.
- Gitleaks passes on the public repo.
- Docs use generic hosts such as `localhost`, `host.example`, or `lmstudio.local`.
- Transcript capture remains opt-in.
- The MIT copyright holder string in `LICENSE` is acceptable.
