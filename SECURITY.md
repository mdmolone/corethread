# Security Policy

## Supported Surface

CoreThread is a local, single-user service. The supported public surface is:

- `POST /v1/chat/completions`
- `GET /v1/models`
- Local UI and observability endpoints under `/v1`

CoreThread does not currently provide authentication, authorization, TLS
termination, multi-user isolation, or public internet deployment hardening.

## Secrets

Do not put API keys in `config.yaml`. Use `api_key_env` to name an environment
variable:

```yaml
frontier:
  api_key_env: OPENAI_API_KEY
```

The `/v1/config` endpoint returns the environment variable name, not the resolved
secret value. CI runs a secret scan, and `scripts/public_audit.py` checks clean
public exports for common key, bearer-token, personal-path, and private-host leaks.

## Prompt And Response Bodies

Trace and stats events are body-free by default. Transcript capture is disabled
unless explicitly enabled:

```yaml
privacy:
  capture_transcripts: true
  transcript_max: 25
```

When enabled, CoreThread keeps prompt and response bodies in memory for local
debugging. Treat that mode as sensitive.

Audit logging under `controls.audit_enabled` is also disabled by default. Audit
events are metadata-only unless `controls.audit_include_request_body` is set to
`true`; enabling request bodies makes the audit log sensitive local data.

## Reporting Issues

For public releases, open a GitHub security advisory or contact the maintainer
privately before posting a suspected secret leak or vulnerability in a public issue.
