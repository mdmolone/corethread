"""Phase 9 / D-14 / D-19 #5 — /v1/_traces_event_schema documentation stub.

Asserts:
- GET /v1/_traces_event_schema returns 410 (FastAPI default HTTPException shape)
- app.openapi() includes the TraceEvent component schema with:
  * 15 properties (the RequestTrace 15-field shape verbatim)
  * additionalProperties: false (the extra='forbid' OpenAPI projection)
  * No forbidden body fields (prompt, message, content, body, request_body, response_body)

Phase 10's `pnpm gen:types` consumes this schema via openapi-typescript;
drift here surfaces as a TypeScript build error in Phase 10 (Pitfall 27 defense).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from corethread.api_ui import router as ui_router

EXPECTED_TRACE_EVENT_FIELDS = {
    "request_id",
    "selected_local_model",
    "judge_model",
    "frontier_model",
    "confidence_score",
    "pivoted",
    "local_latency_ms",
    "judge_latency_ms",
    "frontier_latency_ms",
    "input_tokens",
    "output_tokens",
    "frontier_cost_est",
    "judge_parse_failed",
    "pivot_reason",
    "local_error_class",
}

EXPECTED_ROUTE_ACTIVITY_FIELDS = {
    "request_id",
    "stage",
    "selected_local_model",
    "judge_model",
    "frontier_model",
    "pivoted",
    "confidence_score",
    "error_class",
    "ts_ms",
}


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(ui_router)
    return app


def test_traces_event_schema_endpoint_returns_410() -> None:
    """D-14 — the endpoint exists for OpenAPI publication only; raise 410 on GET."""
    app = _build_test_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/v1/_traces_event_schema")
        assert resp.status_code == 410, resp.text
        # FastAPI default HTTPException shape is {"detail": "..."}; we accept it.
        body = resp.json()
        assert "detail" in body
        assert "/v1/traces/stream" in body["detail"]


def test_app_openapi_publishes_trace_event_schema() -> None:
    """Pitfall 27 — TraceEvent appears in app.openapi()['components']['schemas'].

    Asserts:
    - TraceEvent schema present under components/schemas
    - additionalProperties is False (extra='forbid' projection)
    - All 15 expected RequestTrace fields are in 'properties'
    - No forbidden body fields (prompt, message, content, body, request_body, response_body)
    """
    app = _build_test_app()
    spec = app.openapi()

    schemas = spec.get("components", {}).get("schemas", {})
    assert "TraceEvent" in schemas, f"TraceEvent missing; have: {list(schemas)}"

    trace_schema = schemas["TraceEvent"]
    # extra='forbid' projects to additionalProperties: False in the OpenAPI schema
    assert trace_schema.get("additionalProperties") is False, (
        f"TraceEvent additionalProperties = {trace_schema.get('additionalProperties')!r} "
        f"(expected False — D-14 schema lock + Pitfall 27)"
    )

    actual_fields = set(trace_schema.get("properties", {}).keys())
    assert actual_fields == EXPECTED_TRACE_EVENT_FIELDS, (
        f"Field set drift: {actual_fields ^ EXPECTED_TRACE_EVENT_FIELDS}"
    )

    # Pitfall 18 — assert NONE of the forbidden body fields appear
    forbidden = {"prompt", "message", "content", "body", "request_body", "response_body"}
    assert forbidden.isdisjoint(actual_fields), (
        f"FORBIDDEN body fields appeared in TraceEvent schema: {forbidden & actual_fields}"
    )

    # pivot_reason carries the Literal enum constraint (Phase 10 type-gen consumer)
    pr = trace_schema["properties"]["pivot_reason"]
    # Pydantic 2 emits Literal as enum: list-of-strings
    if "enum" in pr:
        assert set(pr["enum"]) == {
            "none",
            "low_score",
            "local_truncated",
            "local_error",
            "judge_error",
        }


def test_route_activity_event_schema_endpoint_returns_410() -> None:
    app = _build_test_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/v1/_route_activity_event_schema")
        assert resp.status_code == 410, resp.text
        body = resp.json()
        assert "/v1/route/stream" in body["detail"]


def test_app_openapi_publishes_route_activity_schema() -> None:
    app = _build_test_app()
    spec = app.openapi()

    schemas = spec.get("components", {}).get("schemas", {})
    assert "RouteActivityEventView" in schemas

    route_schema = schemas["RouteActivityEventView"]
    assert route_schema.get("additionalProperties") is False
    actual_fields = set(route_schema.get("properties", {}).keys())
    assert actual_fields == EXPECTED_ROUTE_ACTIVITY_FIELDS

    forbidden = {"prompt", "message", "content", "body", "request_body", "response_body"}
    assert forbidden.isdisjoint(actual_fields)
