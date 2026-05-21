import asyncio

import pytest

from corethread.route_activity import (
    RouteActivityBus,
    emit_route_activity,
    route_activity_event,
    set_route_activity_bus,
)


@pytest.mark.asyncio
async def test_route_activity_bus_publishes_body_free_events() -> None:
    bus = RouteActivityBus()
    event = route_activity_event(
        request_id="req-1",
        stage="judge_started",
        selected_local_model="local",
        judge_model="judge",
        frontier_model="frontier",
    )

    async with bus.subscribe(replay=True) as q:
        bus.publish_nowait(event)
        received = await asyncio.wait_for(q.get(), timeout=1.0)

    assert received["request_id"] == "req-1"
    assert received["stage"] == "judge_started"
    assert not {
        "prompt",
        "message",
        "content",
        "request_body",
        "response_body",
    }.intersection(received)


@pytest.mark.asyncio
async def test_emit_route_activity_noops_without_bus() -> None:
    set_route_activity_bus(None)
    event = route_activity_event(
        request_id="req-no-bus",
        stage="received",
        selected_local_model="local",
        judge_model="judge",
        frontier_model="frontier",
    )

    emit_route_activity(event)


@pytest.mark.asyncio
async def test_emit_route_activity_tees_to_configured_bus() -> None:
    bus = RouteActivityBus()
    set_route_activity_bus(bus)
    event = route_activity_event(
        request_id="req-global",
        stage="frontier_started",
        selected_local_model="local",
        judge_model="judge",
        frontier_model="frontier",
        pivoted=True,
    )

    try:
        async with bus.subscribe(replay=True) as q:
            emit_route_activity(event)
            received = await asyncio.wait_for(q.get(), timeout=1.0)
    finally:
        set_route_activity_bus(None)

    assert received["request_id"] == "req-global"
    assert received["pivoted"] is True
