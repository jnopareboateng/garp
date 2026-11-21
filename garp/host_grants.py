"""Recheck optional host grants before a durable factory builds a component."""

import asyncio
from collections.abc import Collection, Mapping

from agno.factory import FactoryPermissionError, QueuedRequest
from agno.os.scopes import AgentOSScope, has_required_scopes, parse_scope

from garp.host_auth import is_host_principal


async def check_queued_grants(ctx, resolver):
    if resolver is None or not isinstance(ctx.request, QueuedRequest):
        return
    if not is_host_principal(ctx.user_id):
        raise FactoryPermissionError("A verified host principal is required")
    try:
        current = await asyncio.to_thread(resolver, ctx.user_id)
    except Exception:  # noqa: BLE001 - host callback failures must fail closed
        raise FactoryPermissionError("Current host grant is unavailable") from None
    if (
        not isinstance(current, Collection)
        or isinstance(current, (str, bytes, Mapping))
        or any(not isinstance(scope, str) for scope in current)
        or AgentOSScope.ADMIN.value in current
    ):
        raise FactoryPermissionError("Current host grant is unavailable")
    for scope in ctx.trusted.scopes:
        parsed = parse_scope(scope)
        if scope in current:
            continue
        if (
            parsed.scope_type == "per_resource"
            and not parsed.is_wildcard_resource
            and has_required_scopes(list(current), [scope], parsed.resource, parsed.resource_id)
        ):
            continue
        raise FactoryPermissionError("Queued run permissions have changed")
