"""Host-configured extensions for the primary AgentOS chat."""

import asyncio
import ipaddress
import logging
import re
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx2
from agno.approval import approval
from agno.os.scopes import has_required_scopes
from agno.skills import Skills
from agno.tools.function import Function
from agno.tools.mcp import MCPTools
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from garp.host_auth import is_host_principal

logger = logging.getLogger(__name__)

_NATIVE_TOOL_NAMES = {
    "delegate_task",
    "read_specialist_result",
    "search_past_sessions",
    "search_chat_transcripts",
    "read_past_session",
    "get_skill_instructions",
    "get_skill_reference",
    "get_skill_script",
    "search_knowledge_base",
}


async def _wait_for_health_check(stop, interval):
    try:
        await asyncio.wait_for(stop.wait(), timeout=interval)
    except TimeoutError:
        return True
    return False


async def _maintain_connection(toolkit, first_attempt, stop):
    retrying = False
    while not stop.is_set():
        try:
            async with asyncio.timeout(toolkit.timeout_seconds + 1):
                connected = await toolkit.connect(force=True)
            if connected:
                if retrying:
                    logger.info("MCP extension %s reconnected", toolkit.name)
                retrying = False
                for function in toolkit.functions.values():
                    function.bound_toolkit = toolkit
                    function.source_toolkit = toolkit
                first_attempt.set()
                while await _wait_for_health_check(stop, 5):
                    if not await toolkit.is_alive():
                        break
        except Exception:  # noqa: BLE001 - A failed transport must leave other tenants running.
            if not retrying:
                logger.warning("MCP extension %s health check failed", toolkit.name)
            retrying = True
        finally:
            toolkit.functions.clear()
            first_attempt.set()
            try:
                await toolkit.close()
            except Exception:  # noqa: BLE001 - A cleanup failure must not stop other connections.
                logger.warning("MCP extension %s cleanup failed", toolkit.name)
        if not stop.is_set() and not retrying:
            logger.warning("MCP extension %s is unavailable", toolkit.name)
        retrying = True
        await _wait_for_health_check(stop, 2)


def _bound_remote_call(entrypoint, principal):
    async def call_remote(_agno_run_context=None, **kwargs):
        if _agno_run_context is None or _agno_run_context.user_id != principal:
            raise PermissionError("MCP principal mismatch")
        return await entrypoint(_agno_run_context=_agno_run_context, **kwargs)

    return call_remote


class _RemoteMCPTools(MCPTools):
    def _remote_client(self):
        def http_client(*, headers=None, auth=None, **kwargs):
            return httpx2.AsyncClient(
                headers=headers,
                auth=auth,
                follow_redirects=False,
                trust_env=False,
                timeout=kwargs.get("timeout") or httpx2.Timeout(30, read=300),
            )

        transport = StreamableHttpTransport(
            self.url, headers=self._merge_http_headers(), httpx_client_factory=http_client
        )
        return Client(transport, timeout=self.timeout_seconds, mode=self.protocol_mode)

    async def _connect(self):
        self._session_context = self._remote_client()
        self.session = await self._session_context.__aenter__()
        await self.initialize()

    @asynccontextmanager
    async def get_temporary_session_for_run(self, run_context=None, agent=None, team=None):
        async with self._remote_client() as session:
            yield session


def _validate_remote_url(url):
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        _ = parsed.port
        if not host or parsed.username is not None or parsed.password is not None:
            raise ValueError
        if parsed.query or parsed.fragment or "\\" in url or any(ord(char) <= 32 for char in url):
            raise ValueError
        httpx2.URL(url)
        if parsed.scheme == "https":
            return
        if parsed.scheme == "http" and (
            host == "localhost" or ipaddress.ip_address(host).is_loopback
        ):
            return
    except (TypeError, ValueError, httpx2.InvalidURL):
        pass
    raise ValueError("Remote MCP requires HTTPS, or HTTP on localhost or a loopback IP address")


@dataclass(frozen=True)
class RemoteMCP:
    """A host-owned HTTP endpoint with credentials resolved for each principal."""

    name: str
    url: str
    include_tools: tuple[str, ...]
    headers_for_principal: Callable[[str], dict[str, str]]
    tool_name_prefix: str | None = None


@dataclass(frozen=True)
class Extensions:
    skills: Skills | None = None
    mcp: tuple[MCPTools, ...] = ()
    retrieve: Callable | None = None
    approval_tools: tuple[Callable, ...] = ()
    remote_mcp: tuple[RemoteMCP, ...] = ()

    def __post_init__(self):
        if _NATIVE_TOOL_NAMES.intersection(fn.__name__ for fn in self.approval_tools):
            raise ValueError("Approval actions cannot use native tool names")
        for names in (
            [tool.name for tool in (*self.mcp, *self.remote_mcp)],
            [fn.__name__ for fn in self.approval_tools],
        ):
            if len(set(names)) != len(names) or any(
                not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name) for name in names
            ):
                raise ValueError("Extension names must be unique lowercase identifiers")
        if any(not tool.include_tools for tool in (*self.mcp, *self.remote_mcp)):
            raise ValueError("MCP extensions require explicit tool names")
        names = _NATIVE_TOOL_NAMES | {fn.__name__ for fn in self.approval_tools}
        for toolkit in (*self.mcp, *self.remote_mcp):
            if isinstance(toolkit, RemoteMCP):
                _validate_remote_url(toolkit.url)
                prefix = f"{toolkit.tool_name_prefix}_" if toolkit.tool_name_prefix else ""
                exposed = {prefix + name for name in toolkit.include_tools}
                if names.intersection(exposed):
                    raise ValueError(f"MCP extension {toolkit.name} has conflicting tool names")
                names.update(exposed)
                continue
            if toolkit.session is not None:
                raise ValueError("Configure an MCP transport for Garp to connect and maintain")
            if toolkit.header_provider is not None and not toolkit.refresh_connection:
                raise ValueError(
                    "Dynamic MCP headers require temporary sessions (refresh_connection=True)"
                )
            prefix = f"{toolkit.tool_name_prefix}_" if toolkit.tool_name_prefix else ""
            exposed = {prefix + name for name in toolkit.include_tools}
            if names.intersection(exposed):
                raise ValueError(f"MCP extension {toolkit.name} has conflicting tool names")
            names.update(exposed)

    @asynccontextmanager
    async def connected(self):
        stop = asyncio.Event()
        attempted = {toolkit.name: asyncio.Event() for toolkit in self.mcp}
        async with asyncio.TaskGroup() as tasks:
            for toolkit in self.mcp:
                tasks.create_task(_maintain_connection(toolkit, attempted[toolkit.name], stop))
            try:
                yield attempted
            finally:
                stop.set()

    async def for_run(self, ctx, hosted, *, approval_tools, mcp_attempted):
        scopes = list(ctx.trusted.scopes)

        def allowed(scope, kind=None, name=None):
            return not hosted or has_required_scopes(scopes, [scope], kind, name)

        tools, instructions = [], []
        for toolkit in self.mcp:
            if allowed("mcp:run", "mcp", toolkit.name) and toolkit.name in mcp_attempted:
                await mcp_attempted[toolkit.name].wait()
                if toolkit.initialized:
                    # The lifecycle task owns reconnection. Agno receives its native functions.
                    tools.extend(toolkit.functions.values())
                if not toolkit.functions:
                    instructions.append(f"The {toolkit.name} MCP extension is unavailable.")
        for remote in self.remote_mcp:
            if not hosted or not is_host_principal(ctx.user_id):
                continue
            if not allowed("mcp:run", "mcp", remote.name):
                continue
            # Resolve outside Agno's header provider, which logs provider exceptions.
            try:
                headers = await asyncio.wait_for(
                    asyncio.to_thread(remote.headers_for_principal, ctx.user_id), timeout=10
                )
                if not isinstance(headers, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
                ):
                    raise ValueError("Invalid headers")
            except Exception:  # noqa: BLE001 - Do not log credentials from host errors.
                instructions.append(f"The {remote.name} MCP extension is unavailable.")
                continue
            toolkit = _RemoteMCPTools(
                name=remote.name,
                url=remote.url,
                transport="streamable-http",
                include_tools=list(remote.include_tools),
                tool_name_prefix=remote.tool_name_prefix,
                refresh_connection=True,
                header_provider=lambda *, headers=headers: headers,
            )
            try:
                if await toolkit.connect() and toolkit.functions:
                    for function in toolkit.functions.values():
                        function.entrypoint = _bound_remote_call(function.entrypoint, ctx.user_id)
                        tools.append(function)
                else:
                    instructions.append(f"The {remote.name} MCP extension is unavailable.")
            finally:
                await toolkit.close()
        if approval_tools:
            tools.extend(
                approval(Function.from_callable(fn))
                for fn in self.approval_tools
                if allowed("actions:run", "actions", fn.__name__)
            )
        if self.skills is not None and allowed("skills:read"):
            # Agno's third skill tool can run scripts on the service host.
            tools.extend(
                tool for tool in self.skills.get_tools() if tool.name != "get_skill_script"
            )
            names = ", ".join(
                f"{item.name}: {item.description}" for item in self.skills.get_all_skills()
            )
            if names:
                instructions.append(
                    f"Use get_skill_instructions to load these installed skills: {names}"
                )

        retriever = self.retrieve if allowed("knowledge:read") else None
        return tools, instructions, retriever
