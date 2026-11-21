# Garp extensions

Configure extensions when embedding AgentOS. A hosted deployment maps tenant IDs to `Extensions` configurations. The verified JWT subject selects the tenant. The default local workspace has no configured extensions. An operator embedding local mode can pass a single `Extensions` instance.

```python
from agno.skills import Skills
from agno.skills.loaders import LocalSkills
from agno.tools.mcp import MCPTools
from garp.agentos import create_agentos_app
from garp.extensions import Extensions, RemoteMCP

app = create_agentos_app(
    dsn,
    host_auth=host_auth,
    extensions={
        "org-1": Extensions(
            skills=Skills([LocalSkills("/srv/garp/org-1/skills")]),
            mcp=(MCPTools(name="catalog", url=catalog_url, include_tools=["lookup_item"]),),
            remote_mcp=(
                RemoteMCP(
                    name="account",
                    url=account_url,
                    include_tools=("lookup_account",),
                    headers_for_principal=account_headers,
                ),
            ),
            retrieve=search_org_1_knowledge,
            approval_tools=(record_review_note,),
        )
    },
)
```

| Extension | Grant | Behavior |
|---|---|---|
| Skills | `skills:read` | The agent loads skill instructions and references with Agno's read-only tools. |
| MCP | `mcp:catalog:run` | Each tenant supplies an Agno transport and its allowed tool names. A failed connection leaves other tenants available. Garp reconnects a lost session and offers its tools on later runs. |
| Knowledge | `knowledge:read` | Agno's native search tool calls `retrieve(query, num_documents, user_id)`. Return documents with their source names and content. The host owns retrieval, document permissions, and storage. This path requires no embedding service. |
| Action | `actions:record_review_note:run` | Garp adds native Agno approval to the configured function. Hosted actions appear only on durable text runs. |

Use `RemoteMCP` for hosted, user-specific HTTP credentials. Its endpoint and tool allowlist are host configuration, never request input. Endpoints require HTTPS, except literal localhost and loopback addresses; redirects are rejected. `headers_for_principal` receives the verified tenant-qualified principal and returns HTTP headers; it must check the principal's entitlement and keep its own logs free of secrets. Garp checks the native MCP scope before resolving credentials. It discovers tools using a fresh client and closes it immediately, then Agno opens a temporary client for each tool call. Credentials stay in runtime memory and are not included in the tool schema or arguments. The remote service must enforce its own user permissions and must not return credentials in tool results, which are visible to the model. Credential rotation takes effect on the next run. This integration uses streamable HTTP.

For shared tenant connections, use `mcp` only for tools that do not expose user-specific state across calls. Extension code runs with the operator's authority. The primary Garp agent receives extensions. Add `extension_scopes=("skills:read", "knowledge:read", "mcp:catalog:run")` to a `Specialist` to opt that specialist into named capabilities. Garp intersects those scopes with the parent's verified grant when it queues the child. Inline Team members remain model-only. Approval-gated specialist actions are not enabled because their native continuation route is bound to the primary agent.

## Resolve a hosted approval

1. The requester starts a durable text run with `background=true` and the action grant. Agno stores the paused run and its approval record in Postgres.
2. The requester lists the record with `GET /approvals?run_id=RUN_ID` using `approvals:read`. In a connected CLI, `/approvals` lists records for the paused chat.
3. A reviewer with `approvals:write` discovers pending decisions at `GET /garp/approvals`, inspects `GET /garp/approvals/ID` and submits `POST /garp/approvals/ID/resolve` with `{"status":"approved"}` or `{"status":"rejected"}`. This grant enables reviewing other users' requests and seeing their proposed tool arguments within the same tenant.
4. The requester posts to `/garp/approvals/ID/continue` or enters `/continue ID` in the CLI. The new token must carry the paused run's original grants. AgentOS queues continuation of the existing run. Use `/history` or the session runs API for the result.

Resolution uses Agno's compare-and-set update. Continuation requires an approval from the current pause and admits one request at a time per run across replicas. Deleting the source chat removes its approval records and makes them unavailable to reviewers. A rejecting reviewer stops the protected action and the run can still explain that outcome. Use an idempotency key accepted by the downstream service for actions that can be retried after process failure.

Official references [Skills](https://docs.agno.com/skills/loading-skills) · [MCP](https://docs.agno.com/tools/mcp/overview) · [Custom retrieval](https://docs.agno.com/knowledge/concepts/search-and-retrieval/custom-retriever) · [Approvals](https://docs.agno.com/hitl/approval)
