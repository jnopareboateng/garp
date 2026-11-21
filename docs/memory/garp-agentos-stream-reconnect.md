---
name: garp-agentos-stream-reconnect
description: 'AgentOS SSE reconnect needs a background stream and a database on Agno 3.2.1'
type: contract
updated: 2026-11-20
---
Agno 3.2.1 can reconnect a client to native AgentOS SSE when the original run is submitted with `stream=true` and `background=true` against a database-backed component. The installed `AgentOSClient` can start a stream and lacks a resume method.

**Why:** The native `/resume` route rejects registered factories in 3.2.1. A real Postgres and Uvicorn test recovered a factory stream through AgentOS's durable queue by reusing the original submission and idempotency key. Only one native job and one turn inference were recorded.
**How to apply:** Keep the last indexed event and the original submission key. Prefer the native resume endpoint. Reattach by key only for a confirmed durable, queueable submission to the same service. Test the actual event store, authentication and worker configuration before promising replay across replicas or restarts. [Agno reconnect example](https://docs.agno.com/examples/agent-os/run-lifecycle/sse-reconnect)
