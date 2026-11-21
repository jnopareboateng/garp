---
name: garp-host-identity
description: 'Hosted AgentOS uses one explicit JWT trust root and a tenant-qualified subject'
type: contract
updated: 2026-11-20
---
Hosted Garp uses the host JWT's verified `sub` as the session, memory, and check owner. The subject must contain both the tenant and user. Local operator access uses a separate app configuration.

**Why:** Agno 3.2.1 merges ambient verification keys into explicit configuration and its admin scope removes tenant filtering. Native authorization handles supported route and resource scopes. Garp requires expiry, refuses tenant admin bypass, and limits hosted access to routes with an audited ownership and model policy.
**How to apply:** Keep issuer, audience, and verification key explicit. Use an RSA public key when the host holds the private signing key. Verify each new native route under two tenant subjects before adding it to the hosted surface.
**Evidence:** The Postgres host integration tests and the pinned AgentOS JWT middleware.
