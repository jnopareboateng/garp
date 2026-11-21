---
name: garp-delegation-boundary
description: 'Independent specialists use AgentOS durable runs with a reduced caller grant'
type: contract
updated: 2026-11-20
---
Garp uses native Team member execution for inline results. Independently resumable children use native AgentOS Agent runs on the durable queue. The handoff records the parent session and run, the child's IDs, and the verified owner. The parent can read the result from its own chat after queue retention removes the ticket.

Specialists opt into named Skills, Knowledge, and MCP scopes. The queued child receives only the intersection with the parent's verified grant and a bounded specialist run scope. Inline Team members remain model-only. Approval-gated specialist actions are disabled until their continuation is bound to the original specialist component.

**Why:** Local Team member runs share the team session. Remote members support separate sessions but need a bearer supplied at runtime. A queued parent no longer has that bearer. Submitting the child to the native queue with a reduced grant preserves separate history without storing or minting a host credential.
**How to apply:** Keep queue admission idempotent for one parent run, specialist and task. Consult both the specialist grant and the parent's identity before scheduling. A queued model-only specialist can be reclaimed once under the same run ID. Require idempotency for any future side-effecting child tool before allowing it on this retry path.
**Evidence:** The hosted delegation and delegated-extensions integration tests exercise owner isolation, reduced extension grants, result linkage, independent resume, cancellation, and stale lease reclaim.
