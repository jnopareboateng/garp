---
name: garp-session-authority
description: 'Garp owns committed session history and one-shot results across runtime attempts'
type: decision
updated: 2026-11-20
---
Garp owns the committed session history and each scheduled occurrence's public result. Runtime state is useful for execution or recovery only when later turns cannot see writes from an expired attempt.

**Why:** An agent runtime can finish after Garp has revoked its lease. Publishing its transcript independently would bypass the same checks that protect the occurrence result.
**How to apply:** Use the event commit path when introducing an executor or persistent checkpoint store. Verify stale-attempt rejection before allowing a new store to supply context.
**Evidence:** The process-death and stale-turn regressions in the harness and store tests.
