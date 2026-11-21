---
name: garp-host-session-revisions
description: 'Postgres revision fences preserve hosted session edits against stale native saves'
type: contract
updated: 2026-11-21
---
Hosted PATCH and rename operations record per-field revisions in session metadata. The Postgres
`garp_host_session_revision_fence` trigger rejects stale native snapshots for those fields while
allowing current native updates and preserving unrelated fields. Direct SQL writers bypass this
contract.

**Why:** AgentOS workers can save an in-memory session after a host edit. A database trigger keeps
the host's acknowledged fields authoritative across the native persistence paths.
**How to apply:** Use the hosted PATCH and rename routes and keep the trigger installed. Test stale
worker saves across the native persistence paths. Treat direct SQL and native-native last-write
ordering as outside this guarantee.
**Evidence:** `garp/host_sessions.py`, `tests/test_host_session_metadata.py::test_worker_saves_preserve_host_edits`.
