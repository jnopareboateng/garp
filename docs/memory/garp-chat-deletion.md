---
name: garp-chat-deletion
description: 'Postgres fences native chat writes and queued replays after a committed session deletion'
type: contract
updated: 2026-11-23
---
AgentOS owns single and bulk deletion. Garp writes an ID-only marker and removes queue payloads in the same Postgres transaction. Owner-scoped queue-key digests prevent replay without the old session ID for 24 hours.

Native check occurrences reference the session through a Postgres foreign key. Deletion cascades baselines and outcomes and fences late admissions. Legacy occurrences have no native reference.

Chat deletion removes approvals in the same transaction and prevents late approval writes.

The deletion trigger captures native AgentOS run IDs in permanent `garp_deleted_runs` tombstones before the session's foreign-key cascade. Replay and tail reads fail closed when a tombstone is present, and raise when the deletion authority is unavailable so a client can retry. A pooled Postgres session-row lock orders acknowledged event publication against deletion. Redis can still finish a command whose network outcome is unknown.

**Why:** A native run can save after deletion. Job and schedule admission lock source rows to order writes against deletion. Schedule cleanup runs after commit because an admitted claim can hold the schedule lock.
**How to apply:** Use native session APIs. Preserve the marker and versioned migration across replicas and restores. Retry locked schedules and attempts at startup and during queue maintenance. Cancel dispatched work before deleting when required. New child admissions lock and check their parent.
**Evidence:** `tests/test_session_deletion.py`, `tests/test_replay_retention.py`, `tests/test_scheduled_prompts.py::test_deleted_chat_schedules_retry_cleanup_without_waiting_on_a_claim`, and `tests/test_native_checks.py::test_native_check_observations_follow_chat_deletion`.
