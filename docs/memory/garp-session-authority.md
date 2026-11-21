---
name: garp-session-authority
description: 'AgentOS owns native chat history and Garp owns due times and committed job observations'
type: decision
updated: 2026-11-20
---
AgentOS owns session and run history for the default CLI. AgentOS also owns hosted cron recurrence, claims, and attempt history. Garp owns one-shot occurrence leases, due times, and observations. The worker commits its observation before appending it to the native chat with a deterministic run ID.

**Why:** A stale lease must never replace a winning result. Replaying a committed observation after a crash must also leave one visible native run.
**How to apply:** Use AgentOS persistence for turns. Derive hosted check IDs from the native schedule ID and its stored `next_run_at`. Keep that time when shutdown interrupts a dispatch, and fence check admission with the native claim and enabled state in the insert transaction. Cron coalesces downtime into the stored overdue tick, with the deadline measured from that time. A finished attempt budget advances to the next future run. Publish only committed results, and replay with the same occurrence ID and evidence. Keep the source legacy rows available for recovery during the cutover.
**Evidence:** The native scheduler, hosted identity and scheduling, import, and stale-lease integration tests.

Enqueue scheduled instructions in existing chats when they become due. Agno 3.2.1 sequences a future `available_at` ticket by acceptance order, so it can block immediate turns in that chat. The Postgres future-ticket probe verified this with two sessions.

A one-shot prompt retains its occurrence ID and deadline when re-enabled after dispatch failure. Admission or expiry closes it. Pause and delete affect future admissions. Append admitted runs as pending with the native append-if-absent primitive so AgentOS can verify ownership on cancellation before execution. The Postgres scheduling tests cover recovery and cancellation.
