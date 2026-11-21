---
name: garp-transcript-search
description: 'Transcript search reads completed user-visible text from owned AgentOS sessions'
type: contract
updated: 2026-11-20
---
Garp searches completed top-level prompts and replies in AgentOS storage. Search uses the tenant-qualified owner and `sessions:read`, or the local workspace owner. It keeps no separate index. AgentOS merges any legacy run data with current run rows.

**Why:** Agno 3.2.1 session filters search metadata and the built-in past-chat tool returns bounded recent previews. Neither can find a phrase in an arbitrary older conversation.
**How to apply:** Keep native title matching and session reads. Scope the query at storage, use only visible text, and keep tool output and reasoning out of search. The CLI searches agent chats it can reopen. The hosted API and the model tool also search Team chats. Deletion makes the source invisible to search. Copies already quoted in another conversation follow that conversation's lifecycle.
**Evidence:** `tests/test_scoped_memory.py` and `tests/test_transcript_search.py` exercise hosted access, the model tool, the CLI, legacy runs, and source deletion on Postgres.
