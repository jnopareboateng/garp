---
name: garp-chat-titles
description: 'The Agno executor proposes three-word titles and Garp saves them with the turn'
type: decision
updated: 2026-11-20
---
Garp stores chat titles with its own session state and caps generated titles at three words. The title keeps its original topic across retries.

**Why:** Agno 3.2.1's built-in session naming prompts for five words and writes through Agno's session store. Garp's executor works with Garp-owned history and no Agno database.
**How to apply:** Use the existing configured Agno model route to propose the title. Validate it at the Garp boundary and commit it with the reply under the turn lease.
**Evidence:** The title transport and session replay tests.
