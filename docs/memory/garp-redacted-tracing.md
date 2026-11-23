---
name: garp-redacted-tracing
description: 'AgentOS trace storage is opt-in and keeps only allowlisted execution metadata'
type: contract
updated: 2026-11-23
---
Use native AgentOS traces and trace reads. Garp binds the instrumentation to the app's request and worker contexts and filters spans before the database writer. A verified hosted subject can read its own trace summaries and spans with `traces:read`.

**Why:** Agno 3.2.1 tracing is process-wide. A host OpenInference context can supply an unrelated trace owner. Native input and output masking can leave tool exception details in span status and tool arguments in logs.
**How to apply:** Use one tracing-enabled Garp app per process and the dedicated Garp trace tables. Keep execution metadata on the export allowlist and test any new span field using synthetic markers. Set a uniform `trace_retention_days` across replicas sharing those tables. Garp deletes traces and their spans together based on the latest span end. The host owns other logs and trace collectors. The local CLI and demo filter Agno's logger instances while the embedded service runs. Uvicorn resets disabled flags on startup and Agno adjusts log levels during runs, so the terminal uses scoped filters.
**Evidence:** The native OpenInference probe and the hosted Postgres trace integration test.
