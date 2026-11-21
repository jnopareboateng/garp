# Scheduling

Pass a job adapter as `jobs` when creating the hosted AgentOS app. The adapter implements `authorize(owner, job_ref)` and `observe(owner, job_ref, now)`. The owner is the tenant-qualified JWT subject. Implement `authorize` against current grants, including account or resource revocation.

## Schedule a check in a chat

The session must belong to the caller and use the `garp` agent. The following request requires `checks:write` and `schedules:write`.

```http
POST /garp/sessions/SESSION_ID/recurring-checks
Authorization: Bearer HOST_JWT
Content-Type: application/json

{
  "name": "Watch batch progress",
  "cron_expr": "*/15 * * * *",
  "timezone": "UTC",
  "job_ref": "batch",
  "delay_seconds": 60,
  "deadline_seconds": 300,
  "max_retries": 1,
  "retry_delay_seconds": 10
}
```

Cron uses five fields and an IANA timezone. The tick establishes the due time and deadline. Garp observes the authorized job at admission to establish a baseline and checks it again at dispatch. A check that is already past its deadline records an expired outcome. The same payload fields for job, delay, and deadline also work on the one-shot `POST /garp/sessions/SESSION_ID/checks` route.

| Operation | Endpoint | Required scope |
|---|---|---|
| List or read schedules | `GET /schedules` or `GET /schedules/ID` | `schedules:read` |
| Inspect attempts | `GET /schedules/ID/runs` | `schedules:read` |
| Pause | `POST /schedules/ID/disable` | `schedules:write` |
| Resume | `POST /garp/recurring-checks/ID/enable` | `schedules:write`, `checks:write` |
| Delete | `DELETE /schedules/ID` | `schedules:delete` |
| Inspect an admitted check | `GET /garp/checks/CHECK_ID` | `checks:read` |

An attempt marked `success` returns `output.check_id` for a newly admitted or previously committed check. Check status records the observation outcome and delivery to the original chat. The job adapter and current session ownership are checked when admitting a tick and before reading the job at dispatch.

Deleting the chat removes its admitted checks, baselines, and outcomes in the same Postgres transaction. A check that finishes its baseline observation after deletion returns 404 during admission. An adapter already reading evidence can finish its read, and the worker discards the result.

The first startup with this change rewrites the check table and removes native checks whose chat is missing. Chat writes can wait for the migration to finish. Restore the AgentOS session and Garp check tables together because native check rows now reference their chat.

Agno coalesces a missed backlog into the oldest stored overdue tick, then evaluates the next future cron time. Hosted checks default to one retry after 60 seconds. A retry or recovered scheduler claim reuses the same check ID for that tick. After exhausting retries, the failed attempts remain in history and cron advances. Distinct ticks may produce checks with overlapping due windows. Pause and delete take effect at admission. Cancel a previously admitted check through `DELETE /garp/sessions/SESSION_ID/checks/CHECK_ID`. [AgentOS scheduler](https://docs.agno.com/agent-os/scheduler/overview)

## Schedule a prompt

For a local service, start `garp serve` and export the same `GARP_TOKEN` for the client. A hosted service enables prompts by passing `scheduled_scopes(owner)` to `create_agentos_app`. Supply a synchronous lookup of the tenant-qualified subject's current grants, with a bounded timeout. A revoked identity should return an empty collection. Garp saves the scope names held at creation and intersects them with live grants again at queue admission and execution. The schedule contains no bearer token.

```sh
garp schedule-prompt "Check my batch progress" --name "Batch follow-up" --session SESSION_ID --after 3600
garp schedule-prompt "Summarize progress" --name "Daily summary" --cron '0 9 * * *' --timezone UTC
```

Inside a connected chat, use `/schedule 3600 Check my batch progress` for a one-shot follow-up in that chat.

The API is `POST /garp/prompt-schedules`. Send `name`, `message`, and one of `after_seconds` or `cron_expr`. Optional fields include `session_id`, `timezone`, `deadline_seconds` measured after the due time, and `options` with `model_id`, `effort`, and `web_search`. A missing session opens a new chat for each occurrence. A supplied session must belong to the caller. Omitting `options` uses that chat's saved settings. The schedule pins its options and validates current provider policy at execution.

Hosted creation and resume require `schedules:write` and `agents:garp:run`. Resume prompts with `POST /garp/prompt-schedules/ID/enable`, which preserves one-shot due times. Use the same native list, attempts, pause and delete routes above. A successful dispatch returns `session_id` and `run_id` in its attempt. Inspect the outcome through `GET /agents/garp/runs/RUN_ID?session_id=SESSION_ID` with `agents:read` and the run grant, or reopen the chat. Pause and delete prevent future admissions. Cancel an admitted run with `POST /agents/garp/runs/RUN_ID/cancel?session_id=SESSION_ID`.

A scheduled prompt joins the native queue when due. The queue orders it behind earlier work in the same chat. Missed cron ticks follow the same coalescing policy as job checks. A one-shot prompt remains due after downtime. If pre-admission retries run out, re-enable the prompt to retry the same occurrence against its original deadline. An admitted or expired occurrence is closed. Retries reuse the same scheduled run ID. Tool calls and approvals follow the agent's configured permissions at worker start.

Deleting a chat removes its linked prompt schedules, recurring check definitions, and native run schedules with an explicit `session_id`, along with their attempt history. If cleanup encounters a locked schedule or a database error, the definition and attempts can remain visible until the next service startup or native queue maintenance pass, which runs about hourly. When creating schedules directly through AgentOS, use the canonical `/garp/sessions/SESSION_ID/checks` or `/agents/AGENT_ID/runs` path and the chat ID returned by Garp. After upgrading, startup also removes schedules and attempts for chats with an existing deletion marker. Schedules that open a fresh chat on each occurrence remain available.
