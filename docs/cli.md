# Garp CLI

## Open a chat

With `OPENROUTER_API_KEY` exported and `GARP_OPENROUTER_ZDR_ONLY=true` set after verifying your ZDR settings, run `garp`. From this checkout, if you keep your key in `.env`, run:

```sh
GARP_OPENROUTER_ZDR_ONLY=true uv run --env-file .env --locked garp
```

Type a message. Text arrives as the model generates it. Use `/history` to show the active conversation, `/resume` to pick a saved chat, and `/new` to start another. Ctrl-C interrupts generation. Completed replies are kept in your conversation.

Use `/find TEXT` to search the messages in saved chats and open a match. The search accepts a literal word or phrase and shows up to 20 chats with a matching excerpt. Use `/resume TITLE` to search by title.

Use `/remember FACT` to save a fact for future chats, `/memory` to list saved facts, and `/forget` to remove one. Saved facts are supplied to the model during later turns. Removing a fact stops future memory recall. Text already recorded in a chat follows that chat's lifecycle.

Local Garp starts AgentOS and a private Postgres database on Linux or WSL. It stores chats under `${XDG_DATA_HOME:-$HOME/.local/share}/garp` and opens one local workspace at a time. The first native start imports committed local chats and carries forward pending checks.

## Delegate to a specialist

Ask Garp to hand off a task to the reviewer or researcher. A delegation returns a link you can paste as `/resume ID` to open the specialist's chat. Use `/cancel` there to stop its active run. Separate specialist chats keep running if you cancel the parent. Return to the parent and ask for the specialist's result. Use `garp serve` for work that needs an independently running process after the terminal closes.

## Run the demo

From the Garp worktree, run:

```sh
uv run --locked garp demo
```

Run `/check demo 5`, wait five seconds, then run `/history` to see the check result. The demo uses scripted replies, a synthetic progress counter, and a disposable workspace. `/exit` closes it and removes its chats. PostgreSQL 16 tools are detected on Ubuntu. Set `GARP_POSTGRES_BIN` if they are installed elsewhere.

## Start with a specific model

From the same checkout, try the AgentOS path with:

```sh
GARP_OPENROUTER_ZDR_ONLY=true uv run --env-file .env --locked garp chat \
  --model openai/gpt-6.1-terra
```

Set the ZDR flag after verifying your OpenRouter settings. Use `/resume [TITLE]` to search saved chats by title.

## Try your OpenRouter route

Set `OPENROUTER_API_KEY` and `GARP_OPENROUTER_ZDR_ONLY=true` in your shell. Verify that the OpenRouter account or guardrail uses **ZDR only**.

```sh
uv run --locked garp demo --live --model anthropic/claude-haiku-5.5
```

Use `/model` to open the picker in an interactive terminal. Enter a model number or ID, or press Enter to keep the current model. You can also switch directly with `/model NUMBER` or `/model ID`. The selection persists when you reopen the chat. Without a saved or configured choice, new chats use DeepSeek V4.1 Flash 1017.

Run `/effort` to see the levels supported by that model and `/effort high` to select one. `/effort default` restores the provider default. Xiaomi does not advertise selectable levels. Switching models resets effort and search to their defaults.

Terra on Azure supports native search. In the OpenRouter workspace, lock `openrouter:web_search` to the **native** engine under Server Tools, then export `GARP_OPENROUTER_NATIVE_SEARCH_ONLY=true` before starting Garp. Select Terra, run `/search on`, and send a question that needs current sources. Garp keeps search off by default and retains provider citations in chat history. OpenRouter can charge for search. [OpenRouter native search](https://openrouter.ai/docs/guides/features/server-tools/web-search#workspace-settings)

| Model | Pinned route |
|---|---|
| `meta/muse-ember` | Unavailable under strict ZDR. The only ZDR catalog listing allowed Safety Retention when checked on 20 November 2026. |
| `deepseek/deepseek-v4.1-flash-1017` | `fireworks/us` |
| `xiaomi/mimo-v2.6-flash` | `cohere/fp8` |
| `anthropic/claude-haiku-5.5` | `amazon-bedrock` |
| `openai/gpt-6.1-terra` | `azure` |

Set `GARP_OPENROUTER_MODEL` to change the default for chats without an explicit selection. A custom model requires `GARP_OPENROUTER_ENDPOINT` too. A conflicting endpoint for a built-in preset is rejected. Garp pins requests to one endpoint with fallbacks disabled. [OpenRouter ZDR endpoints](https://openrouter.ai/docs/api/api-reference/endpoints/list-endpoints-zdr)

The selected model generates the reply and its title through Agno. Titles use a small budget without web search. Garp saves up to three words. If titling fails, the reply still appears and Garp can retry once on the next turn.

## Scripts and job checks

```sh
uv run --locked garp session list
uv run --locked garp chat SESSION_ID
```

Configure `GARP_JOBS_DIR` and `GARP_POLICY_FILE` with a grant for `local` to use `/check JOB SECONDS`. Use `/history` to see delivered results. The one-shot commands return JSON for scripts.

To run checks while the CLI is closed, export the same `GARP_TOKEN` in both shells and keep AgentOS running independently:

```sh
uv run --env-file .env --locked garp serve --port 8787
```

In another shell:

```sh
uv run --locked garp
```

`garp` connects to localhost port 8787 when `GARP_TOKEN` is set. Set `GARP_URL` if you serve a different port. `garp serve` uses the same private workspace by default. Set `GARP_DATABASE_URL` for a dedicated Postgres instance. The service enables AgentOS’s native `/schedules` API for recurring triggers. It currently uses a trusted operator bearer token and binds to localhost.

## Connecting to a service

Set `GARP_URL` (or pass `--url`) and `GARP_TOKEN` to connect to AgentOS. Remote URLs must use HTTPS; local HTTP is supported. The CLI checks the authenticated model catalog. A service exposing the older `/v1` catalog remains supported through the compatibility client. Authentication failures do not select the older API.

`garp events SESSION --after CURSOR` remains available for older `/v1` services. AgentOS stores runs rather than the older session event sequence, so that command reports an error on AgentOS; use `/history` to view the conversation.
