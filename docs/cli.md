# Garp CLI

## Open a chat

Once the [persistent stack](architecture.md#run-locally) is configured and running, type `garp` to start a new conversation. From the source checkout, use `uv run --locked garp`. Type a message at the prompt. The server keeps your conversation after you exit.

## Run the demo

From the Garp worktree, run:

```sh
uv run --locked garp demo
```

Enter a message to see the chat title. Run `/check demo 5`, wait five seconds, then run `/history` to see the check result. Use `/exit` to close the demo. It uses scripted replies, a synthetic progress counter, a real local worker loop, and disposable Postgres. Workspace data is removed on exit. PostgreSQL 16 tools are detected on Ubuntu. Set `GARP_POSTGRES_BIN` if they are installed elsewhere.

## Try your OpenRouter route

Set `OPENROUTER_API_KEY` and `GARP_OPENROUTER_ZDR_ONLY=true` in your shell. Verify that the OpenRouter account or guardrail uses **ZDR only**.

```sh
uv run --locked garp demo --live --model anthropic/claude-haiku-5.5
```

Use `/model` to list presets, then `/model NUMBER` or `/model ID` to change the current chat. The selection persists when you reopen the chat. Without a saved or configured choice, new chats use DeepSeek V4.1 Flash 1017.

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

## Keep chats

With the [persistent local stack](architecture.md#run-locally) running, use:

```sh
uv run --locked garp
uv run --locked garp session list
uv run --locked garp chat SESSION_ID
```

Use `/history` to replay new events and `/check JOB SECONDS` to schedule a check. The one-shot commands remain JSON for scripts. Run `uv run --locked pytest -q` for the test suite.
