# Garp

Garp is an embeddable agent harness. This slice runs stateless Agno turns and durable one-shot job checks with a CLI and local HTTP API.

With the [Garp services](docs/architecture.md#run-locally) running, open a saved conversation with `garp` (or `uv run --locked garp` from this checkout).

To try a disposable demo:

```sh
uv run --locked garp demo
```

Type a message, then try `/check demo 5` and `/history`. The demo uses scripted replies and a real local worker and database. See the [CLI guide](docs/cli.md) for live OpenRouter mode, reopening chats, and tests.
