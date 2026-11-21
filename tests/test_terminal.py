from fastapi.testclient import TestClient

from garp.agentos import create_agentos_app
from garp.native_client import NativeChatClient
from garp.terminal import chat, native_chat


def test_resume_uses_saved_title_and_replays_safe_text(monkeypatch, capsys):
    calls = []
    inputs = iter(("/history", "/refresh", "/exit"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    def request(url, *_args):
        calls.append(url)
        if url.endswith("/session"):
            return {"title": "Worker Lease Review"}
        if url.endswith("?after=0"):
            return {
                "cursor": 1,
                "events": [
                    {
                        "kind": "turn.message",
                        "data": {"role": "assistant", "content": "\x1b[31mcompleted\u202e"},
                    }
                ],
            }
        return {"cursor": 1, "events": []}

    assert chat("http://test", "session", request) == 0
    output = capsys.readouterr().out
    assert "Worker Lease Review" in output and output.count("completed") == 2
    assert "Conversation" in output and "No new activity." in output
    assert "\x1b" not in output and "\u202e" not in output
    assert calls[-1].endswith("?after=1")


def test_stream_prints_each_delta_without_echoing_user_turn(monkeypatch, capsys):
    inputs = iter(("/history", "Tell me something", "/exit"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    events = []

    def request(url, *_args, on_delta=None):
        if "/events?" in url:
            return {"cursor": len(events), "events": events}
        if url.endswith("/turns"):
            on_delta("Partial ")
            assert capsys.readouterr().out.endswith("Partial ")
            on_delta("reply")
            events.extend(
                [
                    {
                        "seq": 1,
                        "kind": "turn.message",
                        "data": {"role": "user", "content": "prompt"},
                    },
                    {
                        "seq": 2,
                        "kind": "turn.message",
                        "data": {"role": "assistant", "content": "Partial reply"},
                    },
                ]
            )
            return {"events": events}
        return {"title": None}

    assert chat("http://test", "session", request) == 0
    assert capsys.readouterr().out == "reply\n"


def test_native_terminal_memory_survives_restart(postgres_dsn, monkeypatch, capsys):
    token = "local-fixture-token"
    headers = {"Authorization": f"Bearer {token}"}

    def run(*commands):
        app = create_agentos_app(postgres_dsn, token)
        client = NativeChatClient("http://localhost", token)
        client.close()
        with TestClient(app, base_url="http://localhost", headers=headers) as client.client:
            inputs = iter(commands)
            monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
            assert native_chat(client) == 0
        return capsys.readouterr().out

    assert "Saved to memory." in run("/remember My preferred tea is rooibos", "/exit")
    output = run("/memory", "/forget", "1", "/memory", "/exit")
    assert output.count("My preferred tea is rooibos") == 2
    assert "Removed from memory." in output
    assert "No saved facts." in output
