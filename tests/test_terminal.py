from garp.terminal import chat


def test_resume_uses_saved_title_and_replays_safe_text(monkeypatch, capsys):
    calls = []
    inputs = iter(("/history", "/exit"))
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
    assert "Worker Lease Review" in output and output.count("completed") == 1
    assert "\x1b" not in output and "\u202e" not in output
    assert calls[-1].endswith("?after=1")
