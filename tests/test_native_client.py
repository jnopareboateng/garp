import json
from urllib.parse import parse_qs

import httpx
import pytest

from garp.native_client import NativeChatClient


def frame(index, kind, **values):
    return f"data: {json.dumps({'event_index': index, 'event': kind, 'run_id': 'run', **values})}\n\n".encode()


def client(handler, *, durable=False, active=False):
    instance = NativeChatClient("http://127.0.0.1:8000", "token")
    instance.client.close()

    def dispatch(request):
        if request.url.path == "/garp/models":
            return httpx.Response(
                200,
                json={
                    "durable_stream_reconnect": durable,
                    "default": "m",
                    "models": [
                        {"id": "m", "available": True, "efforts": ["high"], "native_search": True},
                        {"id": "n", "available": True, "efforts": [], "native_search": False},
                    ],
                },
            )
        if request.url.path.endswith("/runs") and request.url.params.get_list("status") == [
            "PENDING",
            "RUNNING",
            "PAUSED",
        ]:
            return httpx.Response(200, json=[{"status": "RUNNING"}] if active else [])
        return handler(request)

    instance.client = httpx.Client(transport=httpx.MockTransport(dispatch))
    return instance


def test_stream_resume_deduplicates_and_does_not_resubmit():
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            assert request.url.path == "/agents/garp/runs"
            data = parse_qs(request.content.decode())
            assert data["background"] == ["true"]
            assert json.loads(data["factory_input"][0]) == {
                "model_id": "m",
                "effort": "high",
                "web_search": True,
            }
            return httpx.Response(
                200, content=frame(0, "RunStarted") + frame(1, "RunContent", content="Hello ")
            )
        assert request.url.path == "/agents/garp/runs/run/resume"
        assert parse_qs(request.content.decode()) == {
            "session_id": ["s"],
            "last_event_index": ["1"],
        }
        return httpx.Response(
            200,
            content=frame(1, "RunContent", content="Hello ")
            + frame(2, "RunContent", content="world")
            + frame(3, "RunCompleted", content="Hello world"),
        )

    output = []
    result = client(handler).turn(
        "s", "hi", on_delta=output.append, model="m", effort="high", web_search=True
    )
    assert output == ["Hello ", "world"]
    assert result == {"content": "Hello world", "sources": [], "run_id": "run"}
    assert len(requests) == 2


def test_unknown_start_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadError("secret")

    with pytest.raises(RuntimeError, match="connection closed"):
        client(handler).turn("s", "hi", on_delta=lambda _: None)
    assert len(calls) == 1


def test_specialist_link_survives_parent_cancellation():
    child_id = "8c82f7e8-fdb9-486d-8d5b-4ad51955ec74"
    child = {"agent_id": "reviewer", "session_id": child_id}

    def handler(_request):
        return httpx.Response(
            200,
            content=frame(0, "RunStarted")
            + frame(
                1,
                "ToolCallCompleted",
                tool={"tool_name": "delegate_task", "result": json.dumps(child)},
            )
            + frame(2, "RunCancelled"),
        )

    links = []
    with pytest.raises(RuntimeError, match="agent turn failed"):
        client(handler).turn("s", "Delegate", on_delta=lambda _: None, on_delegation=links.append)
    assert links == [child]


def test_approval_pause_is_a_successful_terminal_event():
    def handler(_request):
        return httpx.Response(
            200,
            content=frame(0, "RunStarted") + frame(1, "RunPaused", content="Waiting for review"),
        )

    result = client(handler).turn("s", "Record this note", on_delta=lambda _: None)
    assert result == {
        "content": "Waiting for review",
        "run_id": "run",
        "sources": [],
        "paused": True,
    }


def test_session_options_preserve_state_and_history():
    def handler(request):
        if request.url.path.endswith("/runs"):
            assert request.url.params["status"] == "COMPLETED"
            return httpx.Response(
                200,
                json=[
                    {"run_input": "hi", "content": "hello"},
                    {"parent_run_id": "parent", "run_input": "internal", "content": "internal"},
                ],
            )
        if request.method == "PATCH":
            assert json.loads(request.content) == {
                "session_state": {"other": 1, "garp_options": {"model": "m", "web_search": True}}
            }
            return httpx.Response(200, json={})
        return httpx.Response(
            200,
            json={
                "session_id": "s",
                "session_name": "Title",
                "session_state": {"other": 1, "garp_options": {"model": "m"}},
            },
        )

    c = client(handler)
    assert c.session("s")["model"] == "m"
    assert c.history("s") == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    c.set_options("s", {"web_search": True})


def test_options_stay_unchanged_with_active_turn():
    def unexpected(_request):
        pytest.fail("option changes must stop before reading or writing session state")

    with pytest.raises(ValueError, match="Finish or cancel"):
        client(unexpected, active=True).set_options("s", {"effort": "high"})


def test_cancel_known_run_on_interrupt():
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.endswith("/cancel"):
            assert request.url.params["session_id"] == "s"
            return httpx.Response(200, json={})
        return httpx.Response(
            200, content=frame(0, "RunStarted") + frame(1, "RunContent", content="a")
        )

    def interrupt(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        client(handler).turn("s", "hi", on_delta=interrupt)
    assert len(calls) == 2


def test_durable_factory_resume_400_reattaches_with_identical_key_and_body():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                200, content=frame(0, "RunStarted") + frame(1, "RunContent", content="a")
            )
        if len(calls) == 2:
            assert request.url.path.endswith("/resume")
            return httpx.Response(400)
        assert request.url.path == calls[0].url.path
        assert request.content == calls[0].content
        assert request.headers["Idempotency-Key"] == calls[0].headers["Idempotency-Key"]
        return httpx.Response(
            200,
            content=frame(0, "RunStarted")
            + frame(1, "RunContent", content="a")
            + frame(2, "RunContent", content="b")
            + frame(3, "RunCompleted", content="ab"),
        )

    output = []
    assert client(handler, durable=True).turn("s", "hi", on_delta=output.append)["content"] == "ab"
    assert output == ["a", "b"]
    assert len(calls) == 3


def test_durable_unknown_start_reattaches_once_with_same_key():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadError("disconnected")
        assert request.content == calls[0].content
        assert request.headers["Idempotency-Key"] == calls[0].headers["Idempotency-Key"]
        return httpx.Response(
            200, content=frame(0, "RunStarted") + frame(1, "RunCompleted", content="done")
        )

    assert (
        client(handler, durable=True).turn("s", "hi", on_delta=lambda _: None)["content"] == "done"
    )
    assert len(calls) == 2


def test_sources_filter_unsafe_urls():
    def handler(request):
        return httpx.Response(
            200,
            json=[
                {
                    "content": "a",
                    "citations": {
                        "urls": [
                            {"url": "javascript:alert(1)"},
                            {"url": "https://user:pass@example.com"},
                            {"url": "https://example.com", "title": "Example"},
                        ]
                    },
                }
            ],
        )

    assert client(handler).history("s")[0]["sources"] == [
        {"title": "Example", "url": "https://example.com"}
    ]


def test_model_change_resets_options_and_invalid_settings_are_not_written():
    writes = []

    def handler(request):
        if request.method == "PATCH":
            writes.append(json.loads(request.content))
            return httpx.Response(200, json={})
        return httpx.Response(
            200,
            json={
                "session_state": {
                    "garp_options": {"model": "m", "effort": "high", "web_search": True}
                }
            },
        )

    c = client(handler)
    assert c.set_options("s", {"model": "m"}) == {
        "model": "m",
        "effort": "high",
        "web_search": True,
    }
    assert c.set_options("s", {"model": "n"}) == {"model": "n", "effort": None, "web_search": False}
    with pytest.raises(ValueError):
        c.set_options("s", {"model": "n", "effort": "high"})
    assert len(writes) == 2
