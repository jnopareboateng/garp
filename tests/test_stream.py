import asyncio
import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from test_executor import ENV, mock_transport

from garp import executor
from garp.api import create_app


class ProviderStream(httpx.AsyncByteStream):
    def __init__(self):
        self.release = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        def chunk(delta, finish=None):
            return (
                "data: "
                + json.dumps(
                    {
                        "id": "test",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "vendor/model",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                )
                + "\n\n"
            ).encode()

        yield chunk({"role": "assistant", "content": "Hello "})
        await self.release.wait()
        yield chunk(
            {
                "content": "world",
                "annotations": [
                    {
                        "type": "url_citation",
                        "url_citation": {
                            "url": "https://example.org/source",
                            "title": "Source",
                            "start_index": 0,
                            "end_index": 5,
                        },
                    }
                ],
            }
        )
        yield chunk({}, "stop")
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


def test_real_agno_stream_delivers_before_completion():
    asyncio.run(check_real_agno_stream())


async def check_real_agno_stream():
    provider = ProviderStream()
    requests = []

    async def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, stream=provider, headers={"content-type": "text/event-stream"})

    with patch.dict("os.environ", ENV, clear=True), mock_transport(handle):
        stream = executor.stream_turn("owner", "session", "run", [], "hello")
        first = await asyncio.wait_for(anext(stream), 5)
        assert first == "Hello "
        assert not provider.closed
        provider.release.set()
        remaining = [item async for item in stream]
    assert remaining[0] == "world"
    assert remaining[-1] == executor.Reply(
        "Hello world", [{"url": "https://example.org/source", "title": "Source"}]
    )
    assert requests[0]["stream"] is True
    assert requests[0]["provider"] == {
        "zdr": True,
        "only": ["provider/endpoint"],
        "allow_fallbacks": False,
    }
    assert provider.closed


class Store:
    def initialize(self):
        pass

    def start_turn(self, owner, session, text):
        return {
            "token": "lease",
            "model": None,
            "effort": None,
            "web_search": False,
            "history": [],
            "title": "Existing",
        }

    def finish_turn(self, owner, session, token, text, title, details):
        self.finished = text
        return [{"seq": 1, "data": {"content": text}}] if text is not None else None


def test_api_ndjson_and_failure_cleanup():
    store = Store()

    async def good(*args):
        return "answer"

    async def bad(*args):
        raise ValueError("secret")

    path = "/v1/sessions/00000000-0000-0000-0000-000000000001/turns"
    headers = {"Authorization": "Bearer token", "Accept": "application/x-ndjson"}
    for run, expected in [(good, "done"), (bad, "error")]:
        with TestClient(create_app(store, None, {"token": "owner"}, executor=run)) as client:
            response = client.post(path, json={"text": "hi"}, headers=headers)
            records = [json.loads(line) for line in response.text.splitlines()]
            assert records[-1]["type"] == expected
            assert "secret" not in response.text
            assert store.finished == ("answer" if run is good else None)
            if run is good:
                assert records[0] == {"type": "delta", "text": "answer"}
                assert records[-1]["cursor"] == 1


def test_provider_stream_closes_on_cancellation():
    async def check():
        provider = ProviderStream()

        async def handle(request):
            return httpx.Response(
                200, stream=provider, headers={"content-type": "text/event-stream"}
            )

        with patch.dict("os.environ", ENV, clear=True), mock_transport(handle):
            stream = executor.stream_turn("owner", "session", "run", [], "hello")
            assert await anext(stream) == "Hello "
            await stream.aclose()
        assert provider.closed

    asyncio.run(check())


@pytest.mark.parametrize("while_titling", [False, True])
def test_http_disconnect_finalizes_turn(while_titling):
    async def check():
        started = asyncio.Event()

        async def blocked(*args):
            started.set()
            await asyncio.Future()

        async def done(*args):
            return "completed reply"

        class UntitledStore(Store):
            def start_turn(self, owner, session, text):
                return {**super().start_turn(owner, session, text), "title": None}

        store = UntitledStore()
        response_started = asyncio.Event()
        body_received = False

        async def client_receive():
            nonlocal body_received
            if not body_received:
                body_received = True
                return {"type": "http.request", "body": b'{"text":"hello"}'}
            await started.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                assert message["status"] == 200
                response_started.set()

        app = create_app(
            store,
            None,
            {"token": "owner"},
            executor=done if while_titling else blocked,
            titler=blocked if while_titling else None,
        )
        await asyncio.wait_for(
            app(
                {
                    "type": "http",
                    "asgi": {"spec_version": "2.3"},
                    "http_version": "1.1",
                    "method": "POST",
                    "path": "/v1/sessions/00000000-0000-0000-0000-000000000001/turns",
                    "query_string": b"",
                    "headers": [
                        (b"authorization", b"Bearer token"),
                        (b"accept", b"application/x-ndjson"),
                        (b"content-type", b"application/json"),
                    ],
                },
                client_receive,
                send,
            ),
            5,
        )
        assert response_started.is_set()
        assert store.finished == ("completed reply" if while_titling else None)

    asyncio.run(check())


def test_expired_lease_cannot_report_done():
    store = Store()
    store.finish_turn = lambda *args: False

    async def good(*args):
        return "answer"

    with TestClient(create_app(store, None, {"token": "owner"}, executor=good)) as client:
        response = client.post(
            "/v1/sessions/00000000-0000-0000-0000-000000000001/turns",
            json={"text": "hi"},
            headers={"Authorization": "Bearer token", "Accept": "application/x-ndjson"},
        )
    records = [json.loads(line) for line in response.text.splitlines()]
    assert records[-1] == {"type": "error", "message": "turn lease expired"}
