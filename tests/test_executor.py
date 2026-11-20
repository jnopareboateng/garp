import asyncio
import copy
import json
import os
import unittest
from unittest.mock import patch

import httpx

from garp import executor

ENV = {
    "OPENROUTER_API_KEY": "test-placeholder-key",
    "GARP_OPENROUTER_MODEL": "vendor/model",
    "GARP_OPENROUTER_ENDPOINT": "provider/endpoint",
    "GARP_OPENROUTER_ZDR_ONLY": "true",
}


def mock_transport(handler):
    class FixtureClient(httpx.AsyncClient):
        def __init__(self, **options):
            super().__init__(transport=httpx.MockTransport(handler), **options)

    return patch.object(executor, "AsyncClient", FixtureClient)


class ExecutorTest(unittest.IsolatedAsyncioTestCase):
    async def test_title_agent_adds_three_word_instruction_on_the_same_route(self):
        requests = []

        async def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "title",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "vendor/model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Reliable Worker Leases"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        with patch.dict(os.environ, ENV, clear=True), mock_transport(handle):
            self.assertEqual(
                await executor.generate_title("Design worker leases"), "Reliable Worker Leases"
            )
        body = requests[0]
        self.assertEqual(
            body["provider"], {"zdr": True, "only": ["provider/endpoint"], "allow_fallbacks": False}
        )
        self.assertEqual(body["max_tokens"], 32)
        self.assertIn("at most three words", body["messages"][0]["content"])
        self.assertEqual(body["messages"][-1]["content"], "Design worker leases")

    def test_selected_model_uses_its_own_endpoint(self):
        with patch.dict(os.environ, ENV, clear=True):
            self.assertEqual(
                executor.openrouter_config("xiaomi/mimo-v2.6-flash")[1:],
                ("xiaomi/mimo-v2.6-flash", "cohere/fp8"),
            )
            self.assertEqual(
                executor.openrouter_config("anthropic/claude-haiku-5.5")[1:],
                ("anthropic/claude-haiku-5.5", "amazon-bedrock"),
            )
            with self.assertRaisesRegex(ValueError, "strict-ZDR"):
                executor.openrouter_config("meta/muse-ember")
            with (
                patch.dict(os.environ, {"GARP_OPENROUTER_MODEL": "openai/gpt-6.1-terra"}),
                self.assertRaisesRegex(ValueError, "conflicts"),
            ):
                executor.openrouter_config("openai/gpt-6.1-terra")

    async def test_request_and_history(self):
        requests = []

        async def handle(request):
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "completion",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "vendor/model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "answer"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        history = [
            {"role": "user", "content": "previous"},
            {"role": "assistant", "content": "reply"},
        ]
        original = copy.deepcopy(history)
        actual_arun = executor.Agent.arun
        run_args = {}

        async def capture_run(agent, *args, **kwargs):
            run_args.update(kwargs)
            return await actual_arun(agent, *args, **kwargs)

        with (
            patch.dict(os.environ, ENV, clear=True),
            mock_transport(handle),
            patch.object(executor.Agent, "arun", capture_run),
        ):
            self.assertEqual(
                (await executor.execute_turn("owner", "session", "run", history, "next")).text,
                "answer",
            )
        self.assertEqual(
            {key: run_args[key] for key in ("user_id", "session_id", "run_id")},
            {"user_id": "owner", "session_id": "session", "run_id": "run"},
        )
        self.assertEqual(history, original)
        self.assertEqual(len(requests), 1)
        self.assertEqual(str(requests[0].url), "https://openrouter.ai/api/v1/chat/completions")
        body = json.loads(requests[0].content)
        self.assertEqual(body["model"], "vendor/model")
        self.assertNotIn("user", body)
        self.assertNotIn("owner", requests[0].content.decode())
        self.assertEqual(
            body["provider"], {"zdr": True, "only": ["provider/endpoint"], "allow_fallbacks": False}
        )
        self.assertEqual(
            [(m["role"], m["content"]) for m in body["messages"] if m["role"] != "system"],
            [("user", "previous"), ("assistant", "reply"), ("user", "next")],
        )

    async def test_cancellation_propagates(self):
        started = asyncio.Event()

        async def handle(request):
            started.set()
            await asyncio.Event().wait()

        with patch.dict(os.environ, ENV, clear=True), mock_transport(handle):
            task = asyncio.create_task(
                executor.execute_turn("owner", "session", "run", [], "hello")
            )
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_terra_sends_native_search_and_keeps_citations(self):
        requests = []

        async def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "terra",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "openai/gpt-6.1-terra",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "Verified source",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url_citation": {
                                            "url": "https://example.org/source",
                                            "title": "Example source",
                                            "start_index": 0,
                                            "end_index": 15,
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            )

        options = {"model_id": "openai/gpt-6.1-terra", "effort": "high", "web_search": True}
        with patch.dict(os.environ, ENV, clear=True), mock_transport(handle):
            with self.assertRaisesRegex(ValueError, "native-only"):
                await executor.execute_turn(
                    "owner", "session", "run", [], "Find a source", **options
                )
            self.assertEqual(requests, [])
            with patch.dict(os.environ, {"GARP_OPENROUTER_NATIVE_SEARCH_ONLY": "true"}):
                reply = await executor.execute_turn(
                    "owner", "session", "run", [], "Find a source", **options
                )
                await executor.generate_title("Find a source", model_id=options["model_id"])
        self.assertEqual(
            reply.sources, [{"url": "https://example.org/source", "title": "Example source"}]
        )
        self.assertNotIn("max_tokens", requests[0])
        self.assertNotIn("max_completion_tokens", requests[0])
        self.assertNotIn("max_tokens", requests[1])
        self.assertEqual(requests[1]["max_completion_tokens"], 32)
        self.assertEqual(requests[1]["reasoning"], {"effort": "none"})
        self.assertNotIn("tools", requests[1])
        self.assertEqual(requests[0]["reasoning"], {"effort": "high"})
        self.assertEqual(
            requests[0]["tools"],
            [{"type": "openrouter:web_search", "parameters": {"engine": "native"}}],
        )
        self.assertEqual(
            requests[0]["provider"],
            {"zdr": True, "only": ["azure"], "allow_fallbacks": False, "require_parameters": True},
        )

    async def test_redirect_does_not_forward_prompt(self):
        hosts = []

        async def handle(request):
            hosts.append(request.url.host)
            return httpx.Response(307, headers={"Location": "https://other.example/collect"})

        with (
            patch.dict(os.environ, ENV, clear=True),
            mock_transport(handle),
            self.assertRaisesRegex(RuntimeError, "agent turn did not complete"),
        ):
            await executor.execute_turn("owner", "session", "run", [], "fixture prompt")
        self.assertEqual(hosts, ["openrouter.ai"])

    def test_configuration_fails_closed(self):
        for name in ENV:
            with (
                self.subTest(name=name),
                patch.dict(os.environ, {k: v for k, v in ENV.items() if k != name}, clear=True),
                self.assertRaises(ValueError),
            ):
                executor._model(None)
        with (
            patch.dict(os.environ, {**ENV, "GARP_OPENROUTER_ZDR_ONLY": "false"}, clear=True),
            self.assertRaises(ValueError),
        ):
            executor._model(None)
