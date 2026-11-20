import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from garp import cli


class CliTests(unittest.TestCase):
    def invoke(self, argv):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = cli.main(argv)
        return code, output.getvalue(), errors.getvalue()

    def test_commands_build_contract_urls_and_json_output(self):
        calls = []

        def request(url, method="GET", body=None):
            calls.append((url, method, body))
            return {"ok": True}

        with patch.object(cli, "_request", side_effect=request):
            self.assertEqual(self.invoke(["session", "new"])[0], 0)
            self.assertEqual(self.invoke(["turn", "a/b", "hello"])[0], 0)
            self.assertEqual(self.invoke(["schedule", "s", "job", "--after", "2"])[0], 0)
            self.assertEqual(self.invoke(["status", "q"])[0], 0)
            self.assertEqual(self.invoke(["events", "s", "--after", "12"])[0], 0)
            self.assertEqual(
                calls,
                [
                    ("http://127.0.0.1:8787/v1/sessions", "POST", {}),
                    ("http://127.0.0.1:8787/v1/sessions/a%2Fb/turns", "POST", {"text": "hello"}),
                    (
                        "http://127.0.0.1:8787/v1/sessions/s/schedules",
                        "POST",
                        {"job_ref": "job", "delay_seconds": 2},
                    ),
                    ("http://127.0.0.1:8787/v1/schedules/q", "GET", None),
                    ("http://127.0.0.1:8787/v1/sessions/s/events?after=12", "GET", None),
                ],
            )
            self.assertEqual(json.loads(self.invoke(["session", "new"])[1]), {"ok": True})

    def test_token_is_optional_and_never_in_error_output(self):
        captured = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return b'{"id":"x"}'

        class Opener:
            def open(self, request, timeout):
                captured.append(request)
                return Response()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(cli, "build_opener", return_value=Opener()),
        ):
            cli._request("http://127.0.0.1:8787/v1/sessions")
        self.assertIsNone(captured[-1].get_header("Authorization"))
        with (
            patch.dict(os.environ, {"GARP_TOKEN": "secret-token"}),
            patch.object(cli, "build_opener", return_value=Opener()),
        ):
            cli._request("http://127.0.0.1:8787/v1/sessions")
        self.assertEqual(captured[-1].get_header("Authorization"), "Bearer secret-token")

        with patch.object(
            cli, "_request", side_effect=RuntimeError("could not reach server secret-token")
        ):
            code, _, err = self.invoke(["session", "new"])
        self.assertEqual(code, 1)
        self.assertNotIn("secret-token", err)

    def test_url_policy_and_redirect_protection(self):
        self.assertEqual(cli._base_url("http://localhost:1234/"), "http://localhost:1234")
        self.assertEqual(cli._base_url("https://example.com/api"), "https://example.com/api")
        for value in ("http://example.com", "https://user:pass@example.com", "file:///tmp/x"):
            with self.assertRaises(ValueError):
                cli._base_url(value)
        self.assertIsNone(
            cli._NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.test")
        )

    def test_schedule_delay_must_be_nonnegative(self):
        code, _, err = self.invoke(["schedule", "s", "job", "--after", "-1"])
        self.assertEqual(code, 1)
        self.assertIn("non-negative", err)

    def test_no_command_opens_chat_on_configured_server(self):
        with (
            patch.dict(os.environ, {"GARP_URL": "https://example.com/"}),
            patch("garp.terminal.chat", return_value=0) as chat,
        ):
            self.assertEqual(self.invoke([])[0], 0)
            chat.assert_called_once_with("https://example.com", None, cli._request, model=None)

    def test_chat_reports_when_server_is_unreachable(self):
        with patch.object(cli, "_request", side_effect=RuntimeError("could not reach server")):
            code, _, err = self.invoke([])
        self.assertEqual(code, 1)
        self.assertIn("Start garp-api or set GARP_URL", err)


if __name__ == "__main__":
    unittest.main()
