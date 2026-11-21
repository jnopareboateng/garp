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
            if url.endswith("/garp/models"):
                raise cli._HTTPFailure("server returned HTTP 404")
            if url.endswith("/v1/models"):
                return {"models": []}
            calls.append((url, method, body))
            return {"ok": True}

        with (
            patch.dict(os.environ, {"GARP_URL": "http://127.0.0.1:8787"}),
            patch.object(cli, "_request", side_effect=request),
        ):
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

        with (
            patch.dict(os.environ, {"GARP_URL": "http://127.0.0.1:8787"}),
            patch.object(
                cli, "_request", side_effect=RuntimeError("could not reach server secret-token")
            ),
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
            patch.object(
                cli,
                "_request",
                side_effect=[cli._HTTPFailure("server returned HTTP 404"), {"models": []}],
            ),
            patch("garp.terminal.chat", return_value=0) as chat,
        ):
            self.assertEqual(self.invoke([])[0], 0)
            chat.assert_called_once_with("https://example.com", None, cli._request, model=None)

    def test_bare_with_token_connects_to_native_service(self):
        with (
            patch.dict(os.environ, {"GARP_TOKEN": "fixture"}, clear=True),
            patch("garp.native_client.NativeChatClient") as client,
            patch("garp.terminal.native_chat", return_value=0) as chat,
        ):
            self.assertEqual(self.invoke([])[0], 0)
            client.assert_called_once_with("http://127.0.0.1:8787", "fixture")
            chat.assert_called_once_with(client.return_value, None, model=None)

    def test_bare_with_url_detects_native_service(self):
        with (
            patch.dict(os.environ, {"GARP_TOKEN": "fixture"}, clear=True),
            patch.object(cli, "_request", return_value={"models": []}) as probe,
            patch("garp.native_client.NativeChatClient") as client,
            patch("garp.terminal.native_chat", return_value=0) as chat,
        ):
            self.assertEqual(self.invoke(["--url", "https://example.com"])[0], 0)
            probe.assert_called_once_with("https://example.com/garp/models")
            client.assert_called_once_with("https://example.com", "fixture")
            chat.assert_called_once_with(client.return_value, None, model=None)

    def test_chat_reports_when_server_is_unreachable(self):
        with (
            patch.dict(os.environ, {"GARP_URL": "http://127.0.0.1:8787"}),
            patch.object(cli, "_request", side_effect=RuntimeError("could not reach server")),
        ):
            code, _, err = self.invoke([])
        self.assertEqual(code, 1)
        self.assertIn("Start garp serve or set GARP_URL", err)

    def test_bare_command_starts_local_workspace_without_service_config(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("garp.demo.native_workspace") as launch,
            patch("garp.native_client.NativeChatClient") as client,
            patch("garp.terminal.native_chat", return_value=0) as chat,
        ):
            launch.return_value = contextlib.nullcontext(("http://127.0.0.1:9876", "fixture"))
            self.assertEqual(self.invoke([])[0], 0)
            launch.assert_called_once_with(model=None, needs_model=False)
            client.assert_called_once_with("http://127.0.0.1:9876", "fixture")
            chat.assert_called_once_with(client.return_value, None, model=None)


if __name__ == "__main__":
    unittest.main()


def test_attached_native_commands_and_authentication(postgres_dsn, tmp_path):
    import subprocess

    from garp.agentos import create_agentos_app
    from garp.demo import _scripted_model, _serve_app
    from garp.jobs import LocalJobs

    (tmp_path / "policy.json").write_text(json.dumps({"local": ["job"]}))
    (tmp_path / "job.json").write_text(
        json.dumps(
            {"instance": "fixture", "observed_at": "2026-01-01T00:00:00+00:00", "state": "running"}
        )
    )
    jobs = LocalJobs(tmp_path, tmp_path / "policy.json")
    app = create_agentos_app(postgres_dsn, "fixture-token", jobs=jobs, _demo_model=_scripted_model)
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG") if key in os.environ}
    env["GARP_TOKEN"] = "fixture-token"
    with _serve_app(app) as base:

        def run(*args, token="fixture-token"):
            return subprocess.run(
                [sys.executable, "-m", "garp.cli", "--url", base, *args],
                check=False,
                capture_output=True,
                text=True,
                env={**env, "GARP_TOKEN": token},
                timeout=30,
            )

        created = run("session", "new")
        assert created.returncode == 0, created.stderr
        sid = json.loads(created.stdout)["id"]
        listed = run("session", "list")
        assert sid in [row["id"] for row in json.loads(listed.stdout)["sessions"]]
        turned = run("turn", sid, "hello")
        assert turned.returncode == 0, turned.stderr
        assert "scripted reply" in json.loads(turned.stdout)["content"]
        scheduled = run("schedule", sid, "job", "--after", "0")
        assert scheduled.returncode == 0, scheduled.stderr
        check_id = json.loads(scheduled.stdout)["id"]
        status = run("status", check_id)
        assert status.returncode == 0, status.stderr
        assert json.loads(status.stdout)["id"] == check_id
        events = run("events", sid, "--after", "1")
        assert events.returncode == 1
        assert "events is unavailable on AgentOS" in events.stderr
        denied = run("session", "list", token="wrong-token")
        assert denied.returncode == 1
        assert "HTTP 401" in denied.stderr
        assert "wrong-token" not in denied.stderr


def test_fallback_requires_valid_legacy_catalog():
    with (
        patch.dict(os.environ, {"GARP_TOKEN": "fixture"}, clear=True),
        patch.object(
            cli,
            "_request",
            side_effect=[cli._HTTPFailure("server returned HTTP 404"), {"unexpected": True}],
        ),
        patch.object(cli, "_dispatch") as legacy,
    ):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            assert cli.main(["--url", "http://localhost:1234", "session", "list"]) == 1
        assert "server returned invalid JSON" in errors.getvalue()
        legacy.assert_not_called()


def test_native_turn_routes_to_session_specialist():
    from argparse import Namespace
    from unittest.mock import Mock

    client = Mock()
    client.session.return_value = {"agent_id": "reviewer", "model": "fixture"}
    client.turn.return_value = {"content": "done"}
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli._dispatch_native(Namespace(command="turn", session="s", text="hi"), client) == 0
    assert client.turn.call_args.kwargs["agent_id"] == "reviewer"


def test_native_auth_failure_never_falls_back():
    for code in (401, 403):
        with (
            patch.dict(os.environ, {"GARP_TOKEN": "fixture"}, clear=True),
            patch.object(
                cli, "_request", side_effect=cli._HTTPFailure(f"server returned HTTP {code}")
            ),
            patch.object(cli, "_dispatch") as legacy,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            assert cli.main(["--url", "http://localhost:1234", "session", "list"]) == 1
            legacy.assert_not_called()
