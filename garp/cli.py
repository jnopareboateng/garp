"""Command line client for the Garp HTTP API."""

import argparse
import json
import os
import sys
from contextlib import nullcontext
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

_MODEL_ERRORS = {
    "Use an exact OpenRouter model ID without presets or routing suffixes",
    "No strict-ZDR endpoint is approved for this model",
    "This model has no approved OpenRouter endpoint",
    "This model does not support that reasoning effort",
    "Set GARP_OPENROUTER_MODEL to bind the custom endpoint",
    "The configured endpoint conflicts with this model's strict-ZDR pin",
    "OPENROUTER_API_KEY is required",
    "GARP_OPENROUTER_ZDR_ONLY must explicitly attest true",
    "Native search requires Terra on Azure and a verified native-only workspace",
}


class _HTTPFailure(RuntimeError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _base_url(value):
    try:
        parts = urlsplit(value)
        host = (parts.hostname or "").lower()
        valid_host = host in {"localhost", "127.0.0.1", "::1"}
        if (
            (parts.scheme != "https" and not (parts.scheme == "http" and valid_host))
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError
        return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
    except (TypeError, ValueError):
        raise ValueError("URL must use local HTTP or remote HTTPS") from None


def _request(url, method="GET", body=None, *, token=None, on_delta=None):
    stream = "application/x-ndjson"
    headers = {"Accept": stream if on_delta else "application/json"}
    token = os.environ.get("GARP_TOKEN") if token is None else token
    if token:
        if not token.isascii() or not token.isprintable() or any(c.isspace() for c in token):
            raise ValueError("invalid GARP_TOKEN")
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    req = Request(url, data=data, headers=headers, method=method)
    try:
        handlers = [_NoRedirect()]
        if urlsplit(url).scheme == "http":
            handlers.append(ProxyHandler({}))
        with build_opener(*handlers).open(req, timeout=100) as response:
            if on_delta and response.headers.get_content_type() == stream:
                while line := response.readline(4 * 1024 * 1024):
                    if not line.endswith(b"\n"):
                        break
                    frame = json.loads(line)
                    kind = frame.get("type") if isinstance(frame, dict) else None
                    if kind == "delta" and isinstance(frame.get("text"), str):
                        on_delta(frame["text"])
                    elif kind == "done":
                        return frame
                    elif kind == "error":
                        message = frame.get("message")
                        if isinstance(message, str) and message in {
                            "agent turn failed",
                            "agent turn timed out",
                            "turn lease expired",
                        }:
                            raise RuntimeError(message)
                        raise RuntimeError("agent turn failed")
                    else:
                        raise RuntimeError("received an invalid response stream")
                raise RuntimeError("connection closed before the reply was confirmed")
            return json.loads(response.read())
    except HTTPError as exc:
        detail = ""
        if exc.code in (422, 503):
            try:
                message = json.loads(exc.read(4096)).get("detail")
                if isinstance(message, str) and message in _MODEL_ERRORS:
                    detail = f" ({message})"
            except (AttributeError, ValueError, OSError):
                pass  # Keep the status when a response has no safe error detail.
        raise _HTTPFailure(f"server returned HTTP {exc.code}{detail}") from None
    except (URLError, TimeoutError, OSError, HTTPException):
        raise RuntimeError("could not reach server") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("server returned invalid JSON") from None


def _parser():
    parser = argparse.ArgumentParser(
        prog="garp", description="Open a new Garp chat when run without a command."
    )
    parser.set_defaults(session=None, model=None)
    parser.add_argument(
        "--url", default=os.environ.get("GARP_URL"), help="Connect to a Garp service"
    )
    commands = parser.add_subparsers(dest="command")
    session = commands.add_parser("session", help="Create or list saved chats as JSON")
    actions = session.add_subparsers(dest="session_action", required=True)
    actions.add_parser("new")
    actions.add_parser("list")
    demo = commands.add_parser("demo", help="Try Garp with a disposable local stack")
    demo.add_argument("--live", action="store_true", help="Use your configured OpenRouter route")
    demo.add_argument("--model", help="Select the starting OpenRouter model")
    chat = commands.add_parser("chat", help="Open a new or saved interactive chat")
    chat.add_argument("session", nargs="?")
    chat.add_argument("--model", help="Select the starting OpenRouter model")
    native = commands.add_parser("agentos", help="Open a chat backed by native AgentOS")
    native.add_argument("session", nargs="?")
    native.add_argument("--model", help="Select the starting OpenRouter model")
    serve = commands.add_parser("serve", help="Run AgentOS and its scheduler on localhost")
    serve.add_argument("--port", type=int, default=8787)
    turn = commands.add_parser("turn")
    turn.add_argument("session")
    turn.add_argument("text")
    schedule = commands.add_parser("schedule")
    schedule.add_argument("session")
    schedule.add_argument("job")
    schedule.add_argument("--after", type=int, required=True)
    schedule.add_argument("--deadline", type=int, help="Latest start in seconds from now")
    prompt = commands.add_parser(
        "schedule-prompt", help="Schedule a prompt on a running AgentOS service"
    )
    prompt.add_argument("text")
    prompt.add_argument("--name", required=True)
    prompt.add_argument("--session")
    timing = prompt.add_mutually_exclusive_group(required=True)
    timing.add_argument("--after", type=int, dest="after_seconds")
    timing.add_argument("--cron", dest="cron_expr")
    prompt.add_argument("--timezone", default="UTC")
    prompt.add_argument("--deadline", type=int, dest="deadline_seconds")
    status = commands.add_parser("status")
    status.add_argument("schedule")
    events = commands.add_parser("events")
    events.add_argument("session")
    events.add_argument("--after", type=int)
    return parser


def _dispatch(args, base, request):
    if args.command in (None, "chat"):
        from garp.terminal import chat

        return chat(base, args.session, request, model=args.model)
    if args.command == "session" and args.session_action == "list":
        result = request(f"{base}/v1/sessions")
    elif args.command == "session":
        result = request(f"{base}/v1/sessions", "POST", {})
    elif args.command == "turn":
        result = request(
            f"{base}/v1/sessions/{quote(args.session, safe='')}/turns",
            "POST",
            {"text": args.text},
        )
    elif args.command == "schedule":
        body = {"job_ref": args.job, "delay_seconds": args.after}
        if args.deadline is not None:
            body["deadline_seconds"] = args.deadline
        result = request(
            f"{base}/v1/sessions/{quote(args.session, safe='')}/schedules",
            "POST",
            body,
        )
    elif args.command == "status":
        result = request(f"{base}/v1/schedules/{quote(args.schedule, safe='')}")
    else:
        path = f"{base}/v1/sessions/{quote(args.session, safe='')}/events"
        if args.after is not None:
            path += "?" + urlencode({"after": args.after})
        result = request(path)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _dispatch_native(args, client):
    if args.command in (None, "agentos", "chat"):
        from garp.terminal import native_chat

        return native_chat(client, args.session, model=args.model)
    if args.command == "session":
        result = (
            {"id": client.create_session()}
            if args.session_action == "new"
            else {"sessions": client.list_sessions()}
        )
    elif args.command == "turn":
        settings = client.session(args.session)
        result = client.turn(
            args.session,
            args.text,
            on_delta=lambda _delta: None,
            model=settings.get("model"),
            effort=settings.get("effort"),
            web_search=settings.get("web_search", False),
            agent_id=settings.get("agent_id", "garp"),
        )
    elif args.command == "schedule":
        result = client.schedule(args.session, args.job, args.after, args.deadline)
    elif args.command == "schedule-prompt":
        result = client.schedule_prompt(
            args.text,
            name=args.name,
            session_id=args.session,
            after_seconds=args.after_seconds,
            cron_expr=args.cron_expr,
            timezone=args.timezone,
            deadline_seconds=args.deadline_seconds,
        )
    elif args.command == "status":
        result = client.check_status(args.schedule)
    else:
        raise ValueError("events is unavailable on AgentOS; use chat /history")
    print(json.dumps(result, ensure_ascii=False))
    return 0


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "demo":
            from garp.demo import demo

            return demo(live=args.live, model=args.model)
        if args.command == "serve":
            from garp.agentos import serve

            return serve(args.port)
        if args.command == "schedule" and args.after < 0:
            raise ValueError("--after must be non-negative")
        token = os.environ.get("GARP_TOKEN")
        base = _base_url(args.url) if args.url else None
        if token and base is None and args.command != "agentos":
            base = "http://127.0.0.1:8787"
        if args.command == "schedule-prompt" and not base:
            raise ValueError("Start garp serve and set GARP_TOKEN to schedule a prompt")
        if args.url and args.command != "agentos":
            try:
                _request(f"{base}/garp/models")
            except _HTTPFailure as exc:
                if str(exc) == "server returned HTTP 404":
                    if args.command == "schedule-prompt":
                        raise ValueError("Scheduled prompts require AgentOS") from None
                    catalog = _request(f"{base}/v1/models")
                    if not isinstance(catalog, dict) or not isinstance(catalog.get("models"), list):
                        raise RuntimeError("server returned invalid JSON") from None
                    return _dispatch(args, base, _request)
                raise

        from garp.demo import native_workspace
        from garp.native_client import NativeChatClient

        if base and not token:
            raise ValueError("GARP_TOKEN is required to connect to an AgentOS service")
        native = (
            nullcontext((base, token))
            if base
            else native_workspace(model=args.model, needs_model=args.command == "turn")
        )
        with native as (base, token):
            client = NativeChatClient(base, token)
            try:
                return _dispatch_native(args, client)
            finally:
                client.close()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except ValueError as exc:
        hint = ""
        if str(exc) == "OPENROUTER_API_KEY is required":
            hint = "\nExport the key or run uv run --env-file .env --locked garp."
        print(f"garp: {exc}{hint}", file=sys.stderr)
        return 1
    except _HTTPFailure as exc:
        print(f"garp: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        message = str(exc)
        if message == "could not reach server" and args.command in (None, "chat"):
            print(
                "garp: could not reach server. Start garp serve or set GARP_URL.", file=sys.stderr
            )
            return 1
        if message not in {"could not reach server", "server returned invalid JSON"} and not (
            message.startswith("server returned HTTP ") and message[21:].isdigit()
        ):
            message = "request failed"
        print(f"garp: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
