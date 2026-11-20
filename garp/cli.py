"""Command line client for the Garp HTTP API."""

import argparse
import json
import os
import sys
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


def _request(url, method="GET", body=None, *, token=None):
    headers = {"Accept": "application/json"}
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
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("could not reach server") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("server returned invalid JSON") from None


def _parser():
    parser = argparse.ArgumentParser(
        prog="garp", description="Open a new Garp chat when run without a command."
    )
    parser.set_defaults(session=None, model=None)
    parser.add_argument("--url", default=os.environ.get("GARP_URL", "http://127.0.0.1:8787"))
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
    turn = commands.add_parser("turn")
    turn.add_argument("session")
    turn.add_argument("text")
    schedule = commands.add_parser("schedule")
    schedule.add_argument("session")
    schedule.add_argument("job")
    schedule.add_argument("--after", type=int, required=True)
    schedule.add_argument("--deadline", type=int, help="Latest start in seconds from now")
    status = commands.add_parser("status")
    status.add_argument("schedule")
    events = commands.add_parser("events")
    events.add_argument("session")
    events.add_argument("--after", type=int)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "demo":
            from garp.demo import demo

            return demo(live=args.live, model=args.model)
        base = _base_url(args.url)
        if args.command in (None, "chat"):
            from garp.terminal import chat

            return chat(base, args.session, _request, model=args.model)
        if args.command == "session" and args.session_action == "list":
            result = _request(f"{base}/v1/sessions")
        elif args.command == "session":
            result = _request(f"{base}/v1/sessions", "POST", {})
        elif args.command == "turn":
            result = _request(
                f"{base}/v1/sessions/{quote(args.session, safe='')}/turns",
                "POST",
                {"text": args.text},
            )
        elif args.command == "schedule":
            if args.after < 0:
                raise ValueError("--after must be non-negative")
            body = {"job_ref": args.job, "delay_seconds": args.after}
            if args.deadline is not None:
                body["deadline_seconds"] = args.deadline
            result = _request(
                f"{base}/v1/sessions/{quote(args.session, safe='')}/schedules",
                "POST",
                body,
            )
        elif args.command == "status":
            result = _request(f"{base}/v1/schedules/{quote(args.schedule, safe='')}")
        else:
            path = f"{base}/v1/sessions/{quote(args.session, safe='')}/events"
            if args.after is not None:
                path += "?" + urlencode({"after": args.after})
            result = _request(path)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except ValueError as exc:
        print(f"garp: {exc}", file=sys.stderr)
        return 1
    except _HTTPFailure as exc:
        print(f"garp: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        message = str(exc)
        if message == "could not reach server" and args.command in (None, "chat"):
            print("garp: could not reach server. Start garp-api or set GARP_URL.", file=sys.stderr)
            return 1
        if message not in {"could not reach server", "server returned invalid JSON"} and not (
            message.startswith("server returned HTTP ") and message[21:].isdigit()
        ):
            message = "request failed"
        print(f"garp: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
