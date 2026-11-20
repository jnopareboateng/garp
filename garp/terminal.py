"""Interactive chat using the same API as the JSON commands."""

import os
import sys
import unicodedata
from urllib.parse import quote

_COMMANDS = (
    "/model [NUMBER|ID]  Show or switch the model preset\n"
    "/effort [LEVEL]     Show or set reasoning effort\n"
    "/search [on|off]    Show or toggle native web search\n"
    "/check JOB SECONDS  Schedule a job check\n"
    "/history            Refresh chat\n"
    "/exit               Close chat"
)


def _visible(value: str) -> str:
    return "".join(
        char for char in value if char in "\n\t" or not unicodedata.category(char).startswith("C")
    )


def _label(value: str) -> str:
    if sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb":
        return f"\033[1m{value}\033[0m"
    return value


def chat(base, session, request, *, demo_mode=None, model=None):
    if model:
        choices = request(f"{base}/v1/models")["models"]
        choice = next((row for row in choices if row["id"] == model), None)
        if choice is None:
            raise ValueError("Unknown model. Open a chat and use /model to see the list.")
        if not choice["available"]:
            raise ValueError(_visible(choice["reason"]))
    if session is None:
        session = request(f"{base}/v1/sessions", "POST", {})["id"]
    path = f"{base}/v1/sessions/{quote(session, safe='')}"
    if model:
        request(f"{path}/model", "PUT", {"model": model})
    info = request(path)
    title = info.get("title")
    if demo_mode:
        print(demo_mode)
    print(_label(f"Garp  /  {_visible(title or 'New chat')}"))
    if current := info.get("model"):
        print(f"Model preset  {_visible(current)}")
    print(f"Session  {_visible(session)}")
    print("/help for commands")
    after = 0

    def replay():
        nonlocal after
        while True:
            page = request(f"{path}/events?after={after}")
            if not page["events"]:
                return
            for event in page["events"]:
                data = event["data"]
                if event["kind"] == "turn.message":
                    speaker = "You" if data["role"] == "user" else "Garp"
                    print(f"\n{_label(speaker)}\n{_visible(data['content'])}")
                    if used := data.get("model"):
                        effort = data.get("effort") or "default"
                        search = " · native search enabled" if data.get("web_search") else ""
                        print(f"{_visible(used)} · effort {_visible(effort)}{search}")
                    if sources := data.get("sources"):
                        print(_label("Sources"))
                        for source in sources:
                            print(f"  {_visible(source['title'])}  {_visible(source['url'])}")
                elif event["kind"] == "schedule.result":
                    evidence = data["evidence"]
                    state = evidence.get("state", data["outcome"])
                    progress = evidence.get("progress")
                    summary = f"{state} · progress {progress}" if progress else state
                    print(f"\n{_label('Job check')}  {_visible(summary)}")
                    if observed := evidence.get("last_seen_at"):
                        print(f"Last observed  {_visible(observed)}")
            after = page["cursor"]
            if len(page["events"]) < 200:
                return

    replay()
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text == "/exit":
            return 0
        try:
            if text == "/history":
                replay()
            elif text == "/help":
                print(_COMMANDS)
            elif text.split()[0] == "/model":
                entries = request(f"{base}/v1/models")["models"]
                parts = text.split(maxsplit=1)
                if len(parts) == 1:
                    current = request(path).get("model")
                    for index, entry in enumerate(entries, 1):
                        marker = "*" if entry["id"] == current else " "
                        print(f"{marker} {index}. {_visible(entry['id'])}")
                        if not entry["available"]:
                            print(f"     {_visible(entry['reason'])}")
                    print("Use /model NUMBER or /model ID to switch.")
                    continue
                value = parts[1]
                choice = next(
                    (
                        entry
                        for index, entry in enumerate(entries, 1)
                        if value in (str(index), entry["id"])
                    ),
                    None,
                )
                if choice is None:
                    print("Unknown model. Use /model to see the list.")
                elif not choice["available"]:
                    print(_visible(choice["reason"]))
                else:
                    selected = request(f"{path}/model", "PUT", {"model": choice["id"]})
                    print(f"Model preset  {_visible(selected['model'])}")
            elif text.split()[0] == "/effort":
                parts = text.split(maxsplit=1)
                if len(parts) == 1:
                    current = request(path)
                    entries = request(f"{base}/v1/models")["models"]
                    preset = next((row for row in entries if row["id"] == current.get("model")), {})
                    print(f"Reasoning effort  {_visible(current.get('effort') or 'default')}")
                    options = ", ".join(preset.get("efforts", ()))
                    print(
                        f"Choices  default, {_visible(options)}"
                        if options
                        else "This route exposes no selectable effort levels."
                    )
                    continue
                effort = None if parts[1] == "default" else parts[1]
                current = request(f"{path}/settings", "PATCH", {"effort": effort})
                print(f"Reasoning effort  {_visible(current['effort'] or 'default')}")
            elif text.split()[0] == "/search":
                parts = text.split(maxsplit=1)
                if len(parts) == 1:
                    current = request(path)
                    print("Native web search  " + ("on" if current.get("web_search") else "off"))
                    continue
                if parts[1] not in ("on", "off"):
                    print("Use /search on or /search off.")
                    continue
                current = request(f"{path}/settings", "PATCH", {"web_search": parts[1] == "on"})
                print("Native web search  " + ("on" if current["web_search"] else "off"))
            elif text.split()[0] == "/check":
                parts = text.split()
                try:
                    if len(parts) != 3 or int(parts[2]) < 0:
                        raise ValueError
                    delay = int(parts[2])
                except ValueError:
                    print("Use /check JOB SECONDS, with a nonnegative number of seconds.")
                    continue
                result = request(
                    f"{path}/schedules", "POST", {"job_ref": parts[1], "delay_seconds": delay}
                )
                print(f"Check scheduled. Use /history to see the result.  {_visible(result['id'])}")
            elif text.startswith("/"):
                print("Unknown command. Use /help to see available commands.")
            else:
                turn = request(f"{path}/turns", "POST", {"text": text})
                replay()
                if turn.get("title") and turn["title"] != title:
                    title = turn["title"]
                    print(f"\n{_label('Chat')}  {_visible(title)}")
        except RuntimeError as exc:
            print(
                f"Request failed: {_visible(str(exc))}. Try again or use /history.", file=sys.stderr
            )
