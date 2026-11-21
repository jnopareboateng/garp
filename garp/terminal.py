"""Interactive chat using the same API as the JSON commands."""

import os
import sys
import unicodedata
from urllib.parse import quote
from uuid import UUID, uuid4

try:
    import readline  # noqa: F401 - Enable editing and in-process command history.
except ImportError:
    pass

_COMMANDS = (
    "/model [NUMBER|ID]  Choose a model\n"
    "/effort [LEVEL]     Show or set reasoning effort\n"
    "/search [on|off]    Show or toggle native web search\n"
    "/check JOB SECONDS  Schedule a job check\n"
    "/schedule SECONDS PROMPT  Schedule a prompt in this chat\n"
    "/history            Show this conversation\n"
    "/remember FACT      Save a fact for future chats\n"
    "/memory             List saved facts\n"
    "/forget             Remove a saved fact\n"
    "/cancel             Cancel a run in this chat\n"
    "/approvals          Show approvals for this chat\n"
    "/continue ID        Continue an approved or rejected action\n"
    "/resume             Open a saved chat\n"
    "/find TEXT          Search saved chat messages\n"
    "/new                Start a new chat\n"
    "/refresh            Check for new activity\n"
    "/exit               Close Garp"
)


def _visible(value: str) -> str:
    return "".join(
        char for char in value if char in "\n\t" or not unicodedata.category(char).startswith("C")
    )


def _label(value: str) -> str:
    if sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb":
        return f"\033[1m{value}\033[0m"
    return value


class _LegacyChatClient:
    def __init__(self, base, request):
        self.base, self.request = base, request

    def _session_url(self, session_id):
        return f"{self.base}/v1/sessions/{quote(session_id, safe='')}"

    def create_session(self):
        return self.request(f"{self.base}/v1/sessions", "POST", {})["id"]

    def list_sessions(self):
        return self.request(f"{self.base}/v1/sessions")["sessions"]

    def session(self, session_id):
        return self.request(self._session_url(session_id))

    def catalog(self):
        return self.request(f"{self.base}/v1/models")

    def set_options(self, session_id, options):
        path = self._session_url(session_id)
        if "model" in options:
            return self.request(f"{path}/model", "PUT", {"model": options["model"]})
        return self.request(f"{path}/settings", "PATCH", options)

    def events(self, session_id, after):
        return self.request(f"{self._session_url(session_id)}/events?after={after}")

    def turn(self, session_id, text, *, on_delta):
        result = self.request(
            f"{self._session_url(session_id)}/turns", "POST", {"text": text}, on_delta=on_delta
        )
        return {**result, **result["events"][-1]["data"]}

    def schedule(self, session_id, job, delay):
        return self.request(
            f"{self._session_url(session_id)}/schedules",
            "POST",
            {"job_ref": job, "delay_seconds": delay},
        )


def chat(base, session, request, *, demo_mode=None, model=None):
    return _chat(_LegacyChatClient(base, request), session, demo_mode=demo_mode, model=model)


def native_chat(client, session=None, *, model=None, demo_mode=None):
    return _chat(client, session, model=model, native=True, demo_mode=demo_mode)


def _choose(items, prompt):
    try:
        choice = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not choice:
        return None
    try:
        index = int(choice)
        if 1 <= index <= len(items):
            return items[index - 1]
    except ValueError:
        pass
    print("Choose a number from the list.")
    return None


def _chat(client, session, *, demo_mode=None, model=None, native=False):
    catalog = client.catalog() if native else {}
    check_enabled = not native or catalog.get("job_checks", False)
    scheduling_enabled = catalog.get("prompt_schedules", False)
    approvals_enabled = catalog.get("hosted_approvals", False)
    if model:
        choices = client.catalog()["models"]
        choice = next((row for row in choices if row["id"] == model), None)
        if choice is None:
            raise ValueError("Unknown model. Open a chat and use /model to see the list.")
        if not choice["available"]:
            raise ValueError(_visible(choice["reason"]))
    after = 0
    active = None
    title = None

    def active_session():
        nonlocal active
        if active is None:
            active = client.create_session()
        return active

    def open_chat(session_id, starting_model=None):
        nonlocal after, active, title
        if starting_model:
            session_id = session_id or client.create_session()
            client.set_options(session_id, {"model": starting_model})
        info = client.session(session_id) if session_id else {"model": client.catalog()["default"]}
        active, after = session_id, 0
        title = info.get("title")
        heading = _visible(title or "New chat")
        print(f"\n{_label('Garp')}  ·  {heading}")
        if native and (specialist := info.get("agent_id")) and specialist != "garp":
            print(f"Specialist {_visible(specialist)}")
        if demo_mode:
            print(demo_mode)
        current = info.get("model") or (client.catalog()["default"] if native else None)
        if current and demo_mode != "Local demo · scripted replies":
            print(_visible(current))
        print("Type a message   /resume saved chats   /help")
        replay()

    def replay(*, since=None, skip=()):
        nonlocal after
        if active is None:
            return False
        if native:
            history = client.history(active)
            for message in history:
                _show_message(message)
            return bool(history)
        cursor = after if since is None else since
        shown = False
        while True:
            page = client.events(active, cursor)
            for event in page["events"]:
                if event.get("seq") in skip:
                    continue
                data = event["data"]
                if event["kind"] == "turn.message":
                    _show_message(data)
                    shown = True
                elif event["kind"] == "schedule.result":
                    evidence = data["evidence"]
                    state = evidence.get("state", data["outcome"])
                    progress = evidence.get("progress")
                    summary = f"{state} · progress {progress}" if progress else state
                    print(f"\n{_label('Job check')}  {_visible(summary)}")
                    if observed := evidence.get("last_seen_at"):
                        print(f"Last observed  {_visible(observed)}")
                    shown = True
            cursor = page["cursor"]
            after = max(after, cursor)
            if len(page["events"]) < 200:
                return shown

    open_chat(session, model)
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text == "/exit":
            return 0
        try:
            if text == "/history":
                print(_label("Conversation"))
                if not replay(since=0):
                    print("No messages yet. Type a message to start.")
            elif text == "/refresh" and not native:
                if not replay():
                    print("No new activity.")
            elif text == "/help":
                for line in _COMMANDS.splitlines():
                    if native and line.startswith("/resume"):
                        print("/resume [TITLE|ID]  Find or open a saved chat")
                        continue
                    if line.startswith("/refresh") and native:
                        continue
                    if line.startswith("/check") and not check_enabled:
                        continue
                    if line.startswith("/schedule") and not scheduling_enabled:
                        continue
                    if not approvals_enabled and line.startswith(("/approvals", "/continue")):
                        continue
                    if not native and line.startswith(
                        ("/remember", "/memory", "/forget", "/cancel", "/find")
                    ):
                        continue
                    print(line)
            elif native and text.split(maxsplit=1)[0] == "/remember":
                parts = text.split(maxsplit=1)
                if len(parts) == 1:
                    print("Use /remember followed by the fact to save.")
                    continue
                client.remember(parts[1])
                print("Saved to memory.")
            elif native and text in ("/memory", "/forget"):
                memories = client.memories()
                if not memories:
                    print("No saved facts. Use /remember to save one.")
                    continue
                print(_label("Saved facts"))
                for index, memory in enumerate(memories, 1):
                    print(f"  {index:>2}  {_visible(memory['memory'])}")
                if text == "/forget" and (
                    item := _choose(memories, "Remove fact number (Enter to cancel): ")
                ):
                    client.forget(item["memory_id"])
                    print("Removed from memory.")
            elif native and text == "/cancel":
                if active and client.cancel_pending(active):
                    print("Cancellation requested. Separate specialist chats keep running.")
                else:
                    print("No run to cancel in this chat.")
            elif approvals_enabled and text == "/approvals":
                rows = client.approvals(active) if active else []
                if not rows:
                    print("No paused approvals in this chat.")
                    continue
                for row in rows:
                    print(
                        f"{_visible(row['status'])}  {_visible(row.get('tool_name') or 'action')}  {_visible(row['id'])}"
                    )
            elif approvals_enabled and text.split(maxsplit=1)[0] == "/continue":
                parts = text.split(maxsplit=1)
                if len(parts) == 1:
                    print("Use /continue followed by an approval ID from /approvals.")
                    continue
                client.continue_approval(parts[1])
                print("Continuation queued. Use /history to see the result.")
            elif text == "/resume" or (
                native and text.split(maxsplit=1)[0] in ("/resume", "/find")
            ):
                command, *args = text.split(maxsplit=1)
                query = args[0] if args else ""
                find = command == "/find"
                if find and not 1 <= len(query) <= 200:
                    print("Use /find followed by 1 to 200 characters to search saved chats.")
                    continue
                if native and not find and len(query) == 36:
                    try:
                        session_id = str(UUID(query))
                    except ValueError:
                        pass
                    else:
                        open_chat(session_id)
                        continue
                if find:
                    result = client.find_chats(query)
                    saved = [{**item, "id": item["session_id"]} for item in result["matches"]]
                else:
                    sessions = client.list_sessions(query) if query else client.list_sessions()
                    saved = [item for item in sessions if item["id"] != active]
                if not saved:
                    print("No matching saved chats." if query else "No other chats yet.")
                    continue
                print(_label("Saved chats"))
                for index, item in enumerate(saved, 1):
                    print(f"  {index:>2}  {_visible(item.get('title') or 'Untitled chat')}")
                    if find:
                        print(f"      {_visible(' '.join(item['excerpt'].split()))}")
                if find and result["more"]:
                    print("More chats match. Narrow your search to see other results.")
                if item := _choose(saved, "Chat number (Enter to cancel): "):
                    open_chat(item["id"])
            elif text == "/new":
                open_chat(None)
            elif text.split()[0] == "/model":
                catalog = client.catalog()
                entries = catalog["models"]
                parts = text.split(maxsplit=1)
                if len(parts) == 1:
                    current = (client.session(active).get("model") if active else None) or catalog[
                        "default"
                    ]
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
                    selected = client.set_options(active_session(), {"model": choice["id"]})
                    print(f"Model  {_visible(selected['model'])}")
            elif text.split()[0] == "/effort":
                parts = text.split(maxsplit=1)
                if len(parts) == 1:
                    catalog = client.catalog()
                    current = client.session(active) if active else {}
                    model_id = current.get("model") or catalog["default"]
                    preset = next((row for row in catalog["models"] if row["id"] == model_id), {})
                    print(f"Reasoning effort  {_visible(current.get('effort') or 'default')}")
                    options = ", ".join(preset.get("efforts", ()))
                    print(
                        f"Choices  default, {_visible(options)}"
                        if options
                        else "This route exposes no selectable effort levels."
                    )
                    continue
                effort = None if parts[1] == "default" else parts[1]
                current = client.set_options(active_session(), {"effort": effort})
                print(f"Reasoning effort  {_visible(current['effort'] or 'default')}")
            elif text.split()[0] == "/search":
                parts = text.split(maxsplit=1)
                if len(parts) == 1:
                    current = client.session(active) if active else {}
                    print("Native web search  " + ("on" if current.get("web_search") else "off"))
                    continue
                if parts[1] not in ("on", "off"):
                    print("Use /search on or /search off.")
                    continue
                current = client.set_options(active_session(), {"web_search": parts[1] == "on"})
                print("Native web search  " + ("on" if current["web_search"] else "off"))
            elif text.split()[0] == "/check" and check_enabled:
                parts = text.split()
                try:
                    if len(parts) != 3 or int(parts[2]) < 0:
                        raise ValueError
                    delay = int(parts[2])
                except ValueError:
                    print("Use /check JOB SECONDS, with a nonnegative number of seconds.")
                    continue
                scheduled = client.schedule(active_session(), parts[1], delay)
                follow_up = "/history" if native else "/refresh"
                print(f"Check scheduled in {delay}s. Use {follow_up} to see the result.")
                print(f"Check ID  {_visible(scheduled['id'])}")
            elif text.split()[0] == "/schedule" and scheduling_enabled:
                parts = text.split(maxsplit=2)
                try:
                    delay = int(parts[1]) if len(parts) == 3 else -1
                    if not 0 <= delay <= 31_536_000:
                        raise ValueError
                except ValueError:
                    print("Use /schedule SECONDS PROMPT, with a delay of 0 to 31536000 seconds.")
                    continue
                scheduled = client.schedule_prompt(
                    parts[2],
                    name=f"Scheduled prompt {uuid4().hex[:8]}",
                    session_id=active_session(),
                    after_seconds=delay,
                )
                print(f"Prompt scheduled in {delay}s. Use /history to see the result.")
                print(f"Schedule ID  {_visible(scheduled['id'])}")
            elif text.startswith("/"):
                print("Unknown command. Use /help to see available commands.")
            else:
                session_id = active_session()
                options = {}
                if native:
                    selected = client.session(session_id)
                    options = {
                        "model": selected.get("model"),
                        "effort": selected.get("effort"),
                        "web_search": selected.get("web_search", False),
                        "agent_id": selected.get("agent_id", "garp"),
                        "on_delegation": _announce_child,
                    }
                if turn := _reply(client, session_id, text, options):
                    new_title = (
                        client.session(session_id).get("title") if native else turn.get("title")
                    )
                    if new_title and new_title != title:
                        title = new_title
                        print(f"Saved as {_visible(title)}")
                    if not native:
                        replay(skip={event["seq"] for event in turn["events"]})
        except (RuntimeError, ValueError) as exc:
            hint = "/history" if native else "/refresh"
            print(f"{_visible(str(exc))}. Use {hint} to check for updates.", file=sys.stderr)


def _show_message(data):
    if data.get("kind") == "job_check":
        speaker = "Job check"
    else:
        speaker = "You" if data["role"] == "user" else "Garp"
    print(f"\n{_label(speaker)}\n{_visible(data['content'])}")
    _show_sources(data)


def _announce_child(child):
    print(
        f"\nSpecialist {_visible(child['agent_id'])}  /resume {_visible(child['session_id'])}",
        flush=True,
    )


def _show_sources(data, *, show_children=True):
    if sources := data.get("sources"):
        print(_label("Sources"))
        for source in sources:
            print(f"  {_visible(source['title'])}  {_visible(source['url'])}")

    if show_children:
        for child in data.get("children", []):
            _announce_child(child)


def _reply(client, session_id, text, options):
    print(f"\n{_label('Garp')}", flush=True)
    started = False

    def write_delta(delta):
        nonlocal started
        started = True
        print(_visible(delta), end="", flush=True)

    try:
        turn = client.turn(session_id, text, on_delta=write_delta, **options)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed replies are kept in /history.")
        return None
    except RuntimeError:
        if started:
            print("\nReply interrupted.")
        raise

    if not started:
        print(_visible(turn["content"]), end="")
    print()
    if turn.get("paused"):
        print("Run paused for review.")
    _show_sources(turn, show_children="on_delegation" not in options)
    return turn
