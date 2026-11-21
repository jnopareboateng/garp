"""Synchronous client for the native AgentOS chat endpoints."""

import json
from urllib.parse import quote, urlsplit
from uuid import UUID, uuid4

import httpx

from garp.cli import _base_url


def _sources(event):
    citations = event.get("citations") or {}
    urls = citations.get("urls", []) if isinstance(citations, dict) else []
    result = []
    for item in urls or []:
        if not isinstance(item, dict) or not isinstance(item.get("url"), str):
            continue
        try:
            parts = urlsplit(item["url"])
            if (
                parts.scheme not in ("http", "https")
                or not parts.hostname
                or parts.username
                or parts.password
            ):
                continue
            _ = parts.port
        except ValueError:
            continue
        result.append({"title": item.get("title") or item["url"], "url": item["url"]})
    return result


def _delegated_child(tool):
    if not isinstance(tool, dict) or tool.get("tool_name") != "delegate_task":
        return None
    try:
        result = json.loads(tool.get("result"))
        child = {"agent_id": result["agent_id"], "session_id": str(UUID(result["session_id"]))}
        if isinstance(child["agent_id"], str):
            return child
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


class NativeChatClient:
    def __init__(self, base: str, token: str):
        self.base = _base_url(base)
        if token and (
            not token.isascii() or not token.isprintable() or any(c.isspace() for c in token)
        ):
            raise ValueError("invalid GARP_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        local = urlsplit(self.base).hostname in {"localhost", "127.0.0.1", "::1"}
        self.client = httpx.Client(
            headers=headers, follow_redirects=False, trust_env=not local, timeout=100
        )

    def _request(self, method, path, **kwargs):
        try:
            response = self.client.request(method, self.base + path, **kwargs)
            if not 200 <= response.status_code < 300:
                raise RuntimeError(f"server returned HTTP {response.status_code}")
            if response.status_code == 204:
                return None
            return response.json()
        except httpx.HTTPError:
            raise RuntimeError("could not reach server") from None
        except ValueError:
            raise RuntimeError("server returned invalid JSON") from None

    def close(self):
        self.client.close()

    def catalog(self):
        return self._request("GET", "/garp/models")

    def create_session(self):
        return self._request(
            "POST", "/sessions?type=agent", json={"agent_id": "garp", "user_id": "local"}
        )["session_id"]

    def remember(self, fact):
        self._request("POST", "/memories", json={"user_id": "local", "memory": fact})

    def memories(self):
        memories = []
        page = 1
        while True:
            response = self._request(
                "GET", "/memories", params={"user_id": "local", "page": page, "limit": 100}
            )
            memories.extend(response["data"])
            if page >= response["meta"]["total_pages"]:
                return memories
            page += 1

    def forget(self, memory_id):
        self._request("DELETE", f"/memories/{quote(memory_id, safe='')}?user_id=local")

    def list_sessions(self, title=None):
        result, page = [], 1
        while True:
            params = {"type": "agent", "page": page, "limit": 100}
            if title:
                params["session_name"] = title
            response = self._request(
                "GET",
                "/sessions",
                params=params,
            )
            rows = response["data"]
            result.extend(
                {"id": row["session_id"], "title": row.get("session_name")} for row in rows
            )
            if len(rows) < 100:
                return result
            page += 1

    def find_chats(self, query):
        return self._request(
            "POST", "/garp/transcripts/search", json={"query": query, "session_type": "agent"}
        )

    def _session(self, session_id):
        return self._request("GET", f"/sessions/{quote(session_id, safe='')}?type=agent")

    def session(self, session_id):
        data = self._session(session_id)
        state = data.get("session_state") or {}
        options = state.get("garp_options") or {}
        return {
            "id": data["session_id"],
            "title": data.get("session_name"),
            "agent_id": data.get("agent_id") or "garp",
            **{key: options[key] for key in ("model", "effort", "web_search") if key in options},
        }

    def set_options(self, session_id, options):
        if self._runs(session_id, ["PENDING", "RUNNING", "PAUSED"]):
            raise ValueError("Finish or cancel the active turn before changing settings")
        data = self._session(session_id)
        state = data.get("session_state") or {}
        current = state.get("garp_options") or {}
        catalog = self.catalog()
        previous_model = current.get("model") or catalog["default"]
        selected = {**current, **options}
        model = selected.get("model") or catalog["default"]
        if model != previous_model:
            if "effort" not in options:
                selected["effort"] = None
            if "web_search" not in options:
                selected["web_search"] = False
        choice = next((row for row in catalog["models"] if row["id"] == model), None)
        if choice is None or not choice.get("available"):
            raise ValueError("Unknown or unavailable model")
        if selected.get("effort") is not None and selected["effort"] not in choice.get(
            "efforts", []
        ):
            raise ValueError("This model does not support that reasoning effort")
        if selected.get("web_search") and not choice.get("native_search"):
            raise ValueError("Native search is unavailable for this model")
        state["garp_options"] = selected
        self._request(
            "PATCH",
            f"/sessions/{quote(session_id, safe='')}?type=agent",
            json={"session_state": state},
        )
        return state["garp_options"]

    def _runs(self, session_id, status):
        return self._request(
            "GET",
            f"/sessions/{quote(session_id, safe='')}/runs",
            params={"type": "agent", "status": status},
        )

    def cancel_pending(self, session_id):
        runs = self._runs(session_id, ["PENDING", "RUNNING", "PAUSED"])
        if not runs:
            return False
        agent_id = self.session(session_id)["agent_id"]
        run_id = runs[0]["run_id"]
        self._request(
            "POST",
            f"/agents/{quote(agent_id, safe='')}/runs/{quote(run_id, safe='')}/cancel",
            params={"session_id": session_id},
        )
        return True

    def approvals(self, session_id):
        records = []
        for run in self._runs(session_id, "PAUSED"):
            result = self._request("GET", "/approvals", params={"run_id": run["run_id"]})
            records.extend(result["data"])
        return records

    def continue_approval(self, approval_id):
        return self._request(
            "POST", f"/garp/approvals/{quote(str(UUID(approval_id)), safe='')}/continue"
        )

    def history(self, session_id):
        result = []
        for run in self._runs(session_id, "COMPLETED"):
            if run.get("parent_run_id"):
                continue
            prompt, answer = run.get("run_input"), run.get("content")
            if isinstance(prompt, str) and prompt:
                result.append({"role": "user", "content": prompt})
            if isinstance(answer, str):
                row = {"role": "assistant", "content": answer}
                if (metadata := run.get("metadata") or {}).get("garp_schedule_id") or metadata.get(
                    "garp_legacy_schedule_result"
                ):
                    row["kind"] = "job_check"
                if sources := _sources(run):
                    row["sources"] = sources
                if children := [
                    item for tool in run.get("tools") or [] if (item := _delegated_child(tool))
                ]:
                    row["children"] = children
                result.append(row)
        return result

    def schedule(self, session_id, job, delay, deadline=None):
        body = {"job_ref": job, "delay_seconds": delay}
        if deadline is not None:
            body["deadline_seconds"] = deadline
        return self._request(
            "POST",
            f"/garp/sessions/{quote(session_id, safe='')}/checks",
            json=body,
        )

    def check_status(self, check_id):
        return self._request("GET", f"/garp/checks/{quote(check_id, safe='')}")

    def schedule_prompt(self, text, **options):
        if not self.catalog().get("prompt_schedules", False):
            raise ValueError("Scheduled prompts are unavailable on this service")
        return self._request("POST", "/garp/prompt-schedules", json={"message": text, **options})

    def turn(
        self,
        session_id,
        text,
        *,
        on_delta,
        model=None,
        effort=None,
        web_search=False,
        agent_id="garp",
        on_delegation=None,
    ):
        start_path = f"/agents/{quote(agent_id, safe='')}/runs"
        path = start_path
        options = {"web_search": web_search}
        if model is not None:
            options["model_id"] = model
        if effort is not None:
            options["effort"] = effort
        data = {
            "message": text,
            "session_id": session_id,
            "user_id": "local",
            "stream": "true",
            "background": "true",
            "factory_input": json.dumps(options),
        }
        durable = self.catalog().get("durable_stream_reconnect") is True
        key = str(uuid4())
        original_data = data
        run_id, last_index, chunks, sources = None, -1, [], []
        children = {}
        failures = 0
        try:
            while True:
                try:
                    with self.client.stream(
                        "POST",
                        self.base + path,
                        data=data,
                        headers={"Accept": "text/event-stream", "Idempotency-Key": key},
                    ) as response:
                        if response.status_code == 400 and path.endswith("/resume") and durable:
                            path, data = start_path, original_data
                            continue
                        if not 200 <= response.status_code < 300:
                            raise RuntimeError(f"server returned HTTP {response.status_code}")
                        lines = []
                        for line in response.iter_lines():
                            if line:
                                if line.startswith("data:"):
                                    lines.append(line[5:].lstrip())
                                continue
                            if not lines:
                                continue
                            try:
                                event = json.loads("\n".join(lines))
                            except ValueError:
                                raise RuntimeError("received an invalid response stream") from None
                            lines = []
                            if not isinstance(event, dict):
                                raise RuntimeError("received an invalid response stream")  # noqa: TRY004
                            incoming_id = event.get("run_id")
                            if incoming_id:
                                if run_id and incoming_id != run_id:
                                    raise RuntimeError("received an invalid response stream")
                                run_id = incoming_id
                            kind = event.get("event")
                            if kind in ("replay", "catch_up"):
                                continue
                            index = event.get("event_index")
                            if (
                                path == start_path
                                and failures
                                and last_index >= 0
                                and not isinstance(index, int)
                            ):
                                raise RuntimeError("received an invalid response stream")
                            if isinstance(index, int):
                                if index <= last_index:
                                    continue
                                last_index = index
                            if kind == "RunContent" and isinstance(event.get("content"), str):
                                chunks.append(event["content"])
                                on_delta(event["content"])
                            if found := _sources(event):
                                sources = found
                            if kind == "ToolCallCompleted" and (
                                child := _delegated_child(event.get("tool"))
                            ):
                                child_id = child["session_id"]
                                if child_id not in children:
                                    children[child_id] = child
                                    if on_delegation:
                                        on_delegation(child)
                            if kind in {"RunCompleted", "RunPaused"}:
                                content = event.get("content")
                                return {
                                    "content": content
                                    if isinstance(content, str)
                                    else "".join(chunks),
                                    "sources": sources,
                                    "run_id": run_id,
                                    **({"paused": True} if kind == "RunPaused" else {}),
                                    **({"children": list(children.values())} if children else {}),
                                }
                            if kind in {"RunError", "RunCancelled"}:
                                raise RuntimeError("agent turn failed")
                    raise httpx.ReadError("stream ended")
                except httpx.TransportError:
                    if failures >= 3 or (not durable and (not run_id or last_index < 0)):
                        raise RuntimeError(
                            "connection closed before the reply was confirmed"
                        ) from None
                    failures += 1
                    if run_id and last_index >= 0:
                        path = f"{start_path}/{quote(run_id, safe='')}/resume"
                        data = {"session_id": session_id, "last_event_index": str(last_index)}
                    else:
                        path, data = start_path, original_data
        except KeyboardInterrupt:
            if run_id:
                try:
                    self._request(
                        "POST",
                        f"{start_path}/{quote(run_id, safe='')}/cancel",
                        params={"session_id": session_id},
                    )
                except RuntimeError:
                    pass
            raise
