from pathlib import Path

from garp import demo
from garp.native_client import NativeChatClient


def test_demo_starts_and_removes_its_workspace(postgres_dsn, tmp_path, monkeypatch):
    directory = tmp_path / "demo"
    directory.mkdir(mode=0o700)
    monkeypatch.setattr(demo, "mkdtemp", lambda **_kwargs: str(directory))
    from garp import agentos

    clients = []
    original_client = agentos.AsyncClient

    def capture_client(**kwargs):
        client = original_client(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(agentos, "AsyncClient", capture_client)

    def inspect_demo(client, session=None, *, demo_mode, model):
        assert session is None
        assert model is None
        assert demo_mode == "Local demo · scripted replies"
        assert Path(directory / "pg" / "postmaster.pid").exists()
        sid = client.create_session()
        deltas = []
        result = client.turn(sid, "hello", on_delta=deltas.append)
        assert "scripted reply" in result["content"]
        assert "scripted reply" in "".join(deltas)
        assert client.session(sid)["title"] == "Local Harness Demo"
        assert "scripted reply" in client.turn(sid, "again", on_delta=lambda _: None)["content"]
        assert len(clients) == 1 and not clients[0].is_closed
        check = client.schedule(sid, "demo", 0)
        import time

        for _ in range(100):
            if any(row.get("kind") == "job_check" for row in client.history(sid)):
                break
            time.sleep(0.1)
        else:
            raise AssertionError(f"job check did not complete: {client.check_status(check['id'])}")
        return 0

    monkeypatch.setattr(demo, "native_chat", inspect_demo)
    assert demo.demo() == 0
    assert not directory.exists()
    assert len(clients) == 1 and clients[0].is_closed


def test_local_workspace_persists_and_reopens(postgres_dsn, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.delenv("GARP_JOBS_DIR", raising=False)
    monkeypatch.delenv("GARP_POLICY_FILE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with demo.native_workspace(needs_model=False) as (base, token):
        client = NativeChatClient(base, token)
        try:
            session = client.create_session()
            assert (tmp_path / "garp/pg/postmaster.pid").exists()
            assert (tmp_path / "garp").stat().st_mode & 0o777 == 0o700
            try:
                with demo.native_workspace(needs_model=False):
                    assert False, "concurrent workspace should fail"
            except ValueError as exc:
                assert "already in use" in str(exc)
        finally:
            client.close()
    assert not (tmp_path / "garp/pg/postmaster.pid").exists()
    with demo.native_workspace(needs_model=False) as (base, token):
        client = NativeChatClient(base, token)
        try:
            assert client.session(session)["id"] == session
        finally:
            client.close()
    assert not (tmp_path / "garp/pg/postmaster.pid").exists()
