from pathlib import Path

from garp import demo


def test_demo_starts_and_removes_its_workspace(postgres_dsn, tmp_path, monkeypatch):
    directory = tmp_path / "demo"
    directory.mkdir(mode=0o700)
    monkeypatch.setattr(demo, "mkdtemp", lambda **_kwargs: str(directory))

    def inspect_demo(base, session, request, *, demo_mode, model):
        assert session is None
        assert model is None
        assert demo_mode == "Local demo · scripted replies"
        assert Path(directory / "pg" / "postmaster.pid").exists()
        sid = request(f"{base}/v1/sessions", "POST", {})["id"]
        assert request(f"{base}/v1/sessions/{sid}")["title"] is None
        return 0

    monkeypatch.setattr(demo, "chat", inspect_demo)
    assert demo.demo() == 0
    assert not directory.exists()
