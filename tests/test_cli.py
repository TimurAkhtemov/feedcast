import pytest

from feedcast import cli
from feedcast.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "feedcast" in capsys.readouterr().out


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for command in ("ingest", "edition", "ask", "watch", "feeds", "launchd", "judgments"):
        assert command in out


def test_ask_requires_a_topic():
    with pytest.raises(SystemExit) as exc:
        main(["ask"])
    assert exc.value.code == 2


def test_feeds_passes_through_to_feedcheck(monkeypatch):
    seen = {}
    monkeypatch.setattr("feedcast.feedcheck.main", lambda argv: seen.setdefault("argv", argv) and 0)
    main(["feeds", "check", "https://example.com/rss"])
    assert seen["argv"] == ["check", "https://example.com/rss"]


def test_edition_reports_failure_exit_code(monkeypatch):
    from feedcast import pipeline

    class Ep:
        failed, mp3, duration, segments = True, "x-FAILED.mp3", 4.0, 0

    monkeypatch.setattr(cli, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(cli, "build_services", lambda: object())
    monkeypatch.setattr(pipeline, "edition", lambda svc, **kw: Ep())
    assert main(["edition", "--no-ingest"]) == 1
