import pytest

from feedcast.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "feedcast" in capsys.readouterr().out


def test_stub_commands_exit_nonzero():
    assert main(["ingest"]) == 1
    assert main(["ask", "climate policy"]) == 1
