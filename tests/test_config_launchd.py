import pytest

from feedcast import config, launchd, paths


def test_config_file_overrides_defaults(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("judge:\n  ad_threshold: 0.9\nedition:\n  target_minutes: 15\n")
    cfg = config.load_config(f)
    assert cfg["judge"]["ad_threshold"] == 0.9
    assert cfg["judge"]["topic_threshold"] == config.DEFAULTS["judge"]["topic_threshold"]
    assert cfg["edition"]["target_minutes"] == 15


def test_unknown_config_keys_are_rejected(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("judge:\n  ad_treshold: 0.9\n")
    with pytest.raises(ValueError, match="ad_treshold"):
        config.load_config(f)


def test_repo_config_loads():
    cfg = config.load_config()
    assert cfg["edition"]["target_minutes"] == 12


def test_topics(tmp_path):
    f = tmp_path / "topics.yaml"
    f.write_text("topics:\n  - AI tooling for developers\n  - ''\n")
    assert config.load_topics(f) == ["AI tooling for developers"]
    assert config.load_topics(tmp_path / "missing.yaml") == []


def test_negation_warning():
    assert config.negation_warning("news that is not about sports")
    assert config.negation_warning("non-fiction books")
    assert config.negation_warning("battery recycling") is None
    assert config.negation_warning("nothing burger") is None


def test_launchd_agents():
    agents = launchd.agents()
    ingest, edition, watch = (agents[f"com.feedcast.{n}"] for n in ("ingest", "edition", "watch"))
    assert ingest["StartInterval"] == 3600
    assert ingest["ProgramArguments"][1:] == ["ingest", "--warm"]
    assert edition["StartCalendarInterval"] == {"Hour": 5, "Minute": 30}
    assert watch["WatchPaths"] == [str(paths.REQUESTS_DIR)]
    for spec in agents.values():
        assert spec["ProgramArguments"][0].endswith(".venv/bin/feedcast")
        assert "/opt/homebrew/bin" in spec["EnvironmentVariables"]["PATH"]
        assert "/usr/local/bin" in spec["EnvironmentVariables"]["PATH"]  # docker for SearXNG
