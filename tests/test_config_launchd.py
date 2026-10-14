from datetime import datetime, timedelta, timezone

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
    assert edition["StartCalendarInterval"] == [
        {"Weekday": 1, "Hour": 5, "Minute": 30},
        {"Weekday": 3, "Hour": 5, "Minute": 30},
        {"Weekday": 5, "Hour": 5, "Minute": 30},
    ]
    assert watch["WatchPaths"] == [str(paths.REQUESTS_DIR)]
    for spec in agents.values():
        assert spec["ProgramArguments"][0].endswith(".venv/bin/feedcast")
        assert "/opt/homebrew/bin" in spec["EnvironmentVariables"]["PATH"]
        assert "/usr/local/bin" in spec["EnvironmentVariables"]["PATH"]  # docker for SearXNG


def test_launchd_edition_schedule_from_config():
    settings = {**config.DEFAULTS["edition"], "days": ["sun", "Saturday"], "time": "6:05"}
    spec = launchd.agents(settings)["com.feedcast.edition"]
    assert spec["StartCalendarInterval"] == [{"Weekday": 6, "Hour": 6, "Minute": 5},
                                             {"Weekday": 0, "Hour": 6, "Minute": 5}]


@pytest.mark.parametrize("days, time", [(["mon", "funday"], "05:30"), (["mon"], "5.30"), (["mon"], "25:00")])
def test_edition_schedule_rejects_typos(days, time):
    with pytest.raises(ValueError, match="edition"):
        config.edition_schedule({"days": days, "time": time})


EDT = timezone(timedelta(hours=-4))


@pytest.mark.parametrize("now, expected", [
    (datetime(2026, 10, 12, 5, 30, 4, tzinfo=EDT), datetime(2026, 10, 9, 5, 30, tzinfo=EDT)),   # Mon: since Fri
    (datetime(2026, 10, 14, 5, 30, 4, tzinfo=EDT), datetime(2026, 10, 12, 5, 30, tzinfo=EDT)),  # Wed: since Mon
    (datetime(2026, 10, 16, 5, 30, 4, tzinfo=EDT), datetime(2026, 10, 14, 5, 30, tzinfo=EDT)),  # Fri: since Wed
    (datetime(2026, 10, 12, 8, 15, tzinfo=EDT), datetime(2026, 10, 9, 5, 30, tzinfo=EDT)),      # late wake
    (datetime(2026, 10, 13, 14, 0, tzinfo=EDT), datetime(2026, 10, 12, 5, 30, tzinfo=EDT)),     # manual Tue
])
def test_edition_window_reaches_back_to_the_previous_edition(now, expected):
    assert config.edition_since(now, config.DEFAULTS["edition"]) == expected


def test_edition_window_daily_and_unscheduled():
    now = datetime(2026, 10, 14, 5, 30, 4, tzinfo=EDT)
    daily = {**config.DEFAULTS["edition"], "days": list(config.DAYS)}
    assert config.edition_since(now, daily) == datetime(2026, 10, 13, 5, 30, tzinfo=EDT)
    never = {**config.DEFAULTS["edition"], "days": []}
    assert config.edition_since(now, never) == now - timedelta(hours=24)
