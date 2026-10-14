"""Settings from config.yaml layered over defaults, plus the standing topics.

Thresholds live here rather than in code so they can be tuned against the judgment log
without touching the pipeline.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from feedcast import paths

CONFIG_FILE = paths.PROJECT_ROOT / "config.yaml"

DEFAULTS: dict[str, dict[str, Any]] = {
    "judge": {
        "ad_threshold": 0.8,
        "topic_threshold": 0.5,
        "skip_threshold": 0.8,
        "batch_size": 16,
        "shortlist": 400,
    },
    "ingest": {
        "max_entries_per_feed": 40,
        "max_age_hours": 72,
        "keep_days": 21,
        "concurrency": 8,
        "timeout_seconds": 20,
    },
    "group": {"embed_model": "embeddinggemma", "similarity": 0.75},
    "edition": {
        "target_minutes": 12,
        "days": ["mon", "wed", "fri"],
        "time": "05:30",
        "min_window_hours": 24,
        "topic_stories": 2,
    },
    "ask": {
        "target_minutes": 6,
        "window_hours": 96,
        "min_matches": 5,
        "search_queries": 3,
        "max_stories": 5,
    },
    "write": {
        "model": "qwen3.6:35b",
        "temperature": 0.4,
        "words_per_minute": 130,
        "max_source_chars": 14000,
    },
    "voice": {
        "model": "mlx-community/Kokoro-82M-bf16",
        "voice": "af_heart",
        "speed": 1.0,
        "bitrate": "96k",
        "target_lufs": -16,
    },
    "services": {
        "ollama_url": "http://127.0.0.1:11434",
        "searxng_url": "http://127.0.0.1:8888",
    },
}

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")  # datetime.weekday() order

NEGATIONS = ("not ", "no ", "non-", "without ", "except ", "excluding ", "other than ", "n't ")


def load_config(path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = path or CONFIG_FILE
    config = copy.deepcopy(DEFAULTS)
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        for section, values in data.items():
            if section not in config:
                raise ValueError(f"{path.name}: unknown section {section!r}")
            unknown = set(values or {}) - set(config[section])
            if unknown:
                raise ValueError(f"{path.name}: unknown keys in {section}: {sorted(unknown)}")
            config[section].update(values or {})
    return config


def edition_schedule(edition: dict[str, Any]) -> tuple[list[int], int, int]:
    """The edition's (weekdays, hour, minute), weekdays as datetime.weekday() numbers."""
    days = [str(d).strip().lower()[:3] for d in edition["days"]]
    bad = [d for d in days if d not in DAYS]
    if bad:
        raise ValueError(f"edition.days: unknown day(s) {bad}; use {', '.join(DAYS)}")
    try:
        hour, minute = (int(part) for part in str(edition["time"]).split(":"))
    except ValueError:
        raise ValueError(f"edition.time: expected HH:MM, got {edition['time']!r}") from None
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"edition.time: expected HH:MM, got {edition['time']!r}")
    return sorted({DAYS.index(d) for d in days}), hour, minute


def edition_since(now: datetime, edition: dict[str, Any]) -> datetime:
    """Start of an edition's news window: the last scheduled edition at least
    min_window_hours before now.

    With Monday, Wednesday and Friday editions, Monday's covers the weekend from Friday
    morning and the other two cover two days each. A manual run covers everything since
    the last scheduled edition that's at least min_window_hours old.
    """
    latest = now - timedelta(hours=edition["min_window_hours"])
    weekdays, hour, minute = edition_schedule(edition)
    slot = latest.replace(hour=hour, minute=minute, second=0, microsecond=0)
    for _ in range(8):
        if slot <= latest and slot.weekday() in weekdays:
            return slot
        slot -= timedelta(days=1)
    return latest  # no days scheduled


def _topics_list(path: Path | None, key: str) -> list[str]:
    path = path or paths.TOPICS_FILE
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    return [str(t).strip() for t in data.get(key) or [] if str(t).strip()]


def load_topics(path: Path | None = None) -> list[str]:
    """Standing topics: each gets its own stories in every edition."""
    return _topics_list(path, "topics")


def load_skip(path: Path | None = None) -> list[str]:
    """Subjects left out of the edition's top stories."""
    return _topics_list(path, "skip")


def negation_warning(topic: str) -> str | None:
    """Laya handles negation badly, so a topic phrased negatively gets flagged, not judged silently."""
    padded = f" {topic.lower()} "
    for word in NEGATIONS:
        if f" {word}" in padded or (word.startswith("non") and "non-" in padded):
            return f"Topic {topic!r} contains a negation ({word.strip()!r}); phrase it positively."
    return None
