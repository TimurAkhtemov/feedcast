"""Settings from config.yaml layered over defaults, plus the standing topics.

Thresholds live here rather than in code so they can be tuned against the judgment log
without touching the pipeline.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from feedcast import paths

CONFIG_FILE = paths.PROJECT_ROOT / "config.yaml"

DEFAULTS: dict[str, dict[str, Any]] = {
    "judge": {"ad_threshold": 0.8, "topic_threshold": 0.5, "batch_size": 16, "shortlist": 400},
    "ingest": {
        "max_entries_per_feed": 40,
        "max_age_hours": 72,
        "keep_days": 21,
        "concurrency": 8,
        "timeout_seconds": 20,
    },
    "group": {"embed_model": "embeddinggemma", "similarity": 0.75},
    "edition": {"target_minutes": 12, "window_hours": 24, "topic_stories": 2},
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
        "words_per_minute": 150,
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


def load_topics(path: Path | None = None) -> list[str]:
    path = path or paths.TOPICS_FILE
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    return [str(t).strip() for t in data.get("topics") or [] if str(t).strip()]


def negation_warning(topic: str) -> str | None:
    """Laya handles negation badly, so a topic phrased negatively gets flagged, not judged silently."""
    padded = f" {topic.lower()} "
    for word in NEGATIONS:
        if f" {word}" in padded or (word.startswith("non") and "non-" in padded):
            return f"Topic {topic!r} contains a negation ({word.strip()!r}); phrase it positively."
    return None
