from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

import feedcast.voice as voice
from feedcast.voice import Segment, probe_chapters, probe_duration, render_episode


class FakeSynthesizer:
    sample_rate = 24_000

    def __init__(self, amplitude: float = 0.3) -> None:
        self.amplitude = amplitude

    def synthesize(self, text: str) -> np.ndarray:
        sample_count = round(0.05 * len(text) * self.sample_rate)
        time = np.arange(sample_count, dtype=np.float32) / self.sample_rate
        return (
            self.amplitude * np.sin(2 * np.pi * 440 * time)
        ).astype(np.float32)


class BrokenSynthesizer:
    sample_rate = 24_000

    def synthesize(self, text: str) -> np.ndarray:
        raise RuntimeError("voice unavailable")


def test_renders_duration_and_chapters(tmp_path: Path) -> None:
    segments = [
        Segment("First", ["alpha beta", "gamma delta"]),
        Segment("Second", ["epsilon zeta", "eta theta"]),
        Segment("Third", ["iota kappa", "lambda mu"]),
    ]
    output = tmp_path / "episode.mp3"
    updates: list[tuple[int, int]] = []

    result = render_episode(
        segments,
        output,
        synth=FakeSynthesizer(),
        title="Morning Feed",
        progress=lambda complete, total: updates.append((complete, total)),
    )

    paragraph_seconds = sum(
        0.05 * len(paragraph)
        for segment in segments
        for paragraph in segment.paragraphs
    )
    expected_duration = paragraph_seconds + 3 * 0.35 + 2 * 0.9 + 0.5
    expected_starts = [
        0.25,
        0.25 + 0.5 + 0.35 + 0.55 + 0.9,
        0.25 + 0.5 + 0.35 + 0.55 + 0.9 + 0.6 + 0.35 + 0.45 + 0.9,
    ]
    assert output.exists()
    assert result.path == output
    assert result.duration == pytest.approx(expected_duration, abs=0.25)
    assert [chapter.title for chapter in result.chapters] == [
        "First",
        "Second",
        "Third",
    ]
    assert all(
        first.start < second.start
        for first, second in zip(result.chapters, result.chapters[1:])
    )
    for chapter, expected_start in zip(result.chapters, expected_starts):
        assert chapter.start == pytest.approx(expected_start, abs=0.1)
    assert result.chapters[-1].end == pytest.approx(result.duration, abs=0.1)
    assert updates == [(index, 6) for index in range(1, 7)]


def test_writes_id3_tags(tmp_path: Path) -> None:
    output = tmp_path / "tagged.mp3"
    render_episode(
        [Segment("Story", ["a sufficiently long fake spoken paragraph"])],
        output,
        synth=FakeSynthesizer(),
        title="Daily Brief",
        metadata={"artist": "Feedcast Desk", "date": "2026-10-13"},
    )

    completed = subprocess.run(
        [
            "/opt/homebrew/bin/ffprobe",
            "-v",
            "error",
            "-show_format",
            "-of",
            "json",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    tags = {
        key.lower(): value
        for key, value in json.loads(completed.stdout)["format"]["tags"].items()
    }
    assert tags["title"] == "Daily Brief"
    assert tags["artist"] == "Feedcast Desk"
    assert tags["album"] == "feedcast"
    assert tags["genre"] == "Podcast"


@pytest.mark.parametrize("amplitude", [0.3, 0.015])
def test_normalizes_integrated_loudness(tmp_path: Path, amplitude: float) -> None:
    output = tmp_path / f"normalized-{amplitude}.mp3"
    render_episode(
        [Segment("Level", ["x" * 100])],
        output,
        synth=FakeSynthesizer(amplitude),
        title="Level check",
        target_lufs=-16.0,
    )

    assert _integrated_loudness(output) == pytest.approx(-16.0, abs=1.5)


def test_atomic_write_replaces_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "episode.mp3"
    output.write_bytes(b"old partial episode")

    render_episode(
        [Segment("Story", ["replacement audio is long enough"])],
        output,
        synth=FakeSynthesizer(),
        title="Replacement",
    )

    assert output.read_bytes()[:3] == b"ID3"
    assert not any(path.name.endswith(".tmp") for path in tmp_path.iterdir())


def test_empty_input_raises_and_blank_content_is_skipped(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing to say"):
        render_episode(
            [], tmp_path / "empty.mp3", synth=FakeSynthesizer(), title="Empty"
        )
    with pytest.raises(ValueError, match="nothing to say"):
        render_episode(
            [Segment("Blank", ["", "  \n "])],
            tmp_path / "blank.mp3",
            synth=FakeSynthesizer(),
            title="Blank",
        )

    result = render_episode(
        [
            Segment("Empty", [" "]),
            Segment("Kept", ["", "spoken words", "\n"]),
        ],
        tmp_path / "filtered.mp3",
        synth=FakeSynthesizer(),
        title="Filtered",
    )
    assert [chapter.title for chapter in result.chapters] == ["Kept"]


def test_notice_uses_tones_when_voices_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(voice, "SaySynthesizer", BrokenSynthesizer)
    output = tmp_path / "failure.mp3"

    result = voice.render_notice(
        ["The edition failed."],
        output,
        title="Feedcast failure",
        synth=BrokenSynthesizer(),
    )

    assert output.exists()
    assert result.duration > 0.5
    assert [chapter.title for chapter in result.chapters] == ["Feedcast failure"]


def test_probe_helpers_read_produced_file(tmp_path: Path) -> None:
    output = tmp_path / "probed.mp3"
    result = render_episode(
        [Segment("Probe me", ["this text makes a deterministic tone"])],
        output,
        synth=FakeSynthesizer(),
        title="Probe",
    )

    assert probe_duration(output) == result.duration
    assert probe_chapters(output) == result.chapters


def _integrated_loudness(path: Path) -> float:
    completed = subprocess.run(
        [
            "/opt/homebrew/bin/ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    blocks = re.findall(r"\{[^{}]+\}", completed.stderr, flags=re.DOTALL)
    return float(json.loads(blocks[-1])["input_i"])
