from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from feedcast.voice import (
    KokoroSynthesizer,
    SaySynthesizer,
    Segment,
    render_episode,
)


@pytest.mark.skipif(
    os.environ.get("FEEDCAST_REAL_TTS") != "1",
    reason="set FEEDCAST_REAL_TTS=1 to run Kokoro",
)
def test_real_kokoro_renders_two_chapters(tmp_path: Path) -> None:
    result = render_episode(
        [
            Segment(
                "Weather",
                ["A steady rain will move through the city this afternoon."],
            ),
            Segment(
                "Science",
                ["Researchers published a new map of the night sky today."],
            ),
        ],
        tmp_path / "kokoro.mp3",
        synth=KokoroSynthesizer(),
        title="Kokoro check",
    )

    assert result.duration > 3
    assert [chapter.title for chapter in result.chapters] == ["Weather", "Science"]


@pytest.mark.skipif(shutil.which("say") is None, reason="macOS say is unavailable")
def test_say_synthesizer() -> None:
    try:
        audio = SaySynthesizer().synthesize("Feedcast voice check.")
    except RuntimeError as error:
        if "generated no audio" in str(error):
            pytest.skip("say cannot produce audio in this sandbox")
        raise

    assert audio.ndim == 1
    assert audio.size > 0
    assert audio.dtype.name == "float32"
