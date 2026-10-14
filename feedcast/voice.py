"""Turn written feed segments into a reliably delivered narrated MP3."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class Segment:
    title: str
    paragraphs: list[str]


@dataclass
class Chapter:
    title: str
    start: float
    end: float


@dataclass
class RenderResult:
    path: Path
    duration: float
    chapters: list[Chapter]


class Synthesizer(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray: ...


class KokoroSynthesizer:
    def __init__(
        self,
        model: str = "mlx-community/Kokoro-82M-bf16",
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
    ) -> None:
        self.model_name = model
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code
        self.sample_rate = 24_000
        self._model: object | None = None

    def synthesize(self, text: str) -> np.ndarray:
        if self._model is None:
            from mlx_audio.tts.utils import load_model

            self._model = load_model(model_path=self.model_name)
            self.sample_rate = int(
                getattr(self._model, "sample_rate", self.sample_rate)
            )

        chunks = [
            np.asarray(result.audio, dtype=np.float32).reshape(-1)
            for result in self._model.generate(
                text=text,
                voice=self.voice,
                speed=self.speed,
                lang_code=self.lang_code,
            )
        ]
        if not chunks:
            raise RuntimeError("Kokoro generated no audio")
        return np.concatenate(chunks).astype(np.float32, copy=False)


class SaySynthesizer:
    sample_rate = 24_000

    def synthesize(self, text: str) -> np.ndarray:
        say = shutil.which("say")
        if say is None:
            raise RuntimeError("macOS say is not installed")

        with tempfile.TemporaryDirectory(prefix="feedcast-say-") as temporary:
            wav_path = Path(temporary) / "speech.wav"
            result = subprocess.run(
                [
                    say,
                    "-o",
                    str(wav_path),
                    "--data-format=LEI16@24000",
                    text,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(_command_error("say", result.stderr))
            try:
                with wave.open(str(wav_path), "rb") as source:
                    if source.getnchannels() != 1 or source.getsampwidth() != 2:
                        raise RuntimeError("say returned an unsupported WAV format")
                    self.sample_rate = source.getframerate()
                    samples = np.frombuffer(
                        source.readframes(source.getnframes()), dtype="<i2"
                    )
            except (OSError, EOFError, wave.Error) as error:
                raise RuntimeError(f"could not read audio from say: {error}") from error
        if samples.size == 0:
            raise RuntimeError("say generated no audio")
        return (samples.astype(np.float32) / 32_768.0).astype(np.float32)


def render_episode(
    segments: list[Segment],
    out_path: Path,
    *,
    synth: Synthesizer,
    title: str,
    metadata: dict[str, str] | None = None,
    paragraph_pause: float = 0.35,
    segment_pause: float = 0.9,
    target_lufs: float = -16.0,
    bitrate: str = "96k",
    workdir: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> RenderResult:
    prepared = [
        (
            segment.title,
            [
                text
                for paragraph in segment.paragraphs
                if (text := paragraph.strip())
            ],
        )
        for segment in segments
    ]
    prepared = [(chapter_title, texts) for chapter_title, texts in prepared if texts]
    if not prepared:
        raise ValueError("there is nothing to say")
    if paragraph_pause < 0 or segment_pause < 0:
        raise ValueError("audio pauses cannot be negative")

    total = sum(len(texts) for _, texts in prepared)
    complete = 0
    rendered: list[tuple[str, list[np.ndarray]]] = []
    for chapter_title, texts in prepared:
        audio_parts = []
        for text in texts:
            audio_parts.append(_mono_audio(synth.synthesize(text)))
            complete += 1
            if progress is not None:
                progress(complete, total)
        rendered.append((chapter_title, audio_parts))

    sample_rate = int(synth.sample_rate)
    if sample_rate <= 0:
        raise ValueError("synthesizer sample_rate must be positive")
    audio, chapters = _join_audio(
        rendered, sample_rate, paragraph_pause, segment_pause
    )

    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="feedcast-voice-") as temporary:
            return _encode_episode(
                audio,
                sample_rate,
                chapters,
                out_path,
                title,
                metadata,
                target_lufs,
                bitrate,
                Path(temporary),
            )

    workdir.mkdir(parents=True, exist_ok=True)
    return _encode_episode(
        audio,
        sample_rate,
        chapters,
        out_path,
        title,
        metadata,
        target_lufs,
        bitrate,
        workdir,
    )


def render_notice(
    lines: list[str],
    out_path: Path,
    *,
    title: str,
    synth: Synthesizer | None = None,
) -> RenderResult:
    texts = [line.strip() for line in lines if line.strip()]
    if not texts:
        texts = ["Feedcast could not create this episode."]

    candidates = ([synth] if synth is not None else []) + [SaySynthesizer()]
    for candidate in candidates:
        try:
            prepared = _PreparedSynthesizer(candidate, texts)
        except Exception:
            continue
        return render_episode(
            [Segment(title, texts)], out_path, synth=prepared, title=title
        )

    return render_episode(
        [Segment(title, ["failure tone"])],
        out_path,
        synth=_ToneSynthesizer(),
        title=title,
    )


def probe_duration(path: Path) -> float:
    result = _run(
        [
            _executable("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        "ffprobe",
    )
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("ffprobe did not report an audio duration") from error


def probe_chapters(path: Path) -> list[Chapter]:
    result = _run(
        [
            _executable("ffprobe"),
            "-v",
            "error",
            "-show_chapters",
            "-of",
            "json",
            str(path),
        ],
        "ffprobe",
    )
    try:
        raw_chapters = json.loads(result.stdout).get("chapters", [])
        return [
            Chapter(
                title=chapter.get("tags", {}).get("title", ""),
                start=float(chapter["start_time"]),
                end=float(chapter["end_time"]),
            )
            for chapter in raw_chapters
        ]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("ffprobe returned invalid chapter data") from error


def _mono_audio(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("synthesizer audio must be mono")
    if samples.size == 0:
        raise ValueError("synthesizer returned empty audio")
    return np.clip(np.nan_to_num(samples), -1.0, 1.0)


def _join_audio(
    rendered: list[tuple[str, list[np.ndarray]]],
    sample_rate: int,
    paragraph_pause: float,
    segment_pause: float,
) -> tuple[np.ndarray, list[Chapter]]:
    lead = np.zeros(round(0.25 * sample_rate), dtype=np.float32)
    paragraph_gap = np.zeros(round(paragraph_pause * sample_rate), dtype=np.float32)
    segment_gap = np.zeros(round(segment_pause * sample_rate), dtype=np.float32)
    parts = [lead]
    sample_count = len(lead)
    chapter_starts: list[tuple[str, int]] = []

    for segment_index, (title, paragraphs) in enumerate(rendered):
        chapter_starts.append((title, sample_count))
        for paragraph_index, paragraph in enumerate(paragraphs):
            if paragraph_index:
                parts.append(paragraph_gap)
                sample_count += len(paragraph_gap)
            parts.append(paragraph)
            sample_count += len(paragraph)
        if segment_index < len(rendered) - 1:
            parts.append(segment_gap)
            sample_count += len(segment_gap)

    parts.append(lead)
    sample_count += len(lead)
    audio = np.concatenate(parts)
    chapters = [
        Chapter(
            title=chapter_title,
            start=start / sample_rate,
            end=(
                chapter_starts[index + 1][1] / sample_rate
                if index + 1 < len(chapter_starts)
                else sample_count / sample_rate
            ),
        )
        for index, (chapter_title, start) in enumerate(chapter_starts)
    ]
    return audio, chapters


def _encode_episode(
    audio: np.ndarray,
    sample_rate: int,
    chapters: list[Chapter],
    out_path: Path,
    title: str,
    metadata: dict[str, str] | None,
    target_lufs: float,
    bitrate: str,
    workdir: Path,
) -> RenderResult:
    source_path = workdir / "voice-source.wav"
    metadata_path = workdir / "voice-metadata.txt"
    encoded_path = workdir / "voice-encoded.mp3"
    _write_wav(source_path, audio, sample_rate)

    measurement = _measure_loudness(source_path, target_lufs)
    _write_metadata(metadata_path, title, metadata, chapters)
    loudnorm = (
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:"
        f"measured_I={measurement['input_i']}:"
        f"measured_TP={measurement['input_tp']}:"
        f"measured_LRA={measurement['input_lra']}:"
        f"measured_thresh={measurement['input_thresh']}:"
        f"offset={measurement['target_offset']}:linear=true:print_format=summary"
    )
    _run(
        [
            _executable("ffmpeg"),
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(source_path),
            "-f",
            "ffmetadata",
            "-i",
            str(metadata_path),
            "-map",
            "0:a:0",
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-af",
            loudnorm,
            "-ar",
            "44100",
            "-ac",
            "1",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            "-id3v2_version",
            "3",
            str(encoded_path),
        ],
        "ffmpeg",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    delivery_path = out_path.with_name(f".{out_path.name}.tmp")
    try:
        shutil.copyfile(encoded_path, delivery_path)
        os.replace(delivery_path, out_path)
    finally:
        delivery_path.unlink(missing_ok=True)

    duration = probe_duration(out_path)
    written_chapters = probe_chapters(out_path)
    if len(written_chapters) != len(chapters):
        raise RuntimeError(
            f"ffmpeg wrote {len(written_chapters)} chapters; expected {len(chapters)}"
        )
    return RenderResult(out_path, duration, written_chapters)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32_767).astype("<i2")
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(pcm.tobytes())


def _measure_loudness(path: Path, target_lufs: float) -> dict[str, str]:
    result = _run(
        [
            _executable("ffmpeg"),
            "-hide_banner",
            "-nostdin",
            "-i",
            str(path),
            "-af",
            f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        "ffmpeg",
    )
    blocks = re.findall(r"\{[^{}]+\}", result.stderr, flags=re.DOTALL)
    for block in reversed(blocks):
        try:
            measurement = json.loads(block)
        except json.JSONDecodeError:
            continue
        required = {
            "input_i",
            "input_tp",
            "input_lra",
            "input_thresh",
            "target_offset",
        }
        if required <= measurement.keys():
            return {key: str(measurement[key]) for key in required}
    raise RuntimeError("ffmpeg loudnorm did not return its measurements")


def _write_metadata(
    path: Path,
    title: str,
    metadata: dict[str, str] | None,
    chapters: list[Chapter],
) -> None:
    tags = {"artist": "feedcast", "album": "feedcast", "genre": "Podcast"}
    if metadata:
        tags.update(metadata)
    tags["title"] = title
    lines = [";FFMETADATA1"]
    lines.extend(f"{key}={_metadata_value(value)}" for key, value in tags.items())
    for chapter in chapters:
        lines.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={round(chapter.start * 1000)}",
                f"END={round(chapter.end * 1000)}",
                f"title={_metadata_value(chapter.title)}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metadata_value(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
    )


def _executable(name: str) -> str:
    executable = shutil.which(name)
    homebrew = Path("/opt/homebrew/bin") / name
    if executable is not None:
        return executable
    if homebrew.exists():
        return str(homebrew)
    raise FileNotFoundError(f"{name} is not installed")


def _run(command: list[str], name: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{name} is not installed") from error
    if result.returncode:
        raise RuntimeError(_command_error(name, result.stderr))
    return result


def _command_error(name: str, stderr: str) -> str:
    tail = "\n".join(stderr.strip().splitlines()[-20:])
    return f"{name} failed" + (f":\n{tail}" if tail else "")


class _PreparedSynthesizer:
    def __init__(self, synth: Synthesizer, texts: list[str]) -> None:
        self._audio = {text: _mono_audio(synth.synthesize(text)) for text in texts}
        self.sample_rate = synth.sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        return self._audio[text]


class _ToneSynthesizer:
    sample_rate = 24_000

    def synthesize(self, text: str) -> np.ndarray:
        beep_samples = round(0.18 * self.sample_rate)
        gap = np.zeros(round(0.12 * self.sample_rate), dtype=np.float32)
        time = np.arange(beep_samples, dtype=np.float32) / self.sample_rate
        beep = (0.25 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
        return np.concatenate([beep, gap, beep, gap, beep])
