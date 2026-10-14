"""Show notes: segment titles with timestamps and source links, as Markdown."""

from __future__ import annotations

from dataclasses import dataclass, field

from feedcast.voice import Chapter
from feedcast.write import Dropped, Segment


@dataclass
class FeedSuggestion:
    site: str
    url: str
    title: str | None
    topic: str


@dataclass
class RunReport:
    kind: str                                   # "edition" or "ask"
    title: str
    topics: list[str] = field(default_factory=list)
    stats: dict[str, int | str] = field(default_factory=dict)
    dropped: list[Dropped] = field(default_factory=list)
    suggestions: list[FeedSuggestion] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # warnings worth showing the listener


def stamp(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 \
        else f"{seconds // 60}:{seconds % 60:02d}"


def show_notes(report: RunReport, segments: list[Segment], chapters: list[Chapter],
               duration: float, intro: str, outro: str) -> str:
    lines = [f"# {report.title}", ""]
    meta = [report.kind, f"{stamp(duration)} long", f"{len(segments)} stories"]
    lines += [" · ".join(meta), ""]
    if report.topics:
        label = "Topics" if report.kind == "ask" else "Standing topics"
        lines += [f"{label}: {', '.join(report.topics)}", ""]
    for note in report.notes:
        lines.append(f"> {note}")
    if report.notes:
        lines.append("")

    starts = {c.title: c.start for c in chapters}
    lines += ["## Chapters", "", f"- {stamp(starts.get('Intro', 0))} Intro: {intro}", ""]
    for n, seg in enumerate(segments, 1):
        at = stamp(chapters[n].start) if n < len(chapters) else "?"
        topic = f" (topic: {seg.topic})" if seg.topic else ""
        outlets = f"{len(seg.outlets)} outlet{'s' if len(seg.outlets) != 1 else ''}"
        lines.append(f"### {n}. {seg.headline}{topic}")
        lines.append(f"{at} · {outlets}")
        lines.append("")
        for outlet, title, url in seg.links:
            lines.append(f"- {outlet}: [{_md(title)}]({url})")
        lines.append("")
    lines += [f"- Outro: {outro}", ""]

    if report.dropped:
        lines += ["## Left out", "",
                  "Written but dropped because the grounding check failed after one rewrite:", ""]
        for d in report.dropped:
            first = d.links[0] if d.links else None
            link = f" ([{_md(first[1])}]({first[2]}))" if first else ""
            lines.append(f"- {d.headline}: {d.reason}{link}")
        lines.append("")

    if report.suggestions:
        lines += ["## Feeds worth adding", "",
                  "Found through web search for topics the pool covered thinly. Nothing was "
                  "added automatically; to add one, put it in feeds.yaml and run "
                  "`feedcast feeds verify`.", ""]
        for s in report.suggestions:
            title = f" ({_md(s.title)})" if s.title else ""
            lines.append(f"- {s.site}{title}: {s.url} (for “{s.topic}”)")
        lines.append("")

    if report.stats:
        lines += ["## Run", ""]
        lines += [f"- {k.replace('_', ' ')}: {v}" for k, v in report.stats.items()]
        lines.append("")
    return "\n".join(lines)


def failure_notes(title: str, kind: str, step: str, error: str, detail: str) -> str:
    return "\n".join([
        f"# {title}", "",
        f"**The {kind} run failed while {step}.**", "",
        f"Error: `{error}`", "",
        "```", detail.strip()[-4000:], "```", "",
        "Logs: `data/logs/` in the feedcast folder.", "",
    ])


def _md(text: str) -> str:
    return text.replace("[", "(").replace("]", ")")
