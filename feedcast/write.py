"""Qwen writes each story from its cluster's text only, then code checks the result.

A segment that names a number or proper noun its sources don't contain gets one
rewrite with the offending terms listed; if the rewrite still fails, the segment is
dropped. The outlet line is added in code after the check, so it can't be invented.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from feedcast import grounding
from feedcast.group import Cluster
from feedcast.llm import LLM
from feedcast.text import strip_tracking, strip_urls, word_count

log = logging.getLogger("feedcast.write")

MAX_SOURCES = 6
MIN_SOURCE_CHARS = 1500

SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "paragraphs": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 6},
    },
    "required": ["headline", "paragraphs"],
}

SEGMENT_SYSTEM = """\
You write one story for Feedcast, a spoken morning news briefing. The listener hears it \
and cannot see it.

Rules:
- Use only facts stated in the source articles you are given. No background knowledge: \
no names, places, dates, or numbers that the sources do not contain.
- Write for the ear. Short sentences, mostly under 20 words. Plain words. No lists, \
headings, parentheses, emoji, or URLs.
- Numbers: write the figure in digits and the unit in words, the way a newsreader \
says it: "3.2 billion dollars", "45 percent", "1,200 people". Never use the symbols $, \
%, £ or €. Use figures exactly as the sources give them; do not calculate new ones.
- Spell names exactly as the sources do.
- Open with what happened. Then why it matters. Then what happens next, if the sources \
say. If sources disagree, say so plainly.
- Do not mention "sources", "articles", or "reports" by name, and do not name the \
outlets: attribution is added separately. No greeting and no sign-off.
- Length matters: this story fills about {seconds} seconds of audio, so write {words} \
words (never fewer than {min_words}), in 3 to 5 paragraphs of 2 to 4 sentences each. Use \
the detail the sources give: who, what, where, the key figures, quotes in brief, \
reactions, and what comes next.
- headline: a short chapter title, at most 8 words, in sentence case (capitalize only \
the first word and names), using only names from the sources."""

INTRO_SCHEMA = {
    "type": "object",
    "properties": {"intro": {"type": "string"}, "outro": {"type": "string"}},
    "required": ["intro", "outro"],
}

INTRO_SYSTEM = """\
You write the opening and closing lines of Feedcast, a spoken news briefing.
- intro: at most 60 words. Say "This is Feedcast" and the date exactly as given, then \
preview the first three stories in a few words each. Use only names from the headlines \
and story text given.
- outro: at most 30 words. A brief, warm close. Mention that show notes link every source.
No URLs, no emoji, no numbers except the date. Write for the ear."""

QUERIES_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}

TOPICS_SCHEMA = {
    "type": "object",
    "properties": {"topics": {"type": "array", "items": {"type": "string"}}},
    "required": ["topics"],
}


@dataclass
class SourcePack:
    text: str                           # exactly what Qwen sees; grounding checks against it
    outlets: list[str]                  # every outlet in the cluster, in priority order
    links: list[tuple[str, str, str]]   # (outlet, title, url) for every item in the cluster


@dataclass
class Segment:
    headline: str
    paragraphs: list[str]               # written form, outlet line last
    outlets: list[str]
    links: list[tuple[str, str, str]]
    topic: str | None = None
    rewrites: int = 0
    first_issues: list[str] = field(default_factory=list)

    @property
    def words(self) -> int:
        return sum(word_count(p) for p in self.paragraphs)


@dataclass
class Dropped:
    headline: str
    reason: str
    links: list[tuple[str, str, str]]


def build_pack(cluster: Cluster, max_chars: int = 14000) -> SourcePack:
    """One article per outlet first (best material first), then extra copies."""
    firsts, extras, seen = [], [], set()
    for it in cluster.items:
        (extras if it.outlet in seen else firsts).append(it)
        seen.add(it.outlet)
    chosen = (firsts + extras)[:MAX_SOURCES]
    budget = max(MIN_SOURCE_CHARS, max_chars // len(chosen))
    blocks = []
    for it in chosen:
        body = _cut(it.text or it.lead, budget)
        # No numbering in the labels: the grounding check reads this text, and "Source 2"
        # would make any 2 in a segment look supported.
        blocks.append(f"=== From {it.outlet} ===\nHeadline: {it.title}\n{body}")
    links = [(it.outlet, it.title, strip_tracking(it.url)) for it in cluster.items]
    return SourcePack("\n\n".join(blocks)[: max_chars + 400], cluster.outlets, links)


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind("\n\n"), cut.rfind(". "))
    return cut[: end + 1] if end > limit // 2 else cut


def outlet_line(outlets: list[str]) -> str:
    if len(outlets) == 1:
        return f"This story comes from {outlets[0]}."
    named = outlets[:4]
    rest = len(outlets) - len(named)
    if rest:
        others = "one other outlet" if rest == 1 else f"{rest} other outlets"
        return f"This story was reported by {', '.join(named)}, and {others}."
    return f"This story was reported by {', '.join(named[:-1])} and {named[-1]}."


def target_words(cluster: Cluster, words_per_minute: int) -> int:
    """60 to 120 seconds: bigger stories and richer sources get more time."""
    seconds = 68 + 7 * min(cluster.size - 1, 6)  # 68 to 110 s, leaving room under 120
    if sum(len(it.text) for it in cluster.items) < 1500:
        seconds = 68
    return round(seconds / 60 * words_per_minute)


def _clean(draft: dict) -> tuple[str, list[str]]:
    headline = strip_urls(str(draft.get("headline", ""))).strip().strip('"').rstrip(".")
    paras = [strip_urls(str(p)).strip() for p in draft.get("paragraphs") or []]
    return headline, [re.sub(r"\s+", " ", p) for p in paras if p]


def segment_issues(headline: str, paras: list[str], pack: SourcePack) -> list[str]:
    # No allowlist: the outlets Qwen saw are already named in pack.text, and the
    # attribution line is added after this check.
    body = grounding.unsupported("\n\n".join(paras), pack.text)
    head = grounding.unsupported(headline, pack.text, title_case=grounding.looks_title_case(headline))
    return body + [w for w in head if w not in body]


def write_segment(llm: LLM, cluster: Cluster, *, words: int, max_chars: int = 14000,
                  topic: str | None = None) -> Segment | Dropped:
    pack = build_pack(cluster, max_chars)
    system = (SEGMENT_SYSTEM.replace("{words}", str(words))
              .replace("{min_words}", str(round(words * 0.75)))
              .replace("{seconds}", str(round(words / 2.5))))
    focus = f"\nThe listener asked about: {topic}. Focus on what relates to it.\n" if topic else ""
    user = f"Source articles:\n\n{pack.text}\n{focus}\nWrite the {words}-word story as JSON."
    headline, paras = _clean(llm.json(system, user, SEGMENT_SCHEMA))
    issues = segment_issues(headline, paras, pack)
    n_words = sum(word_count(p) for p in paras)
    length_off = n_words < 0.6 * words or n_words > 1.3 * words
    first_issues = list(issues)
    rewrites = 0
    if issues or length_off or not paras:
        rewrites = 1
        notes = []
        if issues:
            notes.append(
                "These names or numbers in your draft do not appear in the source articles: "
                + ", ".join(issues)
                + ". Remove them or replace them with wording the sources support. Drop any "
                "claim you cannot support from the sources."
            )
        if length_off or not paras:
            notes.append(
                f"Your draft was {n_words} words, which is the wrong length. Write {words} words: "
                "3 to 5 paragraphs of 2 to 4 sentences, using more of the detail in the sources."
            )
        draft = "\n\n".join([headline, *paras])
        retry = (
            f"Source articles:\n\n{pack.text}\n{focus}\nYour previous draft:\n\n{draft}\n\n"
            + "\n".join(notes)
            + "\n\nRewrite the story as JSON, following every rule."
        )
        headline, paras = _clean(llm.json(system, retry, SEGMENT_SCHEMA))
        issues = segment_issues(headline, paras, pack)
        n_words = sum(word_count(p) for p in paras)
        # Thin sources can't fill a full slot and that's fine; a few words isn't a story.
        if issues or n_words < 0.3 * words:
            if issues:
                reason = "unsupported after rewrite: " + ", ".join(issues)
            else:
                reason = f"too short after rewrite: {n_words} words for a {words}-word slot"
            log.info("dropping %r: %s", cluster.title, reason)
            return Dropped(headline or cluster.title, reason, pack.links)
    while sum(word_count(p) for p in paras) > 1.3 * words and len(paras) > 2:
        paras = paras[:-1]
    return Segment(headline or cluster.title, [*paras, outlet_line(pack.outlets)], pack.outlets,
                   pack.links, topic, rewrites, first_issues)


def write_intro_outro(llm: LLM, *, date_text: str, segments: list[Segment],
                      topics: list[str] | None = None, special: bool = False) -> tuple[str, str]:
    """Qwen's intro and outro if they pass the grounding check, else a plain template."""
    stories = "\n".join(f"- {s.headline}: {s.paragraphs[0]}" for s in segments)
    kind = f"a special edition on: {', '.join(topics or [])}" if special else "the daily edition"
    extra = f"\nStanding topics today: {', '.join(topics)}." if topics and not special else ""
    user = f"Date: {date_text}\nThis is {kind}.{extra}\nStories in order:\n{stories}\n\nWrite JSON."
    source = "\n".join([date_text, *(topics or []), *(s.headline for s in segments),
                        *(p for s in segments for p in s.paragraphs)])
    fallback = template_intro_outro(date_text, segments, topics, special)
    try:
        draft = llm.json(INTRO_SYSTEM, user, INTRO_SCHEMA)
    except Exception as exc:
        log.warning("intro/outro failed, using template: %s", exc)
        return fallback
    intro, outro = strip_urls(str(draft.get("intro", ""))), strip_urls(str(draft.get("outro", "")))
    issues = grounding.unsupported(f"{intro}\n\n{outro}", source, allowed=["Feedcast"])
    if issues or not intro or not outro or word_count(intro) > 90 or word_count(outro) > 50:
        log.info("intro/outro rejected (%s), using template", ", ".join(issues) or "length")
        return fallback
    return intro, outro


def template_intro_outro(date_text: str, segments: list[Segment], topics: list[str] | None,
                         special: bool) -> tuple[str, str]:
    heads = [s.headline for s in segments[:3]]
    if special:
        intro = f"This is a Feedcast special on {_join(topics or [])}, for {date_text}."
    else:
        intro = f"Good morning. This is Feedcast for {date_text}."
    if heads:
        intro += f" Coming up: {_join(heads)}."
    outro = "That's Feedcast. Show notes with a link to every source are in the episode folder."
    return intro, outro


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def search_queries(llm: LLM, topic: str, n: int = 3, today: str = "") -> list[str]:
    system = (
        f"Write {n} distinct web search queries that would find English-language news from "
        "the past week about the topic. Each 2 to 6 words, no quotes or search operators. "
        "Do not put a year in the query."
        + (f" Today is {today}." if today else "")
    )
    data = llm.json(system, f"Topic: {topic}\nReturn JSON.", QUERIES_SCHEMA)
    queries = [q.strip().strip('"') for q in data.get("queries") or [] if q.strip()]
    return (queries or [topic])[:n]


def extract_topics(llm: LLM, request: str, max_topics: int = 4) -> list[str]:
    system = (
        "A listener dictated a request for a news briefing. List the distinct news topics "
        f"they asked for, at most {max_topics}, each as a short noun phrase in their own "
        "words. Phrase every topic positively: if they said what they don't want, leave "
        "that out."
    )
    data = llm.json(system, f"Request: {request}\nReturn JSON.", TOPICS_SCHEMA)
    return [t.strip().rstrip(".") for t in data.get("topics") or [] if t.strip()][:max_topics]
