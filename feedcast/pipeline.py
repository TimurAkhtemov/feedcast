"""The two editions, wired end to end: pool -> judge -> group -> write -> voice -> deliver.

Every model and network dependency arrives through `Services`, so the end-to-end test
runs the same code with Laya, Ollama, and the web stubbed out. Any failure still
produces a short MP3 and notes in the episodes folder that say what broke.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import shutil
import subprocess
import time
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np

from feedcast import paths
from feedcast import text as textutil
from feedcast.config import negation_warning
from feedcast.db import Item, Store
from feedcast.feedcheck import FeedCheck, FeedSpec, normalize_feed_url
from feedcast.group import Cluster, cluster_items, embed_items, query_text, shortlist
from feedcast.ingest import body_text, extract_article, ingest, pick_canonical
from feedcast.judge import Backend, Judge
from feedcast.llm import LLM, Embedder
from feedcast.notes import FeedSuggestion, RunReport, failure_notes, show_notes
from feedcast.search import SearchResult, SearchUnavailable
from feedcast.speak import spoken
from feedcast import voice
from feedcast.write import (
    Dropped,
    Segment,
    extract_topics,
    search_queries,
    target_words,
    write_intro_outro,
    write_segment,
)

log = logging.getLogger("feedcast.pipeline")

INTRO_OUTRO_WORDS = 80
FAILED_MP3 = Path(__file__).parent / "assets" / "failed.mp3"  # when even ffmpeg is broken
LOCK_WAIT_SECONDS = 45 * 60


@dataclass
class Services:
    store: Store
    config: dict
    llm: LLM
    embedder: Embedder
    backend_factory: Callable[[], Backend]
    synth_factory: Callable[[], voice.Synthesizer]
    http: httpx.Client
    feeds: list[FeedSpec]
    episodes_dir: Path = paths.EPISODES_DIR
    search: Callable[..., list[SearchResult]] | None = None
    searx_up: Callable[[], bool] = lambda: False
    check_feed: Callable[[str], FeedCheck] | None = None
    candidate_feeds: Callable[[str], list[str]] = lambda url: []
    now: Callable[[], datetime] = lambda: datetime.now().astimezone()
    _backend: Backend | None = field(default=None, repr=False)

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = self.backend_factory()
        return self._backend


class Step:
    """Tracks what a run is doing, so a failure notice can say where it stopped."""

    def __init__(self, kind: str):
        self.kind = kind
        self.name = "starting"

    def __call__(self, name: str) -> None:
        self.name = name
        log.info("%s: %s", self.kind, name)


@dataclass
class Episode:
    mp3: Path
    notes: Path
    duration: float
    segments: int
    failed: bool = False


# ---------------------------------------------------------------- date and naming


def date_text(now: datetime) -> str:
    return f"{now:%A}, {now:%B} {now.day}"


def episode_stem(now: datetime, kind: str, label: str = "") -> str:
    suffix = f"-{textutil.slug(label)}" if label else ""
    return f"{now:%Y-%m-%d-%H%M}-{kind}{suffix}"


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


@contextmanager
def build_lock(wait: float = LOCK_WAIT_SECONDS) -> Iterator[None]:
    """One episode build at a time: they share Qwen, Laya, and the GPU."""
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(paths.DATA_DIR / "build.lock", "w") as fh:
        deadline = time.time() + wait
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() > deadline:
                    raise TimeoutError("another feedcast build held the lock for 45 minutes")
                time.sleep(5)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ---------------------------------------------------------------- shared steps


def drop_ads(svc: Services, judge: Judge, items: list[Item], report: RunReport) -> list[Item]:
    verdicts = judge.ads(items, svc.config["judge"]["ad_threshold"])
    ads = verdicts.yes
    report.stats["ads_filtered"] = int(report.stats.get("ads_filtered", 0)) + len(ads)
    return [it for it in items if it.id not in ads]


def write_plan(svc: Services, plan: list[tuple[Cluster, str | None]], budget: int,
               report: RunReport, step: Step) -> list[Segment]:
    """Write segments in plan order until the word budget is used up."""
    wpm = svc.config["write"]["words_per_minute"]
    max_chars = svc.config["write"]["max_source_chars"]
    segments: list[Segment] = []
    used = 0
    for n, (cluster, topic) in enumerate(plan, 1):
        if used >= budget - 0.5 * target_words(cluster, wpm):
            break
        # Never below ~68 s worth: Kokoro's pace varies, and segments must stay over 60 s.
        words = min(target_words(cluster, wpm), max(round(68 / 60 * wpm), budget - used))
        step(f"writing story {n} of up to {len(plan)}: {cluster.title[:60]}")
        result = write_segment(svc.llm, cluster, words=words, max_chars=max_chars, topic=topic)
        if isinstance(result, Dropped):
            report.dropped.append(result)
            continue
        segments.append(result)
        used += result.words
    report.stats["segments_rewritten"] = sum(s.rewrites for s in segments)
    report.stats["segments_dropped"] = len(report.dropped)
    return segments


def render_and_deliver(svc: Services, report: RunReport, segments: list[Segment], stem: str,
                       step: Step) -> Episode:
    now = svc.now()
    step("writing intro and outro")
    special = report.kind == "ask"
    intro, outro = write_intro_outro(svc.llm, date_text=date_text(now), segments=segments,
                                     topics=report.topics, special=special)
    audio = [voice.Segment("Intro", [spoken(intro)])]
    for seg in segments:
        paras = list(seg.paragraphs)
        if seg.topic and not special:
            paras.insert(0, f"Now, on your standing topic, {seg.topic}.")
        audio.append(voice.Segment(seg.headline, [spoken(p) for p in paras]))
    audio.append(voice.Segment("Outro", [spoken(outro)]))

    step("synthesizing audio")
    vc = svc.config["voice"]
    mp3 = svc.episodes_dir / f"{stem}.mp3"
    result = voice.render_episode(
        audio, mp3, synth=svc.synth_factory(), title=report.title,
        metadata={"date": f"{now:%Y-%m-%d}", "comment": ", ".join(s.headline for s in segments)},
        target_lufs=float(vc["target_lufs"]), bitrate=str(vc["bitrate"]),
    )
    step("writing show notes")
    notes = svc.episodes_dir / f"{stem}.md"
    _write_atomic(notes, show_notes(report, segments, result.chapters, result.duration, intro, outro))
    _save_transcript(stem, audio, result)
    log.info("delivered %s (%.0f s, %d stories)", mp3.name, result.duration, len(segments))
    return Episode(mp3, notes, result.duration, len(segments))


def _save_transcript(stem: str, audio: list[voice.Segment], result: voice.RenderResult) -> None:
    """Spoken text with chapter timings, kept locally for tuning words_per_minute."""
    rows = []
    for seg, ch in zip(audio, result.chapters):
        words = sum(textutil.word_count(p) for p in seg.paragraphs)
        seconds = ch.end - ch.start
        rows.append({"title": seg.title, "words": words, "seconds": round(seconds, 1),
                     "wpm": round(words / seconds * 60) if seconds else None,
                     "paragraphs": seg.paragraphs})
    out = paths.DATA_DIR / "transcripts" / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"duration": result.duration, "chapters": rows}, indent=2))


def write_failure(svc: Services, kind: str, label: str, step: str, exc: BaseException) -> Episode:
    """A short spoken notice plus notes, so silence is never mistaken for an episode."""
    now = svc.now()
    stem = episode_stem(now, kind, label) + "-FAILED"
    what = "today's edition" if kind == "edition" else (f"your request about {label}" if label else kind)
    error = f"{type(exc).__name__}: {exc}"
    lines = [
        f"This is Feedcast. Something went wrong, so {what} was not made.",
        f"The {kind} run failed while {step}.",
        f"The error was: {_speakable(str(exc) or type(exc).__name__)}.",
        "The show notes have the details.",
    ]
    # Notes first: they need nothing but the filesystem.
    notes = svc.episodes_dir / f"{stem}.md"
    detail = "".join(traceback.format_exception(exc))
    _write_atomic(notes, failure_notes(f"Feedcast {kind} failed — {date_text(now)}", kind, step,
                                       error, detail))
    mp3 = svc.episodes_dir / f"{stem}.mp3"
    try:
        try:
            synth = svc.synth_factory()
        except Exception:
            synth = None
        result = voice.render_notice([spoken(l) for l in lines], mp3,
                                     title=f"Feedcast: {kind} failed", synth=synth)
        duration = result.duration
    except Exception:
        log.exception("could not render the failure notice; using the bundled one")
        tmp = mp3.with_name(f".{mp3.name}.tmp")
        shutil.copyfile(FAILED_MP3, tmp)
        os.replace(tmp, mp3)
        duration = 0.0
    return Episode(mp3, notes, duration, 0, failed=True)


def write_notice(svc: Services, kind: str, label: str, title: str, lines: list[str],
                 report: RunReport) -> Episode:
    """A short episode for runs that worked but found nothing worth a story."""
    now = svc.now()
    stem = episode_stem(now, kind, label)
    mp3 = svc.episodes_dir / f"{stem}.mp3"
    result = voice.render_notice([spoken(l) for l in lines], mp3, title=title,
                                 synth=svc.synth_factory())
    notes = svc.episodes_dir / f"{stem}.md"
    body = [f"# {title}", "", *[f"{l}  " for l in lines], ""]
    if report.suggestions:
        body += ["## Feeds worth adding", ""]
        body += [f"- {s.site}: {s.url or 'no feed found automatically'} (for “{s.topic}”)"
                 for s in report.suggestions]
        body.append("")
    if report.stats:
        body += ["## Run", "", *[f"- {k.replace('_', ' ')}: {v}" for k, v in report.stats.items()], ""]
    _write_atomic(notes, "\n".join(body))
    return Episode(mp3, notes, result.duration, 0)


def _speakable(message: str) -> str:
    message = re.sub(r"https?://\S+", "a web address", message.splitlines()[0] if message else "")
    return message[:200]


def run_guarded(svc: Services, kind: str, label: str, fn: Callable[[Step], Episode], *,
                lock: bool = True) -> Episode:
    """Run `fn`; on any exception, including waiting for the lock or recording the run,
    deliver a failure notice instead."""
    step = Step(kind)
    run_id = f"{kind}-{svc.now():%Y%m%d-%H%M%S}"
    try:
        step("waiting for another build to finish")
        with build_lock() if lock else nullcontext():
            step("recording the run")
            svc.store.start_run(run_id, kind)
            episode = fn(step)
    except Exception as exc:
        log.exception("%s failed while %s", kind, step.name)
        episode = write_failure(svc, kind, label, step.name, exc)
        _record(svc, run_id, "failed", f"{step.name}: {exc}", episode)
        return episode
    _record(svc, run_id, "ok", f"{episode.segments} stories", episode)
    return episode


def _record(svc: Services, run_id: str, status: str, detail: str, episode: Episode) -> None:
    try:
        svc.store.finish_run(run_id, status, detail, str(episode.mp3))
    except Exception:
        log.exception("could not record run %s", run_id)


# ---------------------------------------------------------------- default edition


def edition(svc: Services, *, topics: list[str], target_minutes: float | None = None,
            refresh: bool = True) -> Episode:
    def build(step: Step) -> Episode:
        cfg = svc.config
        now = svc.now()
        report = RunReport("edition", f"Feedcast — {date_text(now)}", topics=list(topics))
        if refresh:
            step("ingesting feeds")
            try:
                ingested = run_ingest(svc)
                report.stats["new_items_this_morning"] = len(ingested.added)
            except Exception as exc:  # yesterday's pool is still worth an episode
                log.warning("fresh ingest failed: %s", exc)
                report.notes.append(f"The fresh ingest failed ({exc}); this edition uses items already in the pool.")
        step("loading the pool")
        since = now.timestamp() - cfg["edition"]["window_hours"] * 3600
        items = svc.store.items_since(since)
        report.stats["items_in_window"] = len(items)
        if not items:
            raise RuntimeError(f"the pool has no items from the last {cfg['edition']['window_hours']} hours")
        step("judging ads with Laya")
        judge = Judge(svc.store, svc.backend, run_id=f"edition-{now:%Y%m%d-%H%M}")
        items = drop_ads(svc, judge, items, report)
        step("embedding and grouping stories")
        vecs = embed_items(svc.store, svc.embedder, items)
        clusters = cluster_items(items, vecs, cfg["group"]["similarity"])
        report.stats["stories_found"] = len(clusters)
        report.stats["multi_outlet_stories"] = sum(1 for c in clusters if c.size > 1)

        wpm = cfg["write"]["words_per_minute"]
        budget = round((target_minutes or cfg["edition"]["target_minutes"]) * wpm) - INTRO_OUTRO_WORDS
        topic_plan: list[tuple[Cluster, str]] = []
        taken: set[int] = set()
        for topic in topics:
            if warning := negation_warning(topic):
                report.notes.append(warning)
            step(f"judging standing topic {topic!r} with Laya")
            relevance = topic_matches(svc, judge, items, vecs, topic)
            report.stats[f"matches_{textutil.slug(topic, 30)}"] = len(relevance)
            ranked = [c for c in by_relevance(clusters, relevance) if id(c) not in taken]
            for c in ranked[: cfg["edition"]["topic_stories"]]:
                topic_plan.append((c, topic))
                taken.add(id(c))
            if not ranked:
                report.notes.append(f"Nothing in the last day matched the standing topic “{topic}”.")

        topic_budget = min(budget // 2, sum(target_words(c, wpm) for c, _ in topic_plan))
        top_plan = [(c, None) for c in clusters if id(c) not in taken and c.size > 1][:25]
        if len(top_plan) < 6:  # quiet news day: single-outlet stories, newest first
            singles = sorted((c for c in clusters if id(c) not in taken and c.size == 1),
                             key=lambda c: c.latest, reverse=True)
            top_plan += [(c, None) for c in singles[: 12 - len(top_plan)]]
        # Topic stories are written first so the top stories can't eat their share,
        # but they're read after the top stories.
        topic_segments = write_plan(svc, topic_plan, topic_budget, report, step) if topic_plan else []
        top_segments = write_plan(svc, top_plan, budget - sum(s.words for s in topic_segments),
                                  report, step)
        segments = top_segments + topic_segments
        if not segments:
            raise RuntimeError("every story failed the grounding check; nothing left to read")
        return render_and_deliver(svc, report, segments, episode_stem(now, "edition"), step)

    return run_guarded(svc, "edition", "", build)


def topic_matches(svc: Services, judge: Judge, items: list[Item], vecs: dict, topic: str) -> dict[int, float]:
    """Embedding shortlist for recall, then one Laya yes/no question for precision.

    Returns relevance (Laya's P(yes) plus cosine similarity to the topic) for the items
    Laya said yes to. Laya reads compound topics loosely ("AI tooling for developers"
    gets a yes from most AI news), so stories are ranked by relevance, not by size.
    """
    topic_vec = svc.embedder.embed([query_text(topic)])[0]
    topic_vec = topic_vec / max(float(np.linalg.norm(topic_vec)), 1e-12)
    candidates = shortlist(items, vecs, topic_vec, svc.config["judge"]["shortlist"])
    verdicts = judge.about(candidates, topic, svc.config["judge"]["topic_threshold"])
    return {i: verdicts.p[i] + float(vecs[i] @ topic_vec) for i in verdicts.yes}


def by_relevance(clusters: list[Cluster], relevance: dict[int, float]) -> list[Cluster]:
    def score(c: Cluster) -> tuple:
        return (max(relevance.get(i, 0.0) for i in c.ids), c.size, c.latest)
    return sorted((c for c in clusters if c.ids & relevance.keys()), key=score, reverse=True)


# ---------------------------------------------------------------- on-demand


def ask(svc: Services, topics: list[str], *, target_minutes: float | None = None) -> Episode:
    label = ", ".join(topics)

    def build(step: Step) -> Episode:
        cfg = svc.config
        now = svc.now()
        report = RunReport("ask", f"Feedcast special: {label}", topics=list(topics))
        step("loading the pool")
        items = svc.store.items_since(now.timestamp() - cfg["ask"]["window_hours"] * 3600, source=None)
        report.stats["items_in_window"] = len(items)
        step("embedding the pool")
        vecs = embed_items(svc.store, svc.embedder, items)
        step("loading Laya")
        judge = Judge(svc.store, svc.backend, run_id=f"ask-{now:%Y%m%d-%H%M%S}")

        wpm = cfg["write"]["words_per_minute"]
        budget = round((target_minutes or cfg["ask"]["target_minutes"]) * wpm) - INTRO_OUTRO_WORDS
        plans: list[list[tuple[Cluster, str]]] = []
        for topic in topics:
            if warning := negation_warning(topic):
                report.notes.append(warning)
            step(f"judging {topic!r} with Laya")
            relevance = topic_matches(svc, judge, items, vecs, topic)
            matched = [it for it in items if it.id in relevance]
            matched = drop_ads(svc, judge, matched, report)
            report.stats[f"pool_matches_{textutil.slug(topic, 30)}"] = len(matched)
            if len(matched) < cfg["ask"]["min_matches"]:
                step(f"searching the web for {topic!r}")
                found = search_fallback(svc, judge, topic, report, step)
                matched += [it for it in found if it.id not in {m.id for m in matched}]
                vecs.update(embed_items(svc.store, svc.embedder, found))
                relevance.update({it.id: relevance.get(it.id, 1.0) for it in found})
            if not matched:
                report.notes.append(f"No coverage of “{topic}” in the last {cfg['ask']['window_hours']} hours.")
                plans.append([])
                continue
            step(f"grouping stories for {topic!r}")
            clusters = by_relevance(cluster_items(matched, vecs, cfg["group"]["similarity"]), relevance)
            plans.append([(c, topic) for c in clusters[: cfg["ask"]["max_stories"]]])

        # Interleave topics so each gets a fair share of the budget.
        plan: list[tuple[Cluster, str]] = []
        for rank in range(max((len(p) for p in plans), default=0)):
            plan += [p[rank] for p in plans if rank < len(p)]
        segments = write_plan(svc, plan, budget, report, step) if plan else []
        stem = episode_stem(now, "ask", label)
        if not segments:
            step("writing a nothing-found notice")
            lines = [f"This is Feedcast. I looked for news about {label} and found nothing I could report."]
            if report.suggestions:
                lines.append("The show notes list some feeds that cover it, if you want to add them.")
            return write_notice(svc, "ask", label, report.title, lines, report)
        return render_and_deliver(svc, report, segments, stem, step)

    return run_guarded(svc, "ask", label, build)


def search_fallback(svc: Services, judge: Judge, topic: str, report: RunReport,
                    step: Step) -> list[Item]:
    """Fewer than min_matches in the pool: search SearXNG, judge the results with the same
    Laya questions, and suggest (never add) feeds from the sites that had matches."""
    if svc.search is None or not svc.searx_up():
        report.notes.append("Web search was skipped: the local SearXNG instance isn't running.")
        return []
    queries = search_queries(svc.llm, topic, svc.config["ask"]["search_queries"],
                             today=date_text(svc.now()) + f", {svc.now():%Y}")
    report.stats[f"search_queries_{textutil.slug(topic, 30)}"] = " | ".join(queries)
    results: dict[str, SearchResult] = {}
    for q in queries:
        try:
            for r in svc.search(q, limit=10):
                results.setdefault(textutil.canonical_url(r.url), r)
        except SearchUnavailable as exc:
            report.notes.append(f"Search for “{q}” failed: {exc}")
    fresh = {c: r for c, r in results.items() if not svc.store.has_url(c)}
    step(f"reading {len(fresh)} search results")
    with ThreadPoolExecutor(8) as pool:
        texts = list(pool.map(lambda r: extract_article(svc.http, r.url), fresh.values()))
    new_items: list[Item] = []
    for (canon, r), (full, page_canonical) in zip(fresh.items(), texts):
        body = body_text(r.title, full) or r.content
        if not body:
            continue
        item_id = svc.store.add_item(
            outlet=_site_name(r.url), feed_url="search", url=r.url,
            canonical_url=pick_canonical(r.url, page_canonical), title=r.title,
            lead=textutil.lead_text(body) or r.title, text=body,
            content_hash=textutil.content_hash(r.title, body),
            published=(r.published.timestamp() if r.published else svc.now().timestamp()),
            full_text=len(full) >= 400, source="search",
        )
        if item_id:
            new_items.extend(svc.store.items_by_id([item_id]))
    report.stats[f"search_results_{textutil.slug(topic, 30)}"] = len(new_items)
    if not new_items:
        return []
    step(f"judging {len(new_items)} search results with Laya")
    matched_ids = judge.about(new_items, topic, svc.config["judge"]["topic_threshold"]).yes
    matched = drop_ads(svc, judge, [it for it in new_items if it.id in matched_ids], report)
    report.stats[f"search_matches_{textutil.slug(topic, 30)}"] = len(matched)
    step("looking for feeds worth adding")
    report.suggestions += suggest_feeds(svc, matched, topic)
    return matched


def suggest_feeds(svc: Services, items: list[Item], topic: str, limit: int = 5) -> list[FeedSuggestion]:
    if svc.check_feed is None:
        return []
    known_hosts = {httpx.URL(f.url).host.removeprefix("www.") for f in svc.feeds}
    known_urls = {normalize_feed_url(f.url) for f in svc.feeds}
    out: list[FeedSuggestion] = []
    seen_sites: set[str] = set()
    for it in items:
        site = _site_name(it.url)
        if site in seen_sites or site in known_hosts:
            continue
        seen_sites.add(site)
        found = False
        for candidate in svc.candidate_feeds(it.url)[:8]:
            if normalize_feed_url(candidate) in known_urls:
                found = True
                break
            check = svc.check_feed(candidate)
            if check.ok:
                out.append(FeedSuggestion(site, check.final_url or candidate, check.title, topic))
                found = True
                break
        if not found:  # still worth knowing: the site covers this topic
            out.append(FeedSuggestion(site, None, None, topic))
        if sum(1 for s in out if s.url) >= limit:
            break
    return out


def _site_name(url: str) -> str:
    return httpx.URL(url).host.removeprefix("www.")


# ---------------------------------------------------------------- ingest


def run_ingest(svc: Services, *, warm: bool = False):
    cfg = svc.config["ingest"]
    now = svc.now().timestamp()
    report = ingest(svc.store, svc.feeds, client=svc.http, concurrency=cfg["concurrency"],
                    max_age_hours=cfg["max_age_hours"], max_entries=cfg["max_entries_per_feed"],
                    now=now)
    log.info("ingest: %s", report.summary())
    for url, error in report.feeds_failed:
        log.warning("feed failed: %s (%s)", url, error)
    pruned = svc.store.prune(now - cfg["keep_days"] * 86400)
    if pruned:
        log.info("pruned %d old items", pruned)
    if report.feeds_ok == 0 and svc.feeds:
        raise RuntimeError(f"all {len(svc.feeds)} feeds failed; first error: {report.feeds_failed[0][1]}")
    if warm and report.added:
        # Precompute embeddings and ad answers so on-demand requests start warm.
        new = svc.store.items_by_id(report.added)
        try:
            embed_items(svc.store, svc.embedder, new)
            Judge(svc.store, svc.backend, run_id="ingest").ads(new, svc.config["judge"]["ad_threshold"])
        except Exception as exc:
            log.warning("warming caches failed (not fatal): %s", exc)
    return report


# ---------------------------------------------------------------- phone requests


def parse_request(svc: Services, text: str) -> list[str]:
    """Topics from a dictated request: split on explicit separators, else let Qwen pull
    them apart (and drop anything phrased as what the listener doesn't want)."""
    text = " ".join(text.split()) if "\n" not in text else text
    parts = [p.strip(" .") for p in re.split(r"[\n;,]+", text) if p.strip(" .")]
    if len(parts) > 1:
        return parts[:4]
    single = parts[0] if parts else ""
    if not single:
        return []
    if len(single.split()) > 6 or re.search(r"\b(and|also|plus|what's|whats|tell me)\b", single, re.I):
        try:
            topics = extract_topics(svc.llm, single)
            if topics:
                return topics
        except Exception as exc:
            log.warning("topic extraction failed, using the request as one topic: %s", exc)
    return [single]


def download_placeholders(folder: Path, wait: float = 60.0) -> None:
    """iCloud leaves files made on the phone as `.name.txt.icloud` stubs until something
    asks for them. Ask, then wait for the real files to appear."""
    stubs = [p for p in folder.glob(".*.icloud")]
    if not stubs:
        return
    brctl = shutil.which("brctl") or "/usr/bin/brctl"
    for stub in stubs:
        real = stub.with_name(stub.name[1 : -len(".icloud")])
        subprocess.run([brctl, "download", str(real)], capture_output=True, check=False)
    deadline = time.time() + wait
    while time.time() < deadline and any(p.exists() for p in stubs):
        time.sleep(1)


def watch(svc: Services, requests_dir: Path = paths.REQUESTS_DIR) -> list[Episode]:
    """Build an episode for every request file waiting in the iCloud requests folder."""
    requests_dir.mkdir(parents=True, exist_ok=True)
    download_placeholders(requests_dir)
    done_dir, failed_dir = requests_dir / "done", requests_dir / "failed"
    episodes = []
    for req in sorted(requests_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime):
        text = req.read_text(encoding="utf-8", errors="replace").strip()
        stamp = f"{svc.now():%Y%m%d-%H%M%S}"
        try:
            topics = parse_request(svc, text)
        except Exception as exc:
            topics = []
            log.warning("could not parse %s: %s", req.name, exc)
        if not topics:
            report = RunReport("ask", "Feedcast: empty request")
            episodes.append(write_notice(svc, "ask", "empty-request", report.title, [
                f"This is Feedcast. The request file {req.stem} had no topic in it, so there's no episode.",
            ], report))
            _move(req, failed_dir, stamp)
            continue
        log.info("request %s -> topics %s", req.name, topics)
        episode = ask(svc, topics)
        episodes.append(episode)
        _move(req, failed_dir if episode.failed else done_dir, stamp)
    return episodes


def _move(path: Path, folder: Path, stamp: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    os.replace(path, folder / f"{path.stem}-{stamp}{path.suffix}")
