"""Fetch every feed, pull full article text, and add new items to the pool.

Dedupe happens twice: by canonical URL before an article is fetched (cheap), and by a
hash of the extracted text after (catches syndicated copies on different URLs).
"""

from __future__ import annotations

import calendar
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import feedparser
import httpx
import trafilatura

from feedcast import text as textutil
from feedcast.db import Store
from feedcast.feedcheck import FeedSpec

log = logging.getLogger("feedcast.ingest")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/18.0 Safari/605.1.15 feedcast/0.1"
)
MIN_FULL_TEXT_CHARS = 400
PER_HOST = 2  # concurrent article fetches per site; more gets 429s (phys.org did)


@dataclass
class Entry:
    outlet: str
    feed_url: str
    url: str
    title: str
    summary: str
    published: float


@dataclass
class IngestReport:
    feeds_ok: int = 0
    feeds_failed: list[tuple[str, str]] = field(default_factory=list)
    entries_seen: int = 0
    added: list[int] = field(default_factory=list)
    duplicates: int = 0
    full_text: int = 0

    def summary(self) -> str:
        return (
            f"{self.feeds_ok} feeds ok, {len(self.feeds_failed)} failed; "
            f"{self.entries_seen} recent entries, {len(self.added)} new items "
            f"({self.full_text} with full text), {self.duplicates} duplicates"
        )


def make_client(timeout: float = 20.0) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8"},
        follow_redirects=True,
        timeout=timeout,
    )


def fetch_feed(client: httpx.Client, feed: FeedSpec, state: dict) -> tuple[bytes | None, dict]:
    """Conditional GET. Returns (body or None if unchanged, new validators)."""
    headers = {}
    if state.get("etag"):
        headers["If-None-Match"] = state["etag"]
    if state.get("modified"):
        headers["If-Modified-Since"] = state["modified"]
    resp = client.get(feed.url, headers=headers)
    if resp.status_code == 304:
        return None, {}
    resp.raise_for_status()
    validators = {"etag": resp.headers.get("etag"), "modified": resp.headers.get("last-modified")}
    return resp.content, validators


def parse_entries(feed: FeedSpec, body: bytes, *, now: float, max_age_hours: float,
                  limit: int) -> list[Entry]:
    parsed = feedparser.parse(body)
    out = []
    for e in parsed.entries[:limit]:
        link = (e.get("link") or "").strip()
        title = textutil.normalize_paragraphs(textutil.html_to_text(e.get("title", ""))).strip()
        if not link.startswith(("http://", "https://")) or not title:
            continue
        stamp = e.get("published_parsed") or e.get("updated_parsed")
        published = min(calendar.timegm(stamp), now) if stamp else now
        if now - published > max_age_hours * 3600:
            continue
        content = ""
        if e.get("content"):
            content = max((c.get("value", "") for c in e.content), key=len)
        summary = textutil.html_to_text(content or e.get("summary", ""))
        out.append(Entry(feed.name, feed.url, link, title, summary, published))
    return out


def extract_article(client: httpx.Client, url: str) -> tuple[str, str | None]:
    """(full text, canonical URL from the page) or ("", None) when extraction fails."""
    try:
        resp = client.get(url)
        resp.raise_for_status()
        if "html" not in resp.headers.get("content-type", "html"):
            return "", None
        out = trafilatura.extract(
            resp.text,
            url=str(resp.url),
            output_format="json",
            with_metadata=True,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
    except Exception as exc:  # network errors, bad HTML: fall back to the feed summary
        log.debug("extract failed for %s: %s", url, exc)
        return "", None
    if not out:
        return "", None
    data = json.loads(out)
    return data.get("text") or "", data.get("source")


def pick_canonical(link: str, page_canonical: str | None) -> str:
    """Trust the page's canonical link only when it points at an article on the same site."""
    link_c = textutil.canonical_url(link)
    if not page_canonical:
        return link_c
    page_c = textutil.canonical_url(page_canonical)
    same_site = _site(page_c) == _site(link_c)
    has_path = len(httpx.URL(page_c).path) > 1
    return page_c if same_site and has_path else link_c


def _site(url: str) -> str:
    host = httpx.URL(url).host
    return ".".join(host.split(".")[-2:])


def body_text(title: str, text: str) -> str:
    """Cleaned paragraphs, without a leading repeat of the headline."""
    paras = textutil.clean_paragraphs(text)
    if paras and paras[0].strip().lower() == title.strip().lower():
        paras = paras[1:]
    return "\n\n".join(paras)


def ingest(store: Store, feeds: list[FeedSpec], *, client: httpx.Client | None = None,
           concurrency: int = 8, max_age_hours: float = 72, max_entries: int = 40,
           now: float | None = None) -> IngestReport:
    now = now or time.time()
    report = IngestReport()
    own_client = client is None
    client = client or make_client()
    try:
        states = {f.url: store.feed_state(f.url) for f in feeds}

        def get(feed: FeedSpec):
            try:
                return feed, fetch_feed(client, feed, states[feed.url]), None
            except Exception as exc:
                return feed, (None, {}), _describe(exc)

        entries: list[Entry] = []
        with ThreadPoolExecutor(concurrency) as pool:
            for feed, (body, validators), error in pool.map(get, feeds):
                if error:
                    report.feeds_failed.append((feed.url, error))
                    store.save_feed_state(feed.url, error=error)
                    continue
                report.feeds_ok += 1
                store.save_feed_state(feed.url, **validators)
                if body:
                    entries.extend(
                        parse_entries(feed, body, now=now, max_age_hours=max_age_hours,
                                      limit=max_entries)
                    )
        report.entries_seen = len(entries)

        fresh, seen = [], set()
        for e in entries:
            c = textutil.canonical_url(e.url)
            if c in seen or store.has_url(c):
                report.duplicates += 1
                continue
            seen.add(c)
            fresh.append(e)

        host_slots: dict[str, threading.Semaphore] = {}

        def extract(e: Entry):
            # setdefault is atomic, so two threads can't get different semaphores for one host
            slot = host_slots.setdefault(httpx.URL(e.url).host, threading.Semaphore(PER_HOST))
            with slot:
                return extract_article(client, e.url)

        with ThreadPoolExecutor(concurrency) as pool:
            extracted = list(pool.map(extract, fresh))

        for e, (full, page_canonical) in zip(fresh, extracted):
            full_body = body_text(e.title, full)
            has_full = len(full_body) >= MIN_FULL_TEXT_CHARS
            body = full_body if has_full else body_text(e.title, e.summary)
            lead = textutil.lead_text(body) or e.title
            item_id = store.add_item(
                outlet=e.outlet,
                feed_url=e.feed_url,
                url=e.url,
                canonical_url=pick_canonical(e.url, page_canonical),
                title=e.title,
                lead=lead,
                text=body,
                content_hash=textutil.content_hash(e.title, body),
                published=e.published,
                full_text=has_full,
            )
            if item_id is None:
                report.duplicates += 1
            else:
                report.added.append(item_id)
                report.full_text += int(has_full)
    finally:
        if own_client:
            client.close()
    return report


def _describe(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    return f"{type(exc).__name__}: {exc}"[:200]
