"""Search the web through the local SearXNG when the ingested pool is thin on a topic.

Also finds the feeds behind search hits, so the show notes can suggest feeds worth
adding. Nothing here adds feeds or verifies them; callers decide.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from feedcast.paths import PROJECT_ROOT

log = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8888"
COMPOSE_FILE = PROJECT_ROOT / "searxng" / "docker-compose.yml"

# launchd jobs get a bare PATH, so look where Docker Desktop and Homebrew install it.
_DOCKER_FALLBACKS = (
    "/usr/local/bin/docker",
    "/opt/homebrew/bin/docker",
    "/Applications/Docker.app/Contents/Resources/bin/docker",
)
_COMPOSE_TIMEOUT = 300.0  # long enough to pull the image on a first run
_HEALTH_TIMEOUT = 3.0
_POLL_INTERVAL = 1.0

# Generous upper bounds; SearXNG's own ranges are calendar-ish, not exact.
_TIME_RANGES = {
    "day": timedelta(days=1),
    "week": timedelta(days=7),
    "month": timedelta(days=31),
    "year": timedelta(days=366),
}

_FEED_TYPES = {"application/rss+xml", "application/atom+xml"}
_FEED_GUESSES = ("/feed", "/rss", "/rss.xml", "/feed.xml", "/atom.xml", "/index.xml")
# Some news sites refuse requests that do not look like a browser.
_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) feedcast/0.1"


class SearchUnavailable(RuntimeError):
    """SearXNG could not be reached or did not answer with JSON results."""


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    engines: list[str]
    published: datetime | None


def search(
    query: str,
    *,
    base_url: str = DEFAULT_URL,
    categories: str = "news",
    time_range: str | None = "week",
    language: str = "en",
    limit: int = 10,
    client: httpx.Client | None = None,
    timeout: float = 20.0,
) -> list[SearchResult]:
    """Return up to `limit` results for `query`, in SearXNG's ranking order."""
    params = {"q": query, "format": "json", "categories": categories, "language": language}
    if time_range:
        params["time_range"] = time_range
    payload = _get_json(f"{base_url.rstrip('/')}/search", params, client, timeout)

    raw = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        raise SearchUnavailable(f"SearXNG at {base_url} returned JSON without a results list")

    # Some engines ignore time_range (see searxng/settings.yml), so enforce it here.
    span = _TIME_RANGES.get(time_range or "")
    oldest = datetime.now(UTC) - span if span else None
    results: list[SearchResult] = []
    seen: set[str] = set()
    for item in raw:
        if len(results) >= limit:
            break
        result = _parse_result(item)
        if result is None or _too_old(result, oldest):
            continue
        key = _dedupe_key(result.url)
        if key in seen:
            continue
        seen.add(key)
        results.append(result)
    return results


def is_up(base_url: str = DEFAULT_URL, *, client: httpx.Client | None = None) -> bool:
    """True when SearXNG answers its health check."""
    try:
        with _client_or_new(client, _HEALTH_TIMEOUT) as http:
            resp = http.get(f"{base_url.rstrip('/')}/healthz", timeout=_HEALTH_TIMEOUT)
    except (httpx.HTTPError, httpx.InvalidURL):
        return False
    return resp.status_code == 200


def ensure_up(
    base_url: str = DEFAULT_URL,
    *,
    compose_file: Path = COMPOSE_FILE,
    wait_seconds: float = 60.0,
    docker: str | None = None,
) -> bool:
    """Start the SearXNG container if needed. False, never an exception, if it stays down."""
    if is_up(base_url):
        return True
    exe = docker or _find_docker()
    if exe is None:
        log.warning("SearXNG is down and no docker binary was found")
        return False
    if not _compose_up(exe, compose_file):
        return False

    deadline = time.monotonic() + wait_seconds
    while True:
        if is_up(base_url):
            return True
        if time.monotonic() >= deadline:
            log.warning("SearXNG did not come up at %s within %.0fs", base_url, wait_seconds)
            return False
        time.sleep(_POLL_INTERVAL)


def find_feed_links(page_url: str, html: str) -> list[str]:
    """RSS/Atom URLs a page advertises with <link rel="alternate">, resolved and deduped."""
    parser = _FeedLinkParser()
    parser.feed(html)
    parser.close()
    base = _resolve(page_url, parser.base_href) if parser.base_href else page_url
    resolved = (_resolve(base or page_url, href) for href in parser.hrefs)
    return _dedupe([url for url in resolved if url and _is_http(url)])


def candidate_feeds(
    article_url: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = 10.0,
) -> list[str]:
    """Feeds that probably carry `article_url`'s site, unverified.

    Advertised feeds from the article and its homepage; failing that, common feed paths.
    """
    if not _is_http(article_url):
        return []
    homepage = _homepage(article_url)
    found: list[str] = []
    with _client_or_new(client, timeout) as http:
        for page in _dedupe([article_url, homepage]):
            fetched = _fetch_html(http, page, timeout)
            if fetched is not None:
                final_url, html = fetched
                found.extend(find_feed_links(final_url, html))
    return _dedupe(found) or [urljoin(homepage, path) for path in _FEED_GUESSES]


# --- search helpers ---------------------------------------------------------------


def _get_json(
    url: str, params: dict[str, str], client: httpx.Client | None, timeout: float
) -> Any:
    try:
        with _client_or_new(client, timeout) as http:
            resp = http.get(url, params=params, timeout=timeout)
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise SearchUnavailable(f"SearXNG unreachable at {url}: {exc}") from exc
    if resp.status_code != 200:
        raise SearchUnavailable(f"SearXNG answered HTTP {resp.status_code} at {url}")
    try:
        return resp.json()
    except ValueError as exc:
        raise SearchUnavailable(f"SearXNG answered non-JSON at {url}") from exc


def _parse_result(item: Any) -> SearchResult | None:
    if not isinstance(item, dict):
        return None
    url = item.get("url")
    if not isinstance(url, str) or not _is_http(url.strip()):
        return None
    engines = item.get("engines")
    if not isinstance(engines, list):
        engine = item.get("engine")
        engines = [engine] if isinstance(engine, str) else []
    return SearchResult(
        title=_clean_text(item.get("title")),
        url=url.strip(),
        content=_clean_text(item.get("content")),
        engines=[e for e in engines if isinstance(e, str)],
        published=_parse_published(item.get("publishedDate")),
    )


def _parse_published(value: Any) -> datetime | None:
    """SearXNG emits ISO strings, usually naive; naive means UTC (the container's zone)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(value, UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = _parse_datetime_text(value.strip())
    if parsed is None:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _parse_datetime_text(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text)  # RFC 2822, as some feeds and engines use
    except (TypeError, ValueError):
        return None


def _too_old(result: SearchResult, oldest: datetime | None) -> bool:
    # Undated results stay: Bing honours time_range itself but gives no dates.
    return oldest is not None and result.published is not None and result.published < oldest


def _clean_text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _dedupe_key(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


# --- docker helpers ---------------------------------------------------------------


def _find_docker() -> str | None:
    found = shutil.which("docker")
    if found:
        return found
    for candidate in _DOCKER_FALLBACKS:
        if os.access(candidate, os.X_OK):
            return candidate
    return None


def _compose_up(exe: str, compose_file: Path) -> bool:
    # docker shells out to credential helpers that live beside it, which a bare
    # launchd PATH would not find.
    path = os.pathsep.join([str(Path(exe).parent), os.environ.get("PATH", "")])
    env = {**os.environ, "PATH": path}
    try:
        proc = subprocess.run(
            [exe, "compose", "-f", str(compose_file), "up", "-d"],
            capture_output=True,
            text=True,
            env=env,
            timeout=_COMPOSE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not run docker compose: %s", exc)
        return False
    if proc.returncode != 0:
        log.warning("docker compose up failed: %s", proc.stderr.strip()[-500:])
        return False
    return True


# --- feed discovery helpers -------------------------------------------------------


class _FeedLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.base_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: (value or "").strip() for name, value in attrs}
        if tag == "base" and self.base_href is None and values.get("href"):
            self.base_href = values["href"]
            return
        if tag != "link" or not values.get("href"):
            return
        rels = values.get("rel", "").lower().split()
        mime = values.get("type", "").split(";")[0].strip().lower()
        if "alternate" in rels and mime in _FEED_TYPES:
            self.hrefs.append(values["href"])


def _fetch_html(http: httpx.Client, url: str, timeout: float) -> tuple[str, str] | None:
    try:
        resp = http.get(
            url, follow_redirects=True, timeout=timeout, headers={"User-Agent": _USER_AGENT}
        )
        if resp.status_code != 200:
            return None
        return str(resp.url), resp.text
    except (httpx.HTTPError, httpx.InvalidURL, LookupError, UnicodeError):
        return None


def _homepage(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "/", "", ""))


def _resolve(base: str, href: str) -> str | None:
    try:
        return urljoin(base, href)
    except ValueError:  # e.g. a malformed IPv6 host
        return None


def _is_http(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme.lower() in ("http", "https") and bool(parts.netloc)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


@contextmanager
def _client_or_new(client: httpx.Client | None, timeout: float) -> Iterator[httpx.Client]:
    """Use the caller's client as-is, or a throwaway one closed afterwards."""
    if client is not None:
        yield client
        return
    with httpx.Client(timeout=timeout) as owned:
        yield owned
