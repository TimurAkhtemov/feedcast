"""Verify feed health and import subscriptions before the pipeline runs."""

from __future__ import annotations

import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from urllib.parse import SplitResult, urlsplit, urlunsplit
from xml.etree import ElementTree

import feedparser
import httpx
import yaml

from feedcast.paths import FEEDS_FILE


USER_AGENT = "feedcast/0.1 (+local RSS reader)"


@dataclass(frozen=True)
class FeedSpec:
    name: str
    url: str


@dataclass
class FeedCheck:
    url: str
    ok: bool
    final_url: str | None
    title: str | None
    format: str | None
    entries: int
    newest: datetime | None
    error: str | None


def load_feeds(path: Path) -> list[FeedSpec]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []

    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_feeds = document.get("feeds") or []
    feeds: list[FeedSpec] = []
    for index, entry in enumerate(raw_feeds, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"feed entry {index} is not a mapping")
        if not entry.get("url"):
            label = entry.get("name") or index
            raise ValueError(f"feed entry {label!r} lacks a url")
        url = str(entry["url"])
        name = str(entry.get("name") or _url_name(url))
        feeds.append(FeedSpec(name=name, url=url))
    return feeds


def save_feeds(path: Path, feeds: list[FeedSpec], header: str | None = None) -> None:
    if header is None:
        header = _existing_header(path)
    elif header:
        header = "\n".join(
            line if line.startswith("#") else f"# {line}" for line in header.splitlines()
        )

    document = {"feeds": [{"name": feed.name, "url": feed.url} for feed in feeds]}
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    prefix = f"{header.rstrip()}\n" if header else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prefix + body, encoding="utf-8")


def normalize_feed_url(url: str) -> str:
    parsed = urlsplit(url)
    original_scheme = parsed.scheme.lower()
    scheme = "https" if original_scheme in {"http", "https"} else original_scheme
    hostname = (parsed.hostname or "").lower()

    try:
        port = parsed.port
    except ValueError:
        port = None
    default_ports = {"http": 80, "https": 443}
    if port == default_ports.get(original_scheme):
        port = None

    host = f"[{hostname}]" if ":" in hostname else hostname
    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    netloc = f"{userinfo}{host}{f':{port}' if port is not None else ''}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(SplitResult(scheme, netloc, path, parsed.query, ""))


def check_feed(
    url: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = 20.0,
    max_age_days: float = 21.0,
    now: datetime | None = None,
) -> FeedCheck:
    result = FeedCheck(url, False, None, None, None, 0, None, None)
    owned_client = client is None
    http = client or httpx.Client()
    try:
        response = http.get(
            url,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
    except httpx.TimeoutException:
        result.error = "timeout"
        return result
    except httpx.RequestError as exc:
        result.error = f"connection error: {exc}"
        return result
    finally:
        if owned_client:
            http.close()

    result.final_url = str(response.url)
    if response.status_code != 200:
        result.error = f"HTTP {response.status_code}"
        return result

    try:
        parsed = feedparser.parse(response.content)
    except Exception as exc:
        detail = str(exc) or type(exc).__name__
        result.error = f"parse error: {detail}"
        return result
    result.format = parsed.version or None
    result.title = _text_or_none(parsed.feed.get("title"))
    result.entries = sum(
        1 for entry in parsed.entries if entry.get("title") and entry.get("link")
    )
    result.newest = _newest_entry_date(parsed.entries)

    if not result.format:
        content_type = (
            response.headers.get("content-type", "unknown").split(";", 1)[0].strip()
        )
        result.error = f"not a feed ({content_type})"
        return result
    if not result.entries:
        result.error = "no entries"
        return result

    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    else:
        current = current.astimezone(UTC)
    if result.newest is not None:
        age_days = (current - result.newest).total_seconds() / 86_400
        if age_days > max_age_days:
            result.error = f"stale: newest entry {_format_days(age_days)} days old"
            return result

    result.ok = True
    return result


def check_feeds(
    urls: list[str], *, concurrency: int = 8, **kwargs: object
) -> list[FeedCheck]:
    supplied_client = kwargs.pop("client", None)
    if supplied_client is not None and not isinstance(supplied_client, httpx.Client):
        raise TypeError("client must be an httpx.Client")

    def run(http: httpx.Client) -> list[FeedCheck]:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            return list(
                executor.map(
                    lambda url: check_feed(url, client=http, **kwargs),
                    urls,
                )
            )

    if supplied_client is not None:
        return run(supplied_client)
    with httpx.Client() as client:
        return run(client)


def parse_opml(source: Path | bytes | str) -> list[FeedSpec]:
    if isinstance(source, Path):
        root = ElementTree.fromstring(source.read_bytes())
    elif isinstance(source, bytes):
        root = ElementTree.fromstring(source)
    elif source.lstrip().startswith("<"):
        root = ElementTree.fromstring(source)
    else:
        root = ElementTree.fromstring(Path(source).read_bytes())

    feeds: list[FeedSpec] = []
    seen: set[str] = set()
    for outline in root.iter():
        if outline.tag.rsplit("}", 1)[-1] != "outline":
            continue
        url = outline.get("xmlUrl")
        if not url:
            continue
        normalized = normalize_feed_url(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        name = outline.get("title") or outline.get("text") or _url_name(url)
        feeds.append(FeedSpec(name=name, url=url))
    return feeds


def import_opml(
    opml: Path,
    feeds_file: Path,
    *,
    verify: bool = True,
    check: Callable[[list[str]], list[FeedCheck]] = check_feeds,
) -> tuple[list[FeedSpec], list[FeedCheck]]:
    existing = load_feeds(feeds_file)
    known = {normalize_feed_url(feed.url) for feed in existing}
    candidates = [
        feed for feed in parse_opml(opml) if normalize_feed_url(feed.url) not in known
    ]

    failed: list[FeedCheck] = []
    if verify and candidates:
        checks = check([feed.url for feed in candidates])
        added = []
        for feed, result in zip(candidates, checks, strict=True):
            if result.ok:
                added.append(feed)
            else:
                failed.append(result)
    else:
        added = candidates

    if added:
        save_feeds(feeds_file, existing + added)
    return added, failed


def format_report(checks: list[FeedCheck]) -> str:
    lines: list[str] = []
    now = datetime.now(UTC)
    for check in checks:
        status = "OK" if check.ok else "FAIL"
        if check.newest is None:
            age = "-"
        else:
            age_days = (now - check.newest.astimezone(UTC)).total_seconds() / 86_400
            age = f"{_format_days(age_days)}d"
        detail = check.title if check.ok and check.title else check.error or check.title or "-"
        lines.append(f"{status:<4} {check.entries:>7} {age:>8}  {detail}  {check.url}")
    ok_count = sum(check.ok for check in checks)
    lines.append(f"{ok_count} ok, {len(checks) - ok_count} failed")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="feedcast feeds")
    sub = parser.add_subparsers(dest="command", required=True)

    verify_parser = sub.add_parser("verify", help="verify every configured feed")
    verify_parser.add_argument("--feeds", type=Path, default=FEEDS_FILE)
    verify_parser.add_argument("--max-age-days", type=float, default=21.0)

    import_parser = sub.add_parser("import-opml", help="import subscriptions from OPML")
    import_parser.add_argument("file", type=Path)
    import_parser.add_argument("--feeds", type=Path, default=FEEDS_FILE)
    import_parser.add_argument("--no-verify", action="store_true")

    check_parser = sub.add_parser("check", help="verify one or more feed URLs")
    check_parser.add_argument("urls", nargs="+")

    args = parser.parse_args(argv)
    if args.command == "verify":
        checks = check_feeds(
            [feed.url for feed in load_feeds(args.feeds)],
            max_age_days=args.max_age_days,
        )
        print(format_report(checks))
        return 0 if all(check.ok for check in checks) else 1
    if args.command == "import-opml":
        added, failed = import_opml(
            args.file,
            args.feeds,
            verify=not args.no_verify,
        )
        print(f"Added {len(added)} feed{'s' if len(added) != 1 else ''}")
        for feed in added:
            print(f"+ {feed.name}: {feed.url}")
        if failed:
            print(format_report(failed))
        return 0

    checks = check_feeds(args.urls)
    print(format_report(checks))
    return 0 if all(check.ok for check in checks) else 1


def _existing_header(path: Path) -> str:
    if not path.exists():
        return ""
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        lines.append(line)
    return "\n".join(lines)


def _newest_entry_date(entries: list[object]) -> datetime | None:
    dates: list[datetime] = []
    for entry in entries:
        for field in ("published_parsed", "updated_parsed", "created_parsed"):
            value = entry.get(field)  # type: ignore[union-attr]
            if value:
                dates.append(datetime.fromtimestamp(calendar.timegm(value), UTC))
                break
    return max(dates, default=None)


def _text_or_none(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _url_name(url: str) -> str:
    return urlsplit(url).hostname or url


def _format_days(days: float) -> str:
    rounded = round(days)
    return str(rounded) if abs(days - rounded) < 0.05 else f"{days:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
