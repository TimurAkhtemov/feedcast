from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from feedcast.feedcheck import (
    FeedCheck,
    FeedSpec,
    check_feed,
    check_feeds,
    format_report,
    import_opml,
    load_feeds,
    main,
    normalize_feed_url,
    parse_opml,
    save_feeds,
)


FIXTURES = Path(__file__).parent / "fixtures" / "feeds"
NOW = datetime(2026, 10, 10, 12, tzinfo=UTC)


class FeedHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/rss.xml")
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def feed_server() -> Iterator[str]:
    handler = partial(FeedHandler, directory=str(FIXTURES))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    ("filename", "version", "entries", "newest"),
    [
        ("rss.xml", "rss20", 2, datetime(2026, 9, 30, 12, tzinfo=UTC)),
        ("atom.xml", "atom10", 1, datetime(2026, 10, 1, 12, tzinfo=UTC)),
        ("rdf.xml", "rss10", 1, datetime(2026, 10, 2, 12, tzinfo=UTC)),
    ],
)
def test_feed_formats_are_accepted(
    feed_server: str,
    filename: str,
    version: str,
    entries: int,
    newest: datetime,
) -> None:
    result = check_feed(f"{feed_server}/{filename}", now=NOW)

    assert result.ok
    assert result.format == version
    assert result.entries == entries
    assert result.newest == newest
    assert result.newest.tzinfo is UTC
    assert result.error is None


def test_html_is_rejected_as_not_a_feed(feed_server: str) -> None:
    result = check_feed(f"{feed_server}/not-feed.html", now=NOW)

    assert not result.ok
    assert result.error == "not a feed (text/html)"


def test_404_is_rejected(feed_server: str) -> None:
    result = check_feed(f"{feed_server}/missing.xml", now=NOW)

    assert not result.ok
    assert result.error == "HTTP 404"


def test_redirect_is_followed_and_final_url_is_recorded(feed_server: str) -> None:
    result = check_feed(f"{feed_server}/redirect", now=NOW)

    assert result.ok
    assert result.final_url == f"{feed_server}/rss.xml"


def test_stale_feed_can_be_rejected_or_accepted(feed_server: str) -> None:
    url = f"{feed_server}/stale.xml"

    stale = check_feed(url, now=NOW)
    accepted = check_feed(url, now=NOW, max_age_days=41)

    assert not stale.ok
    assert stale.error == "stale: newest entry 40 days old"
    assert accepted.ok


def test_empty_feed_is_rejected(feed_server: str) -> None:
    result = check_feed(f"{feed_server}/empty.xml", now=NOW)

    assert not result.ok
    assert result.format == "rss20"
    assert result.error == "no entries"


def test_undated_feed_is_accepted() -> None:
    content = b"""<rss version="2.0"><channel><title>Undated</title>
        <item><title>Story</title><link>https://example.com/story</link></item>
        </channel></rss>"""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=content)
    )
    client = httpx.Client(transport=transport)
    try:
        result = check_feed("https://example.com/feed", client=client, now=NOW)
    finally:
        client.close()

    assert result.ok
    assert result.newest is None


@pytest.mark.parametrize(
    ("exception", "error"),
    [
        (httpx.ReadTimeout("too slow"), "timeout"),
        (httpx.ConnectError("refused"), "connection error: refused"),
    ],
)
def test_network_errors_do_not_raise(exception: Exception, error: str) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        exception.request = request  # type: ignore[attr-defined]
        raise exception

    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        result = check_feed("https://example.com/feed", client=client)

    assert not result.ok
    assert result.error == error


def test_check_feed_sends_descriptive_user_agent() -> None:
    seen: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["user-agent"])
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        check_feed("https://example.com/feed", client=client)

    assert seen == ["feedcast/0.1 (+local RSS reader)"]


def test_check_feeds_preserves_input_order(feed_server: str) -> None:
    urls = [
        f"{feed_server}/rdf.xml",
        f"{feed_server}/missing.xml",
        f"{feed_server}/atom.xml",
    ]

    checks = check_feeds(urls, concurrency=3, now=NOW)

    assert [check.url for check in checks] == urls
    assert [check.ok for check in checks] == [True, False, True]


@pytest.mark.parametrize(
    ("url", "normalized"),
    [
        ("HTTP://Example.COM:80/feed/#part", "https://example.com/feed"),
        ("https://Example.COM:443", "https://example.com/"),
        ("http://example.com/feed?view=full#top", "https://example.com/feed?view=full"),
        ("https://example.com/feed///", "https://example.com/feed"),
        ("https://example.com:80/feed", "https://example.com:80/feed"),
    ],
)
def test_normalize_feed_url(url: str, normalized: str) -> None:
    assert normalize_feed_url(url) == normalized


def test_parse_opml_walks_nested_groups_and_deduplicates() -> None:
    feeds = parse_opml(FIXTURES / "subscriptions.opml")

    assert feeds == [
        FeedSpec("First Feed", "HTTP://Example.com:80/feed/"),
        FeedSpec("Second Feed", "https://second.example/atom?view=full#latest"),
        FeedSpec("third.example", "https://third.example/rss"),
    ]


def test_parse_opml_accepts_xml_bytes_and_text() -> None:
    content = b'<opml><body><outline text="A" xmlUrl="https://a.example/rss" /></body></opml>'

    assert parse_opml(content) == [FeedSpec("A", "https://a.example/rss")]
    assert parse_opml(content.decode()) == [FeedSpec("A", "https://a.example/rss")]


def test_import_opml_adds_only_new_verified_feeds(tmp_path: Path) -> None:
    feeds_file = tmp_path / "feeds.yaml"
    save_feeds(
        feeds_file,
        [FeedSpec("Already here", "https://example.com/feed")],
        header="Subscriptions",
    )

    def verify(urls: list[str]) -> list[FeedCheck]:
        return [
            FeedCheck(urls[0], True, urls[0], "Second", "atom10", 1, NOW, None),
            FeedCheck(urls[1], False, urls[1], None, None, 0, None, "HTTP 404"),
        ]

    added, failed = import_opml(
        FIXTURES / "subscriptions.opml", feeds_file, check=verify
    )

    assert added == [
        FeedSpec("Second Feed", "https://second.example/atom?view=full#latest")
    ]
    assert [result.url for result in failed] == ["https://third.example/rss"]
    assert load_feeds(feeds_file) == [
        FeedSpec("Already here", "https://example.com/feed"),
        FeedSpec("Second Feed", "https://second.example/atom?view=full#latest"),
    ]
    assert feeds_file.read_text().startswith("# Subscriptions\n")


def test_import_opml_leaves_file_untouched_when_nothing_is_added(tmp_path: Path) -> None:
    feeds_file = tmp_path / "feeds.yaml"
    feeds_file.write_text(
        "# Preserve this exact file\nfeeds:\n- name: First\n  url: https://example.com/feed\n"
        "- name: Second\n  url: https://second.example/atom?view=full\n"
        "- name: Third\n  url: https://third.example/rss\n"
    )
    before = feeds_file.read_bytes()

    added, failed = import_opml(
        FIXTURES / "subscriptions.opml", feeds_file, verify=False
    )

    assert added == []
    assert failed == []
    assert feeds_file.read_bytes() == before


def test_save_and_load_round_trip_and_preserve_header(tmp_path: Path) -> None:
    path = tmp_path / "feeds.yaml"
    path.write_text("# Curated feeds\n# Keep this note\nfeeds: []\n")
    feeds = [
        FeedSpec("News: daily", "https://example.com/rss?edition=morning"),
        FeedSpec("Unicode — feed", "https://example.org/atom"),
    ]

    save_feeds(path, feeds)

    assert load_feeds(path) == feeds
    assert path.read_text().startswith("# Curated feeds\n# Keep this note\n")


def test_load_feeds_handles_missing_and_names_invalid_entry(tmp_path: Path) -> None:
    path = tmp_path / "feeds.yaml"
    assert load_feeds(path) == []
    path.write_text("feeds:\n- name: Broken feed\n")

    with pytest.raises(ValueError, match="Broken feed"):
        load_feeds(path)


def test_format_report_has_one_line_per_feed_and_summary() -> None:
    checks = [
        FeedCheck("https://ok", True, "https://ok", "Good", "rss20", 2, None, None),
        FeedCheck("https://bad", False, "https://bad", None, None, 0, None, "HTTP 404"),
    ]

    lines = format_report(checks).splitlines()

    assert len(lines) == 3
    assert lines[0].startswith("OK") and "Good" in lines[0]
    assert lines[1].startswith("FAIL") and "HTTP 404" in lines[1]
    assert lines[2] == "1 ok, 1 failed"


def test_main_verify_exit_codes(
    feed_server: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    feeds_file = tmp_path / "feeds.yaml"
    save_feeds(feeds_file, [FeedSpec("RSS", f"{feed_server}/rss.xml")])

    assert main(["verify", "--feeds", str(feeds_file), "--max-age-days", "100000"]) == 0
    assert "1 ok, 0 failed" in capsys.readouterr().out

    save_feeds(feeds_file, [FeedSpec("Missing", f"{feed_server}/missing.xml")])
    assert main(["verify", "--feeds", str(feeds_file)]) == 1
    assert "0 ok, 1 failed" in capsys.readouterr().out
