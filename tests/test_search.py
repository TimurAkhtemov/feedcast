from datetime import UTC, datetime, timedelta

import httpx
import pytest

from feedcast import search as search_mod
from feedcast.search import (
    COMPOSE_FILE,
    SearchUnavailable,
    candidate_feeds,
    ensure_up,
    find_feed_links,
    is_up,
    search,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _json_client(results, seen: list | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json={"query": "q", "results": results})

    return _client(handler)


def _hit(url: str, **extra) -> dict:
    return {"url": url, "title": f"title {url}", "content": "snippet", "engines": ["bing news"], **extra}


def test_compose_file_ships_with_project():
    assert COMPOSE_FILE.is_file()
    assert COMPOSE_FILE.parent.name == "searxng"


# --- search ------------------------------------------------------------------------


def test_search_sends_expected_query_params():
    seen: list[httpx.Request] = []
    search(
        "chip export rules",
        base_url="http://sx.test:9999/",
        categories="news",
        time_range="month",
        language="en-US",
        client=_json_client([], seen),
    )
    (request,) = seen
    assert request.method == "GET"
    assert request.url.host == "sx.test"
    assert request.url.port == 9999
    assert request.url.path == "/search"
    params = request.url.params
    assert params["q"] == "chip export rules"
    assert params["format"] == "json"
    assert params["categories"] == "news"
    assert params["time_range"] == "month"
    assert params["language"] == "en-US"


def test_search_defaults_and_no_time_range():
    seen: list[httpx.Request] = []
    search("x", client=_json_client([], seen))
    search("x", time_range=None, client=_json_client([], seen))
    first, second = (r.url.params for r in seen)
    assert seen[0].url.host == "127.0.0.1" and seen[0].url.port == 8888
    assert first["categories"] == "news"
    assert first["time_range"] == "week"
    assert first["language"] == "en"
    assert "time_range" not in second


def test_search_parses_fields():
    raw = [
        {
            "url": "https://example.com/a",
            "title": "  Chips   and\npolicy ",
            "content": "Some snippet.",
            "engines": ["bing news", "duckduckgo news"],
            "publishedDate": "2026-10-10T08:30:00",
        }
    ]
    (result,) = search("q", time_range=None, client=_json_client(raw))
    assert result.title == "Chips and policy"
    assert result.url == "https://example.com/a"
    assert result.content == "Some snippet."
    assert result.engines == ["bing news", "duckduckgo news"]
    assert result.published == datetime(2026, 10, 10, 8, 30, tzinfo=UTC)


def test_search_tolerates_missing_fields():
    raw = [{"url": "https://example.com/bare"}, {"url": "https://example.com/one", "engine": "yahoo news"}]
    bare, single = search("q", time_range=None, client=_json_client(raw))
    assert bare.title == ""
    assert bare.content == ""
    assert bare.engines == []
    assert bare.published is None
    assert single.engines == ["yahoo news"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-10-10T08:30:00", datetime(2026, 10, 10, 8, 30, tzinfo=UTC)),
        ("2026-10-10T08:30:00.482377", datetime(2026, 10, 10, 8, 30, 0, 482377, tzinfo=UTC)),
        ("2026-10-10T08:30:00Z", datetime(2026, 10, 10, 8, 30, tzinfo=UTC)),
        ("2026-10-10T10:30:00+02:00", datetime(2026, 10, 10, 8, 30, tzinfo=UTC)),
        ("2026-10-10 08:30:00", datetime(2026, 10, 10, 8, 30, tzinfo=UTC)),
        ("2026-10-10", datetime(2026, 10, 10, tzinfo=UTC)),
        ("Sat, 10 Oct 2026 08:30:00 GMT", datetime(2026, 10, 10, 8, 30, tzinfo=UTC)),
        (1791621000, datetime(2026, 10, 10, 8, 30, tzinfo=UTC)),
        (None, None),
        ("", None),
        ("yesterday-ish", None),
        (True, None),
        ({"nested": 1}, None),
    ],
)
def test_search_published_date_variants(value, expected):
    raw = [_hit("https://example.com/a", publishedDate=value)]
    (result,) = search("q", time_range=None, client=_json_client(raw))
    assert result.published == expected
    if result.published is not None:
        assert result.published.utcoffset() == timedelta(0)


def test_search_drops_non_http_urls():
    raw = [
        _hit("ftp://example.com/file"),
        _hit("javascript:alert(1)"),
        _hit("/relative/path"),
        _hit("http://[broken"),
        {"title": "no url at all"},
        {"url": 42},
        "not a dict",
        _hit("http://example.com/plain-http"),
        _hit("https://example.com/secure"),
    ]
    results = search("q", time_range=None, client=_json_client(raw))
    assert [r.url for r in results] == ["http://example.com/plain-http", "https://example.com/secure"]


def test_search_dedupes_keeps_order_and_caps():
    raw = [
        _hit("https://a.example/1"),
        _hit("https://b.example/2"),
        _hit("https://a.example/1"),
        _hit("https://A.example/1#comments"),
        _hit("https://c.example/3"),
        _hit("https://d.example/4"),
    ]
    results = search("q", time_range=None, limit=3, client=_json_client(raw))
    assert [r.url for r in results] == [
        "https://a.example/1",
        "https://b.example/2",
        "https://c.example/3",
    ]
    assert search("q", time_range=None, limit=0, client=_json_client(raw)) == []


def test_search_enforces_time_range_on_dated_results():
    now = datetime.now(UTC)
    raw = [
        _hit("https://ex.com/fresh", publishedDate=(now - timedelta(days=2)).isoformat()),
        _hit("https://ex.com/stale", publishedDate=(now - timedelta(days=40)).isoformat()),
        _hit("https://ex.com/undated"),
    ]
    week = search("q", time_range="week", client=_json_client(raw))
    assert [r.url for r in week] == ["https://ex.com/fresh", "https://ex.com/undated"]
    anytime = search("q", time_range=None, client=_json_client(raw))
    assert len(anytime) == 3


def test_search_connection_error_raises_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(SearchUnavailable):
        search("q", client=_client(handler))


def test_search_timeout_raises_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(SearchUnavailable):
        search("q", client=_client(handler))


def test_search_http_500_raises_unavailable():
    with pytest.raises(SearchUnavailable, match="500"):
        search("q", client=_client(lambda request: httpx.Response(500, text="boom")))


def test_search_json_disabled_403_raises_unavailable():
    # What SearXNG answers when the json format is not enabled in settings.yml.
    with pytest.raises(SearchUnavailable, match="403"):
        search("q", client=_client(lambda request: httpx.Response(403, text="Forbidden")))


def test_search_bad_json_raises_unavailable():
    page = httpx.Response(200, text="<html>not json</html>", headers={"content-type": "text/html"})
    with pytest.raises(SearchUnavailable):
        search("q", client=_client(lambda request: page))


def test_search_json_without_results_raises_unavailable():
    with pytest.raises(SearchUnavailable):
        search("q", client=_client(lambda request: httpx.Response(200, json={"error": "nope"})))


# --- is_up / ensure_up -------------------------------------------------------------


def test_is_up_true_on_healthz_ok():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, text="OK")

    assert is_up("http://sx.test", client=_client(handler)) is True
    assert paths == ["/healthz"]


def test_is_up_false_on_error_status():
    assert is_up(client=_client(lambda request: httpx.Response(502))) is False


def test_is_up_false_on_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert is_up(client=_client(handler)) is False


def test_ensure_up_short_circuits_when_already_up(monkeypatch):
    monkeypatch.setattr(search_mod, "is_up", lambda base_url=None, **kw: True)
    monkeypatch.setattr(search_mod.subprocess, "run", _fail_if_called)
    assert ensure_up(docker="/nonexistent/docker") is True


def test_ensure_up_false_when_docker_missing(monkeypatch):
    monkeypatch.setattr(search_mod, "is_up", lambda base_url=None, **kw: False)
    monkeypatch.setattr(search_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(search_mod, "_DOCKER_FALLBACKS", ())
    monkeypatch.setattr(search_mod.subprocess, "run", _fail_if_called)
    assert ensure_up(wait_seconds=0.1) is False


def test_ensure_up_false_when_docker_path_does_not_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(search_mod, "is_up", lambda base_url=None, **kw: False)
    compose = tmp_path / "docker-compose.yml"
    assert ensure_up(docker="/nonexistent/bin/docker", compose_file=compose, wait_seconds=0.1) is False


def test_ensure_up_false_when_compose_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(search_mod, "is_up", lambda base_url=None, **kw: False)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return search_mod.subprocess.CompletedProcess(cmd, 1, "", "Cannot connect to the Docker daemon")

    monkeypatch.setattr(search_mod.subprocess, "run", fake_run)
    compose = tmp_path / "docker-compose.yml"
    assert ensure_up(docker="/usr/bin/docker", compose_file=compose, wait_seconds=0.1) is False
    assert calls == [["/usr/bin/docker", "compose", "-f", str(compose), "up", "-d"]]


def test_ensure_up_polls_until_healthy(monkeypatch, tmp_path):
    answers = iter([False, False, True])
    monkeypatch.setattr(search_mod, "is_up", lambda base_url=None, **kw: next(answers))
    monkeypatch.setattr(search_mod, "_POLL_INTERVAL", 0.0)
    monkeypatch.setattr(
        search_mod.subprocess,
        "run",
        lambda cmd, **kw: search_mod.subprocess.CompletedProcess(cmd, 0, "", ""),
    )
    assert ensure_up(docker="/usr/bin/docker", compose_file=tmp_path / "c.yml", wait_seconds=5) is True
    assert (tmp_path / ".env").read_text().startswith("SEARXNG_SECRET=")


def test_ensure_up_gives_up_after_wait(monkeypatch, tmp_path):
    monkeypatch.setattr(search_mod, "is_up", lambda base_url=None, **kw: False)
    monkeypatch.setattr(search_mod, "_POLL_INTERVAL", 0.0)
    monkeypatch.setattr(
        search_mod.subprocess,
        "run",
        lambda cmd, **kw: search_mod.subprocess.CompletedProcess(cmd, 0, "", ""),
    )
    assert ensure_up(docker="/usr/bin/docker", compose_file=tmp_path / "c.yml", wait_seconds=0.05) is False


def test_ensure_secret_creates_a_private_key_once(tmp_path):
    env = tmp_path / ".env"
    assert search_mod.ensure_secret(env) is True
    first = env.read_text()
    assert len(first.strip().split("=", 1)[1]) == 64
    assert env.stat().st_mode & 0o777 == 0o600
    assert search_mod.ensure_secret(env) is False and env.read_text() == first


def test_committed_settings_hold_no_secret():
    settings = (COMPOSE_FILE.parent / "settings.yml").read_text()
    assert "secret_key" not in settings.replace("# secret_key", "")
    assert "SEARXNG_SECRET" in COMPOSE_FILE.read_text()


def _fail_if_called(*args, **kwargs):
    raise AssertionError("docker should not have been run")


# --- feed discovery ----------------------------------------------------------------

PAGE = """<!doctype html>
<html><head>
  <link rel="stylesheet" href="/style.css">
  <link rel="alternate" type="application/rss+xml" title="All" href="/feed/">
  <LINK REL="Alternate" TYPE="application/atom+xml" HREF="atom.xml">
  <link rel="alternate" type="text/html" hreflang="fr" href="/fr/">
  <link rel="alternate" type="application/json" href="/feed.json">
  <link rel="icon" type="application/rss+xml" href="/not-a-feed.xml">
  <link rel="alternate" type="application/rss+xml" href="https://feeds.example.net/world.rss">
  <link rel="alternate" type="application/rss+xml" href="/feed/">
  <link rel="alternate" type="application/rss+xml">
</head><body><a rel="alternate" type="application/rss+xml" href="/body-link.xml">x</a></body></html>
"""


def test_find_feed_links_resolves_filters_and_dedupes():
    assert find_feed_links("https://news.example.com/world/story.html", PAGE) == [
        "https://news.example.com/feed/",
        "https://news.example.com/world/atom.xml",
        "https://feeds.example.net/world.rss",
    ]


def test_find_feed_links_honours_base_and_type_params():
    page = """<head><base href="https://cdn.example.org/site/">
    <link rel="alternate home" type="application/rss+xml; charset=utf-8" href="rss.xml"></head>"""
    assert find_feed_links("https://example.org/a/b", page) == ["https://cdn.example.org/site/rss.xml"]


def test_find_feed_links_empty_and_broken_html():
    assert find_feed_links("https://example.com/", "") == []
    assert find_feed_links("https://example.com/", "<link rel=alternate type=application/rss+xml") == []


def _site(pages: dict[str, httpx.Response | Exception]):
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requested.append(url)
        answer = pages.get(url, httpx.Response(404))
        if isinstance(answer, Exception):
            raise answer
        return answer

    return _client(handler), requested


def _html(body: str) -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8"})


def test_candidate_feeds_discovers_from_article_and_homepage():
    article = '<link rel="alternate" type="application/rss+xml" href="/section/tech/rss">'
    home = (
        '<link rel="alternate" type="application/rss+xml" href="/rss">'
        '<link rel="alternate" type="application/rss+xml" href="/section/tech/rss">'
    )
    client, requested = _site(
        {
            "https://site.example/2026/10/story": _html(article),
            "https://site.example/": _html(home),
        }
    )
    feeds = candidate_feeds("https://site.example/2026/10/story", client=client)
    assert feeds == ["https://site.example/section/tech/rss", "https://site.example/rss"]
    assert requested == ["https://site.example/2026/10/story", "https://site.example/"]


def test_candidate_feeds_follows_redirects_before_resolving():
    client, _ = _site(
        {
            "https://short.example/s/1": httpx.Response(
                301, headers={"location": "https://www.long.example/news/1"}
            ),
            "https://www.long.example/news/1": _html(
                '<link rel="alternate" type="application/atom+xml" href="feed.atom">'
            ),
        }
    )
    feeds = candidate_feeds("https://short.example/s/1", client=client)
    assert feeds == ["https://www.long.example/news/feed.atom"]


def test_candidate_feeds_falls_back_to_guesses():
    client, _ = _site({"https://quiet.example/a/b?x=1": _html("<html><head></head></html>")})
    assert candidate_feeds("https://quiet.example/a/b?x=1", client=client) == [
        "https://quiet.example/feed",
        "https://quiet.example/rss",
        "https://quiet.example/rss.xml",
        "https://quiet.example/feed.xml",
        "https://quiet.example/atom.xml",
        "https://quiet.example/index.xml",
    ]


def test_candidate_feeds_survives_network_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    feeds = candidate_feeds("https://down.example/story", client=_client(handler))
    assert feeds[0] == "https://down.example/feed"
    assert len(feeds) == 6


def test_candidate_feeds_keeps_what_it_found_when_homepage_fails():
    request = httpx.Request("GET", "https://half.example/")
    client, _ = _site(
        {
            "https://half.example/p/1": _html(
                '<link rel="alternate" type="application/rss+xml" href="/p/feed">'
            ),
            "https://half.example/": httpx.ReadTimeout("slow", request=request),
        }
    )
    assert candidate_feeds("https://half.example/p/1", client=client) == ["https://half.example/p/feed"]


def test_candidate_feeds_ignores_non_http():
    assert candidate_feeds("mailto:someone@example.com", client=_client(_fail_if_called)) == []
