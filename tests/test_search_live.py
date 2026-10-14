"""Hits the real local SearXNG. Run with FEEDCAST_LIVE_SEARXNG=1 once the container is up."""

import os

import pytest

from feedcast.search import is_up, search

pytestmark = pytest.mark.skipif(
    os.environ.get("FEEDCAST_LIVE_SEARXNG") != "1",
    reason="set FEEDCAST_LIVE_SEARXNG=1 to query the local SearXNG",
)


def test_live_news_search_returns_results():
    assert is_up(), "SearXNG is not answering; docker compose -f searxng/docker-compose.yml up -d"
    results = search("semiconductor export controls", limit=10)
    assert len(results) >= 1
    assert all(r.url.startswith(("http://", "https://")) for r in results)
    assert any(r.title for r in results)
