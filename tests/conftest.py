"""Stand-ins for Laya, Ollama, Kokoro, and the web, so the pipeline runs end to end in tests.

The stubs are deterministic and only as clever as the assertions need: the fake Laya
matches keywords, the fake embedder maps story keywords to fixed directions, and the
fake Qwen writes segments by quoting source sentences (so they pass the grounding
check) unless a source carries a HALLUCINATE marker.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pytest

from feedcast import paths
from feedcast.config import DEFAULTS
from feedcast.db import Store
from feedcast.feedcheck import FeedCheck, FeedSpec
from feedcast.pipeline import Services
from feedcast.search import SearchResult

FIXTURES = Path(__file__).parent / "fixtures" / "e2e"
NOW = datetime(2026, 10, 14, 5, 30, tzinfo=timezone.utc)

FEEDS = [
    FeedSpec("Coastline Times", "https://coastline.example/rss.xml"),
    FeedSpec("Metro Wire", "https://metrowire.example/rss.xml"),
    FeedSpec("Valley Post", "https://valleypost.example/rss.xml"),
    FeedSpec("Orbit Report", "https://orbit.example/rss.xml"),
]
FEED_FILES = {
    "coastline.example": "coastline",
    "metrowire.example": "metro",
    "valleypost.example": "valley",
    "orbit.example": "orbit",
}


def fixture_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if path == "/rss.xml" and host in FEED_FILES:
            body = (FIXTURES / "feeds" / f"{FEED_FILES[host]}.xml").read_bytes()
            return httpx.Response(200, content=body, headers={"content-type": "application/rss+xml"})
        if host in FEED_FILES and path.startswith("/2026/10/"):
            page = FIXTURES / "articles" / f"{FEED_FILES[host]}-{path.rsplit('/', 1)[-1]}.html"
            if page.exists():
                return httpx.Response(200, content=page.read_bytes(), headers={"content-type": "text/html"})
        if host in ("volcanowatch.example", "geonews.example") and path.startswith("/story"):
            page = FIXTURES / "search" / f"{host}.html"
            return httpx.Response(200, content=page.read_bytes(), headers={"content-type": "text/html"})
        return httpx.Response(404, text="not found")

    return httpx.MockTransport(handler)


class KeywordLaya:
    """Fake Laya: yes when the text has the question's keywords."""

    name = "stub-laya"

    def __init__(self):
        self.calls = 0

    def fit(self, text: str, question: str) -> tuple[str, bool]:
        limit = 1500 - len(question)
        return (text, False) if len(text) <= limit else (text[:limit], True)

    def noul_many(self, jobs):
        self.calls += len(jobs)
        out = []
        for state, question in jobs:
            text = state["article"].lower()
            if "advertisement" in question:
                out.append(0.95 if ("sponsored" in text or "shop now" in text) else 0.05)
                continue
            topic = re.search(r"about (.+)\?$", question).group(1).lower()
            words = [w for w in re.findall(r"[a-z]+", topic) if len(w) > 3]
            out.append(0.9 if words and all(w in text for w in words) else 0.1)
        return out


class KeywordEmbedder:
    """Fake embeddinggemma: each story keyword is a fixed direction."""

    name = "stub-embed"
    KEYWORDS = ["bridge", "tessera", "battery", "soundmax", "parking", "lunch", "volcano", "recycling",
                "pelicans"]

    def embed(self, texts):
        out = []
        for t in texts:
            low = t.lower()
            v = np.array([1.0 if k in low else 0.0 for k in self.KEYWORDS] + [0.05], dtype=np.float32)
            out.append(v)
        return np.stack(out)


class FakeQwen:
    """Writes by quoting source sentences, so a normal segment is fully grounded."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.prompts: list[str] = []

    def json(self, system: str, user: str, schema: dict) -> dict:
        self.prompts.append(user)
        if self.fail:
            from feedcast.llm import OllamaError
            raise OllamaError("Ollama is not reachable at http://127.0.0.1:11434/api/chat")
        props = schema["properties"]
        if "paragraphs" in props:
            return self._segment(system, user)
        if "intro" in props:
            date = re.search(r"Date: (.+)", user).group(1)
            return {"intro": f"This is Feedcast for {date}.", "outro": "That's all for today."}
        if "queries" in props:
            topic = re.search(r"Topic: (.+)", user).group(1)
            return {"queries": [topic, f"{topic} news", f"latest {topic}"]}
        if "topics" in props:
            request = re.search(r"Request: (.+)", user).group(1)
            return {"topics": [t.strip() for t in re.split(r"\band\b", request) if t.strip()]}
        raise AssertionError(f"unexpected schema {schema}")

    def _segment(self, system: str, user: str) -> dict:
        words = int(re.search(r"write (\d+) words", system).group(1))
        sources = re.split(r"\nThe listener asked about:|\nWrite the|Your previous draft:", user)[0]
        headline = re.search(r"Headline: (.+)", sources).group(1)
        sentences = []
        for block in re.split(r"=== From .+ ===", sources)[1:]:
            body = "\n".join(l for l in block.splitlines() if not l.startswith("Headline:"))
            for s in re.split(r"(?<=\.)\s+", body):
                s = s.strip()
                if s.endswith(".") and "HALLUCINATE" not in s and s not in sentences and len(s.split()) > 3:
                    sentences.append(s)
        paras, count = [], 0
        for i in range(0, len(sentences), 2):
            if count >= words:
                break
            paras.append(" ".join(sentences[i : i + 2]))
            count += len(paras[-1].split())
        while len(paras) < 3:
            paras.append(paras[-1] if paras else headline + ".")
        retry = "Your previous draft:" in user
        if "HALLUCINATE_ALWAYS" in sources or ("HALLUCINATE_ONCE" in sources and not retry):
            paras[0] = "Officials from Zorblax Industries said 999 people attended. " + paras[0]
        return {"headline": headline, "paragraphs": paras}


class ToneSynth:
    sample_rate = 24000

    def synthesize(self, text: str) -> np.ndarray:
        n = int(self.sample_rate * min(4.0, 0.012 * len(text) + 0.2))
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def fake_search(query: str, **kwargs) -> list[SearchResult]:
    if "volcano" not in query.lower():
        return []
    return [
        SearchResult("Mount Kessel volcano monitoring network expands",
                     "https://volcanowatch.example/story/kessel", "Sensors added", ["stub"], None),
        SearchResult("New sensors boost volcano monitoring at Mount Kessel",
                     "https://geonews.example/story/kessel-sensors", "", ["stub"], None),
    ]


def fake_check_feed(url: str) -> FeedCheck:
    ok = url == "https://volcanowatch.example/feed.xml"
    return FeedCheck(url, ok, url, "Volcano Watch" if ok else None, "rss20", 5 if ok else 0, NOW,
                     None if ok else "HTTP 404")


@pytest.fixture
def services(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    (tmp_path / "data").mkdir()
    config = {k: dict(v) for k, v in DEFAULTS.items()}
    config["edition"]["target_minutes"] = 4
    config["ask"]["target_minutes"] = 2
    svc = Services(
        store=Store(tmp_path / "data" / "test.db"),
        config=config,
        llm=FakeQwen(),
        embedder=KeywordEmbedder(),
        backend_factory=KeywordLaya,
        synth_factory=ToneSynth,
        http=httpx.Client(transport=fixture_transport(), follow_redirects=True),
        feeds=FEEDS,
        episodes_dir=tmp_path / "episodes",
        search=fake_search,
        searx_up=lambda: True,
        check_feed=fake_check_feed,
        candidate_feeds=lambda url: [f"https://{httpx.URL(url).host}/feed.xml"],
        now=lambda: NOW,
    )
    yield svc
    svc.store.close()
