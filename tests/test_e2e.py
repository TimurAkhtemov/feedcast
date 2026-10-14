"""End to end on fixture feeds: ingest -> judge -> group -> write -> voice -> deliver.

Laya, Ollama, Kokoro, and the web are stubbed (see conftest.py); ffmpeg, SQLite,
trafilatura, feedparser, clustering, and the grounding check are real.
"""


from feedcast import voice
from feedcast.pipeline import ask, edition, watch

from conftest import FakeQwen


def _chapters(path):
    return [c.title for c in voice.probe_chapters(path)]


def test_default_edition_end_to_end(services):
    services.config["edition"]["target_minutes"] = 8
    ep = edition(services, topics=["battery recycling"], skip=["sports"])

    assert not ep.failed
    assert ep.mp3.exists() and ep.notes.exists()
    assert ep.mp3.parent == services.episodes_dir
    assert voice.probe_duration(ep.mp3) == ep.duration > 5

    chapters = _chapters(ep.mp3)
    assert chapters[0] == "Intro" and chapters[-1] == "Outro"
    assert len(chapters) == ep.segments + 2
    # Biggest story first: three outlets on the bridge, two on the rover.
    assert "Harbor Bridge" in chapters[1]
    assert "Tessera" in chapters[2] or "Mars" in chapters[2]
    # The standing topic got its own segment.
    assert any("Battery" in c for c in chapters)
    # The sponsored post never became a story.
    assert not any("SoundMax" in c for c in chapters)
    # Two outlets covered the championship final, but sports is on the skip list.
    assert not any("Pelicans" in c for c in chapters)

    notes = ep.notes.read_text()
    assert "battery recycling" in notes
    assert "https://coastline.example/2026/10/harbor-bridge" in notes
    assert "3 outlets" in notes
    # HALLUCINATE_ALWAYS: rewritten once, still ungrounded, dropped and listed.
    assert "## Left out" in notes and "Council debates downtown parking rules" in notes
    assert "Zorblax" in notes
    assert "## Skipped" in notes and "Pelicans win regional championship final" in notes
    assert "sports, 2 outlets" in notes
    # HALLUCINATE_ONCE: fixed by the rewrite and kept.
    assert any("School lunch" in c for c in chapters)

    db = services.store.db
    questions = {q for (q,) in db.execute("SELECT DISTINCT question FROM judgments")}
    assert "Is this article mainly an advertisement or sponsored post?" in questions
    assert "Is this article about battery recycling?" in questions
    assert "Is this article about sports?" in questions
    # Every item in the window got an ad judgment, and each one is logged with its threshold.
    n_items = db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    ad_rows = db.execute(
        "SELECT COUNT(DISTINCT item_id), MIN(threshold) FROM judgments WHERE question LIKE '%advertisement%'"
    ).fetchone()
    assert ad_rows == (n_items, services.config["judge"]["ad_threshold"])


def test_skip_list_needs_most_of_a_story_and_spares_standing_topics(services):
    from feedcast.pipeline import run_ingest

    # Only one of the three Harbor Bridge articles calls the repairs "costly": the story stays.
    ep = edition(services, topics=[], skip=["costly"])
    assert "Harbor Bridge" in _chapters(ep.mp3)[1]
    assert "## Skipped" not in ep.notes.read_text()

    # A standing topic is something you asked for, so the skip list doesn't block it.
    ep = edition(services, topics=["championship final"], skip=["sports"], refresh=False)
    assert any("Pelicans" in c for c in _chapters(ep.mp3))
    assert "## Skipped" not in ep.notes.read_text()

    # The hourly ingest answers the skip-list question ahead of time for new items.
    services.store.db.execute("DELETE FROM items")
    run_ingest(services, warm=True, skip=["sports"])
    warmed = services.store.db.execute(
        "SELECT COUNT(*) FROM judgments WHERE run_id = 'ingest' AND question = 'Is this article about sports?'"
    ).fetchone()[0]
    assert warmed == 11


def test_edition_dedupes_and_extracts_full_text(services):
    edition(services, topics=[])
    db = services.store.db
    urls = [u for (u,) in db.execute("SELECT canonical_url FROM items")]
    assert len(urls) == len(set(urls)) == 11
    assert all("utm_source" not in u for u in urls)
    # Full article text comes from the page, not the one-paragraph feed summary.
    bridge = db.execute("SELECT full_text, text FROM items WHERE title LIKE 'Harbor Bridge reopens%'").fetchone()
    assert bridge[0] == 1 and "Engineers replaced the deck" in bridge[1]
    # Boilerplate stripped from what Laya sees.
    leads = " ".join(l for (l,) in db.execute("SELECT lead FROM items"))
    assert "newsletter" not in leads and "By Staff Reporter" not in leads
    # A second run adds nothing new.
    edition(services, topics=[])
    assert db.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 11


def test_on_demand_falls_back_to_search_and_suggests_feeds(services):
    ep = ask(services, ["volcano monitoring"])

    assert not ep.failed and ep.segments == 1
    chapters = _chapters(ep.mp3)
    assert len(chapters) == 3 and "volcano" in chapters[1].lower()
    notes = ep.notes.read_text()
    assert "## Feeds worth adding" in notes
    assert "https://volcanowatch.example/feed.xml" in notes
    assert "geonews.example/feed.xml" not in notes  # its candidate feed failed the check
    # Suggestions are never added to the pool's feed list.
    assert all("volcanowatch" not in f.url for f in services.feeds)
    assert "Is this article about volcano monitoring?" in "\n".join(
        q for (q,) in services.store.db.execute("SELECT DISTINCT question FROM judgments")
    )


def test_on_demand_with_nothing_found_still_writes_an_episode(services):
    services.search = lambda q, **kw: []
    ep = ask(services, ["deep sea mining"])
    assert not ep.failed and ep.segments == 0
    assert ep.mp3.exists() and "found nothing" in ep.notes.read_text()


def test_failure_still_writes_a_notice(services):
    services.llm = FakeQwen(fail=True)
    ep = edition(services, topics=[])

    assert ep.failed
    assert ep.mp3.name.endswith("-FAILED.mp3") and ep.mp3.exists()
    assert voice.probe_duration(ep.mp3) > 1
    notes = ep.notes.read_text()
    assert "failed while writing story 1" in notes
    assert "Ollama is not reachable" in notes
    status = services.store.db.execute("SELECT status FROM runs").fetchone()[0]
    assert status == "failed"


def test_failure_when_laya_cannot_load(services):
    def broken():
        raise RuntimeError("No Metal device available")

    services.backend_factory = broken
    ep = edition(services, topics=[])
    assert ep.failed and "judging ads with Laya" in ep.notes.read_text()


def test_watch_builds_requests_from_the_phone(services, tmp_path):
    requests = tmp_path / "requests"
    requests.mkdir()
    (requests / "Ask Feedcast.txt").write_text("volcano monitoring\n")
    (requests / "blank.txt").write_text("   \n")

    episodes = watch(services, requests)

    assert len(episodes) == 2
    built = [e for e in episodes if e.segments]
    assert len(built) == 1 and "volcano-monitoring" in built[0].mp3.name
    assert not list(requests.glob("*.txt"))
    assert len(list((requests / "done").glob("Ask Feedcast-*.txt"))) == 1
    assert len(list((requests / "failed").glob("blank-*.txt"))) == 1


def test_lock_timeout_still_writes_a_notice(services, monkeypatch):
    from contextlib import contextmanager

    from feedcast import pipeline

    @contextmanager
    def stuck(wait=0):
        raise TimeoutError("another feedcast build held the lock for 45 minutes")
        yield

    monkeypatch.setattr(pipeline, "build_lock", stuck)
    ep = edition(services, topics=[])
    assert ep.failed and ep.mp3.exists()
    assert "waiting for another build" in ep.notes.read_text()


def test_broken_ffmpeg_falls_back_to_bundled_notice(services, monkeypatch):
    from feedcast import pipeline

    def no_ffmpeg(*args, **kwargs):
        raise FileNotFoundError("ffmpeg is not installed")

    monkeypatch.setattr(pipeline.voice, "render_episode", no_ffmpeg)
    monkeypatch.setattr(pipeline.voice, "render_notice", no_ffmpeg)
    ep = edition(services, topics=[])
    assert ep.failed and ep.mp3.read_bytes() == pipeline.FAILED_MP3.read_bytes()
    assert "ffmpeg is not installed" in ep.notes.read_text()
