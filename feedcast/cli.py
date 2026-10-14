"""Command-line entry point."""

from __future__ import annotations

import argparse
import fcntl
import logging
import sys
import time
from pathlib import Path

from feedcast import __version__, paths

log = logging.getLogger("feedcast")


def setup_logging(verbose: bool = False) -> None:
    (paths.DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in (logging.StreamHandler(sys.stderr),
                    logging.FileHandler(paths.DATA_DIR / "logs" / "feedcast.log")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    for noisy in ("httpx", "httpcore", "trafilatura", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_services(episodes_dir: Path | None = None):
    """The real thing: SQLite pool, Laya on Metal, Qwen and embeddinggemma via Ollama,
    Kokoro via mlx-audio, and the local SearXNG."""
    from feedcast import search
    from feedcast.config import load_config
    from feedcast.db import Store
    from feedcast.feedcheck import check_feed, load_feeds
    from feedcast.ingest import make_client
    from feedcast.judge import LayaBackend
    from feedcast.llm import OllamaEmbedder, OllamaLLM
    from feedcast.pipeline import Services
    from feedcast.voice import KokoroSynthesizer

    cfg = load_config()
    svc_cfg, write_cfg, voice_cfg = cfg["services"], cfg["write"], cfg["voice"]
    http = make_client(cfg["ingest"]["timeout_seconds"])
    synth: list = []

    def synth_factory():
        if not synth:
            synth.append(KokoroSynthesizer(voice_cfg["model"], voice_cfg["voice"], float(voice_cfg["speed"])))
        return synth[0]

    searx = svc_cfg["searxng_url"]
    return Services(
        store=Store(paths.DB_PATH),
        config=cfg,
        llm=OllamaLLM(write_cfg["model"], url=svc_cfg["ollama_url"], temperature=write_cfg["temperature"]),
        embedder=OllamaEmbedder(cfg["group"]["embed_model"], url=svc_cfg["ollama_url"]),
        backend_factory=lambda: LayaBackend(batch_size=cfg["judge"]["batch_size"]),
        synth_factory=synth_factory,
        http=http,
        feeds=load_feeds(paths.FEEDS_FILE),
        episodes_dir=episodes_dir or paths.EPISODES_DIR,
        search=lambda q, **kw: search.search(q, base_url=searx, **kw),
        searx_up=lambda: search.ensure_up(searx),
        check_feed=lambda url: check_feed(url, client=http, max_age_days=14),
        candidate_feeds=lambda url: search.candidate_feeds(url, client=http),
    )


def cmd_ingest(args) -> int:
    from feedcast.pipeline import run_guarded, run_ingest

    lock = open(paths.DATA_DIR / "ingest.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("another ingest is running; skipping")
        return 0
    svc = build_services()
    try:
        report = run_ingest(svc, warm=args.warm)
    except Exception as exc:
        # Hourly job: write a failure notice, but at most once per 12 hours per error,
        # so a long outage doesn't fill the episodes folder.
        detail = f"{type(exc).__name__}: {exc}"
        last = svc.store.last_failure("ingest", detail)
        if last is None or time.time() - last > 12 * 3600:
            run_guarded(svc, "ingest", "", lambda step: _raise(step, exc))
        else:
            log.error("ingest failed again (notice already written): %s", detail)
        return 1
    print(report.summary())
    return 0


def _raise(step, exc):
    step("fetching feeds")
    raise exc


def cmd_edition(args) -> int:
    from feedcast.config import load_topics
    from feedcast.pipeline import edition

    svc = build_services()
    ep = edition(svc, topics=load_topics(), target_minutes=args.minutes, refresh=not args.no_ingest)
    print(f"{'FAILED' if ep.failed else 'ok'}: {ep.mp3} ({ep.duration:.0f} s, {ep.segments} stories)")
    return 1 if ep.failed else 0


def cmd_ask(args) -> int:
    from feedcast.pipeline import ask

    svc = build_services()
    ep = ask(svc, args.topics, target_minutes=args.minutes)
    print(f"{'FAILED' if ep.failed else 'ok'}: {ep.mp3} ({ep.duration:.0f} s, {ep.segments} stories)")
    return 1 if ep.failed else 0


def cmd_watch(args) -> int:
    from feedcast.pipeline import watch

    svc = build_services()
    episodes = watch(svc)
    for ep in episodes:
        print(f"{'FAILED' if ep.failed else 'ok'}: {ep.mp3} ({ep.duration:.0f} s)")
    if not episodes:
        print("no requests waiting")
    return 1 if any(ep.failed for ep in episodes) else 0


def cmd_feeds(args) -> int:
    from feedcast import feedcheck

    return feedcheck.main(args.rest)


def cmd_launchd(args) -> int:
    from feedcast import launchd

    lines = {"install": launchd.install, "uninstall": launchd.uninstall, "status": launchd.status}[args.action]()
    print("\n".join(lines) or "nothing to do")
    return 0


def cmd_judgments(args) -> int:
    """Answer distribution per question, for tuning thresholds in config.yaml."""
    import sqlite3

    db = sqlite3.connect(paths.DB_PATH)
    rows = db.execute(
        "SELECT question, COUNT(*), SUM(verdict), AVG(p), SUM(p >= 0.3), SUM(p >= 0.5), "
        "SUM(p >= 0.7), SUM(p >= 0.9), MAX(threshold), SUM(truncated) FROM judgments "
        "WHERE ts >= strftime('%s','now') - ? * 86400 GROUP BY question ORDER BY COUNT(*) DESC",
        (args.days,),
    ).fetchall()
    print(f"{'n':>6} {'yes':>5} {'mean':>5} {'>=.3':>5} {'>=.5':>5} {'>=.7':>5} {'>=.9':>5} {'thr':>4} {'trunc':>5}  question")
    for q, n, yes, mean, a, b, c, d, thr, trunc in rows:
        print(f"{n:>6} {yes:>5} {mean:>5.2f} {a:>5} {b:>5} {c:>5} {d:>5} {thr:>4.2f} {trunc:>5}  {q}")
    if args.question:
        top = db.execute(
            "SELECT j.p, i.outlet, i.title FROM judgments j JOIN items i ON i.id = j.item_id "
            "WHERE j.question = ? GROUP BY j.item_id ORDER BY j.p DESC LIMIT ?",
            (args.question, args.limit),
        ).fetchall()
        for p, outlet, title in top:
            print(f"  {p:.3f} [{outlet}] {title}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="feedcast",
        description="Condense RSS feeds into a narrated audio episode.",
    )
    parser.add_argument("--version", action="version", version=f"feedcast {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="fetch every feed in feeds.yaml into the local pool")
    p.add_argument("--warm", action="store_true", help="also precompute embeddings and ad answers")
    p.set_defaults(func=cmd_ingest)
    p = sub.add_parser("edition", help="build the default episode from the whole pool")
    p.add_argument("--minutes", type=float, help="target length (default from config.yaml)")
    p.add_argument("--no-ingest", action="store_true", help="skip the fresh ingest first")
    p.set_defaults(func=cmd_edition)
    p = sub.add_parser("ask", help="build an on-demand episode for one or more topics")
    p.add_argument("topics", nargs="+", help="topics, each phrased positively")
    p.add_argument("--minutes", type=float, help="target length (default from config.yaml)")
    p.set_defaults(func=cmd_ask)
    p = sub.add_parser("watch", help="build episodes for topic requests dropped in iCloud")
    p.set_defaults(func=cmd_watch)
    p = sub.add_parser("feeds", help="verify feeds, check URLs, import OPML (see `feeds -h`)",
                       add_help=False)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_feeds)
    p = sub.add_parser("launchd", help="install, remove, or inspect the launchd agents")
    p.add_argument("action", choices=["install", "uninstall", "status"])
    p.set_defaults(func=cmd_launchd)
    p = sub.add_parser("judgments", help="summarize logged Laya answers for threshold tuning")
    p.add_argument("--days", type=float, default=7)
    p.add_argument("--question", help="also list the top-scoring items for this exact question")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_judgments)

    args = parser.parse_args(argv)
    if args.command not in ("feeds", "launchd", "judgments"):
        setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
