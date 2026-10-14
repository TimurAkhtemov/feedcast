# feedcast

Reads a broad pool of RSS feeds, keeps what matters, and turns it into a narrated
audio episode that shows up on your phone. Everything runs on this Mac.

```
feeds ──► ingest ──► judge (Laya) ──► group ──► write (Qwen) ──► voice (Kokoro) ──► iCloud
                          ▲
            topic request from phone (optional)
```

Two modes:

- **Default edition**, built overnight from the whole pool plus any standing topics.
- **On-demand**, built from a topic you send from your phone that morning.

Status: skeleton only. See `CLAUDE.md` for the build brief.

## Layout

```
feedcast/        package (cli.py is the entry point)
feeds.yaml       the feed pool
topics.yaml      standing topics for the default edition
data/            local SQLite and audio scratch (git-ignored)
tests/
```

## Run

```bash
uv sync
uv run feedcast --help
uv run pytest
```
