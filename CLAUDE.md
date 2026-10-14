# feedcast build brief

Build "feedcast": a local pipeline that reads a broad pool of RSS feeds, picks what
matters (by default or for a topic I ask about that morning), and turns it into a
narrated audio episode on my iPhone. Everything runs on this Mac. No cloud model APIs.
Work end to end without checking in between steps.

The skeleton in this folder (pyproject, cli stubs, paths, feeds.yaml, topics.yaml) is
the starting point: build on it. Python 3.12, uv, SQLite, pytest. Commit locally as
you go; don't push unless asked.

## Building blocks

- Laya runs locally through `laya-mlx` (pinned in pyproject.toml). Batch questions
  across many items and cache answers by item text + question (`feedcast/judge.py`).
- Laya model pin: `LAYA_REPO = "aac6fef/laya-mlx"`, revision
  `047678560251f28113ee8f5df4be82102c7bf336`. Do not change it without asking me.
- Ollama with `qwen3.6:35b` and `embeddinggemma`.

## Laya limits to design around (all of them)

- English only. 512-token input cap, question included: judge items on title + first
  paragraph, truncated to fit.
- Handles negation badly: phrase every question positively.
- Confidence isn't calibrated above ~10 options: only ask yes/no questions, one per
  topic. Never ask Laya to choose among feeds or topics.
- Accuracy drops when unrelated text is in the input: strip boilerplate first.

## Pipeline

1. **Ingest** (hourly via launchd): RSS/Atom from `feeds.yaml` (support OPML import).
   Extract full article text (trafilatura). Dedupe by canonical URL and content hash.
   Seed `feeds.yaml` with ~40 broad, reliable feeds (world, US, business, tech,
   science) and verify each URL actually returns a valid feed before keeping it.
2. **Judge** (Laya): per item, "Is this article mainly an advertisement or sponsored
   post?" and, for each requested topic, "Is this article about <topic>?". Batch and
   cache. Keep thresholds in config and log every judgment for tuning.
3. **Group**: embed items with embeddinggemma and cluster near-duplicate coverage so
   ten outlets on one event become one story. Cluster size = importance.
4. **Write** (qwen3.6:35b via Ollama, thinking off; JSON schema output where
   structured): one segment per story, written only from that cluster's article text.
   Written for the ear: short sentences, numbers phrased to be spoken, no URLs, and a
   line naming which outlets reported it. 60-120 seconds per segment, plus a short
   intro and outro. Grounding check in code: every number and proper noun in a segment
   must appear in that cluster's source text, otherwise rewrite once, then drop the
   segment.
5. **Voice**: Kokoro-82M via mlx-audio. Synthesize per paragraph, join, normalize
   loudness, encode MP3 with ffmpeg, and add a chapter marker per segment.
6. **Deliver** to `paths.EPISODES_DIR` (iCloud Drive/Feedcast/episodes): the MP3 plus
   a show-notes .md with segment titles and source links.

## Two modes

- **Default edition** (`feedcast edition`): launchd job at 05:30. Biggest stories
  across the whole pool, plus any standing topics in `topics.yaml`. Target length
  configurable, default 12 minutes.
- **On-demand** (`feedcast ask`, and `feedcast watch` for phone requests): launchd
  WatchPaths on `paths.REQUESTS_DIR`. A .txt file there holds one or more topics. Build
  that episode from the already-ingested pool and aim for it to be ready within a few
  minutes. If fewer than 5 items match a topic, have Qwen write 3 search queries, fetch
  results from a local SearXNG instance (docker compose, include setup), run the same
  Laya filter, and list feeds worth adding in the show notes. Never add feeds
  automatically.

## Failure visibility

If any run fails, still write a short MP3 and .md to the episodes folder saying what
failed, so silence is never mistaken for an episode.

## Done means

- pytest passes, including an end-to-end run on fixture feeds with Laya and Ollama
  stubbed.
- One real default edition and one real on-demand episode generated, with their audio
  length checked.
- README covers: the one-time `sudo pmset repeat wakeorpoweron` command for me to run
  (don't run sudo yourself; the Mac must be plugged in), and how to make an iPhone
  Shortcut "Ask Feedcast" that dictates text and saves it as a .txt into
  iCloud Drive/Feedcast/requests/.
