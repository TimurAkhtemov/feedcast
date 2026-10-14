# feedcast

Reads a broad pool of RSS feeds, keeps what matters, and turns it into a narrated
audio episode that shows up on your phone. Everything runs on this Mac: no cloud model
APIs.

```
feeds.yaml ─► ingest (hourly) ─► SQLite pool
                                   │
             topic request ──────► │  (iCloud/Feedcast/requests/*.txt, or `feedcast ask`)
                                   ▼
        judge: Laya yes/no ─► group: embeddinggemma clusters ─► write: Qwen, grounded
                                   │                                    │
          SearXNG fallback ◄───────┘ (on-demand, < 5 matches)           ▼
                                               voice: Kokoro ─► MP3 + chapters + notes
                                                                        │
                                                       iCloud Drive/Feedcast/episodes
```

Two modes:

- **Default edition** (`feedcast edition`, launchd at 05:30 on Monday, Wednesday and
  Friday): the biggest stories across the whole pool since the previous edition (Monday's
  covers the weekend), plus a story or two for each standing topic in `topics.yaml`,
  minus subjects on its skip list (sports, to start). About 12 minutes. Days, time, and
  length are in `config.yaml` → `edition`.
- **On-demand** (`feedcast ask "topic" ...`, or a `.txt` dropped in
  `iCloud Drive/Feedcast/requests/` from your phone): an episode on just those topics,
  built from the already-ingested pool in a minute or two. If fewer than 5 items match a
  topic, Qwen writes 3 search queries for the local SearXNG, results go through the
  same Laya filter, and the show notes list feeds worth adding. Feeds are never added
  automatically.

## Setup

Requirements, all already on this Mac: Python 3.12 + `uv`, `ffmpeg`, Ollama with
`qwen3.6:35b` and `embeddinggemma`, and Docker Desktop (only for the SearXNG fallback).

```bash
uv sync                                               # Laya, Kokoro (mlx-audio), etc.
echo "SEARXNG_SECRET=$(openssl rand -hex 32)" > searxng/.env   # once: SearXNG's own key
docker compose -f searxng/docker-compose.yml up -d    # local SearXNG on 127.0.0.1:8888
uv run feedcast ingest                                 # first fill of the pool (~30 s)
uv run feedcast edition --no-ingest                    # a first episode
uv run feedcast launchd install                        # hourly ingest, M/W/F edition, request watcher
```

Model weights download on first use into `~/.cache/huggingface`: Laya is pinned to
`aac6fef/laya-mlx` at revision `047678560251f28113ee8f5df4be82102c7bf336`
(`feedcast/judge.py`), and Kokoro is `mlx-community/Kokoro-82M-bf16`.

### One-time: wake the Mac before the morning edition (you run this)

The edition runs at 05:30 on Monday, Wednesday and Friday, but only if the Mac is awake
(if it's asleep, launchd runs it as soon as the Mac next wakes). Schedule a wake at
05:25 on those days. This needs `sudo`, and the Mac must be **plugged in** (scheduled
wakes don't fire on battery):

```bash
sudo pmset repeat wakeorpoweron MWF 05:25:00
pmset -g sched                       # check it took
# sudo pmset repeat cancel           # to undo
```

If you change `edition.days` in `config.yaml`, run `uv run feedcast launchd install` and
the `pmset` command again with the matching days (M T W R F S U).

### One-time: let the launchd jobs use iCloud Drive

macOS asks before a background process touches iCloud Drive. The first time a feedcast
launchd job runs, you'll see a prompt that **python3.12** wants to access files in
iCloud Drive: click **Allow**. Until then the job waits at that prompt, and no other
request runs while it waits. If you missed the prompt, add the interpreter by hand:
System Settings → Privacy & Security → Files & Folders (or Full Disk Access) → add

```
~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3.12
```

(`readlink -f .venv/bin/python3` prints the exact path). Then
`uv run feedcast launchd install` again to restart the jobs. Runs you start from
Terminal already have access.

## Ask from your iPhone: the "Ask Feedcast" Shortcut

1. Open **Shortcuts**, tap **+**, and name the shortcut **Ask Feedcast**.
2. Add **Dictate Text**. Set Language to English and Stop Listening to *After Pause*.
3. Add **Format Date**: Date = *Current Date*, Date Format = *Custom*, `yyyy-MM-dd HHmmss`.
4. Add **Set Name**: set the name of *Dictated Text* to `Ask Feedcast ` followed by
   the *Formatted Date* variable, then `.txt`.
5. Add **Save File**: File = *Renamed Item*. Turn **Ask Where to Save** off. Tap the
   folder and choose **iCloud Drive → Feedcast → requests**. Leave *Overwrite If File
   Exists* off.
6. (Optional) Add **Show Notification**: "Sent to Feedcast. Episode in a few minutes."
7. Run it from the home screen, the Action button, or "Hey Siri, Ask Feedcast", and
   say the topic: *"semiconductor export controls"*. For several topics, pause between
   them or say them as a list: *"AI regulation, the housing market"*. Free-form
   sentences work too; Qwen pulls the topics out. Phrase topics positively ("electric
   cars", not "news that isn't about Tesla"), because Laya handles negation badly.

The Mac's `com.feedcast.watch` agent notices the new file (it asks iCloud to download
it if it's still a placeholder), builds the episode, and moves the request to
`requests/done/` (or `requests/failed/`). The episode appears in
`iCloud Drive/Feedcast/episodes/`. Open it from the Files app, or point a player
that reads iCloud Drive at that folder.

## Standing topics and the skip list

`topics.yaml` holds two lists, both read by every edition:

```yaml
topics:            # each gets up to 2 stories of its own, even small ones
  - semiconductor export controls
skip:              # left out of the top stories
  - sports
```

- A **standing topic** reserves room: up to `edition.topic_stories` of the most relevant
  stories since the last edition, played after the top stories and taking at most half
  the episode. For a one-off focus, ask for an on-demand episode instead.
- The **skip list** removes stories from the top stories. Laya asks "Is this article
  about sports?" of each article, and a story is skipped when its articles' average yes
  is at least `judge.skip_threshold` (0.8). One outlet covering a news story from the
  sports desk won't knock it out. Standing topics and on-demand requests ignore the skip
  list, since those are things you asked for. Every skipped story is listed in the show
  notes under **Skipped**, with its score, so you can tell if it's too aggressive.
- Name the subject itself ("sports", "celebrity gossip"), never "news that isn't
  sports": Laya handles negation badly. Subjects work better than moods ("upsetting
  news"), which Laya judges less consistently.

## What you get

Each episode is two files in `iCloud Drive/Feedcast/episodes/`:

- `2026-10-14-0530-edition.mp3` (or `...-ask-<topic>.mp3`): mono 96 kbps MP3,
  loudness-normalized to -16 LUFS, with an ID3 chapter per segment (Intro, each story,
  Outro).
- the same name with `.md`: show notes with chapter times, every source link per story,
  stories dropped by the grounding check and why, feeds worth adding (on-demand), and
  run stats.

**Failures are never silent.** If a run fails at any step, you still get a short
`...-FAILED.mp3` saying what failed and while doing what, plus a `.md` with the error
and traceback. If Kokoro itself is broken, the notice falls back to macOS `say`, then
to a beep pattern. An on-demand topic with no coverage produces a short "found nothing"
episode instead of nothing.

## How the pieces work

**Ingest** (`feedcast/ingest.py`): conditional GET (ETag / Last-Modified) for every
feed, full article text with trafilatura (feed summary as fallback), dedupe by
canonical URL before fetching and by content hash after (catches syndicated copies).
Bylines, newsletter pitches, photo credits, and links are stripped. At most two
concurrent fetches per site. The hourly job (`ingest --warm`) also precomputes
embeddings and ad judgments so on-demand requests start warm. Items older than 21 days
are pruned.

**Judge** (`feedcast/judge.py`): Laya answers yes/no only:
"Is this article mainly an advertisement or sponsored post?" for every item, and
"Is this article about <topic>?" once per requested topic or skip-list subject. It
never chooses among topics or feeds, because confidence isn't calibrated above about
10 options. Input is
the title plus the cleaned opening paragraph, fitted to Laya's 512-token budget
(question included) with its own tokenizer; truncation is logged. Questions are
batched across items and cached by exact input + question.
Every judgment goes to the `judgments` table with its threshold. For on-demand topics,
an embedding shortlist (400 nearest items) comes first, then Laya decides. Stories are
ranked by Laya's P(yes) plus embedding similarity, because Laya reads compound topics
loosely: "AI tooling for developers" gets a yes from most AI news.

**Group** (`feedcast/group.py`): embeddinggemma vectors, average-link clustering at
cosine 0.75. A story's importance is the number of distinct outlets covering it.

**Write** (`feedcast/write.py`, `feedcast/grounding.py`): Qwen (thinking off, JSON
schema) writes each story only from that cluster's article text: short sentences,
figures as digits with spoken units, 60 to 120 seconds. Code then checks that every
number and proper noun in the segment appears in that source text. Numbers match by
value ("45 percent" = "45%", "3.2 million" = "3,200,000", rounding within 5%), and
names allow common aliases ("U.S."/"United States", "Gov."/"Governor", "Sept."/
"September"). A failing segment gets one rewrite with the unsupported terms listed,
then it's dropped and listed in the show notes. The "reported by ..." line is added in
code after the check. Numbers become words (`feedcast/speak.py`) only right before
synthesis.

**Voice** (`feedcast/voice.py`): Kokoro-82M via mlx-audio, one paragraph at a time,
joined with pauses, two-pass ffmpeg loudnorm, MP3 with ID3 chapters, written to
iCloud atomically, so the phone never syncs a half-written file.

## Known limits

- The grounding check is lexical. It catches a figure, name, or place that no source
  mentions; it can't catch a wrong claim built from words the sources do contain.
- Deliberate blind spots, to keep false alarms down: a lone "one" (usually a pronoun)
  and "first/second/third" aren't checked as numbers; a sentence-initial word that is
  also an ordinary dictionary word ("Apple announced...") isn't treated as a name
  (mid-sentence it is, and all-caps acronyms always are); "-5" is read as 5, because
  hyphens are far more often "COVID-19" or "10-12".
- Laya judges compound topics loosely (see above), and it's English-only. Feeds in
  other languages will be judged badly.
- Sites that don't advertise a feed show up in the show notes as "matching coverage,
  no feed found" rather than as a feed URL.

## Tuning

`config.yaml` holds every threshold and length. To see how Laya is answering:

```bash
uv run feedcast judgments                       # answer distribution per question, last 7 days
uv run feedcast judgments --question "Is this article mainly an advertisement or sponsored post?"
```

Calibration notes from the first real pool (723 items, 2026-10-13):

- **Ad threshold:** at 0.5, 11 items were flagged but only ~3 were deal posts; Laya also
  says yes to articles *about* ads or products. Ads rarely cluster across outlets, so
  0.8 is used.
- **Skip threshold:** "Is this article about sports?" on 997 items was bimodal: 956
  below 0.1, 34 at 0.5 or higher, almost all real sports. Its mistakes sat between 0.5
  and 0.7 (car-finance payouts 0.67, Netflix's World Cup rights 0.52), so 0.8 is used.
  Only 2 multi-outlet sports stories turned up in that pool, and neither would have made
  the edition anyway, so the list rarely changes an episode.
- **Cluster similarity:** at 0.8, a 10-outlet story split in two. At 0.75, clusters
  stayed whole and pure.
- **Speaking rate:** Kokoro `af_heart` reads about 133 words a minute, so
  `words_per_minute` is 130. `data/transcripts/` keeps per-chapter word counts and
  timings from every episode for re-checking.

## Commands

```
feedcast ingest [--warm]           fetch every feed into the pool
feedcast edition [--minutes N] [--no-ingest]
feedcast ask TOPIC [TOPIC ...] [--minutes N]
feedcast watch                     build episodes for waiting phone requests
feedcast feeds verify              check every feed in feeds.yaml (valid, has entries, fresh)
feedcast feeds check URL ...       check candidate feeds before adding them
feedcast feeds import-opml FILE    add verified feeds from an OPML export
feedcast launchd install|uninstall|status
feedcast judgments [--question Q]  Laya answer stats for threshold tuning
```

Logs: `data/logs/feedcast.log` and one log per launchd job. SearXNG: see
`searxng/README.md`.

## Layout

```
feedcast/          package: ingest, judge, group, write, grounding, speak, voice,
                   pipeline (both modes + watcher), feedcheck, search, launchd, cli
feeds.yaml         43 verified feeds, 32 outlets (world, US, business, tech, science)
topics.yaml        standing topics for the default edition
config.yaml        thresholds, lengths, models
searxng/           docker compose + settings for the local search fallback
data/              SQLite pool, logs, transcripts (git-ignored)
tests/             pytest, including an end-to-end run on fixture feeds
```

## Tests

```bash
uv run pytest                                        # everything, models stubbed
FEEDCAST_REAL_TTS=1 uv run pytest tests/test_voice_kokoro.py
FEEDCAST_REAL_LAYA=1 uv run pytest tests/test_judge.py -k real
FEEDCAST_LIVE_SEARXNG=1 uv run pytest tests/test_search_live.py
```

`tests/test_e2e.py` runs both modes end to end on fixture feeds (`tests/fixtures/e2e`)
with Laya, Ollama, Kokoro, and the web stubbed. Ingest, extraction, clustering, the
grounding check, ffmpeg, chapters, notes, failure notices, and the phone-request
watcher are all real.
