# Local SearXNG

feedcast searches here when an on-demand topic has fewer than 5 matching items in the
pool. It listens on 127.0.0.1:8888 only. `feedcast.search.ensure_up()` starts it if needed.

```bash
docker compose -f searxng/docker-compose.yml up -d     # start (restarts with Docker)
docker compose -f searxng/docker-compose.yml down      # stop
curl -s http://127.0.0.1:8888/healthz                   # check: prints OK
docker logs --tail 50 feedcast-searxng                  # engine errors, CAPTCHAs
```

JSON API (news from the past week):

```bash
curl -s 'http://127.0.0.1:8888/search?q=semiconductor+export+controls&format=json&categories=news&time_range=week&language=en' \
  | python3 -c 'import json,sys; [print(r["title"], r["url"]) for r in json.load(sys.stdin)["results"][:5]]'
```

`unresponsive_engines` in the JSON names blocked engines; tune them in `settings.yml`.
