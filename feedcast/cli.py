"""Command-line entry point. Each command is a stub until the pipeline is built."""

import argparse
import sys

from feedcast import __version__


def _todo(name: str) -> int:
    print(f"feedcast {name}: not built yet", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="feedcast",
        description="Condense RSS feeds into a narrated audio episode.",
    )
    parser.add_argument("--version", action="version", version=f"feedcast {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="fetch every feed in feeds.yaml into the local pool")
    sub.add_parser("edition", help="build the default episode from the whole pool")
    ask = sub.add_parser("ask", help="build an on-demand episode for one or more topics")
    ask.add_argument("topics", nargs="+", help="topics, each phrased positively")
    sub.add_parser("watch", help="build episodes for topic requests dropped in iCloud")

    args = parser.parse_args(argv)
    return _todo(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
