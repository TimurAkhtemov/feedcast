"""launchd agents: hourly ingest, the 05:30 edition, and a watcher on the requests folder.

They're user LaunchAgents (no sudo). launchd starts jobs with a bare PATH, so each
agent gets an explicit one that includes Homebrew (ffmpeg) and Docker (SearXNG).
"""

from __future__ import annotations

import os
import plistlib
import subprocess
from pathlib import Path

from feedcast import paths

AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
LOG_DIR = paths.DATA_DIR / "logs"
PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def executable() -> str:
    return str(paths.PROJECT_ROOT / ".venv" / "bin" / "feedcast")


def agents(edition_hour: int = 5, edition_minute: int = 30) -> dict[str, dict]:
    def base(label: str, *args: str) -> dict:
        name = label.rsplit(".", 1)[-1]
        return {
            "Label": label,
            "ProgramArguments": [executable(), *args],
            "WorkingDirectory": str(paths.PROJECT_ROOT),
            "EnvironmentVariables": {"PATH": PATH, "PYTHONUNBUFFERED": "1",
                                     "HF_HUB_DISABLE_TELEMETRY": "1"},
            "StandardOutPath": str(LOG_DIR / f"{name}.log"),
            "StandardErrorPath": str(LOG_DIR / f"{name}.log"),
            "ProcessType": "Background",
        }

    ingest = base("com.feedcast.ingest", "ingest", "--warm")
    ingest["StartInterval"] = 3600
    edition = base("com.feedcast.edition", "edition")
    edition["StartCalendarInterval"] = {"Hour": edition_hour, "Minute": edition_minute}
    edition["ProcessType"] = "Interactive"  # don't let App Nap slow the morning build
    watch = base("com.feedcast.watch", "watch")
    watch["WatchPaths"] = [str(paths.REQUESTS_DIR)]
    watch["RunAtLoad"] = True
    watch["ThrottleInterval"] = 15
    watch["ProcessType"] = "Interactive"
    return {a["Label"]: a for a in (ingest, edition, watch)}


def plist_path(label: str) -> Path:
    return AGENTS_DIR / f"{label}.plist"


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["/bin/launchctl", *args], capture_output=True, text=True, check=False)


def install() -> list[str]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    paths.REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    paths.EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    domain = f"gui/{os.getuid()}"
    done = []
    for label, spec in agents().items():
        path = plist_path(label)
        _launchctl("bootout", f"{domain}/{label}")
        with open(path, "wb") as fh:
            plistlib.dump(spec, fh)
        result = _launchctl("bootstrap", domain, str(path))
        if result.returncode:
            raise RuntimeError(f"launchctl bootstrap {label} failed: {result.stderr.strip()}")
        done.append(f"{label} -> {path}")
    return done


def uninstall() -> list[str]:
    domain = f"gui/{os.getuid()}"
    done = []
    for label in agents():
        _launchctl("bootout", f"{domain}/{label}")
        path = plist_path(label)
        if path.exists():
            path.unlink()
            done.append(f"removed {path}")
    return done


def status() -> list[str]:
    domain = f"gui/{os.getuid()}"
    lines = []
    for label in agents():
        result = _launchctl("print", f"{domain}/{label}")
        if result.returncode:
            lines.append(f"{label}: not loaded")
            continue
        info = dict(
            line.strip().split(" = ", 1)
            for line in result.stdout.splitlines()
            if " = " in line and line.startswith("\t") and not line.startswith("\t\t")
        )
        lines.append(f"{label}: state={info.get('state', '?')} runs={info.get('runs', '0')} "
                     f"last exit={info.get('last exit code', '-')}")
    return lines
