"""Where feedcast reads and writes."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "feedcast.db"
FEEDS_FILE = PROJECT_ROOT / "feeds.yaml"
TOPICS_FILE = PROJECT_ROOT / "topics.yaml"

ICLOUD_DIR = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Feedcast"
EPISODES_DIR = ICLOUD_DIR / "episodes"
REQUESTS_DIR = ICLOUD_DIR / "requests"
