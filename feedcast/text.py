"""Text cleanup shared by ingest, judging, and writing.

Laya's accuracy drops when unrelated text is in its input, so the lead it judges is the
article's opening with bylines, credits, newsletter pitches, and links stripped out.
"""

from __future__ import annotations

import hashlib
import html
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PARAMS = {
    "fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "ocid", "cmpid", "taid",
    "smid", "smtyp", "cmp", "ref", "ref_src", "referrer", "source", "src", "via", "feature",
    "partner", "cid", "icid", "ito", "itm_source", "itm_medium", "itm_campaign", "at_medium",
    "at_campaign", "at_custom1", "at_custom2", "at_custom3", "at_custom4", "spm", "guccounter",
    "guce_referrer", "guce_referrer_sig", "rss", "mod", "traffic_source", "xtor",
}

BOILERPLATE = [
    r"^(advertisement|sponsored|ad|promoted|supported by)\.?$",
    r"^(by|written by|reporting by|edited by|words by)\s+[\w .,'’-]{2,80}$",
    r"^(photo|image|video|illustration|graphic|credit|source)s?\s*(:|by|credit)",
    r"^(getty images|reuters|ap photo|afp|shutterstock|epa)\b.{0,60}$",
    r"(sign up|subscribe)\b.{0,80}\b(newsletter|our|today|now|free)",
    r"^(read more|related|see also|more:|watch:|listen:|recommended)\b",
    r"^(share|tweet|email|print|copy link|save)( this)?( article| story)?\.?$",
    r"^(updated|published|posted|last updated)\b.{0,60}$",
    r"^\d+\s*(min|minute)s?\s*read$",
    r"^(this (article|story) (was|is|has been)|this content is)\b.{0,120}(updated|corrected|produced|sponsored)",
    r"(cookie|cookies)\b.{0,60}\b(accept|consent|policy|settings)",
    r"^follow (us|@)",
    r"©|copyright \d{4}|all rights reserved",
    r"^(click|tap) here\b",
    r"^(listen to|download) (this|the) (article|story|podcast)",
]
BOILERPLATE_RE = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE]
URL_RE = re.compile(r"(https?://\S+|www\.\S+\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\S+@\S+\.\w+")
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"[ \t ​]+")


def canonical_url(url: str) -> str:
    """One spelling per article, so the same story from two feeds dedupes by URL."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if parts.port and parts.port not in (80, 443):
        host = f"{host}:{parts.port}"
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False) if not _tracking(k)]
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if len(path) > 1:
        path = path.rstrip("/")
    return urlunsplit(("https", host, path, urlencode(sorted(query)), ""))


def strip_tracking(url: str) -> str:
    """The URL as published, minus tracking parameters: what show notes link to."""
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _tracking(k)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def _tracking(key: str) -> bool:
    key = key.lower()
    return key.startswith(("utm_", "at_", "itm_")) or key in TRACKING_PARAMS


def html_to_text(fragment: str) -> str:
    """Feed summaries are HTML fragments; keep paragraph breaks, drop tags."""
    if not fragment:
        return ""
    text = re.sub(r"(?i)<\s*(br|/p|/div|/li|/h\d)\s*/?>", "\n\n", fragment)
    text = html.unescape(TAG_RE.sub(" ", text))
    return normalize_paragraphs(text)


def normalize_paragraphs(text: str) -> str:
    paras = [SPACE_RE.sub(" ", p).strip() for p in re.split(r"\n\s*\n|\n", text or "")]
    return "\n\n".join(p for p in paras if p)


def is_boilerplate(paragraph: str) -> bool:
    p = paragraph.strip()
    if not p:
        return True
    return any(rx.search(p) for rx in BOILERPLATE_RE)


def clean_paragraphs(text: str) -> list[str]:
    out = []
    for para in normalize_paragraphs(text).split("\n\n"):
        para = SPACE_RE.sub(" ", EMAIL_RE.sub("", URL_RE.sub("", para))).strip()
        if para and not is_boilerplate(para):
            out.append(para)
    return out


def lead_text(text: str, *, min_words: int = 40, max_words: int = 220) -> str:
    """The article's opening: the first paragraph, extended until it says something.

    Very short first paragraphs are usually a dateline or a one-line teaser, so keep
    adding paragraphs until there are `min_words` words, capped at `max_words`.
    """
    words: list[str] = []
    for para in clean_paragraphs(text):
        words.extend(para.split())
        if len(words) >= min_words:
            break
    return " ".join(words[:max_words])


def content_hash(title: str, text: str) -> str:
    """Hash of the article body, so syndicated copies on different URLs dedupe too."""
    basis = text if len(text) >= 200 else f"{title} {text}"
    norm = re.sub(r"\W+", " ", basis.lower()).strip()
    return hashlib.sha256(norm.encode()).hexdigest()


def strip_urls(text: str) -> str:
    return SPACE_RE.sub(" ", URL_RE.sub("", text)).strip()


def word_count(text: str) -> int:
    return len(text.split())


def slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "topic"
