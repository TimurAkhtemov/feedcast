"""Grounding check: every number and proper noun in a segment must appear in its sources.

The check runs on the written text (Qwen writes numbers as digits; speech conversion
comes later), against exactly the source text the segment was written from. It is a
deliberately simple lexical test. It can't tell whether a claim is true, but it
catches the common fabrication: a figure, name, or place that no source mentions.

Matching rules:
- Numbers match by value, so "45 percent" matches "45%", "3.2 million" matches
  "3,200,000", and "twelve" matches "12". A rounded figure matches when it is the
  source figure rounded to the precision written ("3.2 billion" for "3.24 billion"),
  within 5%.
- A proper noun is a capitalized word mid-sentence, or a sentence-initial capitalized
  word whose lowercase form is not used as an ordinary word. It matches when the word
  appears in the source, ignoring case and possessives; dotted acronyms and a few common
  country aliases ("U.S."/"US"/"United States") are treated as one name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

SYSTEM_WORDS = Path("/usr/share/dict/words")

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
SCALES = {"hundred": 100, "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
ABBREV_SCALES = {"k": 1e3, "m": 1e6, "mn": 1e6, "bn": 1e9, "b": 1e9, "tn": 1e12, "t": 1e12}

DIGITS_RE = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?!\d)"
    r"(?:\s+(thousand|million|billion|trillion)\b|(bn|mn|tn|m|k|b|t)\b)?",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’.&-]*")
SENTENCE_RE = re.compile(r"(?<=[.!?:;])[\"'”’)\]]*\s+|\n+")

ALIASES = [
    {"us", "u.s", "u.s.", "usa", "united states", "america", "american", "americans"},
    {"uk", "u.k", "u.k.", "britain", "british", "united kingdom"},
    {"eu", "e.u.", "european union"},
    {"un", "u.n.", "united nations"},
    {"gov", "gov.", "governor"}, {"sen", "sen.", "senator"}, {"rep.", "representative"},
    {"pres.", "president"}, {"gen.", "general"}, {"lt.", "lieutenant"}, {"sgt.", "sergeant"},
    {"capt.", "captain"}, {"prof.", "professor"}, {"dr.", "doctor"}, {"mt.", "mount"},
    {"st.", "saint"}, {"ft.", "fort"}, {"dept.", "department"}, {"corp.", "corporation"},
    {"atty.", "attorney"}, {"supt.", "superintendent"},
]
ALWAYS_OK = {"i", "feedcast", "ok", "tv", "ai", "ceo", "gdp", "covid"}
# Words that start sentences and are never names, so a sentence-initial capital on
# them says nothing. Anything else sentence-initial counts as a name unless its
# lowercase form appears as an ordinary word in the segment or the sources.
COMMON = set("""
a an the this that these those it its it's he she they we you his her their our your
but and or so yet if then than when while where why how what who whom whose which
in on at for with without after before during since until about against among between
into onto over under through from to by of as also still now today tonight yesterday
tomorrow meanwhile however instead later earlier first second third finally next last
some many most more several both each every other others another all any few much
no not none nothing there here such only even just already again once once
officials experts critics analysts researchers scientists police authorities lawmakers
investors prices shares markets people residents voters leaders workers students
according despite although though because unless whether including amid
monday's tuesday's
""".split())


@lru_cache(maxsize=1)
def _dictionary() -> frozenset[str]:
    """Lowercase entries of the system word list: ordinary English words, not names."""
    try:
        words = SYSTEM_WORDS.read_text().split()
    except OSError:
        return frozenset()
    return frozenset(w for w in words if w[:1].islower())


def is_dictionary_word(word: str) -> bool:
    lw = word.lower().replace("’", "'")
    words = _dictionary()
    if lw in COMMON or lw in words:
        return True
    # The word list has no inflections: try the obvious stems.
    for suffix, repl in (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"),
                         ("ing", ""), ("ing", "e"), ("ly", ""), ("er", ""), ("est", "")):
        if lw.endswith(suffix) and lw[: -len(suffix)] + repl in words:
            return True
    return False


@dataclass(frozen=True)
class NumberMention:
    raw: str
    value: float     # as written, before any scale word
    scale: float     # 1 when no scale word
    decimals: int

    @property
    def scaled(self) -> float:
        return self.value * self.scale


def numbers_in(text: str) -> list[NumberMention]:
    out = []
    for m in DIGITS_RE.finditer(text):
        whole, frac, word, abbr = m.groups()
        value = float(whole.replace(",", "") + (f".{frac}" if frac else ""))
        scale = SCALES.get((word or "").lower()) or ABBREV_SCALES.get((abbr or "").lower(), 1.0)
        out.append(NumberMention(m.group(0), value, scale, len(frac or "")))
    # Mask digit matches so "3.2 million" doesn't also yield a bare "million".
    out.extend(_spelled_numbers(DIGITS_RE.sub(" ", text)))
    return out


def _spelled_numbers(text: str) -> list[NumberMention]:
    words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", text)
    out: list[NumberMention] = []
    run: list[str] = []

    def flush():
        if run and run != ["one"]:
            out.append(NumberMention(" ".join(run), _spelled_value(run), 1.0, 0))
        run.clear()

    for w in words:
        parts = w.lower().split("-")
        if all(p in UNITS or p in TENS or p in SCALES for p in parts):
            run.extend(parts)
        elif w.lower() == "and" and run:
            continue
        else:
            flush()
    flush()
    return out


def _spelled_value(words: list[str]) -> float:
    total, current = 0.0, 0.0
    for w in words:
        if w in UNITS:
            current += UNITS[w]
        elif w in TENS:
            current += TENS[w]
        elif w == "hundred":
            current = (current or 1) * 100
        else:
            total += (current or 1) * SCALES[w]
            current = 0.0
    return total + current


def number_supported(mention: NumberMention, source_numbers: list[NumberMention],
                     source_lower: str) -> bool:
    if mention.raw.lower() in source_lower and not mention.raw[0].isdigit():
        return True  # spelled out exactly as the source spells it
    targets = {mention.value, mention.scaled}
    for s in source_numbers:
        for sv in {s.value, s.scaled}:
            if any(abs(sv - t) <= 1e-9 * max(1.0, abs(t)) for t in targets):
                return True
            for scale in {1.0, mention.scale}:
                x = sv / scale
                if x and round(x, mention.decimals) == mention.value and abs(x - mention.value) <= 0.05 * abs(x):
                    return True
    return False


def proper_nouns_in(text: str, context_lower: str, *, title_case: bool = False) -> list[str]:
    """Capitalized words that look like names.

    A sentence-initial capital says nothing, so such a word only counts when it is not
    an ordinary word: not in the system dictionary and not used in lowercase in
    `context_lower` (segment plus sources). With `title_case` (a headline), every word
    gets that treatment.
    """
    out = []
    for sentence in SENTENCE_RE.split(text):
        for idx, m in enumerate(WORD_RE.finditer(sentence.lstrip("\"'“‘(["))):
            word = _clean_word(m.group(0))
            if not word or not word[0].isupper() or word.lower() in ALWAYS_OK:
                continue
            if all(p in UNITS or p in TENS or p in SCALES for p in word.lower().split("-")):
                continue  # "Twelve" starting a sentence is a number, checked as one
            if idx == 0 or title_case:
                lw = word.lower()
                if is_dictionary_word(word) or re.search(
                    rf"(?<![A-Za-z]){re.escape(lw)}(?![A-Za-z])", context_lower
                ):
                    continue
            out.append(word)
    return out


def looks_title_case(text: str) -> bool:
    words = [w for w in re.findall(r"[A-Za-z][\w'’-]*", text) if len(w) > 3]
    return len(words) >= 2 and sum(w[0].isupper() for w in words) >= 0.6 * len(words)


def _clean_word(word: str) -> str:
    word = re.sub(r"(['’]s|['’])$", "", word)
    if re.fullmatch(r"(?:[A-Za-z]\.){2,}", word):
        return word  # dotted acronym like U.S.
    return word.rstrip(".-&")


def name_supported(word: str, source_lower: str) -> bool:
    lw = word.lower()
    variants = {lw, lw.replace(".", "")}
    for group in ALIASES:
        if lw in group or lw.replace(".", "") in group:
            variants |= group
    for v in variants:
        if v and re.search(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])", source_lower):
            return True
    if "-" in word:  # "Biden-era": check the capitalized parts
        parts = [p for p in word.split("-") if p[:1].isupper()]
        return bool(parts) and all(name_supported(p, source_lower) for p in parts)
    return False


def _has_phrase(phrase: str, text_lower: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text_lower) is not None


def unsupported(text: str, source: str, *, allowed: Iterable[str] = (),
                title_case: bool = False) -> list[str]:
    """Numbers and names in `text` that `source` doesn't support, deduped, in order.
    Pass `title_case=True` for a headline written in Title Case."""
    source_lower = source.lower()
    allowed_lower = " ".join(allowed).lower()
    source_numbers = numbers_in(source)
    problems: list[str] = []
    for n in numbers_in(text):
        if not number_supported(n, source_numbers, source_lower) and n.raw not in problems:
            problems.append(n.raw)
    # Multi-word aliases ("United States" for a source's "U.S.") are matched as phrases
    # before names are split into single words.
    names_text = text
    for group in ALIASES:
        if any(_has_phrase(v, source_lower) for v in group):
            for v in sorted(group, key=len, reverse=True):
                names_text = re.sub(rf"(?<![A-Za-z]){re.escape(v)}(?![A-Za-z])", " ", names_text,
                                    flags=re.IGNORECASE)
    context_lower = f"{text}\n{source}"
    # Only lowercase occurrences show a word is ordinary, so compare against a copy
    # where capitalized words are masked out.
    ordinary = re.sub(r"\b[A-Z][A-Za-z]*", " ", context_lower)
    for word in proper_nouns_in(names_text, ordinary, title_case=title_case):
        if name_supported(word, source_lower) or name_supported(word, allowed_lower):
            continue
        if word not in problems:
            problems.append(word)
    return problems
