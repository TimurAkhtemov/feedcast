"""Written text to spoken text: numbers, symbols, and abbreviations as a newsreader says them.

Qwen writes figures as digits so the grounding check can compare them with the
sources; this turns them into words right before synthesis, so Kokoro never has to guess
how to read "$1.2bn" or "2026".
"""

from __future__ import annotations

import re

from num2words import num2words

from feedcast.text import strip_urls

CURRENCY = {"$": ("dollar", "dollars"), "£": ("pound", "pounds"), "€": ("euro", "euros")}
SCALE_WORDS = {"k": "thousand", "m": "million", "mn": "million", "bn": "billion",
               "b": "billion", "tn": "trillion", "t": "trillion"}
MONTHS = {"jan": "January", "feb": "February", "mar": "March", "apr": "April", "aug": "August",
          "sep": "September", "sept": "September", "oct": "October", "nov": "November",
          "dec": "December"}
ABBREVIATIONS = {
    r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is",
    r"\bvs\.?": "versus",
    r"\betc\.": "and so on",
    r"\bapprox\.": "approximately",
    r"\bNo\. (?=\d)": "number ",
}


def _num(text: str) -> str:
    text = text.replace(",", "")
    value = float(text) if "." in text else int(text)
    return _clean(num2words(value))


def _clean(words: str) -> str:
    return words.replace(",", "").replace(" and ", " ")


def _currency(m: re.Match) -> str:
    symbol, amount, scale = m.group(1), m.group(2), (m.group(3) or "").lower()
    singular, plural = CURRENCY[symbol]
    scale_word = SCALE_WORDS.get(scale, scale)
    unit = singular if amount == "1" and not scale_word else plural
    return " ".join(p for p in (_num(amount), scale_word, unit) if p)


def _time(m: re.Match) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    if minute == 0:
        return _clean(num2words(hour))
    if minute < 10:
        return f"{_clean(num2words(hour))} oh {_clean(num2words(minute))}"
    return f"{_clean(num2words(hour))} {_clean(num2words(minute))}"


def _year(m: re.Match) -> str:
    return _clean(num2words(int(m.group(0)), to="year"))


def spoken(text: str) -> str:
    text = strip_urls(text)
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\b(Jan|Feb|Mar|Apr|Aug|Sept?|Oct|Nov|Dec)\.(?=\s*\d)",
                  lambda m: MONTHS[m.group(1).lower()], text)
    text = re.sub(
        r"([$£€])\s?(\d[\d,]*(?:\.\d+)?)(?:\s?(thousand|million|billion|trillion)\b|(?:(bn|mn|tn|m|k|b|t)\b))?",
        lambda m: _currency(_merge_scale(m)),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s?%", r"\1 percent", text)
    text = re.sub(  # "October 4" is read "October fourth"
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
        r" (\d{1,2})\b(?![,.]?\d)",
        lambda m: f"{m.group(1)} {_clean(num2words(int(m.group(2)), to='ordinal'))}",
        text,
    )
    text = re.sub(r"(?<![\w.])(\d{1,2}):(\d{2})(?!\d)", _time, text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b",
                  lambda m: _clean(num2words(int(m.group(1)), to="ordinal")), text)
    text = re.sub(r"(?<=\d)\s?[–-]\s?(?=\d)", " to ", text)
    text = re.sub(r"(?<![\d$£€,.])\b(1[1-9]\d\d|20\d\d)\b(?![,.]\d)", _year, text)
    text = re.sub(r"(?<![\w.])\d[\d,]*(?:\.\d+)?|(?<![\w.])\d+",
                  lambda m: _num(m.group(0).rstrip(",")) if m.group(0).rstrip(",") else m.group(0),
                  text)
    text = text.replace("&", " and ").replace("%", " percent")
    return re.sub(r"\s{2,}", " ", text).strip()


class _Merged:
    def __init__(self, symbol: str, amount: str, scale: str | None):
        self._groups = (symbol, amount, scale)

    def group(self, i: int):
        return self._groups[i - 1]


def _merge_scale(m: re.Match) -> _Merged:
    return _Merged(m.group(1), m.group(2), m.group(3) or m.group(4))
