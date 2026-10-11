"""Indonesian text normalizer for narration.

Turns written forms into what a narrator would say: numbers, currency, percentages, dates,
Hijri/Gregorian years, clock times, ordinals, Quran references, abbreviations and Islamic terms.
Lexicon-driven parts (symbols, abbreviations, pronunciations) come from config/lexicon.yaml.
"""
from __future__ import annotations

import re
import unicodedata

from app.text.lexicon import LexiconData, compile_replacements
from app.text.numbers_id import decimal_to_words, digits_to_words, number_to_words, ordinal_to_words

MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
          "September", "Oktober", "November", "Desember"]

_NUM = r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?"
_SCALE = r"ribu|juta|miliar|triliun"

_QS_NAMED = re.compile(
    r"\bQS\.?\s*([A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*)?)\s*(?:\[\s*\d+\s*\])?\s*[:,]\s*(?:ayat\s*)?(\d+)"
    r"(?:\s*[-–]\s*(\d+))?"
)
_QS_NUMERIC = re.compile(r"\bQS\.?\s*(\d+)\s*:\s*(\d+)(?:\s*[-–]\s*(\d+))?")
_TIME = re.compile(r"\b(pukul|jam)\s+(\d{1,2})[.:](\d{2})\b", re.IGNORECASE)
_CURRENCY = re.compile(rf"\bRp\.?\s?({_NUM})(?:\s+({_SCALE}))?(?:,-)?")
_PERCENT = re.compile(rf"({_NUM})\s?%")
_DATE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
_ERA = re.compile(r"\b(\d{1,4})\s?(SM|AH|H|M)(?=[\s,.;:!?)]|$)")
_ORDINAL = re.compile(r"\bke-?(\d+)\b")
_RANGE = re.compile(r"\b(\d+)\s?[-–]\s?(\d+)\b")
_ENGLISH_DECIMAL = re.compile(r"\b(\d+)\.(\d{1,2})\b(?!\.\d)")
_NUMBER = re.compile(rf"(?<![\w.,])({_NUM})(?![\w])")
_LEADING_ZERO = re.compile(r"\b0\d{3,}\b")

_ERA_WORDS = {"H": "Hijriah", "AH": "Hijriah", "M": "Masehi", "SM": "sebelum Masehi"}


def _num_words(token: str) -> str:
    integer, _, fraction = token.replace(".", "").partition(",")
    if len(integer) > 15:
        return digits_to_words(integer)
    return decimal_to_words(integer, fraction or None)


def _qs_named(m: re.Match) -> str:
    out = f"Al-Qur'an surah {m.group(1)} ayat {number_to_words(int(m.group(2)))}"
    if m.group(3):
        out += f" sampai {number_to_words(int(m.group(3)))}"
    return out


def _qs_numeric(m: re.Match) -> str:
    out = f"Al-Qur'an surah nomor {number_to_words(int(m.group(1)))} ayat {number_to_words(int(m.group(2)))}"
    if m.group(3):
        out += f" sampai {number_to_words(int(m.group(3)))}"
    return out


def _time(m: re.Match) -> str:
    hour, minute = int(m.group(2)), int(m.group(3))
    out = f"{m.group(1)} {number_to_words(hour)}"
    if minute:
        out += f" {number_to_words(minute)}"
    return out


def _currency(m: re.Match) -> str:
    scale = f" {m.group(2)}" if m.group(2) else ""
    return f"{_num_words(m.group(1))}{scale} rupiah"


def _date(m: re.Match) -> str:
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return m.group(0)
    return f"{number_to_words(day)} {MONTHS[month - 1]} {number_to_words(year)}"


def _era(m: re.Match) -> str:
    era = m.group(2)
    if era == "M" and len(m.group(1)) < 3:
        return m.group(0)  # "5 M" is more likely "5 meter/miliar" than a year
    return f"{number_to_words(int(m.group(1)))} {_ERA_WORDS[era]}"


def _abbreviation(replacement: str, m: re.Match, text: str) -> str:
    """An abbreviation ending in '.' at the very end of the text also ends the sentence."""
    if m.group(0).endswith(".") and not text[m.end():].strip():
        return replacement + "."
    return replacement


def _match_case(replacement: str, original: str) -> str:
    if original[:1].isupper() and replacement[:1].islower():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def apply_symbols(text: str, lex: LexiconData) -> str:
    """Replace symbols (e.g. ﷺ) before segmentation so they are not mistaken for Arabic verses."""
    for symbol in sorted(lex.symbols, key=len, reverse=True):
        text = text.replace(symbol, f" {lex.symbols[symbol]} ")
    return text


def _ensure_sentence_breaks(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip().lstrip("#*->•").strip()
        if not line:
            continue
        if line[-1] not in ".!?…:;,":
            line += "."
        lines.append(line)
    return " ".join(lines)


def _cleanup(text: str) -> str:
    text = re.sub(r"[*_#`~|^<>{}\[\]]", " ", text)
    text = re.sub(r"\s*[()]\s*", ", ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:…])", r"\1", text)
    text = re.sub(r",\s*(?=[,.!?;:])", "", text)
    text = re.sub(r"([.!?])\s*,", r"\1", text)
    text = re.sub(r"^\s*[,.;:]\s*", "", text)
    return text.strip()


def normalize(text: str, lex: LexiconData) -> str:
    """Normalize one speech segment (no Arabic script, no [AYAT] blocks)."""
    text = unicodedata.normalize("NFC", text).replace("’", "'").replace("“", '"').replace("”", '"')
    text = apply_symbols(text, lex)
    text = _ensure_sentence_breaks(text)

    text = _QS_NAMED.sub(_qs_named, text)
    text = _QS_NUMERIC.sub(_qs_numeric, text)
    for pattern, replacement in compile_replacements(lex.abbreviations, ignore_case=False):
        text = pattern.sub(lambda m, r=replacement: _abbreviation(r, m, text), text)

    text = _TIME.sub(_time, text)
    text = _CURRENCY.sub(_currency, text)
    text = _PERCENT.sub(lambda m: f"{_num_words(m.group(1))} persen", text)
    text = _DATE.sub(_date, text)
    text = _ERA.sub(_era, text)
    text = _ORDINAL.sub(lambda m: ordinal_to_words(int(m.group(1))), text)
    text = _RANGE.sub(lambda m: f"{number_to_words(int(m.group(1)))} sampai {number_to_words(int(m.group(2)))}", text)
    text = _LEADING_ZERO.sub(lambda m: digits_to_words(m.group(0)), text)
    text = _ENGLISH_DECIMAL.sub(lambda m: decimal_to_words(m.group(1), m.group(2)), text)
    text = _NUMBER.sub(lambda m: _num_words(m.group(1)), text)

    for pattern, replacement in compile_replacements(lex.pronunciations, ignore_case=True):
        text = pattern.sub(lambda m, r=replacement: _match_case(r, m.group(0)), text)
    return _cleanup(text)
