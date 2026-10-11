"""Hot-reloadable lexicon (config/lexicon.yaml).

The file is re-read whenever its modification time changes, so edits apply to the next job
without restarting the service. An invalid file is rejected and the last good version is kept.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LexiconData:
    symbols: dict[str, str] = field(default_factory=dict)
    abbreviations: dict[str, str] = field(default_factory=dict)
    pronunciations: dict[str, str] = field(default_factory=dict)
    version: float = 0.0


def _string_map(raw: object, section: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"section '{section}' must be a mapping")
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str) or not key:
            raise ValueError(f"section '{section}': keys and values must be non-empty strings ({key!r})")
        out[key] = value
    return out


def parse_lexicon(text: str, version: float = 0.0) -> LexiconData:
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError("lexicon root must be a mapping")
    return LexiconData(
        symbols=_string_map(raw.get("symbols"), "symbols"),
        abbreviations=_string_map(raw.get("abbreviations"), "abbreviations"),
        pronunciations=_string_map(raw.get("pronunciations"), "pronunciations"),
        version=version,
    )


class Lexicon:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data = LexiconData()
        self._mtime: float | None = None
        self.last_error: str | None = None
        self.get()
        if self.last_error is not None:
            raise FileNotFoundError(f"lexicon not found or invalid: {self.path} ({self.last_error})")

    def get(self) -> LexiconData:
        with self._lock:
            try:
                mtime = self.path.stat().st_mtime
            except OSError as exc:
                self.last_error = str(exc)
                return self._data
            if mtime != self._mtime:
                try:
                    self._data = parse_lexicon(self.path.read_text(encoding="utf-8"), version=mtime)
                    self.last_error = None
                    log.info("lexicon loaded: %s", self.path)
                except (yaml.YAMLError, ValueError) as exc:
                    self.last_error = str(exc)
                    log.error("lexicon rejected, keeping previous version: %s", exc)
                self._mtime = mtime
            return self._data


def compile_replacements(mapping: dict[str, str], ignore_case: bool) -> list[tuple[re.Pattern, str]]:
    """Longest keys first, matched as whole words (a key may itself end with '.')."""
    flags = re.IGNORECASE if ignore_case else 0
    rules = []
    for key in sorted(mapping, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(key)}(?!\w)", flags)
        rules.append((pattern, mapping[key]))
    return rules
