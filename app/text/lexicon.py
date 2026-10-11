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
    arabic_terms: dict[str, str] = field(default_factory=dict)
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
        arabic_terms=_string_map(raw.get("arabic_terms"), "arabic_terms"),
        version=version,
    )


class Lexicon:
    """config/lexicon.yaml plus an optional local override file (data/lexicon.local.yaml, git-ignored).

    The local file holds decisions made at runtime (e.g. from the Telegram pronunciation check):
    `arabic_terms_off: [latin spelling, ...]` turns curated Arabic terms back to the Latin reading.
    """

    def __init__(self, path: Path, local_path: Path | None = None):
        self.path = Path(path)
        self.local_path = Path(local_path) if local_path else None
        self._lock = threading.Lock()
        self._base = LexiconData()
        self._data = LexiconData()
        self._off: frozenset[str] = frozenset()
        self._local_pron: dict[str, str] = {}
        self._mtimes: tuple[float | None, float | None] = (None, None)
        self.last_error: str | None = None
        self.get()
        if self.last_error is not None:
            raise FileNotFoundError(f"lexicon not found or invalid: {self.path} ({self.last_error})")

    def _mtime(self, path: Path | None) -> float | None:
        try:
            return path.stat().st_mtime if path else None
        except OSError:
            return None

    def get(self) -> LexiconData:
        with self._lock:
            base_m, local_m = self._mtime(self.path), self._mtime(self.local_path)
            if base_m is None:
                self.last_error = f"cannot read {self.path}"
                return self._data
            if (base_m, local_m) == self._mtimes:
                return self._data
            if base_m != self._mtimes[0]:
                try:
                    self._base = parse_lexicon(self.path.read_text(encoding="utf-8"), version=base_m)
                    self.last_error = None
                    log.info("lexicon loaded: %s", self.path)
                except (yaml.YAMLError, ValueError) as exc:
                    self.last_error = str(exc)
                    log.error("lexicon rejected, keeping previous version: %s", exc)
            if local_m != self._mtimes[1]:
                local = self._read_local()
                self._off = frozenset(t.lower() for t in local.get("arabic_terms_off", []))
                self._local_pron = local.get("pronunciations", {})
            self._mtimes = (base_m, local_m)
            terms = {k: v for k, v in self._base.arabic_terms.items() if k.lower() not in self._off}
            pronunciations = {**self._base.pronunciations, **self._local_pron}  # local respellings win
            self._data = LexiconData(self._base.symbols, self._base.abbreviations, pronunciations,
                                     terms, version=max(base_m, local_m or 0))
            return self._data

    def _read_local(self) -> dict:
        if not self.local_path or not self.local_path.exists():
            return {}
        try:
            raw = yaml.safe_load(self.local_path.read_text(encoding="utf-8")) or {}
            off = raw.get("arabic_terms_off", [])
            if not isinstance(off, list) or not all(isinstance(t, str) for t in off):
                raise ValueError("arabic_terms_off must be a list of strings")
            return {"arabic_terms_off": off,
                    "pronunciations": _string_map(raw.get("pronunciations"), "pronunciations")}
        except (yaml.YAMLError, ValueError, AttributeError) as exc:
            self.last_error = f"local lexicon ignored: {exc}"
            log.error("local lexicon rejected: %s", exc)
            return {}

    def all_arabic_terms(self) -> list[dict]:
        """Every curated term with its current state (for the pronunciation check)."""
        self.get()
        respell = {k.lower(): v for k, v in self._local_pron.items()}
        return [{"latin": k, "arabic": v, "enabled": k.lower() not in self._off, "respelling": respell.get(k.lower())}
                for k, v in self._base.arabic_terms.items()]

    def set_arabic_off(self, terms: list[str]) -> None:
        known = {k.lower(): k for k in self._base.arabic_terms}
        self._write_local(off=sorted({known[t.lower()] for t in terms if t.lower() in known}))

    def set_respelling(self, term: str, respelling: str | None) -> None:
        """Local pronunciation override: `term` is read as `respelling` (None removes it)."""
        self.get()
        pron = {k: v for k, v in self._local_pron.items() if k.lower() != term.lower()}
        if respelling:
            pron[term] = respelling
        self._write_local(pronunciations=pron)

    def _write_local(self, off: list[str] | None = None, pronunciations: dict[str, str] | None = None) -> None:
        if not self.local_path:
            raise RuntimeError("no local lexicon path configured")
        current = self._read_local()
        data = {"arabic_terms_off": off if off is not None else current.get("arabic_terms_off", []),
                "pronunciations": pronunciations if pronunciations is not None else current.get("pronunciations", {})}
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.local_path.with_suffix(".tmp")
        tmp.write_text("# Written by the KotakSuara API (Telegram /cek_pelafalan): Latin readings and respellings.\n"
                       + yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        tmp.replace(self.local_path)


def compile_replacements(mapping: dict[str, str], ignore_case: bool) -> list[tuple[re.Pattern, str]]:
    """Longest keys first, matched as whole words (a key may itself end with '.')."""
    flags = re.IGNORECASE if ignore_case else 0
    rules = []
    for key in sorted(mapping, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(key)}(?!\w)", flags)
        rules.append((pattern, mapping[key]))
    return rules
