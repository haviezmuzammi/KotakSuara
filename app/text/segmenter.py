"""Split a script into speech segments and non-synthesized ayat segments.

Quran verses are never synthesized. `[AYAT]...[/AYAT]` blocks and any Arabic-script span become
`ayat` segments, which the pipeline renders as silence and reports with timestamps so a qari
recording can be inserted later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_ARABIC = r"؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿"
_ARABIC_SPAN = re.compile(
    rf"[{_ARABIC}](?:[{_ARABIC}\s‌‍.,;:!?()\[\]«»\"'\-0-9٠-٩]*[{_ARABIC}])?"
)
_TAG = re.compile(r"\[(/?)AYAT\]", re.IGNORECASE)


class AyatTagError(ValueError):
    pass


@dataclass(frozen=True)
class Segment:
    kind: str  # "speech" | "ayat"
    text: str
    source: str = ""  # for ayat: "tag" | "arabic"


def _split_arabic(text: str) -> list[Segment]:
    out: list[Segment] = []
    pos = 0
    for m in _ARABIC_SPAN.finditer(text):
        if m.start() > pos:
            out.append(Segment("speech", text[pos:m.start()]))
        out.append(Segment("ayat", m.group(0).strip(), "arabic"))
        pos = m.end()
    if pos < len(text):
        out.append(Segment("speech", text[pos:]))
    return out


def split_segments(text: str) -> list[Segment]:
    """Return ordered segments; empty speech segments are dropped. Raises AyatTagError on bad tags."""
    raw: list[Segment] = []
    pos = 0
    open_at: int | None = None
    for m in _TAG.finditer(text):
        closing = m.group(1) == "/"
        if not closing:
            if open_at is not None:
                raise AyatTagError("nested [AYAT] tag")
            raw.extend(_split_arabic(text[pos:m.start()]))
            open_at = m.end()
        else:
            if open_at is None:
                raise AyatTagError("[/AYAT] without matching [AYAT]")
            raw.append(Segment("ayat", text[open_at:m.start()].strip(), "tag"))
            open_at = None
        pos = m.end()
    if open_at is not None:
        raise AyatTagError("[AYAT] block is not closed with [/AYAT]")
    raw.extend(_split_arabic(text[pos:]))

    segments: list[Segment] = []
    for seg in raw:
        if seg.kind == "speech" and not re.search(r"\w", seg.text):
            continue
        segments.append(seg)
    return segments
