"""Split normalized text into TTS-sized chunks at sentence boundaries."""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")


def _split_long(piece: str, max_chars: int) -> list[str]:
    """Split an over-long sentence at clause marks, then at word boundaries."""
    if len(piece) <= max_chars:
        return [piece]
    parts: list[str] = []
    for clause in _CLAUSE_END.split(piece):
        if len(clause) <= max_chars:
            parts.append(clause)
            continue
        current = ""
        for word in clause.split():
            if current and len(current) + 1 + len(word) > max_chars:
                parts.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            parts.append(current)
    return _pack(parts, max_chars)


def _pack(pieces: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, max_chars: int = 250) -> list[str]:
    if max_chars < 20:
        raise ValueError("max_chars must be at least 20")
    sentences = [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]
    pieces: list[str] = []
    for sentence in sentences:
        pieces.extend(_split_long(sentence, max_chars))
    return _pack(pieces, max_chars)
