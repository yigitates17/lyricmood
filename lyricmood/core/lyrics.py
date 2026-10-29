"""Lyrics text processing. No LLM: only facts computable from the text."""

import re
from collections import Counter

from pydantic import BaseModel

from lyricmood.core.text import normalize

_WORD = re.compile(r"\w+(?:'\w+)?", re.UNICODE)


def stanzas(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"\n\s*\n", text.strip()) if s.strip()]


def truncate(text: str, max_chars: int) -> tuple[str, bool]:
    """Cut at a stanza boundary (else a line boundary) to fit `max_chars`."""
    if len(text) <= max_chars:
        return text, False
    kept: list[str] = []
    size = 0
    for stanza in stanzas(text):
        if size + len(stanza) + 2 > max_chars:
            if not kept:  # a single huge stanza: fall back to whole lines
                lines, size = [], 0
                for line in stanza.splitlines():
                    if size + len(line) + 1 > max_chars:
                        break
                    lines.append(line)
                    size += len(line) + 1
                kept.append("\n".join(lines))
            break
        kept.append(stanza)
        size += len(stanza) + 2
    return "\n\n".join(kept), True


class LyricsStats(BaseModel):
    lines: int
    stanzas: int
    words: int
    unique_word_ratio: float
    # Lines sung 2+ times, most frequent first: usually the chorus or hook.
    repeated_lines: list[str]


def stats(text: str) -> LyricsStats:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = [w.casefold() for w in _WORD.findall(text)]
    by_norm = Counter(normalize(line) for line in lines)
    first_seen: dict[str, str] = {}
    for line in lines:
        first_seen.setdefault(normalize(line), line)
    repeated = [first_seen[n] for n, c in by_norm.most_common() if c >= 2 and n][:3]
    return LyricsStats(
        lines=len(lines),
        stanzas=len(stanzas(text)),
        words=len(words),
        unique_word_ratio=round(len(set(words)) / len(words), 2) if words else 0.0,
        repeated_lines=repeated,
    )


def excerpt(text: str, max_chars: int) -> str:
    """A short sample of a song: the opening stanza, plus the chorus (the
    stanza containing the most-repeated line) when it is a different one."""
    parts = stanzas(text)
    if not parts:
        return ""
    chosen = [parts[0]]
    repeated = stats(text).repeated_lines
    if repeated:
        chorus = next((s for s in parts if repeated[0] in s), None)
        if chorus and chorus != parts[0]:
            chosen.append(chorus)
    return truncate("\n\n".join(chosen), max_chars)[0]
