"""String helpers for matching the same song across services."""

import re
import unicodedata

from rapidfuzz import fuzz

# Suffixes that name a version of a song, not a different song:
# "Nemo - Remastered 2004", "Song (feat. X)", "Song - Radio Edit".
_VERSION_SUFFIX = re.compile(
    r"\s*(?:-\s*|\(|\[)\s*(?:(?:\d{4}\s+)?remaster(?:ed)?|feat\.?|ft\.?|with |radio edit|"
    r"single version|album version|mono|stereo|explicit|clean|bonus track)[^)\]]*[)\]]?\s*$",
    re.IGNORECASE,
)


def clean_title(title: str) -> str:
    """Strip version suffixes so lyrics lookups match the base song."""
    previous = None
    while previous != title:
        previous, title = title, _VERSION_SUFFIX.sub("", title).strip()
    return title or previous


def normalize(text: str) -> str:
    """Casefold and drop punctuation; keeps letters like ä and ö intact."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def same_name(a: str, b: str, threshold: float = 90) -> bool:
    return fuzz.ratio(normalize(a), normalize(b)) >= threshold
