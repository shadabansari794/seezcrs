"""Query filter extraction for the RAG recommender."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import re

from data.loader import MovieDataLoader


GENRE_VOCAB = {
    "action", "comedy", "drama", "horror", "sci-fi", "romance",
    "thriller", "western", "animation", "documentary", "family",
    "fantasy", "crime", "adventure", "mystery",
}


_DECADE_WORDS = {
    "eighties": (1980, 1989),
    "nineties": (1990, 1999),
    "noughties": (2000, 2009),
    "aughts": (2000, 2009),
    "tens": (2010, 2019),
    "twenties": (2020, 2029),
}

# Matches "80s", "'80s", "80's", "1980s", "1990's", "2000s"
_DECADE_DIGIT_RE = re.compile(r"(?<!\d)(?:(19|20)?(\d{2})['\u2019]?s)(?!\d)")

# Matches a bare 4-digit year in 1900-2099
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _extract_year_filters(lowered: str) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
    """Return (year, year_range). Year wins if both are present."""
    year_match = _YEAR_RE.search(lowered)
    if year_match:
        return int(year_match.group(0)), None

    for word, span in _DECADE_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            return None, span

    digit_match = _DECADE_DIGIT_RE.search(lowered)
    if digit_match:
        century, two = digit_match.groups()
        if century:
            start = int(century + two)
        else:
            two_int = int(two)
            start = 1900 + two_int if two_int >= 30 else 2000 + two_int
        return None, (start, start + 9)

    return None, None


def _longest_name_in(haystack: str, names: set) -> Optional[str]:
    """Return the longest name from ``names`` that appears in ``haystack``."""
    match = None
    for name in names:
        if name and name in haystack and (match is None or len(name) > len(match)):
            match = name
    return match


def extract_filters(query: str, loader: Optional[MovieDataLoader] = None) -> Dict[str, Any]:
    """Extract supported catalog filters from a free-text query.

    ``loader`` is optional: when provided, we also extract director/actor by
    checking for any known full name from the TMDB-enriched catalog.
    """
    filters: Dict[str, Any] = {}
    lowered = query.lower()

    for token in GENRE_VOCAB:
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            filters["genre"] = token
            break

    year, year_range = _extract_year_filters(lowered)
    if year is not None:
        filters["year"] = year
    elif year_range is not None:
        filters["year_range"] = year_range

    if loader is not None:
        director = _longest_name_in(lowered, loader.known_directors)
        actor = _longest_name_in(lowered, loader.known_cast)
        if director:
            filters["director"] = director
            if actor == director:
                actor = None
        if actor:
            filters["actor"] = actor

    return filters
