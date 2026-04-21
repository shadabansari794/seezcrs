"""Heuristic scorer for agent-path candidates.

Signals, combined as a weighted sum:

- similarity: 1 - chroma cosine distance (missing -> 0).
- affinity: +0.6 if the candidate's director overlaps the user's liked
  directors, +0.3 if any genre overlaps.
- negative: -1.0 if the candidate matches a disliked title, -0.5 if its
  director overlaps disliked directors.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _lower(values: Any) -> set:
    if not values:
        return set()
    if isinstance(values, str):
        values = [values]
    return {v.strip().lower() for v in values if isinstance(v, str) and v.strip()}


def score_candidates(
    candidates: List[Dict[str, Any]],
    preferences: Optional[Dict[str, Any]] = None,
    movie_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Score + sort candidates. Returns a new list with a ``score`` field added.

    ``preferences`` may contain ``recent_likes``/``recent_dislikes`` (title
    lists). ``movie_lookup`` maps lowercased title -> movie dict so liked
    titles resolve into director/genre signals. Both optional.
    """
    preferences = preferences or {}
    movie_lookup = movie_lookup or {}

    liked_titles = _lower(preferences.get("recent_likes"))
    disliked_titles = _lower(preferences.get("recent_dislikes"))

    liked_directors: set = set()
    liked_genres: set = set()
    disliked_directors: set = set()
    for title in liked_titles:
        m = movie_lookup.get(title)
        if m:
            liked_directors |= _lower(m.get("director"))
            liked_genres |= _lower(m.get("genres"))
    for title in disliked_titles:
        m = movie_lookup.get(title)
        if m:
            disliked_directors |= _lower(m.get("director"))

    scored: List[Dict[str, Any]] = []
    for movie in candidates:
        dist = movie.get("distance")
        sim = 1.0 - min(max(float(dist), 0.0), 2.0) / 2.0 if dist is not None else 0.0

        directors = _lower(movie.get("director"))
        genres = _lower(movie.get("genres"))

        score = sim
        if directors & liked_directors:
            score += 0.6
        if genres & liked_genres:
            score += 0.3
        if (movie.get("title") or "").strip().lower() in disliked_titles:
            score -= 1.0
        if directors & disliked_directors:
            score -= 0.5

        out = dict(movie)
        out["score"] = round(score, 4)
        scored.append(out)

    scored.sort(key=lambda m: m["score"], reverse=True)
    logger.info(
        "[Agent.rank] scored %d candidates top=%r score=%.3f",
        len(scored),
        scored[0]["title"] if scored else None,
        scored[0]["score"] if scored else 0.0,
    )
    return scored
