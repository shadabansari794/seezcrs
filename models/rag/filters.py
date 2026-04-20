"""Query filter extraction for the RAG recommender."""

from typing import Any, Dict
import re


GENRE_VOCAB = {
    "action", "comedy", "drama", "horror", "sci-fi", "romance",
    "thriller", "western", "animation", "documentary", "family",
    "fantasy", "crime", "adventure", "mystery",
}


def extract_filters(query: str) -> Dict[str, Any]:
    """Extract supported catalog filters from a free-text query."""
    filters: Dict[str, Any] = {}

    lowered = query.lower()
    for token in GENRE_VOCAB:
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            filters["genre"] = token
            break

    return filters
