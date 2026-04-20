"""Recommendation parsing helpers."""

from typing import List, Optional
import re

from app.schemas import MovieRecommendation

_BOLD_TITLE_RE = re.compile(
    r"^\s*(?:[-*]\s+|\d+[.)]\s+)?\*\*([^*\n]+)\*\*\s*(?:[-:\u2013\u2014]\s*(.*))?$"
)


def _confidence_from_text(response_text: str) -> float:
    """Extract coarse confidence language when present."""
    lower_response = response_text.lower()
    if "high confidence" in lower_response:
        return 0.9
    if "medium confidence" in lower_response:
        return 0.7
    if "low confidence" in lower_response:
        return 0.6
    return 0.8


def _clean_title(title: str) -> Optional[str]:
    """Normalize and reject obvious non-title captures."""
    title = title.strip()
    if len(title) <= 3 or len(title) >= 100:
        return None
    return title


def parse_recommendations(response_text: str) -> List[MovieRecommendation]:
    """
    Parse structured recommendations from a conversational response.

    The prompt reserves markdown-bold titles for recommendation mode. Parsing
    only that shape prevents normal prose, hyphenated words, and internal mode
    labels from becoming fake recommendations.
    """
    recommendations = []
    seen_titles = set()
    confidence = _confidence_from_text(response_text)

    for line in response_text.splitlines():
        match = _BOLD_TITLE_RE.match(line)
        if not match:
            continue

        title = _clean_title(match.group(1))
        if not title:
            continue

        title_key = title.casefold()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        reason = (match.group(2) or "Recommended based on your preferences").strip()
        recommendations.append(MovieRecommendation(
            title=title,
            confidence=confidence,
            reason=reason[:200],
        ))

    return recommendations[:5]
