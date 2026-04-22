"""Simple keyword-based reranker.

Replaces the heavy ML cross-encoder with a fast, dependency-free heuristic.
Scores (query, movie_text) pairs based on keyword overlap and boosts the original
vector search score.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MovieReranker:
    """Simple keyword-overlap reranker for search results."""

    def __init__(self, model_name: str = "keyword-based") -> None:
        logger.info("[Reranker] initialized simple keyword-based reranker")

    @staticmethod
    def _extract_content_for_reranking(movie: Dict[str, Any]) -> str:
        """Build a short passage from a movie dict for scoring."""
        title = movie.get("title") or ""
        year = movie.get("year") or ""
        genres = ", ".join(movie.get("genres") or [])
        overview = movie.get("overview") or movie.get("description") or ""
        if overview == f"Movie: {title}":
            overview = ""
        parts = [f"{title} ({year})" if year else title]
        if genres:
            parts.append(genres)
        if overview:
            parts.append(overview[:500])
        return " | ".join(parts)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Score and re-sort candidates based on keyword overlap.

        Calculates keyword overlap between query and movie text,
        then boosts the original search score (max 20% boost).
        """
        if not candidates:
            return candidates

        try:
            # Simple tokenization: lowercase and split on whitespace
            query_terms = set(query.lower().split())

            for movie in candidates:
                content = self._extract_content_for_reranking(movie)
                content_terms = set(content.lower().split()) if content else set()

                overlap_ratio = 0.0
                if query_terms:
                    overlap_ratio = len(query_terms.intersection(content_terms)) / len(query_terms)

                # Original score from Chromadb logic (1.0 - distance, lower is better distance)
                # Ensure the original score exists, default to 0.5 if not
                original_score = movie.get("distance", 1.0) # often in chroma, distance is returned. 
                # Wait, our RAG recommender might just pass the list. Let's rely on movie['score'] or default to 0
                base_score = movie.get("score", movie.get("rerank_score", 0.0))
                
                # If there's no score coming from Chroma, we just use the overlap itself.
                # Since distance might be in the dict, let's check it. In chromadb generally lower distance is better
                # if there is a 'distance' key, score = 1 / (1 + distance)
                if "distance" in movie and "score" not in movie:
                    base_score = 1.0 / (1.0 + movie["distance"])

                keyword_boost = overlap_ratio * 0.2
                new_score = min(base_score + keyword_boost, 1.0)
                
                movie["score"] = new_score
                movie["rerank_score"] = new_score

            # Re-sort by boosted scores
            reranked = sorted(candidates, key=lambda m: m["rerank_score"], reverse=True)

            logger.info(
                "[Reranker] query=%r  in=%d  out=%d  top_score=%.4f  bottom_score=%.4f",
                query[:60],
                len(candidates),
                min(top_k or len(reranked), len(reranked)),
                reranked[0]["rerank_score"],
                reranked[-1]["rerank_score"],
            )

            if top_k:
                return reranked[:top_k]
            return reranked

        except Exception as e:
            logger.error(f"[Reranker] simple re-ranking failed, using original results: {e}")
            if top_k:
                return candidates[:top_k]
            return candidates
