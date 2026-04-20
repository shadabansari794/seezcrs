"""Tool definitions for the agent recommender."""

from typing import Any, Dict, List, Optional
import logging

from langchain_core.tools import tool

from data.loader import MovieDataLoader
from utils.vector_store import MovieVectorStore

logger = logging.getLogger(__name__)


def create_agent_tools(
    vector_store: MovieVectorStore,
    movie_loader: MovieDataLoader,
) -> List[Any]:
    """Create LangChain tools bound to the current vector store and dataset."""

    @tool
    def search_movies(
        query: str,
        top_k: int = 5,
        genre: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with an optional genre filter.

        Pass `genre` when the user's request mentions one. Leave it as None
        otherwise so pure semantic ranking is used.
        """
        logger.info(f"[Agent.tool] search_movies query={query!r} top_k={top_k} genre={genre!r}")
        movies = vector_store.search(query, top_k=top_k, genre=genre)

        enriched = []
        for movie in movies:
            full_movie = movie_loader.get_movie_by_title(movie.get("title", ""))
            enriched.append(full_movie or movie)

        logger.info(f"[Agent.tool] search_movies -> {len(enriched)} results")
        return enriched

    @tool
    def get_movie_details(title: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific movie title."""
        logger.info(f"[Agent.tool] get_movie_details title={title!r}")
        result = movie_loader.get_movie_by_title(title)
        logger.info(f"[Agent.tool] get_movie_details -> {'hit' if result else 'miss'}")
        return result

    @tool
    def filter_by_genre(genre: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Filter catalog movies by inferred genre."""
        logger.info(f"[Agent.tool] filter_by_genre genre={genre!r} top_k={top_k}")
        filtered = []
        for movie in movie_loader.movies:
            if genre.lower() in [g.lower() for g in movie.get("genres", [])]:
                filtered.append(movie)

        filtered.sort(key=lambda x: x.get("rating") or 0, reverse=True)
        logger.info(f"[Agent.tool] filter_by_genre -> {len(filtered[:top_k])} results")
        return filtered[:top_k]

    @tool
    def get_user_history(user_id: str) -> Dict[str, Any]:
        """Fetch the user's recent liked/disliked movies and historical items."""
        logger.info(f"[Agent.tool] get_user_history user_id={user_id!r}")
        result = movie_loader.get_user_history(user_id)
        logger.info(f"[Agent.tool] get_user_history -> keys={list(result.keys()) if result else []}")
        return result

    return [search_movies, get_movie_details, filter_by_genre, get_user_history]
