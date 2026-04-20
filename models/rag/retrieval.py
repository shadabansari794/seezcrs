"""Async retrieval helpers for the RAG recommender."""

from typing import Any, Dict, List, Optional
import asyncio
import logging

from utils.vector_store import MovieVectorStore

logger = logging.getLogger(__name__)


async def retrieve_relevant_movies(
    vector_store: MovieVectorStore,
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run vector search off the event loop and return matching movies."""
    filters = filters or {}
    loop = asyncio.get_running_loop()
    movies = await loop.run_in_executor(
        None,
        lambda: vector_store.search(query, top_k, **filters),
    )

    logger.info(f"Retrieved {len(movies)} movies for query: {query[:50]}... (filters={filters})")
    return movies
