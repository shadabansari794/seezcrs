import logging
from typing import Any, List

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from app.config import settings
from scripts.test_tmdb_lookup import (
    get_movie_details,
    search_movie,
    to_needed_keys,
)

logger = logging.getLogger(__name__)

tavily_tool = None
if settings.tavily_api_key:
    tavily_tool = TavilySearchResults(max_results=3, tavily_api_key=settings.tavily_api_key)


@tool
def search_tmdb(movie_title: str) -> str:
    """
    Look up *established* movie details from The Movie Database — plot/overview,
    director, cast, release year, genres, and keywords. Best when the user names a
    specific known film and wants production-side facts (who's in it, who directed,
    when it came out, what it's about).

    Does NOT have current box-office numbers, upcoming-release schedules, news, or
    trailers — use `search_web` for those.
    """
    try:
        logger.info(f"[Tool] Searching TMDB for: {movie_title}")
        hit = search_movie(movie_title, year=None)
        details = get_movie_details(hit["id"])
        needed = to_needed_keys(details, cast_limit=5, keyword_limit=10)

        title = needed.get("title") or movie_title
        year = needed.get("year") or "?"
        director = ", ".join(needed.get("director") or []) or "Unknown"
        cast = ", ".join(needed.get("cast") or []) or "Unknown"
        genres = ", ".join(needed.get("genres") or []) or "Unknown"
        keywords = ", ".join(needed.get("keywords") or []) or ""
        overview = needed.get("overview") or "No overview available."

        lines = [
            f"Title: {title} ({year})",
            f"Genres: {genres}",
            f"Director: {director}",
            f"Cast: {cast}",
        ]
        if keywords:
            lines.append(f"Keywords: {keywords}")
        lines.append(f"Overview: {overview}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"[Tool] TMDB search failed: {e}")
        return f"No TMDB results found for '{movie_title}': {e}"


@tool
def search_web(query: str) -> str:
    """
    Live web search for *current* or *time-sensitive* info: upcoming releases,
    what's in theaters now, latest trailers, this week's box office, recent awards,
    breaking film news. Use whenever the question depends on facts newer than
    well-established film history.
    """
    if not tavily_tool:
        return "Web search is not configured (TAVILY_API_KEY missing)."

    try:
        logger.info(f"[Tool] Searching Web for: {query}")
        return tavily_tool.invoke({"query": query})
    except Exception as e:
        logger.error(f"[Tool] Web search failed: {e}")
        return f"Failed to perform web search: {e}"


def get_tools() -> List[Any]:
    """Return the static list of agent tools."""
    return [search_tmdb, search_web]
