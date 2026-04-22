import logging
import json
from typing import List, Any

from langchain_core.tools import tool
from imdb import Cinemagoer
from langchain_community.tools.tavily_search import TavilySearchResults

from app.config import settings

logger = logging.getLogger(__name__)

ia = Cinemagoer()
tavily_tool = None
if settings.tavily_api_key:
    tavily_tool = TavilySearchResults(max_results=3, tavily_api_key=settings.tavily_api_key)


@tool
def search_imdb(movie_title: str) -> str:
    """
    Search IMDb for comprehensive movie details such as plot, cast, and directors.
    Use this if a user asks a factual question about a movie.
    """
    try:
        logger.info(f"[Tool] Searching IMDb for: {movie_title}")
        results = ia.search_movie(movie_title)
        if not results:
            return f"No IMDb results found for '{movie_title}'."

        movie = results[0]
        ia.update(movie)

        title = movie.get('title', '')
        year = movie.get('year', '')
        plot = movie.get('plot', ['No plot available'])[0]
        director = ", ".join([d.get('name', '') for d in movie.get('directors', [])])
        cast = ", ".join([c.get('name', '') for c in movie.get('cast', [])[:5]])
        rating = movie.get('rating', 'N/A')

        return f"Title: {title} ({year})\nRating: {rating}/10\nDirector: {director}\nCast: {cast}\nPlot: {plot}"
    except Exception as e:
        logger.error(f"[Tool] IMDb search failed: {e}")
        return f"Failed to retrieve IMDb info: {e}"


@tool
def search_web(query: str) -> str:
    """
    Perform a general web search.
    Use this to find latest news or information not covered by the IMDb tool.
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
    return [search_imdb, search_web]
