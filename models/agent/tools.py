import logging
import json
from typing import Optional, List, Dict, Any

from langchain_core.tools import tool
from imdb import Cinemagoer
from langchain_community.tools.tavily_search import TavilySearchResults

from data.loader import MovieDataLoader
from app.config import settings

logger = logging.getLogger(__name__)

# Singleton instances for tools
ia = Cinemagoer()
tavily_tool = None
if settings.tavily_api_key:
    tavily_tool = TavilySearchResults(max_results=3, tavily_api_key=settings.tavily_api_key)

# We need a way to pass the user ID and data loader to the tool. 
# LangChain tools can be created dynamically via a factory.

def get_tools(movie_loader: MovieDataLoader, user_id: Optional[str]) -> List[Any]:
    """Factory to generate bound tools for the current session."""
    
    @tool
    def search_user_history(query: str = "") -> str:
        """
        Search the user's past interaction history, including past conversations and past system recommendations.
        Use this tool to find out what movies the user liked/disliked or to recall past conversation contexts.
        """
        if not user_id:
            return "No historical user ID provided for this session."
            
        history = movie_loader.get_user_history(user_id)
        if not history:
            return f"No history found for user {user_id}."
            
        full_history = history.get("full_history", [])
        if not full_history:
            return f"User liked: {history.get('recent_likes', [])}\nUser disliked: {history.get('recent_dislikes', [])}"
            
        # Format the rich history
        blocks = []
        for item in full_history:
            lines = [f"--- Past Interaction: {item['id']} ---"]
            if item['likes']:
                lines.append(f"Movies liked: {', '.join(item['likes'])}")
            if item['dislikes']:
                lines.append(f"Movies disliked: {', '.join(item['dislikes'])}")
            if item['rec_items']:
                lines.append(f"System recommended: {', '.join(item['rec_items'])}")
            if item['transcript']:
                lines.append(f"Chat Transcript:\n{item['transcript']}")
            blocks.append("\n".join(lines))
            
        return "\n\n".join(blocks)
        
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
            # The tavily tool returns a JSON string or dict list, we just invoke it
            return tavily_tool.invoke({"query": query})
        except Exception as e:
            logger.error(f"[Tool] Web search failed: {e}")
            return f"Failed to perform web search: {e}"

    return [search_user_history, search_imdb, search_web]
