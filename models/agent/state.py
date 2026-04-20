"""LangGraph state definitions for the agent recommender."""

from typing import Any, Dict, List, TypedDict, Annotated
from operator import add


class AgentState(TypedDict):
    """State carried through the agent graph."""

    messages: Annotated[List, add]
    user_preferences: Dict[str, Any]
    retrieved_movies: List[Dict[str, Any]]
    current_query: str
    recommendations: List[Dict[str, Any]]
