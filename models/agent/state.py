"""LangGraph state definitions for the agent recommender."""

from typing import Any, Dict, List, Optional, TypedDict, Annotated
from operator import add


INTENT_CHAT = "chat"
INTENT_RECOMMEND = "recommend"
INTENT_CLARIFY = "clarify"
INTENT_CLOSING = "closing"
INTENT_RESEARCH = "research"


class AgentState(TypedDict, total=False):
    """Structured state passed between nodes in the recommendation pipeline."""

    # LLM-facing messages (used by chit-chat and explain nodes)
    messages: Annotated[List, add]

    # Request inputs
    current_query: str
    user_id: Optional[str]
    max_recommendations: int
    history: List[Any]                       # List[app.schemas.Message]

    # Pipeline outputs — one node writes each key
    intent: str                              # classify_intent
    rewritten_query: str                     # extract_preferences
    preferences: Dict[str, Any]              # extract_preferences
    candidates: List[Dict[str, Any]]         # retrieve
    ranked: List[Dict[str, Any]]             # rank_score (adds "score" key)
    response_text: str                       # chat_reply / explain
