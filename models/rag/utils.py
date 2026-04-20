"""Prompt and profile utilities for the RAG recommender."""

from typing import Dict, List, Optional, Tuple
import logging

from langchain_core.messages import BaseMessage

from app.schemas import Message
from data.loader import MovieDataLoader

logger = logging.getLogger(__name__)

LC_TO_OPENAI_ROLE = {"system": "system", "human": "user", "ai": "assistant"}


def to_openai_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Convert LangChain BaseMessage objects into OpenAI chat message dicts."""
    return [
        {"role": LC_TO_OPENAI_ROLE.get(m.type, m.type), "content": m.content}
        for m in messages
    ]


def build_user_profile_block(
    movie_loader: MovieDataLoader,
    user_id: Optional[str],
) -> Tuple[str, str]:
    """
    Build a prompt profile block and a retrieval boost string for a known user.

    Returns empty strings when the user is unknown or no dataset-side history
    exists.
    """
    if not user_id:
        return "", ""

    history = movie_loader.get_user_history(user_id, max_items=5)
    if not history:
        logger.info(f"No history for user_id={user_id!r}; falling back to anonymous path")
        return "", ""

    likes = history.get("recent_likes", [])
    dislikes = history.get("recent_dislikes", [])
    historical = history.get("historical_sample", [])

    lines = ["USER PROFILE:"]
    if likes:
        lines.append(f"- Recently liked: {', '.join(likes)}")
    if dislikes:
        lines.append(f"- Recently disliked: {', '.join(dislikes)}")
    if historical:
        lines.append(f"- Historical interactions: {', '.join(historical)}")

    retrieval_boost = " ".join(likes[:3])
    logger.info(f"Personalizing RAG for user={user_id!r} with {len(likes)} likes, {len(dislikes)} dislikes")
    return "\n".join(lines), retrieval_boost


def format_conversation_history(history: List[Message]) -> str:
    """Render recent server-side conversation history for the prompt."""
    if not history:
        return "No previous conversation."

    formatted = []
    for msg in history[-6:]:
        role = "User" if msg.role == "user" else "Assistant"
        formatted.append(f"{role}: {msg.content}")

    return "\n".join(formatted)
