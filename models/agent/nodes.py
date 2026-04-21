"""Node implementations for the structured agent pipeline.

Each node is a closure over the shared dependencies (LLMs, vector store,
data loader, prompt templates) and returns a partial ``AgentState`` update
so LangGraph can merge it into the running state.

Pipeline shape:

    classify_intent ─┬─▶ extract_preferences ─▶ retrieve ─▶ rank_score ─▶ explain ─▶ END
                     └─▶ chat_reply ─▶ END

The chat branch handles chit-chat, clarify, and closing turns so we don't spend
retrieval + ranking tokens on them.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from data.loader import MovieDataLoader
from models.agent.intent import classify_intent
from models.agent.ranking import score_candidates
from models.agent.state import AgentState
from models.rag.filters import extract_filters
from prompts.templates import PromptTemplates
from utils.vector_store import MovieVectorStore

logger = logging.getLogger(__name__)


_CHAT_SYSTEM = """You are a friendly conversational movie assistant. Reply naturally in 1-3 short sentences.

- Acknowledge what the user said; share a brief relevant thought if it fits.
- If the user's request is ambiguous (e.g. "recommend something"), ask ONE short clarifying question.
- If the user is closing the conversation (thanks, goodbye), reply with a single warm one-liner.
- Do NOT list or recommend catalog titles. Do NOT use the `**Title** - reason` format.
- No filler like "Absolutely!" or "Great question!"."""


_EXPLAIN_SYSTEM = """You are a friendly conversational movie assistant. The user is asking for a recommendation. You will be given a ranked list of CANDIDATES and must recommend 1-3 of them.

Rules:
- Pick 1-3 titles from CANDIDATES only. Never invent titles.
- Format each pick on its own line: **Title** - one short sentence on the fit.
- No preamble, no restating the request, no confidence labels, no follow-up question.
- Prefer higher-ranked candidates unless a lower one is a clearly better match for the user's stated preferences."""


def _history_to_messages(history: List[Any]) -> List[Any]:
    """Convert app.schemas.Message objects to LangChain BaseMessages."""
    out: List[Any] = []
    for msg in history or []:
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", "") or ""
        if role == "user":
            out.append(HumanMessage(content=content))
        else:
            out.append(AIMessage(content=content))
    return out


def _format_candidate_block(candidates: List[Dict[str, Any]], limit: int) -> str:
    """Render ranked candidates for the explain prompt."""
    blocks = []
    for idx, movie in enumerate(candidates[:limit], start=1):
        year = movie.get("year") or "N/A"
        genres = ", ".join(movie.get("genres") or []) or "N/A"
        overview = movie.get("overview") or movie.get("description") or ""
        if overview == f"Movie: {movie.get('title')}":
            overview = ""
        director = ", ".join(movie.get("director") or [])
        cast = ", ".join((movie.get("cast") or [])[:3])
        score = movie.get("score")

        lines = [f"{idx}. {movie.get('title')} ({year}) [genres: {genres}]"]
        if overview:
            lines.append(f"   overview: {overview[:220]}")
        if director:
            lines.append(f"   director: {director}")
        if cast:
            lines.append(f"   cast: {cast}")
        if score is not None:
            lines.append(f"   score: {score}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(no candidates)"


def build_nodes(
    llm_intent: Any,
    llm_main: Any,
    vector_store: MovieVectorStore,
    movie_loader: MovieDataLoader,
    prompt_templates: PromptTemplates,
) -> Dict[str, Callable]:
    """Build the five pipeline node callables bound to the given dependencies."""

    async def classify_intent_node(state: AgentState) -> Dict[str, Any]:
        query = state.get("current_query", "")
        history = state.get("history") or []
        intent = await classify_intent(llm_intent, query, history)
        return {"intent": intent}

    def extract_preferences_node(state: AgentState) -> Dict[str, Any]:
        query = state.get("current_query", "")
        user_id = state.get("user_id")

        filters = extract_filters(query, loader=movie_loader)
        user_history = movie_loader.get_user_history(user_id) if user_id else {}

        logger.info(
            "[Agent.extract_preferences] filters=%s user_history_keys=%s",
            filters,
            list(user_history.keys()),
        )
        return {
            "preferences": {
                "filters": filters,
                "user_history": user_history,
            }
        }

    def retrieve_node(state: AgentState) -> Dict[str, Any]:
        query = state.get("current_query", "")
        prefs = state.get("preferences") or {}
        filters = prefs.get("filters") or {}
        max_recs = state.get("max_recommendations") or 5
        top_k = max(max_recs * 2, 6)

        candidates = vector_store.search(query, top_k=top_k, **filters)
        logger.info(
            "[Agent.retrieve] query=%r filters=%s -> %d candidates",
            query[:80], filters, len(candidates),
        )
        return {"candidates": candidates}

    def rank_score_node(state: AgentState) -> Dict[str, Any]:
        candidates = state.get("candidates") or []
        prefs = state.get("preferences") or {}
        user_history = prefs.get("user_history") or {}

        lookup = {
            (m.get("title") or "").strip().lower(): m
            for m in movie_loader.movies
            if m.get("title")
        }
        ranked = score_candidates(candidates, user_history, lookup)
        max_recs = state.get("max_recommendations") or 5
        return {"ranked": ranked[: max(max_recs, 3)]}

    async def explain_node(state: AgentState) -> Dict[str, Any]:
        query = state.get("current_query", "")
        ranked = state.get("ranked") or []
        max_recs = state.get("max_recommendations") or 5
        history_msgs = _history_to_messages(state.get("history") or [])

        candidates_block = _format_candidate_block(ranked, limit=max(max_recs, 3))
        user_content = (
            f"USER MESSAGE: {query}\n\n"
            f"CANDIDATES (ranked best-first):\n{candidates_block}\n\n"
            f"Recommend up to {max_recs} titles using the format "
            f"`**Title** - one short sentence on the fit`."
        )

        prompt = [
            SystemMessage(content=_EXPLAIN_SYSTEM),
            *history_msgs,
            HumanMessage(content=user_content),
        ]

        logger.info(
            "[Agent.explain] ranked=%d history_turns=%d",
            len(ranked), len(history_msgs),
        )
        response = await llm_main.ainvoke(prompt)
        content = getattr(response, "content", "") or ""
        return {
            "response_text": content,
            "messages": [response],
        }

    async def chat_reply_node(state: AgentState) -> Dict[str, Any]:
        query = state.get("current_query", "")
        intent = state.get("intent", "chat")
        history_msgs = _history_to_messages(state.get("history") or [])

        user_content = f"USER MESSAGE ({intent}): {query}"
        prompt = [
            SystemMessage(content=_CHAT_SYSTEM),
            *history_msgs,
            HumanMessage(content=user_content),
        ]

        logger.info(
            "[Agent.chat_reply] intent=%s history_turns=%d",
            intent, len(history_msgs),
        )
        response = await llm_main.ainvoke(prompt)
        content = getattr(response, "content", "") or ""
        return {
            "response_text": content,
            "messages": [response],
        }

    return {
        "classify_intent": classify_intent_node,
        "extract_preferences": extract_preferences_node,
        "retrieve": retrieve_node,
        "rank_score": rank_score_node,
        "explain": explain_node,
        "chat_reply": chat_reply_node,
    }
