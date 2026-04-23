"""Node implementations for the structured agent pipeline.

Each node is a closure over the shared dependencies (LLMs, vector store,
data loader, prompt templates) and returns a partial ``AgentState`` update
so LangGraph can merge it into the running state.

Pipeline shape:

    Query Rewrite-classify_intent ─┬─▶ extract_preferences ─▶ retrieve ─▶ rank_score ─▶ explain ─▶ END
                     └─▶ chat_reply ─▶ END

The chat branch handles chit-chat, clarify, and closing turns so we don't spend
retrieval + ranking tokens on them.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from app.schemas import Message
from data.loader import MovieDataLoader
from models.agent.intent import classify_intent
from models.agent.state import AgentState
from models.agent.tools import get_tools
from models.rag.filters import extract_filters
from models.rag.utils import build_user_profile_block
from models.query_rewrite import rewrite_query
from prompts.templates import PromptTemplates
from utils.vector_store import MovieVectorStore

logger = logging.getLogger(__name__)


def _history_to_string(history: List[Any]) -> str:
    """Format history for raw text templates."""
    lines = []
    for msg in history or []:
        role = "User" if getattr(msg, "role", None) == "user" else "Assistant"
        content = getattr(msg, "content", "") or ""
        lines.append(f"{role}: {content}")
    return "\n".join(lines) or "(No prior turns)"


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
        
        item_parts = [f"MOVIE: {movie.get('title')} ({year}). Genres: {genres}"]
        if director: item_parts.append(f"Director: {director}")
        if cast: item_parts.append(f"Cast: {cast}")
        if overview: item_parts.append(f"Synopsis: {overview[:300]}")
        blocks.append(" | ".join(item_parts))
    
    return "\n---\n".join(blocks) if blocks else "(no candidates)"


def build_nodes(
    llm_intent: Any,
    llm_main: Any,
    vector_store: MovieVectorStore,
    movie_loader: MovieDataLoader,
    prompt_templates: PromptTemplates,
) -> Dict[str, Callable]:
    """Build the five pipeline node callables bound to the given dependencies."""

    async def rewrite_query_node(state: AgentState) -> Dict[str, Any]:
        query = state.get("current_query", "")
        # log_reasoning_step("✍️ Improving query")
        history = state.get("history") or []
        rewritten = await rewrite_query(llm_intent, query, history)
        return {"rewritten_query": rewritten}

    async def classify_intent_node(state: AgentState) -> Dict[str, Any]:
        query = state.get("current_query", "")
        # log_reasoning_step("🧠 Classifying intent", "Determining how to best help you...")
        history = state.get("history") or []
        intent = await classify_intent(llm_intent, query, history)
        return {"intent": intent}

    def extract_preferences_node(state: AgentState) -> Dict[str, Any]:
        rewritten = state.get("rewritten_query") or state.get("current_query", "")
        user_id = state.get("user_id")
        # log_reasoning_step("📋 Extracting filters")

        filters = extract_filters(rewritten, loader=movie_loader)
        profile_text, retrieval_boost = build_user_profile_block(movie_loader, user_id)

        logger.info(
            "[Agent.extract_preferences] filters=%s has_profile=%s boost=%r",
            filters,
            bool(profile_text),
            retrieval_boost,
        )
        return {
            "preferences": {
                "filters": filters,
                "profile_text": profile_text,
                "retrieval_boost": retrieval_boost,
            }
        }

    def retrieve_node(state: AgentState) -> Dict[str, Any]:
        query = state.get("rewritten_query") or state.get("current_query", "")
        prefs = state.get("preferences") or {}
        filters = prefs.get("filters") or {}
        boost = prefs.get("retrieval_boost") or ""
        max_recs = state.get("max_recommendations") or 5
        top_k = max(max_recs * 2, 6)

        retrieval_query = f"{query} {boost}".strip() if boost else query
        candidates = vector_store.search(retrieval_query, top_k=top_k, **filters)
        logger.info(
            "[Agent.retrieve] query=%r boost=%r filters=%s -> %d candidates",
            retrieval_query[:80], boost, filters, len(candidates),
        )
        return {"candidates": candidates}

    def rank_score_node(state: AgentState) -> Dict[str, Any]:
        query = state.get("rewritten_query") or state.get("current_query", "")
        candidates = state.get("candidates") or []

        # Apply simple keyword-overlap reranker exclusively
        from utils.reranker import MovieReranker
        simple_reranker = MovieReranker()
        ranked = simple_reranker.rerank(query, candidates)

        max_recs = state.get("max_recommendations") or 5
        return {"ranked": ranked[: max(max_recs, 3)]}

    async def explain_node(state: AgentState) -> Dict[str, Any]:
        query = state.get("current_query", "")
        ranked = state.get("ranked") or []
        max_recs = state.get("max_recommendations") or 5
        history_str = _history_to_string(state.get("history") or [])
        profile_text = (state.get("preferences") or {}).get("profile_text") or ""
        if profile_text:
            history_str = f"{profile_text}\n\n{history_str}"
        candidates_block = _format_candidate_block(ranked, limit=max(max_recs, 3))
        
        # Build the full prompt text from the template
        prompt_template = prompt_templates.get_explain_prompt()
        full_prompt_text = prompt_template.format(
            history_msgs=history_str,
            query=query,
            candidates_block=candidates_block,
            max_recs=max_recs
        )

        logger.info(
            "[Agent.explain] ranked=%d history_turns=%d",
            len(ranked), len(state.get("history") or []) // 2,
        )

        # Direct LLM call — lets outer astream_events capture token stream.
        response_msg = await llm_main.ainvoke(
            [SystemMessage(content=full_prompt_text), HumanMessage(content=query)]
        )
        content = getattr(response_msg, "content", "") or ""
        return {
            "response_text": content,
            "messages": [response_msg],
        }


    async def chat_reply_node(state: AgentState) -> Dict[str, Any]:
        intent = state.get("intent", "chat")
        query = state.get("current_query", "")
        history_str = _history_to_string(state.get("history") or [])

        prompt_template = prompt_templates.get_chat_reply_prompt()
        full_prompt_text = prompt_template.format(
            history_msgs=history_str,
            intent=intent,
            query=query
        )

        logger.info(
            "[Agent.chat_reply] intent=%s history_turns=%d",
            intent, len(state.get("history") or []) // 2,
        )

        response_msg = await llm_main.ainvoke(
            [SystemMessage(content=full_prompt_text), HumanMessage(content=query)]
        )
        content = getattr(response_msg, "content", "") or ""
        return {
            "response_text": content,
            "messages": [response_msg],
        }


    async def research_node(state: AgentState) -> Dict[str, Any]:
        query = state.get("current_query", "")
        history_str = _history_to_string(state.get("history") or [])

        research_prompt = prompt_templates.get_research_reply_prompt().format(
            history_msgs=history_str,
            query=query,
            today=date.today().isoformat(),
        )

        logger.info(
            "[Agent.research] history_turns=%d tools_available=True",
            len(state.get("history") or []) // 2,
        )

        tools = get_tools()
        agent = create_react_agent(llm_main, tools=tools, prompt=research_prompt)
        response_state = await agent.ainvoke({"messages": [HumanMessage(content=query)]})

        for msg in response_state["messages"]:
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    logger.info(
                        f"[Agent.research] 🛠️ LLM used tool '{tc.get('name')}' with args: {tc.get('args')}"
                    )

        final_message = response_state["messages"][-1]
        content = getattr(final_message, "content", "") or ""
        return {
            "response_text": content,
            "messages": [final_message],
        }

    return {
        "rewrite_query": rewrite_query_node,
        "classify_intent": classify_intent_node,
        "extract_preferences": extract_preferences_node,
        "retrieve": retrieve_node,
        "rank_score": rank_score_node,
        "explain": explain_node,
        "chat_reply": chat_reply_node,
        "research": research_node,
    }
