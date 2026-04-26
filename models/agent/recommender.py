"""Structured agent-based conversational movie recommender.

Runs a 4-stage LangGraph pipeline for recommend turns (intent → preferences →
retrieval → ranking → explanation) and short-circuits chit-chat/clarify/closing
turns to a single conversational reply node. See ``models/agent/graph.py``.
"""

from typing import Any, AsyncGenerator, Dict, List, Optional
import logging

from langchain_openai import ChatOpenAI

from app.config import settings
from app.schemas import Message, MovieRecommendation
from utils.reasoning import log_reasoning_step, clear_reasoning_log
from data.loader import MovieDataLoader
from prompts.templates import PromptTemplates
from utils.vector_store import MovieVectorStore

from models.response import strip_leaked_mode_label
from models.agent.graph import create_agent_graph
from models.agent.nodes import build_nodes
from models.agent.state import AgentState
from models.rag.parser import parse_recommendations as parse_recommendations_text

logger = logging.getLogger(__name__)

REASON_PING = "\x1e"


def _preview(text: Any, limit: int = 400) -> str:
    s = str(text or "")
    return s if len(s) <= limit else s[:limit] + "…"


def _candidate_summary(movie: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": movie.get("title"),
        "year": movie.get("year"),
        "genres": movie.get("genres") or [],
        "score": movie.get("score") or movie.get("rerank_score"),
    }


def _summarize_node_update(node_name: str, update: Any, query: str) -> Dict[str, Any]:
    """Build an expandable payload for the reasoning UI from a node update."""
    update = update or {}
    data: Dict[str, Any] = {"input": {"query": query}}

    if node_name == "rewrite_query":
        data["output"] = {"rewritten_query": update.get("rewritten_query")}
    elif node_name == "classify_intent":
        data["output"] = {"intent": update.get("intent")}
    elif node_name == "extract_preferences":
        prefs = update.get("preferences") or {}
        data["output"] = {
            "filters": prefs.get("filters"),
            "retrieval_boost": prefs.get("retrieval_boost"),
            "has_profile": bool(prefs.get("profile_text")),
        }
    elif node_name == "retrieve":
        candidates = update.get("candidates") or []
        data["output"] = {
            "count": len(candidates),
            "candidates": [_candidate_summary(c) for c in candidates[:10]],
        }
    elif node_name == "rank_score":
        ranked = update.get("ranked") or []
        data["output"] = {
            "count": len(ranked),
            "ranked": [_candidate_summary(c) for c in ranked],
        }
    elif node_name in {"explain", "chat_reply", "research"}:
        data["output"] = {"response_preview": _preview(update.get("response_text"))}
    else:
        data["output"] = {k: _preview(v, 200) for k, v in update.items()}
    return data


class AgentRecommender:
    """Structured recommendation pipeline driven by LangGraph."""

    def __init__(
        self,
        vector_store: MovieVectorStore,
        movie_loader: MovieDataLoader,
        few_shot_examples: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.movie_loader = movie_loader
        self.prompt_templates = PromptTemplates(few_shot_examples=few_shot_examples)

        self.llm_main = ChatOpenAI(
            model=settings.llm_model_main,
            temperature=settings.temperature,
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout,
            streaming=True,
        )
        # Utility tasks (intent, rewrite) need deterministic accuracy + speed.
        self.llm_utility = ChatOpenAI(
            model=settings.llm_model_utility,
            temperature=0.1,  # Set slightly above 0 to avoid potential "0.0 invalid" API issues
            api_key=settings.openai_api_key,
            max_tokens=300, # Increased for query rewrite
            timeout=settings.request_timeout,
            streaming=False, # No need to stream utility logic
        )

        self.nodes = build_nodes(
            llm_intent=self.llm_utility,
            llm_main=self.llm_main,
            vector_store=self.vector_store,
            movie_loader=self.movie_loader,
            prompt_templates=self.prompt_templates,
        )
        self.graph = create_agent_graph(self.nodes)

    async def generate_recommendation(
        self,
        query: str,
        history: List[Message],
        max_recommendations: int = 5,
        user_id: Optional[str] = None,
    ) -> str:
        """Drive the structured pipeline and return the final response text."""
        logger.info(
            f"[Agent] start user_id={user_id!r} history_turns={len(history)} "
            f"max_recs={max_recommendations} query={query[:80]!r}"
        )

        initial_state: AgentState = {
            "current_query": query,
            "user_id": user_id,
            "max_recommendations": max_recommendations,
            "history": list(history or []),
            "messages": [],
        }

        result = await self.graph.ainvoke(initial_state)
        intent = result.get("intent", "?")
        response_text = result.get("response_text") or ""
        response_text = strip_leaked_mode_label(response_text)
        
        logger.info(f"[Agent] [OUTPUT] {response_text[:250]}...")
        logger.info(
            f"[Agent] done intent={intent} "
            f"candidates={len(result.get('candidates') or [])} "
            f"ranked={len(result.get('ranked') or [])} "
            f"response_len={len(response_text)}"
        )
        return response_text

    async def stream_recommendation(
        self,
        query: str,
        history: List[Message],
        max_recommendations: int = 5,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream conversational recommendations by driving the compiled graph.

        Uses ``graph.astream(stream_mode=["updates", "messages"])`` so routing
        and LLM token streaming both come from the same graph as the
        non-streaming path. Emits ``REASON_PING`` sentinels when each node
        finishes (for the reasoning UI) and yields token content from terminal
        nodes (``explain``, ``chat_reply``, ``research``) as it arrives.
        """
        logger.info(f"[Agent-Stream] start user_id={user_id!r}")
        clear_reasoning_log()

        initial_state: AgentState = {
            "current_query": query,
            "user_id": user_id,
            "max_recommendations": max_recommendations,
            "history": list(history or []),
            "messages": [],
            "response_text": "",
        }

        node_reason_steps = {
            "rewrite_query": "Rewrote query",
            "classify_intent": "Classified intent",
            "extract_preferences": "Extracted filters",
            "retrieve": "Retrieved candidates",
            "rank_score": "Ranked candidates",
            "research": "Searched the web",
            "chat_reply": "Generated reply",
            "explain": "Generated response",
        }
        terminal_nodes = {"explain", "chat_reply", "research"}

        log_reasoning_step(
            "Processing request",
            data={"input": {"query": query, "user_id": user_id, "history_turns": len(history or [])}},
        )
        yield REASON_PING

        async for mode, event_data in self.graph.astream(
            initial_state,
            stream_mode=["updates", "messages"],
        ):
            if mode == "updates":
                for node_name, update in event_data.items():
                    if node_name not in node_reason_steps:
                        continue
                    data = _summarize_node_update(node_name, update, query)
                    log_reasoning_step(node_reason_steps[node_name], data=data)
                    yield REASON_PING
            elif mode == "messages":
                chunk, metadata = event_data
                if metadata.get("langgraph_node") not in terminal_nodes:
                    continue
                content = getattr(chunk, "content", "")
                if isinstance(content, str) and content:
                    yield content

    def parse_recommendations(self, response_text: str) -> List[MovieRecommendation]:
        """Parse recommendations from response text."""
        return parse_recommendations_text(response_text)
