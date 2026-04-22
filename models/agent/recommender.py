"""Structured agent-based conversational movie recommender.

Runs a 4-stage LangGraph pipeline for recommend turns (intent → preferences →
retrieval → ranking → explanation) and short-circuits chit-chat/clarify/closing
turns to a single conversational reply node. See ``models/agent/graph.py``.
"""

from typing import AsyncGenerator, List, Optional
import logging

from langchain_openai import ChatOpenAI

from app.config import settings
from app.schemas import Message, MovieRecommendation
from data.loader import MovieDataLoader
from prompts.templates import PromptTemplates
from utils.vector_store import MovieVectorStore

from models.response import strip_leaked_mode_label
from models.agent.graph import create_agent_graph
from models.agent.nodes import build_nodes
from models.agent.state import AgentState
from models.rag.parser import parse_recommendations as parse_recommendations_text

logger = logging.getLogger(__name__)


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
        )
        # Utility tasks (intent, rewrite) need deterministic accuracy + speed.
        self.llm_utility = ChatOpenAI(
            model=settings.llm_model_utility,
            temperature=0,
            api_key=settings.openai_api_key,
            max_tokens=300, # Increased for query rewrite
            timeout=settings.request_timeout,
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
        """Stream conversational recommendations from the agent graph."""
        logger.info(f"[Agent-Stream] start user_id={user_id!r}")
        
        initial_state: AgentState = {
            "current_query": query,
            "user_id": user_id,
            "max_recommendations": max_recommendations,
            "history": list(history or []),
            "messages": [],
            "response_text": "",
        }

        # Use astream_events to catch tokens from the internal LLM calls
        async for event in self.graph.astream_events(initial_state, version="v2"):
            # We filter for 'on_chat_model_stream' events
            # We only care about tokens from the final generation nodes (explain or chat_reply)
            # Metadata can help us differentiate if we tagged the LLM calls, but by default 
            # we'll yield tokens from any streaming chat model call in the graph.
            if event["event"] == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content

    def parse_recommendations(self, response_text: str) -> List[MovieRecommendation]:
        """Parse recommendations from response text."""
        return parse_recommendations_text(response_text)
