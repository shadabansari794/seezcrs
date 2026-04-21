"""Structured agent-based conversational movie recommender.

Runs a 4-stage LangGraph pipeline for recommend turns (intent → preferences →
retrieval → ranking → explanation) and short-circuits chit-chat/clarify/closing
turns to a single conversational reply node. See ``models/agent/graph.py``.
"""

from typing import List, Optional
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
            model=settings.llm_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key,
        )
        # Intent classification needs deterministic routing, not creativity.
        self.llm_intent = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
            max_tokens=8,
        )

        self.nodes = build_nodes(
            llm_intent=self.llm_intent,
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
        logger.info(
            f"[Agent] done intent={intent} "
            f"candidates={len(result.get('candidates') or [])} "
            f"ranked={len(result.get('ranked') or [])} "
            f"response_len={len(response_text)}"
        )
        return strip_leaked_mode_label(response_text)

    def parse_recommendations(self, response_text: str) -> List[MovieRecommendation]:
        """Parse recommendations from response text."""
        return parse_recommendations_text(response_text)
