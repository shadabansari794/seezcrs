"""
RAG-based conversational movie recommender.

The class coordinates the RAG flow while helper modules own filter extraction,
retrieval, prompt-history formatting, user profile rendering, and response
parsing.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from app.config import settings
from app.schemas import Message, MovieRecommendation
from data.loader import MovieDataLoader
from prompts.templates import PromptTemplates
from utils.vector_store import MovieVectorStore

from models.agent.intent import classify_intent
from models.agent.state import INTENT_RECOMMEND
from models.response import strip_leaked_mode_label
from models.rag.filters import extract_filters
from models.rag.parser import parse_recommendations as parse_recommendations_text
from models.rag.retrieval import retrieve_relevant_movies
from models.rag.utils import (
    build_user_profile_block,
    format_conversation_history,
    to_openai_messages,
)

logger = logging.getLogger(__name__)


class RAGRecommender:
    """RAG-based movie recommender using vector retrieval and LLM generation."""

    def __init__(
        self,
        vector_store: MovieVectorStore,
        movie_loader: MovieDataLoader,
        few_shot_examples: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.movie_loader = movie_loader
        self.prompt_templates = PromptTemplates(few_shot_examples=few_shot_examples)
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        # Cheap, deterministic classifier so chit-chat turns skip retrieval.
        self.llm_intent = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
            max_tokens=8,
        )

    def _build_user_profile_block(self, user_id: Optional[str]) -> Tuple[str, str]:
        """Compatibility wrapper around the profile utility."""
        return build_user_profile_block(self.movie_loader, user_id)

    def _format_conversation_history(self, history: List[Message]) -> str:
        """Compatibility wrapper around the history formatter."""
        return format_conversation_history(history)

    async def _retrieve_relevant_movies(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Compatibility wrapper around the retrieval helper."""
        return await retrieve_relevant_movies(self.vector_store, query, top_k, filters)

    async def generate_recommendation(
        self,
        query: str,
        history: List[Message],
        max_recommendations: int = 5,
        user_id: Optional[str] = None,
    ) -> str:
        """Generate a conversational recommendation using the RAG pipeline."""
        logger.info(
            f"[RAG] start user_id={user_id!r} history_turns={len(history)} "
            f"max_recs={max_recommendations}"
        )

        intent = await classify_intent(self.llm_intent, query, history)
        logger.info(f"[RAG] intent={intent!r}")

        profile_text, retrieval_boost = self._build_user_profile_block(user_id)

        if intent != INTENT_RECOMMEND:
            # Chit-chat, clarify, or closing — skip Chroma; the MODE A/C/D branches
            # in the RAG system prompt handle these fine without a movies block.
            retrieved_movies: List[Dict[str, Any]] = []
        else:
            retrieval_query = f"{query} {retrieval_boost}".strip() if retrieval_boost else query
            filters = extract_filters(query, loader=self.movie_loader)
            if filters:
                logger.info(f"[RAG] extracted filters: {filters}")

            retrieved_movies = await self._retrieve_relevant_movies(
                retrieval_query,
                top_k=max_recommendations * 2,
                filters=filters,
            )

        conv_history = self._format_conversation_history(history)
        if profile_text:
            conv_history = f"{profile_text}\n\n{conv_history}"

        chat_prompt = self.prompt_templates.get_rag_chat_prompt()
        messages = chat_prompt.format_messages(
            conversation_history=conv_history or "No previous conversation.",
            movies_context=PromptTemplates.format_movies_context(
                retrieved_movies[:max_recommendations]
            ),
            user_query=query,
        )

        logger.info(
            f"[RAG] calling LLM model={settings.llm_model} "
            f"messages={len(messages)} candidates={len(retrieved_movies[:max_recommendations])}"
        )
        response = await self.client.chat.completions.create(
            model=settings.llm_model,
            messages=to_openai_messages(messages),
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        content = strip_leaked_mode_label(response.choices[0].message.content)
        usage = getattr(response, "usage", None)
        logger.info(
            f"[RAG] LLM done tokens_in={getattr(usage, 'prompt_tokens', '?')} "
            f"tokens_out={getattr(usage, 'completion_tokens', '?')} "
            f"response_len={len(content or '')}"
        )
        return content

    def parse_recommendations(self, response_text: str) -> List[MovieRecommendation]:
        """Parse structured recommendations from response text."""
        return parse_recommendations_text(response_text)
