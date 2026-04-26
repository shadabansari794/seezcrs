"""
RAG-based conversational movie recommender.

The class coordinates the RAG flow while helper modules own filter extraction,
retrieval, prompt-history formatting, user profile rendering, and response
parsing.
"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import logging

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from app.config import settings
from app.schemas import Message, MovieRecommendation
from utils.reasoning import log_reasoning_step, clear_reasoning_log
from data.loader import MovieDataLoader
from prompts.templates import PromptTemplates
from utils.vector_store import MovieVectorStore
from utils.reranker import MovieReranker

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

REASON_PING = "\x1e"


class RAGRecommender:
    """RAG-based movie recommender using vector retrieval and LLM generation."""

    def __init__(
        self,
        vector_store: MovieVectorStore,
        movie_loader: MovieDataLoader,
        reranker: Optional[MovieReranker] = None,
        few_shot_examples: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.movie_loader = movie_loader
        self.reranker = reranker
        self.prompt_templates = PromptTemplates(few_shot_examples=few_shot_examples)
        self.client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=settings.request_timeout)
        # Utility tasks (intent, rewrite) need deterministic accuracy + speed.
        self.llm_utility = ChatOpenAI(
            model=settings.llm_model_utility,
            temperature=0,
            api_key=settings.openai_api_key,
            max_tokens=300, # Sufficient for query rewrite
            timeout=settings.request_timeout,
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
            f"max_recs={max_recommendations} query='{query}'"
        )

        from models.query_rewrite import rewrite_query
        rewritten_query = await rewrite_query(self.llm_utility, query, history)

        intent = await classify_intent(self.llm_utility, rewritten_query, history)
        logger.info(f"[RAG] intent={intent!r}")

        profile_text, retrieval_boost = self._build_user_profile_block(user_id)

        if intent != INTENT_RECOMMEND:
            # Chit-chat, clarify, or closing — skip Chroma; the MODE A/C/D branches
            # in the RAG system prompt handle these fine without a movies block.
            retrieved_movies: List[Dict[str, Any]] = []
        else:
            retrieval_query = f"{rewritten_query} {retrieval_boost}".strip() if retrieval_boost else rewritten_query
            
            filters = extract_filters(rewritten_query, loader=self.movie_loader)
            if filters:
                logger.info(f"[RAG] extracted filters: {filters}")

            retrieved_movies = await self._retrieve_relevant_movies(
                retrieval_query,
                top_k=max_recommendations * 4,
                filters=filters,
            )

            # Rerank candidates with cross-encoder
            if self.reranker and retrieved_movies:
                retrieved_movies = self.reranker.rerank(
                    rewritten_query, retrieved_movies, top_k=max_recommendations
                )

        conv_history = self._format_conversation_history(history)
        if profile_text:
            conv_history = f"{profile_text}\n\n{conv_history}"

        chat_prompt = self.prompt_templates.get_rag_prompt()
        messages = chat_prompt.format_messages(
            conversation_history=conv_history or "No previous conversation.",
            movies_context=PromptTemplates.format_movies_context(
                retrieved_movies[:max_recommendations]
            ),
            user_query=query,
        )

        logger.info(
            f"[RAG] calling LLM model={settings.llm_model_main} "
            f"messages={len(messages)} candidates={len(retrieved_movies[:max_recommendations])}"
        )
        response = await self.client.chat.completions.create(
            model=settings.llm_model_main,
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
        logger.info(f"[RAG] [OUTPUT] {content[:250]}...")
        return content

    async def stream_recommendation(
        self,
        query: str,
        history: List[Message],
        max_recommendations: int = 5,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream conversational recommendations for real-time low-latency UI."""
        logger.info(f"[RAG-Stream] start user_id={user_id!r}")
        clear_reasoning_log()
        log_reasoning_step(
            "Processing request",
            data={"input": {"query": query, "user_id": user_id, "history_turns": len(history or [])}},
        )
        yield REASON_PING

        from models.query_rewrite import rewrite_query
        rewritten_query = await rewrite_query(self.llm_utility, query, history)
        log_reasoning_step(
            "Rewrote query",
            data={"input": {"query": query}, "output": {"rewritten_query": rewritten_query}},
        )
        yield REASON_PING

        intent = await classify_intent(self.llm_utility, rewritten_query, history)
        log_reasoning_step(
            "Classified intent",
            data={"input": {"query": rewritten_query}, "output": {"intent": intent}},
        )
        yield REASON_PING

        profile_text, retrieval_boost = self._build_user_profile_block(user_id)

        if intent != INTENT_RECOMMEND:
            log_reasoning_step(
                "Generated reply",
                data={"input": {"query": query, "intent": intent}, "output": {"path": "chat short-circuit (no retrieval)"}},
            )
            yield REASON_PING
            retrieved_movies: List[Dict[str, Any]] = []
        else:
            filters = extract_filters(rewritten_query, loader=self.movie_loader)
            log_reasoning_step(
                "Extracted filters",
                data={
                    "input": {"query": rewritten_query},
                    "output": {
                        "filters": filters,
                        "retrieval_boost": retrieval_boost,
                        "has_profile": bool(profile_text),
                    },
                },
            )
            yield REASON_PING

            retrieval_query = f"{rewritten_query} {retrieval_boost}".strip() if retrieval_boost else rewritten_query
            retrieved_movies = await self._retrieve_relevant_movies(retrieval_query, top_k=max_recommendations * 4, filters=filters)
            log_reasoning_step(
                "Retrieved candidates",
                data={
                    "input": {"query": retrieval_query, "filters": filters, "top_k": max_recommendations * 4},
                    "output": {
                        "count": len(retrieved_movies),
                        "candidates": [
                            {
                                "title": m.get("title"),
                                "year": m.get("year"),
                                "genres": m.get("genres") or [],
                            }
                            for m in retrieved_movies[:10]
                        ],
                    },
                },
            )
            yield REASON_PING

            if self.reranker and retrieved_movies:
                retrieved_movies = self.reranker.rerank(rewritten_query, retrieved_movies, top_k=max_recommendations)
                log_reasoning_step(
                    "Ranked candidates",
                    data={
                        "input": {"query": rewritten_query, "candidates_in": len(retrieved_movies)},
                        "output": {
                            "count": len(retrieved_movies),
                            "ranked": [
                                {
                                    "title": m.get("title"),
                                    "year": m.get("year"),
                                    "genres": m.get("genres") or [],
                                    "score": m.get("rerank_score") or m.get("score"),
                                }
                                for m in retrieved_movies
                            ],
                        },
                    },
                )
                yield REASON_PING

        conv_history = self._format_conversation_history(history)
        if profile_text:
            conv_history = f"{profile_text}\n\n{conv_history}"

        chat_prompt = self.prompt_templates.get_rag_prompt()
        movies_context = PromptTemplates.format_movies_context(retrieved_movies[:max_recommendations])
        messages = chat_prompt.format_messages(
            conversation_history=conv_history or "No previous conversation.",
            movies_context=movies_context,
            user_query=query,
        )

        log_reasoning_step(
            "Generating response",
            data={
                "input": {
                    "query": query,
                    "candidates_in_prompt": len(retrieved_movies[:max_recommendations]),
                    "has_user_profile": bool(profile_text),
                },
                "output": {"status": "streaming tokens..."},
            },
        )
        yield REASON_PING

        response = await self.client.chat.completions.create(
            model=settings.llm_model_main,
            messages=to_openai_messages(messages),
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            stream=True
        )
        full_response_parts: List[str] = []
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                full_response_parts.append(content)
                yield content

        full_response = "".join(full_response_parts)
        preview = full_response if len(full_response) <= 400 else full_response[:400] + "…"
        log_reasoning_step(
            "Generated response",
            data={"input": {"query": query}, "output": {"response_preview": preview, "length": len(full_response)}},
        )
        yield REASON_PING

    def parse_recommendations(self, response_text: str) -> List[MovieRecommendation]:
        """Parse structured recommendations from response text."""
        return parse_recommendations_text(response_text)
