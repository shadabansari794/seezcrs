"""Structured agent-based conversational movie recommender.

Runs a 4-stage LangGraph pipeline for recommend turns (intent → preferences →
retrieval → ranking → explanation) and short-circuits chit-chat/clarify/closing
turns to a single conversational reply node. See ``models/agent/graph.py``.
"""

from typing import AsyncGenerator, List, Optional
import logging

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from app.config import settings
from app.schemas import Message, MovieRecommendation
from utils.reasoning import log_reasoning_step, clear_reasoning_log
from data.loader import MovieDataLoader
from prompts.templates import PromptTemplates
from utils.vector_store import MovieVectorStore

from models.response import strip_leaked_mode_label
from models.agent.graph import create_agent_graph
from models.agent.intent import classify_intent
from models.agent.nodes import build_nodes, _history_to_string, _format_candidate_block
from models.agent.state import AgentState, INTENT_RECOMMEND
from models.query_rewrite import rewrite_query
from models.rag.filters import extract_filters
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

        # Raw OpenAI client for reliable streaming of the final response
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
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
        """Stream conversational recommendations from the agent graph.

        Runs pipeline stages (rewrite → classify → extract → retrieve → rank)
        synchronously via the node closures, then streams the final LLM
        response directly through the OpenAI API for reliable token delivery.
        """
        logger.info(f"[Agent-Stream] start user_id={user_id!r}")
        clear_reasoning_log()

        # Build the running state that accumulates node outputs
        state: AgentState = {
            "current_query": query,
            "user_id": user_id,
            "max_recommendations": max_recommendations,
            "history": list(history or []),
            "messages": [],
            "response_text": "",
        }

        # --- Stage 1: Rewrite query ---
        log_reasoning_step("Rewriting query")
        state.update(await self.nodes["rewrite_query"](state))

        # --- Stage 2: Classify intent ---
        log_reasoning_step("Classifying intent")
        state.update(await self.nodes["classify_intent"](state))

        intent = state.get("intent", "recommend")

        if intent == INTENT_RECOMMEND:
            # --- Stage 3: Extract preferences ---
            log_reasoning_step("Extracting filters")
            state.update(self.nodes["extract_preferences"](state))

            # --- Stage 4: Retrieve ---
            log_reasoning_step("Searching catalog")
            state.update(self.nodes["retrieve"](state))

            # --- Stage 5: Rank ---
            log_reasoning_step("Ranking candidates")
            state.update(self.nodes["rank_score"](state))

            # Build the explain prompt for the final streamed call
            ranked = state.get("ranked") or []
            history_str = _history_to_string(state.get("history") or [])
            candidates_block = _format_candidate_block(
                ranked, limit=max(max_recommendations, 3)
            )
            prompt_template = self.prompt_templates.get_explain_prompt()
            system_prompt = prompt_template.format(
                history_msgs=history_str,
                query=query,
                candidates_block=candidates_block,
                max_recs=max_recommendations,
            )
        else:
            # Chat / clarify / closing — skip retrieval
            log_reasoning_step("Chatting")
            history_str = _history_to_string(state.get("history") or [])
            prompt_template = self.prompt_templates.get_chat_reply_prompt()
            system_prompt = prompt_template.format(
                history_msgs=history_str,
                intent=intent,
                query=query,
            )

        # --- Final stage: Stream the LLM response directly ---
        log_reasoning_step("Generating response")
        response = await self.client.chat.completions.create(
            model=settings.llm_model_main,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def parse_recommendations(self, response_text: str) -> List[MovieRecommendation]:
        """Parse recommendations from response text."""
        return parse_recommendations_text(response_text)
