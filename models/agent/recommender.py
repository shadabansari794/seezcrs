"""Agent-based conversational movie recommender."""

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
from models.agent.messages import build_agent_messages
from models.agent.state import AgentState
from models.agent.tools import create_agent_tools
from models.rag.parser import parse_recommendations as parse_recommendations_text

logger = logging.getLogger(__name__)


class AgentRecommender:
    """Agent-based movie recommender with LangGraph tool calling."""

    def __init__(
        self,
        vector_store: MovieVectorStore,
        movie_loader: MovieDataLoader,
        few_shot_examples: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.movie_loader = movie_loader
        self.prompt_templates = PromptTemplates(few_shot_examples=few_shot_examples)

        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key,
            streaming=True,
        )
        self.tools = self._create_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._create_agent_graph()

    def _create_tools(self) -> List:
        """Compatibility wrapper around the tool factory."""
        return create_agent_tools(self.vector_store, self.movie_loader)

    def _create_agent_graph(self):
        """Compatibility wrapper around the graph factory."""
        return create_agent_graph(self.llm_with_tools, self.tools)

    def _build_messages(
        self,
        query: str,
        history: List[Message],
        user_id: Optional[str],
    ) -> List:
        """Compatibility wrapper around the message builder."""
        return build_agent_messages(self.prompt_templates, query, history, user_id)

    async def generate_recommendation(
        self,
        query: str,
        history: List[Message],
        max_recommendations: int = 5,
        user_id: Optional[str] = None,
    ) -> str:
        """Generate a conversational recommendation using the agent graph."""
        logger.info(
            f"[Agent] start user_id={user_id!r} history_turns={len(history)} "
            f"max_recs={max_recommendations}"
        )
        messages = self._build_messages(query, history, user_id)

        state = AgentState(
            messages=messages,
            user_preferences={},
            retrieved_movies=[],
            current_query=query,
            recommendations=[],
        )

        result = await self.graph.ainvoke(state)
        final_message = result["messages"][-1]
        logger.info(
            f"[Agent] done steps={len(result['messages'])} "
            f"response_len={len(final_message.content or '')}"
        )
        return strip_leaked_mode_label(final_message.content)

    def parse_recommendations(self, response_text: str) -> List[MovieRecommendation]:
        """Parse recommendations from response text."""
        return parse_recommendations_text(response_text)
