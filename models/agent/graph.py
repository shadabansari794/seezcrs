"""LangGraph construction for the structured agent recommender.

Flow:
                        query rewrute
                              |
                       classify_intent
                      /              \\
           (recommend)                (chat | clarify | closing)
                |                                |
       extract_preferences                  chat_reply ──▶ END
                |
             retrieve
                |
            rank_score
                |
             explain ──▶ END
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from langgraph.graph import END, StateGraph

from models.agent.state import (
    INTENT_CHAT,
    INTENT_CLARIFY,
    INTENT_CLOSING,
    INTENT_RECOMMEND,
    AgentState,
)

logger = logging.getLogger(__name__)


def _route_from_intent(state: AgentState) -> str:
    intent = state.get("intent") or INTENT_RECOMMEND
    if intent == INTENT_RECOMMEND:
        return "extract_preferences"
    return "chat_reply"


def create_agent_graph(nodes: Dict[str, Callable]) -> Any:
    """Compile the structured recommendation graph from a node dict.

    ``nodes`` must contain: classify_intent, extract_preferences, retrieve,
    rank_score, explain, chat_reply — as produced by ``build_nodes``.
    """
    required = {
        "rewrite_query",
        "classify_intent",
        "extract_preferences",
        "retrieve",
        "rank_score",
        "explain",
        "chat_reply",
    }
    missing = required - set(nodes)
    if missing:
        raise ValueError(f"create_agent_graph missing nodes: {sorted(missing)}")

    workflow = StateGraph(AgentState)

    workflow.add_node("rewrite_query", nodes["rewrite_query"])
    workflow.add_node("classify_intent", nodes["classify_intent"])
    workflow.add_node("extract_preferences", nodes["extract_preferences"])
    workflow.add_node("retrieve", nodes["retrieve"])
    workflow.add_node("rank_score", nodes["rank_score"])
    workflow.add_node("explain", nodes["explain"])
    workflow.add_node("chat_reply", nodes["chat_reply"])

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        _route_from_intent,
        {
            "extract_preferences": "extract_preferences",
            "chat_reply": "chat_reply",
        },
    )

    workflow.add_edge("extract_preferences", "retrieve")
    workflow.add_edge("retrieve", "rank_score")
    workflow.add_edge("rank_score", "explain")
    workflow.add_edge("explain", END)
    workflow.add_edge("chat_reply", END)

    logger.info(
        "[Agent.graph] compiled nodes=%s entry=classify_intent intents=%s",
        sorted(required),
        [INTENT_CHAT, INTENT_RECOMMEND, INTENT_CLARIFY, INTENT_CLOSING],
    )
    return workflow.compile()
