"""LangGraph construction for the agent recommender."""

from typing import Any, List
import logging

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from models.agent.state import AgentState

logger = logging.getLogger(__name__)


def create_agent_graph(llm_with_tools: Any, tools: List[Any]) -> Any:
    """Create and compile the tool-calling agent graph."""
    workflow = StateGraph(AgentState)

    def agent_node(state: AgentState) -> AgentState:
        """Run the LLM reasoning step."""
        messages = state["messages"]
        logger.info(f"[Agent] node=agent messages_in={len(messages)}")
        response = llm_with_tools.invoke(messages)
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            names = [tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?") for tc in tool_calls]
            logger.info(f"[Agent] LLM requested {len(tool_calls)} tool call(s): {names}")
        else:
            logger.info(f"[Agent] LLM returned final content len={len(getattr(response, 'content', '') or '')}")
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """Route to tools when the last LLM message requested tool calls."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()
