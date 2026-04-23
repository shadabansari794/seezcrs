"""Render the compiled agent graph with the research ReAct subgraph expanded.

The runtime graph treats ``research_node`` as an opaque function — it builds a
ReAct sub-agent internally at call time. For visualization, this script swaps
the opaque node for the compiled ReAct graph itself, so ``xray=True`` can
expand its internals (agent ⇄ tools loop) inside the outer PNG.

Run inside the API container:

    docker exec movie-crs-api python -m scripts.render_graph

Outputs:
- ``docs/graph_outer.png`` — outer pipeline with research subgraph expanded.
- ``docs/graph_react_subgraph.png`` — the ReAct loop in isolation.
"""
from __future__ import annotations

from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import settings
from models.agent.graph import create_agent_graph
from models.agent.tools import get_tools

OUT_DIR = Path("docs")
OUT_DIR.mkdir(exist_ok=True)


async def _noop_async(state):
    return {}


def _noop_sync(state):
    return {}


def main() -> None:
    llm_main = ChatOpenAI(
        model=settings.llm_model_main,
        api_key=settings.openai_api_key,
        streaming=True,
    )
    react_agent = create_react_agent(llm_main, tools=get_tools())

    viz_nodes = {
        "rewrite_query": _noop_async,
        "classify_intent": _noop_async,
        "extract_preferences": _noop_sync,
        "retrieve": _noop_sync,
        "rank_score": _noop_sync,
        "explain": _noop_async,
        "chat_reply": _noop_async,
        "research": react_agent,
    }
    viz_graph = create_agent_graph(viz_nodes)

    outer_path = OUT_DIR / "graph_outer.png"
    outer_png = viz_graph.get_graph(xray=True).draw_mermaid_png()
    outer_path.write_bytes(outer_png)
    print(f"[render] wrote {outer_path} ({len(outer_png)} bytes)")

    react_path = OUT_DIR / "graph_react_subgraph.png"
    react_png = react_agent.get_graph().draw_mermaid_png()
    react_path.write_bytes(react_png)
    print(f"[render] wrote {react_path} ({len(react_png)} bytes)")


if __name__ == "__main__":
    main()
