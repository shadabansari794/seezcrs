"""
Run the scripted test conversations in-process and produce a step-by-step
trace file showing every pipeline stage, its inputs, and its outputs.

For the RAG path we re-run each stage inline (intent → filters → retrieval →
LLM → parse) and record results. For the Agent path we use
``graph.astream()`` which yields the partial state update produced by each
LangGraph node.

Run inside the Docker container (where the vector store + dataset already
live):

    docker exec -it movie-crs-api python scripts/generate_pipeline_trace.py
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.schemas import Message
from data.loader import MovieDataLoader
from models.agent.intent import classify_intent as agent_classify_intent
from models.agent.recommender import AgentRecommender
from models.agent.state import AgentState, INTENT_RECOMMEND
from models.rag.filters import extract_filters
from models.rag.recommender import RAGRecommender
from models.rag.utils import (
    build_user_profile_block,
    format_conversation_history,
    to_openai_messages,
)
from prompts.templates import PromptTemplates
from utils.vector_store import MovieVectorStore


logger = logging.getLogger(__name__)


# ---------- test cases (mirror run_live_api_transcript.py) ----------


@dataclass
class TestCase:
    number: int
    title: str
    model_type: str
    user_id: str
    profile_hints: str
    turns: List[str]


TEST_CASES: List[TestCase] = [
    TestCase(
        number=0,
        title="New user with RAG",
        model_type="rag",
        user_id="demo-rag-new",
        profile_hints="",
        turns=[
            "Hi, I'm in the mood for something cozy. I recently watched White Christmas and really liked the warm music and family feeling.",
            "Could you recommend something with that same light, older-movie comfort?",
            "That sounds fun. Is it more silly or more sentimental?",
            "Nice. Give me one more option, but keep it family-friendly.",
            "Great, thanks. I will try one of those tonight.",
        ],
    ),
    TestCase(
        number=1,
        title="Existing LLM-Redial user with RAG",
        model_type="rag",
        user_id="A2NBOL825B93OM",
        profile_hints=(
            "liked The Bourne Identity, Master and Commander: The Far Side of the World, "
            "The Lord of the Rings: The Fellowship of the Ring; disliked Driven VHS and Darkness Falls VHS."
        ),
        turns=[
            "I want something exciting, but not as ridiculous as Driven. I liked The Bourne Identity because it stayed tense without feeling dumb.",
            "Can you recommend something that fits that?",
            "I already liked Fellowship, so that makes sense. Is it more fantasy than action?",
            "What if I want something stranger and darker, but not cheap horror?",
            "Good. That is enough for now, thanks.",
        ],
    ),
    TestCase(
        number=2,
        title="New user with Agent",
        model_type="agent",
        user_id="demo-agent-new",
        profile_hints="",
        turns=[
            "Hey, I just watched The Screaming Skull and the print quality was awful. The acting felt grade Z too.",
            "I still want something eerie, just better made. Any ideas?",
            "I like slow-burn horror, but I do not want anything too confusing.",
            "Is that closer to ghost story or slasher?",
            "Perfect. Thanks, goodbye.",
        ],
    ),
    TestCase(
        number=3,
        title="Existing LLM-Redial user with Agent",
        model_type="agent",
        user_id="AQP1VPK16SVWM",
        profile_hints=(
            "liked Stir Of Echoes, Quatermass & The Pit VHS, The Incredibles (Mandarin Chinese Edition), "
            "The Three Musketeers/The Four Musketeers VHS; disliked Apollo 18, Primer, Eraserhead, The Colony."
        ),
        turns=[
            "I want something suspenseful, but please avoid anything like Apollo 18 or Primer. I did not enjoy those.",
            "Can you recommend one movie from that angle?",
            "I already liked that one. Give me a different direction but still with mystery.",
            "What if I want something lighter after that?",
            "Great. That gives me a nice mix. Thanks.",
        ],
    ),
]


# ---------- trace buffer helpers ----------


class TraceWriter:
    def __init__(self) -> None:
        self.buf = io.StringIO()

    def write(self, s: str = "") -> None:
        self.buf.write(s + "\n")

    def section(self, title: str) -> None:
        self.write("")
        self.write(title)
        self.write("-" * len(title))

    def step(self, idx: int, name: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        self.write(f"  Step {idx}. {name}")
        self.write(f"    in:  {_render(inputs)}")
        self.write(f"    out: {_render(outputs)}")

    def value(self) -> str:
        return self.buf.getvalue()


def _render(d: Dict[str, Any], width: int = 180) -> str:
    """Render a dict into a compact, single-line trace. Truncate long strings."""
    parts = []
    for k, v in d.items():
        parts.append(f"{k}={_short(v, width)}")
    return ", ".join(parts)


def _short(v: Any, width: int) -> str:
    if isinstance(v, str):
        s = v.replace("\n", " ⏎ ")
        if len(s) > width:
            return repr(s[: width - 3] + "...")
        return repr(s)
    if isinstance(v, list):
        if not v:
            return "[]"
        if all(isinstance(x, str) for x in v):
            joined = ", ".join(v)
            if len(joined) > width:
                joined = joined[: width - 3] + "..."
            return f"[{joined}]"
        if all(isinstance(x, dict) and "title" in x for x in v):
            titles = [x.get("title", "?") for x in v]
            return f"[{len(v)} movies] " + ", ".join(titles[:5]) + ("..." if len(titles) > 5 else "")
        return f"[{len(v)} items]"
    if isinstance(v, dict):
        if not v:
            return "{}"
        keys = list(v.keys())
        return "{" + ", ".join(f"{k}=...") + "}" if len(v) > 4 else "{" + ", ".join(f"{k}={_short(vv, 40)}" for k, vv in v.items()) + "}"
    return repr(v)


# ---------- RAG trace (re-runs each stage with instrumentation) ----------


async def trace_rag_turn(
    rag: RAGRecommender,
    query: str,
    history: List[Message],
    user_id: str,
    max_recs: int,
    writer: TraceWriter,
) -> str:
    """Re-implements RAGRecommender.generate_recommendation with step records."""
    # Step 1: classify intent
    intent = await agent_classify_intent(rag.llm_intent, query, history)
    writer.step(
        1, "classify_intent (LLM, temp=0)",
        inputs={"query": query, "history_turns": len(history)},
        outputs={"intent": intent},
    )

    # Step 2: user profile + retrieval boost
    profile_text, retrieval_boost = build_user_profile_block(rag.movie_loader, user_id)
    writer.step(
        2, "build_user_profile_block",
        inputs={"user_id": user_id},
        outputs={
            "profile_text_len": len(profile_text),
            "retrieval_boost": retrieval_boost,
        },
    )

    if intent != INTENT_RECOMMEND:
        retrieved: List[Dict[str, Any]] = []
        writer.step(
            3, "retrieve",
            inputs={"skipped": True, "reason": f"intent={intent!r}"},
            outputs={"candidates": 0},
        )
    else:
        # Step 3: filters
        filters = extract_filters(query, loader=rag.movie_loader)
        writer.step(
            3, "extract_filters",
            inputs={"query": query},
            outputs={"filters": filters},
        )

        # Step 4: retrieval
        retrieval_query = f"{query} {retrieval_boost}".strip() if retrieval_boost else query
        retrieved = rag.vector_store.search(
            retrieval_query,
            top_k=max_recs * 2,
            **filters,
        )
        writer.step(
            4, "vector_store.search",
            inputs={"query": retrieval_query, "top_k": max_recs * 2, "filters": filters},
            outputs={"candidates": retrieved},
        )

    # Step 5: build prompt
    conv_history = format_conversation_history(history)
    if profile_text:
        conv_history = f"{profile_text}\n\n{conv_history}"
    chat_prompt = rag.prompt_templates.get_rag_chat_prompt()
    movies_context = PromptTemplates.format_movies_context(retrieved[:max_recs])
    messages = chat_prompt.format_messages(
        conversation_history=conv_history or "No previous conversation.",
        movies_context=movies_context,
        user_query=query,
    )
    writer.step(
        5, "build_rag_prompt",
        inputs={
            "history_turns": len(history),
            "profile_attached": bool(profile_text),
            "movies_in_context": len(retrieved[:max_recs]),
        },
        outputs={"messages": len(messages), "movies_context_chars": len(movies_context)},
    )

    # Step 6: LLM call
    from models.response import strip_leaked_mode_label
    response = await rag.client.chat.completions.create(
        model=settings.llm_model,
        messages=to_openai_messages(messages),
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    content = strip_leaked_mode_label(response.choices[0].message.content)
    usage = getattr(response, "usage", None)
    writer.step(
        6, "OpenAI chat.completions",
        inputs={"model": settings.llm_model, "temperature": settings.temperature},
        outputs={
            "tokens_in": getattr(usage, "prompt_tokens", None),
            "tokens_out": getattr(usage, "completion_tokens", None),
            "response": content,
        },
    )

    # Step 7: parse
    parsed = rag.parse_recommendations(content)
    writer.step(
        7, "parse_recommendations",
        inputs={"response_chars": len(content)},
        outputs={"recommendations": [p.title for p in parsed]},
    )

    return content


# ---------- Agent trace (uses graph.astream) ----------


async def trace_agent_turn(
    agent: AgentRecommender,
    query: str,
    history: List[Message],
    user_id: str,
    max_recs: int,
    writer: TraceWriter,
) -> str:
    """Stream the compiled agent graph and record every node's output."""
    initial_state: AgentState = {
        "current_query": query,
        "user_id": user_id,
        "max_recommendations": max_recs,
        "history": list(history or []),
        "messages": [],
    }

    final: Dict[str, Any] = {}
    step_idx = 0
    async for event in agent.graph.astream(initial_state):
        # event is {node_name: partial_state_update}
        for node_name, update in event.items():
            step_idx += 1
            final.update(update)
            inputs, outputs = _summarize_agent_step(node_name, update, final, query, history)
            writer.step(step_idx, node_name, inputs=inputs, outputs=outputs)

    response_text = final.get("response_text", "")

    # Final parse step
    step_idx += 1
    parsed = agent.parse_recommendations(response_text)
    writer.step(
        step_idx, "parse_recommendations",
        inputs={"response_chars": len(response_text)},
        outputs={"recommendations": [p.title for p in parsed]},
    )
    return response_text


def _summarize_agent_step(
    node: str,
    update: Dict[str, Any],
    cumulative: Dict[str, Any],
    query: str,
    history: List[Message],
) -> (Dict[str, Any], Dict[str, Any]):
    if node == "classify_intent":
        return (
            {"query": query, "history_turns": len(history)},
            {"intent": update.get("intent")},
        )
    if node == "extract_preferences":
        prefs = update.get("preferences") or {}
        user_hist = prefs.get("user_history") or {}
        return (
            {"query": query, "user_id_resolved": bool(user_hist)},
            {
                "filters": prefs.get("filters"),
                "likes": user_hist.get("recent_likes") or [],
                "dislikes": user_hist.get("recent_dislikes") or [],
            },
        )
    if node == "retrieve":
        prefs = cumulative.get("preferences") or {}
        return (
            {"query": query, "filters": prefs.get("filters")},
            {"candidates": update.get("candidates") or []},
        )
    if node == "rank_score":
        ranked = update.get("ranked") or []
        top = [(m.get("title"), m.get("score")) for m in ranked[:5]]
        return (
            {"candidates_in": len(cumulative.get("candidates") or [])},
            {"ranked_top5": top, "total_ranked": len(ranked)},
        )
    if node == "explain":
        return (
            {"ranked_in": len(cumulative.get("ranked") or []), "history_turns": len(history)},
            {"response": update.get("response_text", "")},
        )
    if node == "chat_reply":
        return (
            {"intent": cumulative.get("intent"), "history_turns": len(history)},
            {"response": update.get("response_text", "")},
        )
    return ({}, update)


# ---------- driver ----------


async def run_case(
    case: TestCase,
    rag: RAGRecommender,
    agent: AgentRecommender,
    writer: TraceWriter,
) -> None:
    header = f"Test Case {case.number}: {case.title}"
    writer.write("=" * len(header))
    writer.write(header)
    writer.write("=" * len(header))
    writer.write(f"model_type: {case.model_type}")
    writer.write(f"user_id: {case.user_id}")
    if case.profile_hints:
        writer.write(f"profile_hints: {case.profile_hints}")

    history: List[Message] = []
    max_recs = 5

    for i, query in enumerate(case.turns, start=1):
        writer.section(f"Turn {i}")
        writer.write(f"  User: {query}")
        t0 = time.time()
        if case.model_type == "rag":
            response = await trace_rag_turn(rag, query, history, case.user_id, max_recs, writer)
        else:
            response = await trace_agent_turn(agent, query, history, case.user_id, max_recs, writer)
        dt = (time.time() - t0) * 1000
        writer.write(f"  Final response ({dt:.0f} ms): {response}")

        history.append(Message(role="user", content=query))
        history.append(Message(role="assistant", content=response))

    writer.write("")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="pipeline_trace.txt")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING")  # keep the trace file readable

    data_dir = Path("data/llm_redial")
    movie_loader = MovieDataLoader(
        item_map_path=str(data_dir / "item_map.json"),
        conversations_jsonl_path=str(data_dir / "final_data.jsonl"),
        conversations_txt_path=str(data_dir / "Conversation.txt"),
        user_ids_path=str(data_dir / "user_ids.json"),
        tmdb_enriched_path=str(data_dir / "tmdb_enriched_movies.json"),
    )
    movie_loader.load_movies()

    vector_store = MovieVectorStore(
        embedding_model=settings.embedding_model,
        persist_directory=settings.vector_store_path,
    )

    dialogue_snippets = movie_loader.get_conversation_examples(n=3)
    few_shot = PromptTemplates.format_conversation_examples(dialogue_snippets)

    rag = RAGRecommender(vector_store, movie_loader, few_shot_examples=few_shot)
    agent = AgentRecommender(vector_store, movie_loader, few_shot_examples=few_shot)

    writer = TraceWriter()
    writer.write("Movie CRS Pipeline Trace")
    writer.write("Source: scripts/generate_pipeline_trace.py")
    writer.write("")

    for case in TEST_CASES:
        print(f"[case {case.number}] {case.title}", flush=True)
        await run_case(case, rag, agent, writer)

    out_path = Path(args.output)
    out_path.write_text(writer.value(), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
