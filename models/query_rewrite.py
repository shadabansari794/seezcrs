"""Shared LLM query rewriter for retrieval.

Expands follow-ups like "that", "those", "another", "something lighter" into a
standalone movie search query using recent conversation turns. Used by both
the RAG and Agent paths before vector retrieval, so "that sounds fun, is it
more silly or sentimental?" becomes something the embedder can actually match.

Kept deliberately small — one short LLM call, shared system prompt.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas import Message

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You rewrite the USER'S CURRENT MESSAGE into a standalone movie-search query.

- If it references prior turns ("that", "those", "another", "more like", "lighter", "darker"), expand it using the recent conversation into a self-contained phrase.
- If it is already self-contained (describes a mood, genre, actor, director, or title), return it unchanged.
- Preserve specific names (actors, directors, titles, years, decades) verbatim.
- Output ONLY the rewritten query. No preamble, no quotes, no explanation."""


async def rewrite_query(
    llm,
    query: str,
    history: Optional[List[Message]] = None,
) -> str:
    """Rewrite ``query`` into a standalone search query using recent history."""
    q = (query or "").strip()
    if not q:
        return q

    context_lines: List[str] = []
    for msg in (history or [])[-4:]:
        role = "User" if msg.role == "user" else "Assistant"
        context_lines.append(f"{role}: {msg.content}")
    context_block = "\n".join(context_lines) or "(no prior turns)"

    prompt = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Recent conversation:\n{context_block}\n\n"
                f"Current user message: {q}\n\nRewritten search query:"
            )
        ),
    ]

    response = await llm.ainvoke(prompt)
    rewritten = (getattr(response, "content", "") or "").strip().strip("\"'")
    if not rewritten:
        return q

    if rewritten.lower() != q.lower():
        logger.info(f"[QueryRewrite] {q[:80]!r} -> {rewritten[:120]!r}")
    return rewritten
