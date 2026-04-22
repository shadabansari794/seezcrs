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

from app.schemas import Message
from prompts.templates import PromptTemplates

logger = logging.getLogger(__name__)


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

    chain = PromptTemplates().get_query_rewrite_prompt() | llm
    response = await chain.ainvoke({"context_block": context_block, "query": q})
    rewritten = (getattr(response, "content", "") or "").strip().strip("\"'")
    if not rewritten:
        return q

    if rewritten.lower() != q.lower():
        logger.info(f"[QueryRewrite] {q[:80]!r} -> {rewritten[:120]!r}")
    return rewritten
