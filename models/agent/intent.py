"""LLM-based intent classifier for the agent pipeline.

One short call to a lightweight LLM. Returns the label only; keeps chit-chat
and closing turns out of the retrieval + ranking path so those turns spend the
minimum tokens. Uses temperature=0 for stability.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.schemas import Message
from models.agent.state import (
    INTENT_CHAT,
    INTENT_CLARIFY,
    INTENT_CLOSING,
    INTENT_RECOMMEND,
    INTENT_RESEARCH,
)
from prompts.templates import PromptTemplates

logger = logging.getLogger(__name__)

VALID_INTENTS = {INTENT_CHAT, INTENT_RECOMMEND, INTENT_CLARIFY, INTENT_CLOSING, INTENT_RESEARCH}


async def classify_intent(
    llm,
    query: str,
    history: Optional[List[Message]] = None,
) -> str:
    """Classify the current turn. Returns one of VALID_INTENTS."""
    q = (query or "").strip()
    if not q:
        return INTENT_CLARIFY

    context_lines = []
    if history:
        # Last 2 turns are enough to disambiguate pronouns like "that", "it".
        for msg in history[-4:]:
            role = "User" if msg.role == "user" else "Assistant"
            context_lines.append(f"{role}: {msg.content}")
    context_block = "\n".join(context_lines) or "(no prior turns)"

    chain = PromptTemplates().get_intent_classify_prompt() | llm
    response = await chain.ainvoke({"context_block": context_block, "query": q})
    raw = (getattr(response, "content", "") or "").strip().lower()
    label = raw.split()[0] if raw else ""
    label = label.strip(".,!? \"'")

    if label in VALID_INTENTS:
        logger.info(f"[Agent.intent] classified={label!r} query={q[:80]!r}")
        return label

    logger.warning(f"[Agent.intent] unknown label {raw!r}; defaulting to recommend")
    return INTENT_RECOMMEND
