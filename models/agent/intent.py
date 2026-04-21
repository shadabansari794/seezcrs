"""LLM-based intent classifier for the agent pipeline.

One short call to a lightweight LLM. Returns the label only; keeps chit-chat
and closing turns out of the retrieval + ranking path so those turns spend the
minimum tokens. Uses temperature=0 for stability.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas import Message
from models.agent.state import (
    INTENT_CHAT,
    INTENT_CLARIFY,
    INTENT_CLOSING,
    INTENT_RECOMMEND,
)

logger = logging.getLogger(__name__)

VALID_INTENTS = {INTENT_CHAT, INTENT_RECOMMEND, INTENT_CLARIFY, INTENT_CLOSING}


_SYSTEM_PROMPT = """You classify the USER'S CURRENT MESSAGE in a movie recommendation chat.

Output exactly one label, lowercase, no punctuation, no explanation:

- recommend: user asks for suggestions, names a mood/genre/actor/director they want, asks for "another", "more like", "something else", "what should I watch".
- chat: user greets, shares an opinion, reacts to a movie, small talk, describes something they watched, asks a factual follow-up about an already-recommended movie (e.g. "is it more silly or sentimental?").
- clarify: user's ask is too vague to act on (e.g. a bare "recommend something").
- closing: thanks, goodbye, "that's enough", "will do".

If unsure between chat and recommend, pick recommend only when the user is clearly requesting a new suggestion. Otherwise pick chat.

Return only: recommend | chat | clarify | closing"""


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

    prompt = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Recent conversation:\n{context_block}\n\n"
                f"Current user message: {q}\n\nLabel:"
            )
        ),
    ]

    response = await llm.ainvoke(prompt)
    raw = (getattr(response, "content", "") or "").strip().lower()
    label = raw.split()[0] if raw else ""
    label = label.strip(".,!? \"'")

    if label in VALID_INTENTS:
        logger.info(f"[Agent.intent] classified={label!r} query={q[:80]!r}")
        return label

    logger.warning(f"[Agent.intent] unknown label {raw!r}; defaulting to recommend")
    return INTENT_RECOMMEND
