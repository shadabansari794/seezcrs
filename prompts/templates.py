"""
Prompt templates for conversational movie recommendation.

Built on LangChain's ChatPromptTemplate so both the RAG and Agent paths share
the same structured prompt surface:

- **System** messages are injected as literal `SystemMessage` objects (no
  variable interpolation), which keeps arbitrary content in few-shot dialogues
  safe from `.format()` collisions with stray `{` / `}` characters.
- **User** messages are `HumanMessagePromptTemplate`s with named variables
  (`{conversation_history}`, `{movies_context}`, `{user_query}`, etc.) so call
  sites just pass kwargs to `.format_messages(...)`.

Optionally, real LLM-Redial dialogues can be attached as few-shot examples.
"""
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


_RAG_SYSTEM_BASE = """You are a friendly, conversational movie assistant. You talk like a human who happens to know a lot about movies — not a search engine.

READ THE USER'S INTENT FIRST, THEN CHOOSE A MODE:

MODE A — CHIT-CHAT (greetings, reactions to a movie, opinions, small talk, jokes)
- Reply naturally in 1–3 short sentences.
- Acknowledge what they said; share a brief relevant thought if it fits.
- Do NOT recommend movies. Do NOT list catalog titles. Do NOT use the `**Title** — reason` format.

MODE B — RECOMMENDATION REQUEST (they ask for suggestions, "what should I watch", "any recommendations", describe a mood/genre/actor they want)
- Pick 1–3 movies from the provided catalog only.
- Format each pick on its own line:  **Title** — one short sentence on the fit.
- No preamble, no restating the request, no confidence labels, no follow-up question.
- Never recommend anything outside the catalog.

MODE C — CLARIFY (their request is genuinely ambiguous, e.g. "recommend something")
- Ask ONE short clarifying question. No recommendations this turn.

MODE D — CLOSING (thanks, goodbye, "will do")
- One warm one-liner. No recommendations.

ALWAYS:
- Reference preferences mentioned earlier in the conversation when relevant.
- Keep responses tight. No filler, no "Absolutely!", no "Great question!".
- The mode choice is private. Never mention MODE A/B/C/D or output a mode label."""


_RAG_USER_TEMPLATE = """CONVERSATION HISTORY:
{conversation_history}

AVAILABLE MOVIES FROM CATALOG (use only if the user is asking for recommendations):
{movies_context}

USER MESSAGE: {user_query}

Privately pick the right mode and reply per the rules. Do not mention the mode name or letter. Only use the `**Title** — reason` format if you're in recommendation mode."""


_AGENT_SYSTEM_BASE = """You are a friendly, conversational movie assistant with access to tools. You talk like a human — not a search engine — and only reach for tools when the user actually wants recommendations.

READ THE USER'S INTENT FIRST, THEN CHOOSE A MODE:

MODE A — CHIT-CHAT (greetings, reactions to a movie, opinions, small talk)
- Reply naturally in 1–3 short sentences. No tool calls. No recommendations.

MODE B — RECOMMENDATION REQUEST (they ask for suggestions, describe a mood/genre/actor, want "something to watch")
- Use tools: search_movies / filter_by_genre / get_user_history as appropriate.
- Then recommend 1–3 titles, each on its own line:  **Title** — one short sentence on the fit.
- No preamble, no restating the request, no confidence labels, no follow-up question.

MODE C — CLARIFY (truly ambiguous request)
- Ask ONE short clarifying question. No tool calls this turn.

MODE D — CLOSING (thanks, goodbye)
- One warm one-liner. No tool calls, no recommendations.

ALWAYS:
- Reference preferences mentioned earlier when relevant.
- Keep responses tight. No filler.
- The mode choice is private. Never mention MODE A/B/C/D or output a mode label."""


_AGENT_USER_TEMPLATE = """CONVERSATION SUMMARY:
{conversation_summary}

CURRENT USER MESSAGE: {user_query}

Privately pick the right mode and respond. Do not mention the mode name or letter. Only call tools if you're in recommendation mode."""


class PromptTemplates:
    """Centralized ChatPromptTemplate factories for RAG and Agent CRS."""

    def __init__(self, few_shot_examples: Optional[str] = None) -> None:
        """
        Args:
            few_shot_examples: Pre-formatted block of example dialogues to
                append to system prompts. If None, system prompts are used as-is.
        """
        self.few_shot_examples = few_shot_examples

    @staticmethod
    def format_conversation_examples(snippets: List[str]) -> str:
        """Format raw LLM-Redial dialogue snippets into a few-shot block."""
        if not snippets:
            return ""

        blocks = [f"Example {i}:\n{snippet.strip()}" for i, snippet in enumerate(snippets, start=1)]
        return (
            "EXAMPLE DIALOGUES (from real user conversations — follow this tone and structure):\n\n"
            + "\n\n---\n\n".join(blocks)
        )

    def _system_content(self, base: str) -> str:
        """Attach few-shot examples to a base system prompt if configured."""
        if not self.few_shot_examples:
            return base
        return f"{base}\n\n{self.few_shot_examples}"

    def get_rag_chat_prompt(self) -> ChatPromptTemplate:
        """
        ChatPromptTemplate for the RAG path.

        Variables: conversation_history, movies_context, user_query.
        """
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=self._system_content(_RAG_SYSTEM_BASE)),
            HumanMessagePromptTemplate.from_template(_RAG_USER_TEMPLATE),
        ])

    def get_agent_chat_prompt(self, user_id: Optional[str] = None) -> ChatPromptTemplate:
        """
        ChatPromptTemplate for the Agent path.

        Variables: conversation_summary, user_query. When `user_id` is given,
        a per-request hint is appended to the system prompt so the LLM knows
        to call `get_user_history` with it.
        """
        system = self._system_content(_AGENT_SYSTEM_BASE)
        if user_id:
            system = (
                f"{system}\n\n"
                f"CURRENT USER CONTEXT:\n"
                f'The current user_id is "{user_id}". '
                f"When personalizing, call the `get_user_history` tool with this user_id "
                f"to retrieve their recent liked/disliked movies before making recommendations."
            )
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessagePromptTemplate.from_template(_AGENT_USER_TEMPLATE),
        ])

    @staticmethod
    def format_movies_context(retrieved_movies: List[Dict[str, Any]]) -> str:
        """Render retrieved movies into the block the RAG user prompt expects."""
        return "\n\n".join([
            f"Movie: {m['title']} ({m.get('year', 'N/A')})\n"
            f"Genres: {', '.join(m.get('genres', []))}\n"
            f"Rating: {m.get('rating', 'N/A')}/10\n"
            f"Description: {m.get('description', 'No description')}\n"
            f"Director: {m.get('director', 'Unknown')}"
            for m in retrieved_movies
        ])

    @staticmethod
    def summarize_conversation(messages: List[Dict[str, str]]) -> str:
        """Concise summary of conversation for context efficiency."""
        if not messages:
            return "No prior conversation."

        user_preferences = []
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"].lower()
                if any(genre in content for genre in ["action", "drama", "comedy", "thriller", "sci-fi", "romance"]):
                    user_preferences.append(msg["content"])

        summary_parts = []
        if user_preferences:
            summary_parts.append(f"User preferences: {'; '.join(user_preferences[:3])}")
        summary_parts.append(f"Total exchanges: {len(messages)}")
        return " | ".join(summary_parts)
