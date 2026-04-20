"""Message assembly helpers for the agent recommender."""

from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from app.schemas import Message
from prompts.templates import PromptTemplates


def build_agent_messages(
    prompt_templates: PromptTemplates,
    query: str,
    history: List[Message],
    user_id: Optional[str],
) -> List:
    """
    Assemble [system, *history_turns, user] via ChatPromptTemplate.

    The template produces [system, final_user]; prior turns are spliced between
    them so the LLM sees the full dialogue.
    """
    chat_prompt = prompt_templates.get_agent_chat_prompt(user_id=user_id)
    conv_summary = prompt_templates.summarize_conversation(
        [{"role": m.role, "content": m.content} for m in history]
    )
    templated = chat_prompt.format_messages(
        conversation_summary=conv_summary,
        user_query=query,
    )
    system_msg, final_user_msg = templated[0], templated[-1]

    history_msgs = []
    for msg in history:
        if msg.role == "user":
            history_msgs.append(HumanMessage(content=msg.content))
        else:
            history_msgs.append(AIMessage(content=msg.content))

    return [system_msg, *history_msgs, final_user_msg]
