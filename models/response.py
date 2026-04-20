"""Shared response post-processing helpers."""

import re

_LEAKED_MODE_RE = re.compile(r"^\s*(?:MODE\s*)?[A-D]\s*[-:\u2013\u2014?]\s*(.*)$", re.IGNORECASE)


def strip_leaked_mode_label(response_text: str) -> str:
    """
    Remove accidental prompt-internal mode labels from the start of a response.

    The prompts ask the LLM to choose modes privately. If it still emits a
    leading line like "D - One warm one-liner", prefer the actual assistant
    response on following lines, or strip just the label when that is all there
    is.
    """
    text = (response_text or "").strip()
    if not text:
        return text

    lines = text.splitlines()
    first_line = lines[0].strip()
    match = _LEAKED_MODE_RE.match(first_line)
    if not match:
        return text

    remaining_lines = [line for line in lines[1:] if line.strip()]
    if remaining_lines:
        return "\n".join(remaining_lines).strip()

    return match.group(1).strip()
