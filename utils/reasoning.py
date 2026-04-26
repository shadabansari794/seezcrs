import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

REASONING_LOG_PATH = Path("/app/logs/reasoning.log")


def log_reasoning_step(step_name: str, data: Optional[Dict[str, Any]] = None, detail: str = ""):
    """Append a reasoning entry as one JSON line.

    ``data`` is a free-form dict of inputs/outputs the UI renders inside an
    expandable block. ``detail`` is an optional short one-liner for simple
    entries that don't need the full expander body.
    """
    entry = {
        "ts": datetime.now().strftime("%H:%M:%S"),
        "step": step_name,
        "detail": detail,
        "data": data or {},
    }
    try:
        REASONING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REASONING_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logging.error(f"Failed to write reasoning log: {e}")


def clear_reasoning_log():
    """Clear the reasoning log for a new conversation turn."""
    try:
        if REASONING_LOG_PATH.exists():
            with open(REASONING_LOG_PATH, "w", encoding="utf-8") as f:
                f.write("")
    except Exception as e:
        logging.error(f"Failed to clear reasoning log: {e}")
