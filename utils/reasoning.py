import logging
from pathlib import Path
from datetime import datetime

REASONING_LOG_PATH = Path("/app/logs/reasoning.log")

def log_reasoning_step(step_name: str, detail: str = ""):
    """Log a human-readable reasoning step for the frontend to display."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {step_name}: {detail}\n"
    
    try:
        # Ensure log directory exists
        REASONING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REASONING_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)
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
