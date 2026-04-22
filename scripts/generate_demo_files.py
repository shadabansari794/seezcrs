import requests
import json
import subprocess
import time

API_URL = "http://localhost:8000/recommend"

scenarios = [
    {
        "name": "1. New User Exploration (RAG)",
        "user_id": "new_user_rag_multi",
        "model_type": "rag",
        "turns": [
            "I am looking for a heartwarming family movie.",
            "Can you suggest something similar but more recent?",
            "Actually, I'd prefer if it had some magic or fantasy elements."
        ]
    },
    {
        "name": "2. Factual Deep-Dive (Agent + Tools)",
        "user_id": "agent_tools_multi",
        "model_type": "agent",
        "turns": [
            "Who directed the movie Inception?",
            "What other famous movies did they direct?",
            "Which of those are sci-fi movies?",
            "What is the plot of Interstellar? Use IMDb."
        ]
    },
    {
        "name": "3. History & Preference Recall (Agent)",
        "user_id": "A30Q8X8B1S3GGT", # Real user from dataset
        "model_type": "agent",
        "turns": [
            "I'm in the mood for an action movie like the ones in my history.",
            "I've seen Commando. Suggest something more modern.",
            "Wait, what movies did you recommend to me in our past interactions before this one?",
            "Is there anything similar being announced lately?"
        ]
    },
    {
        "name": "4. Web & News Exploration (Agent)",
        "user_id": "web_explorer_multi",
        "model_type": "agent",
        "turns": [
            "Hi! What are the biggest movie announcements this month?",
            "Are there any updates on the next Marvel movie?",
            "Which actors are confirmed for it?",
            "Search for a trailer or release date for any of those."
        ]
    }
]

transcript_output = []
print("Running Multi-Turn API Demo Transcript Generator...")

for scenario in scenarios:
    name = scenario["name"]
    user_id = scenario["user_id"]
    model_type = scenario["model_type"]
    
    print(f"--- Scenario: {name} ---")
    transcript_output.append(f"### SCENARIO: {name} (User: {user_id}, Mode: {model_type.upper()})")
    
    for idx, query in enumerate(scenario["turns"]):
        print(f"  Turn {idx+1}: {query}")
        transcript_output.append(f"**TURN {idx+1}**")
        transcript_output.append(f"USER: {query}")
        
        payload = {
            "query": query,
            "user_id": user_id,
            "model_type": model_type
        }
        
        # Retry logic to handle transient connection issues (Docker cold-start)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(API_URL, json=payload, timeout=150)
                data = response.json()
                
                transcript_output.append(f"ASSISTANT:\n{data.get('response_text')}")
                recs = data.get('recommendations')
                if recs:
                    transcript_output.append(f"EXTRACTED RECS: {[r['title'] for r in recs]}")
                transcript_output.append(f"*Latency: {data.get('processing_time_ms', 0):.0f}ms*")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    [Retry {attempt+1}] {str(e)}. Retrying in 3s...")
                    time.sleep(3)
                else:
                    transcript_output.append(f"!! ERROR after {max_retries} attempts: {str(e)}")
            
        transcript_output.append("-" * 30)
        time.sleep(1)
    
    transcript_output.append("\n" + "="*80 + "\n")

with open("conversation_live_api_transcript.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(transcript_output))
print("Updated conversation_live_api_transcript.txt")

try:
    # Use errors='replace' to avoid windows character map crashes
    # Increase tail to make sure we get all turns
    result = subprocess.run(
        ["docker", "compose", "logs", "--tail", "2000", "movie-crs-api"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True
    )
    
    # Filter logs to keep the interesting agent traces
    filtered_logs = []
    seen_final_responses = set() 
    import re
    
    # regex to strip "2026-04-21 15:21:30,658 - models.agent.nodes - INFO - "
    # Note: Using \b to handle leading spaces more flexibly if needed, or just .strip()
    log_pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - .*? - INFO - ")

    raw_lines = result.stdout.split('\n')
    for line in raw_lines:
        if "/health" in line or "GET /" in line or "POST /" in line:
            continue
        
        markers = ["[Agent", "[Tool]", "[RAG]", "[QueryRewrite]", "LLM decided to use tool", "[OUTPUT]"]
        if any(kw in line for kw in markers):
            # Extract content from docker log format "service_name | log_content"
            clean_line = line.split("|")[-1].strip() 
            
            # Strip standard python logging prefix
            clean_line = log_pattern.sub("", clean_line).strip()
            
            # Special formatting for the actual response text to the user
            if "[OUTPUT]" in clean_line:
                msg_content = clean_line.split("[OUTPUT]")[-1].strip()
                clean_line = f"RESULT >> {msg_content}"

            # Add a visual break before each new request
            if "[Agent] start" in clean_line or "[RAG] start" in clean_line:
                # Extract query from log "query='...'" or just keep the whole start line
                filtered_logs.append("\n" + "="*80)
                filtered_logs.append(f"REQUEST >> {clean_line}")
                filtered_logs.append("="*80)
            elif clean_line:
                filtered_logs.append(clean_line)
            
    with open("pipeline_trace.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(filtered_logs))
    print(f"Captured {len(filtered_logs)} lines into pipeline_trace.txt")
except Exception as e:
    print(f"Failed to fetch logs: {e}")
