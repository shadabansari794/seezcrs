"""Run the 4 live API test conversations and write transcript + trace files."""
import json, time, httpx, sys

API = "http://localhost:8000/recommend"

TESTS = [
    {
        "label": "New user with RAG",
        "model_type": "rag",
        "user_id": "demo-rag-new",
        "profile_hints": None,
        "turns": [
            # Turn 1: clear recommend intent
            "I want a cozy family movie with warm music, something like White Christmas.",
            # Turn 2: follow-up asking for more (uses history)
            "That was nice. Can you suggest something similar but from the 1950s or 1960s?",
            # Turn 3: chit-chat question about previous recommendation (no new recs expected)
            "That sounds fun. Is it more silly or more sentimental?",
            # Turn 4: explicit new recommendation request with constraint
            "Give me one more option, but keep it family-friendly and animated.",
            # Turn 5: closing
            "Great, thanks. I will try one of those tonight.",
        ],
    },
    {
        "label": "Existing LLM-Redial user with RAG",
        "model_type": "rag",
        "user_id": "A2NBOL825B93OM",
        "profile_hints": "liked The Bourne Identity, Master and Commander: The Far Side of the World, The Lord of the Rings: The Fellowship of the Ring; disliked Driven VHS and Darkness Falls VHS.",
        "turns": [
            # Turn 1: clear request referencing profile
            "I want something exciting and tense like The Bourne Identity, but not as ridiculous as Driven.",
            # Turn 2: follow-up with a new direction
            "Those are good. Now suggest something with naval or military adventure, like Master and Commander.",
            # Turn 3: chit-chat opinion (no new recs expected)
            "I loved Fellowship of the Ring too. The epic scale was amazing.",
            # Turn 4: new request with different mood
            "What if I want something stranger and darker, but not cheap horror?",
            # Turn 5: closing
            "Good. That is enough for now, thanks.",
        ],
    },
    {
        "label": "New user with Agent",
        "model_type": "agent",
        "user_id": "demo-agent-new",
        "profile_hints": None,
        "turns": [
            # Turn 1: pure chit-chat (sharing opinion, no recs expected)
            "Hey, I just watched The Screaming Skull and the print quality was awful. The acting felt grade Z too.",
            # Turn 2: now explicitly asking for recommendations
            "I still want something eerie, just better made. Can you recommend a few?",
            # Turn 3: narrowing preference
            "I prefer slow-burn horror that is not too confusing. What do you have?",
            # Turn 4: chit-chat follow-up about previous rec (no new recs expected)
            "Is that closer to a ghost story or a slasher?",
            # Turn 5: closing
            "Perfect. Thanks, goodbye.",
        ],
    },
    {
        "label": "Existing LLM-Redial user with Agent",
        "model_type": "agent",
        "user_id": "AQP1VPK16SVWM",
        "profile_hints": "liked Stir Of Echoes, Quatermass & The Pit VHS, The Incredibles (Mandarin Chinese Edition), The Three Musketeers/The Four Musketeers VHS; disliked Apollo 18, Primer, Eraserhead, The Colony.",
        "turns": [
            # Turn 1: clear request with dislike constraints
            "I want something suspenseful like Stir of Echoes, but please avoid anything like Apollo 18 or Primer.",
            # Turn 2: narrowing to a single pick
            "Can you pick just one from those and tell me why it fits?",
            # Turn 3: asking for a completely different direction
            "Give me something with mystery instead, but different from what you already suggested.",
            # Turn 4: mood shift
            "What if I want something lighter and fun after all that darkness?",
            # Turn 5: closing
            "Great. That gives me a nice mix. Thanks.",
        ],
    },
]


def parse_recs(text: str):
    import re
    return [m.group(1).strip() for m in re.finditer(r"\*\*(.+?)\*\*", text)]


def main():
    client = httpx.Client(timeout=120)

    transcript_lines = [
        "Movie CRS Live API Conversation Transcript",
        "",
        "Source: live /recommend API",
        "Format: numbered test cases with multi-turn User/Agent exchanges.",
        "",
    ]
    trace_lines = [
        "Movie CRS Pipeline Trace",
        "Source: scripts/run_live_tests.py",
        "",
    ]

    for idx, tc in enumerate(TESTS):
        print(f"\n=== Test Case {idx}: {tc['label']} ===")

        # --- transcript header ---
        transcript_lines.append(str(idx))
        transcript_lines.append("")
        transcript_lines.append(f"Test Case: {tc['label']}")
        transcript_lines.append(f"model_type: {tc['model_type']}")
        transcript_lines.append(f"user_id: {tc['user_id']}")
        if tc["profile_hints"]:
            transcript_lines.append(f"known profile hints: {tc['profile_hints']}")
        transcript_lines.append("")

        # --- trace header ---
        bar = "=" * max(len(f"Test Case {idx}: {tc['label']}") + 4, 30)
        trace_lines.append(bar)
        trace_lines.append(f"Test Case {idx}: {tc['label']}")
        trace_lines.append(bar)
        trace_lines.append(f"model_type: {tc['model_type']}")
        trace_lines.append(f"user_id: {tc['user_id']}")
        if tc["profile_hints"]:
            trace_lines.append(f"profile_hints: {tc['profile_hints']}")
        trace_lines.append("")

        for turn_i, query in enumerate(tc["turns"], start=1):
            print(f"  Turn {turn_i} ...", end=" ", flush=True)
            payload = {
                "query": query,
                "user_id": tc["user_id"],
                "model_type": tc["model_type"],
            }
            t0 = time.time()
            resp = client.post(API, json=payload)
            elapsed_ms = (time.time() - t0) * 1000
            data = resp.json()
            response_text = data.get("response_text", "")
            recs = parse_recs(response_text)
            proc_ms = data.get("processing_time_ms", elapsed_ms)
            print(f"{proc_ms:.0f}ms  recs={len(recs)}")

            # --- transcript turn ---
            transcript_lines.append(f"Turn {turn_i}")
            transcript_lines.append(f"User: {query}")
            transcript_lines.append(f"Agent: {response_text}")
            transcript_lines.append(f"Recommendations: {', '.join(recs)}")
            transcript_lines.append("")

            # --- trace turn ---
            trace_lines.append(f"Turn {turn_i}")
            trace_lines.append("------")
            trace_lines.append(f"  User: {query}")
            trace_lines.append(f"  Processing time: {proc_ms:.0f} ms")
            trace_lines.append(f"  Model used: {data.get('model_used', tc['model_type'])}")
            trace_lines.append(f"  Recommendations parsed: {len(recs)}")
            if recs:
                trace_lines.append(f"  Recommendation titles: {', '.join(recs)}")
            trace_lines.append(f"  Response ({len(response_text)} chars): {response_text}")
            trace_lines.append("")

    # Write files
    with open("conversation_live_api_transcript.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(transcript_lines) + "\n")
    print(f"\nWrote conversation_live_api_transcript.txt")

    with open("pipeline_trace.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(trace_lines) + "\n")
    print(f"Wrote pipeline_trace.txt")


if __name__ == "__main__":
    main()
