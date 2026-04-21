"""
Replay the scripted conversations from conversation_test_cases.txt against the
live /recommend API and write the results to conversation_live_api_transcript.txt.

Usage:
    python scripts/run_live_api_transcript.py
    python scripts/run_live_api_transcript.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import httpx


@dataclass
class TestCase:
    number: int
    title: str
    model_type: str
    user_id: str
    profile_hints: str
    turns: List[str]


TEST_CASES: List[TestCase] = [
    TestCase(
        number=0,
        title="New user with RAG",
        model_type="rag",
        user_id="demo-rag-new",
        profile_hints="",
        turns=[
            "Hi, I'm in the mood for something cozy. I recently watched White Christmas and really liked the warm music and family feeling.",
            "Could you recommend something with that same light, older-movie comfort?",
            "That sounds fun. Is it more silly or more sentimental?",
            "Nice. Give me one more option, but keep it family-friendly.",
            "Great, thanks. I will try one of those tonight.",
        ],
    ),
    TestCase(
        number=1,
        title="Existing LLM-Redial user with RAG",
        model_type="rag",
        user_id="A2NBOL825B93OM",
        profile_hints=(
            "liked The Bourne Identity, Master and Commander: The Far Side of the World, "
            "The Lord of the Rings: The Fellowship of the Ring; disliked Driven VHS and Darkness Falls VHS."
        ),
        turns=[
            "I want something exciting, but not as ridiculous as Driven. I liked The Bourne Identity because it stayed tense without feeling dumb.",
            "Can you recommend something that fits that?",
            "I already liked Fellowship, so that makes sense. Is it more fantasy than action?",
            "What if I want something stranger and darker, but not cheap horror?",
            "Good. That is enough for now, thanks.",
        ],
    ),
    TestCase(
        number=2,
        title="New user with Agent",
        model_type="agent",
        user_id="demo-agent-new",
        profile_hints="",
        turns=[
            "Hey, I just watched The Screaming Skull and the print quality was awful. The acting felt grade Z too.",
            "I still want something eerie, just better made. Any ideas?",
            "I like slow-burn horror, but I do not want anything too confusing.",
            "Is that closer to ghost story or slasher?",
            "Perfect. Thanks, goodbye.",
        ],
    ),
    TestCase(
        number=3,
        title="Existing LLM-Redial user with Agent",
        model_type="agent",
        user_id="AQP1VPK16SVWM",
        profile_hints=(
            "liked Stir Of Echoes, Quatermass &amp; The Pit VHS, The Incredibles (Mandarin Chinese Edition), "
            "The Three Musketeers/The Four Musketeers VHS; disliked Apollo 18, Primer, Eraserhead, The Colony."
        ),
        turns=[
            "I want something suspenseful, but please avoid anything like Apollo 18 or Primer. I did not enjoy those.",
            "Can you recommend one movie from that angle?",
            "I already liked that one. Give me a different direction but still with mystery.",
            "What if I want something lighter after that?",
            "Great. That gives me a nice mix. Thanks.",
        ],
    ),
]


def call_recommend(client: httpx.Client, base_url: str, user_id: str, model_type: str, query: str) -> dict:
    response = client.post(
        f"{base_url}/recommend",
        json={"user_id": user_id, "model_type": model_type, "query": query},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def format_case(case: TestCase, responses: List[dict]) -> str:
    lines: List[str] = [
        str(case.number),
        "",
        f"Test Case: {case.title}",
        f"model_type: {case.model_type}",
        f"user_id: {case.user_id}",
    ]
    if case.profile_hints:
        lines.append(f"known profile hints: {case.profile_hints}")
    lines.append("")

    for i, (query, data) in enumerate(zip(case.turns, responses), start=1):
        recs = data.get("recommendations") or []
        rec_titles = [r.get("title", "") if isinstance(r, dict) else str(r) for r in recs]
        lines.append(f"Turn {i}")
        lines.append(f"User: {query}")
        lines.append(f"Agent: {data.get('response_text', '').strip()}")
        lines.append(f"Recommendations: {', '.join(rec_titles)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay scripted conversations against the live /recommend API.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("conversation_live_api_transcript.txt"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    header = [
        "Movie CRS Live API Conversation Transcript",
        "",
        "Source: live /recommend API",
        "Format: numbered test cases with multi-turn User/Agent exchanges.",
        "",
    ]

    blocks = ["\n".join(header)]

    with httpx.Client() as client:
        for case in TEST_CASES:
            print(f"[case {case.number}] {case.title} user_id={case.user_id} model={case.model_type}", flush=True)
            responses: List[dict] = []
            for i, turn in enumerate(case.turns, start=1):
                t0 = time.time()
                data = call_recommend(client, args.base_url, case.user_id, case.model_type, turn)
                print(f"  turn {i}: {(time.time() - t0):.1f}s, parsed={len(data.get('recommendations') or [])}", flush=True)
                responses.append(data)
            blocks.append(format_case(case, responses))

    args.output.write_text("\n".join(blocks), encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
