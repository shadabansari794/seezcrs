"""
Enrich the LLM-Redial movie catalog with TMDB metadata.

Usage:
    python scripts/enrich_tmdb_catalog.py
    python scripts/enrich_tmdb_catalog.py --limit 25
    python scripts/enrich_tmdb_catalog.py --start 100 --limit 100

Auth:
    Preferred: TMDB_ACCESS_TOKEN=<TMDB API Read Access Token>
    Also supported: TMDB_API_KEY=<TMDB v3 API key>

Output:
    data/llm_redial/tmdb_enriched_movies.json

The output is a dict keyed by the original LLM-Redial item id so the loader can
merge metadata back into the catalog without relying on title matching.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from test_tmdb_lookup import (
    get_movie_details,
    load_dotenv,
    search_movie,
    to_needed_keys,
)

DEFAULT_ITEM_MAP = Path("data/llm_redial/item_map.json")
DEFAULT_OUTPUT = Path("data/llm_redial/tmdb_enriched_movies.json")

_FORMAT_PATTERNS = [
    r"\bVHS\b",
    r"\bDVD\b",
    r"\bBlu[- ]?ray\b",
    r"\bUltraViolet\b",
    r"\bWidescreen\b",
    r"\bFullscreen\b",
    r"\bCollector'?s Edition\b",
    r"\bSpecial Edition\b",
    r"\bDirector'?s Cut\b",
]


def clean_title_for_search(title: str) -> str:
    """Remove common retail/video-format suffixes before TMDB search."""
    cleaned = html.unescape(title)
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    cleaned = re.sub(r"\([^)]*(?:DVD|VHS|Blu|UltraViolet|Widescreen|Fullscreen)[^)]*\)", " ", cleaned, flags=re.I)
    for pattern in _FORMAT_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -:/") 


def load_item_map(path: Path) -> Dict[str, str]:
    """Load LLM-Redial item_map.json."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing_output(path: Path) -> Dict[str, Any]:
    """Load existing enrichment cache for resume support."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_output(path: Path, data: Dict[str, Any]) -> None:
    """Write enrichment output atomically enough for resume-friendly runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def iter_items(item_map: Dict[str, str], start: int, limit: Optional[int]) -> Iterable[Tuple[int, str, str]]:
    """Yield indexed item ids/titles with optional start and limit."""
    items = list(item_map.items())
    end = None if limit is None else start + limit
    for idx, (item_id, title) in enumerate(items[start:end], start=start):
        yield idx, item_id, title


def enrich_one(
    item_id: str,
    source_title: str,
    cast_limit: int,
    keyword_limit: int,
    clean_titles: bool,
) -> Dict[str, Any]:
    """Search and enrich one LLM-Redial movie title.

    With ``clean_titles=True`` we search on the retail-stripped title first and
    fall back to the raw title once if TMDB returned no results. That keeps the
    old behavior on already-good titles while rescuing entries like
    "Outland VHS" / "Movie [Blu-ray]" that the old search missed.
    """
    cleaned = clean_title_for_search(source_title)
    query_title = cleaned if clean_titles else source_title

    try:
        search_result = search_movie(query_title, year=None)
    except RuntimeError:
        if not clean_titles or cleaned == source_title:
            raise
        search_result = search_movie(source_title, year=None)
        query_title = source_title

    details = get_movie_details(search_result["id"])
    needed = to_needed_keys(details, cast_limit=cast_limit, keyword_limit=keyword_limit)
    return {
        "id": item_id,
        "source_title": source_title,
        "query_title": query_title,
        **needed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich LLM-Redial item_map movies with TMDB metadata.")
    parser.add_argument("--item-map", type=Path, default=DEFAULT_ITEM_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=0, help="Start offset in item_map.")
    parser.add_argument("--limit", type=int, help="Max number of items to process.")
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of concurrent TMDB workers (each movie makes 2 sequential requests).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Per-worker sleep after each movie. Default 0; backoff in request_tmdb handles 429s.",
    )
    parser.add_argument("--save-every", type=int, default=25, help="Persist output every N completed items.")
    parser.add_argument("--cast-limit", type=int, default=5)
    parser.add_argument("--keyword-limit", type=int, default=12)
    parser.add_argument("--retry-misses", action="store_true", help="Retry records that already have an error.")
    parser.add_argument(
        "--clean-titles",
        dest="clean_titles",
        action="store_true",
        help="Strip retail suffixes (VHS/DVD/Blu-ray/...) before TMDB search (default).",
    )
    parser.add_argument(
        "--no-clean-titles",
        dest="clean_titles",
        action="store_false",
        help="Search TMDB with the raw LLM-Redial title.",
    )
    parser.set_defaults(clean_titles=True)
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    item_map = load_item_map(args.item_map)
    output = load_existing_output(args.output)
    output_lock = threading.Lock()

    pending: List[Tuple[int, str, str]] = []
    for idx, item_id, source_title in iter_items(item_map, args.start, args.limit):
        existing = output.get(item_id)
        if existing and (existing.get("tmdb_id") or (existing.get("error") and not args.retry_misses)):
            continue
        pending.append((idx, item_id, source_title))

    ok_count = sum(1 for value in output.values() if value.get("tmdb_id"))
    error_count = sum(1 for value in output.values() if value.get("error"))
    print(
        f"TMDB enrichment start: items={len(item_map)} to_process={len(pending)} "
        f"existing_ok={ok_count} existing_errors={error_count} workers={args.workers} "
        f"clean_titles={args.clean_titles} output={args.output}",
        flush=True,
    )

    def worker(idx: int, item_id: str, source_title: str) -> None:
        try:
            record = enrich_one(
                item_id,
                source_title,
                cast_limit=args.cast_limit,
                keyword_limit=args.keyword_limit,
                clean_titles=args.clean_titles,
            )
            status_line = (
                f"[{idx}] ok {item_id} {source_title!r} -> "
                f"{record.get('title')!r} ({record.get('year')})"
            )
            stream = sys.stdout
        except Exception as exc:
            record = {
                "id": item_id,
                "source_title": source_title,
                "query_title": clean_title_for_search(source_title) if args.clean_titles else source_title,
                "tmdb_id": None,
                "error": str(exc),
            }
            status_line = f"[{idx}] error {item_id} {source_title!r}: {exc}"
            stream = sys.stderr

        with output_lock:
            output[item_id] = record

        print(status_line, file=stream, flush=True)

        if args.sleep:
            time.sleep(args.sleep)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(worker, idx, item_id, src) for idx, item_id, src in pending]
            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                if args.save_every and completed % args.save_every == 0:
                    with output_lock:
                        snapshot = dict(output)
                    save_output(args.output, snapshot)
    finally:
        with output_lock:
            final_snapshot = dict(output)
        save_output(args.output, final_snapshot)

    ok_count = sum(1 for value in final_snapshot.values() if value.get("tmdb_id"))
    error_count = sum(1 for value in final_snapshot.values() if value.get("error"))
    print(f"TMDB enrichment done: ok={ok_count} errors={error_count} output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
