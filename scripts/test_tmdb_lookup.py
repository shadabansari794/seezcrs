"""
Test a TMDB movie lookup and print only the fields useful for enrichment.

Usage:
    python scripts/test_tmdb_lookup.py "The Bourne Identity" --year 2002

Auth:
    Preferred: TMDB_ACCESS_TOKEN=<TMDB API Read Access Token>
    Also supported: TMDB_API_KEY=<TMDB v3 API key>

The script also reads simple KEY=VALUE entries from .env if present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

TMDB_BASE_URL = "https://api.themoviedb.org/3"

_client: Optional[httpx.Client] = None
_client_lock = threading.Lock()


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs from .env without adding a dependency."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _get_client() -> httpx.Client:
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                base_url=TMDB_BASE_URL,
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_keepalive_connections=32, max_connections=64),
                headers={"accept": "application/json"},
            )
    return _client


def _tmdb_auth() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (extra_headers, extra_params) based on TMDB_ACCESS_TOKEN / TMDB_API_KEY."""
    access_token = os.getenv("TMDB_ACCESS_TOKEN")
    api_key = os.getenv("TMDB_API_KEY")
    if access_token:
        return {"Authorization": f"Bearer {access_token}"}, {}
    if api_key:
        return {}, {"api_key": api_key}
    raise RuntimeError(
        "Missing TMDB auth. Set TMDB_ACCESS_TOKEN or TMDB_API_KEY in your environment or .env."
    )


def request_tmdb(path: str, params: Optional[Dict[str, Any]] = None, attempts: int = 8) -> Dict[str, Any]:
    """Call TMDB via a shared pooled client with retries on 429/5xx."""
    auth_headers, auth_params = _tmdb_auth()
    merged_params = {**(params or {}), **auth_params}
    client = _get_client()

    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(path, params=merged_params, headers=auth_headers)
        except httpx.RequestError as exc:
            if attempt == attempts:
                raise
            last_error = exc
        else:
            if response.status_code == 200:
                return response.json()
            if response.status_code in {429, 500, 502, 503, 504} and attempt < attempts:
                last_error = RuntimeError(f"TMDB HTTP {response.status_code}")
            else:
                raise RuntimeError(f"TMDB HTTP {response.status_code}: {response.text}")

        time.sleep(min(1.5 * attempt, 12))

    raise RuntimeError(f"TMDB request failed after {attempts} attempts: {last_error}")


def search_movie(title: str, year: Optional[str]) -> Dict[str, Any]:
    """Search TMDB and return the first result."""
    params = {
        "query": title,
        "include_adult": "false",
        "language": "en-US",
        "page": 1,
    }
    if year:
        params["year"] = year

    data = request_tmdb("/search/movie", params)
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"No TMDB movie results for title={title!r} year={year!r}")

    return results[0]


def get_movie_details(tmdb_id: int) -> Dict[str, Any]:
    """Fetch movie details, credits, and keywords in one request."""
    return request_tmdb(
        f"/movie/{tmdb_id}",
        {
            "language": "en-US",
            "append_to_response": "credits,keywords",
        },
    )


def extract_directors(details: Dict[str, Any]) -> List[str]:
    """Extract director names from credits.crew."""
    crew = ((details.get("credits") or {}).get("crew")) or []
    return [
        person.get("name")
        for person in crew
        if person.get("job") == "Director" and person.get("name")
    ]


def extract_cast(details: Dict[str, Any], limit: int) -> List[str]:
    """Extract top cast names."""
    cast = ((details.get("credits") or {}).get("cast")) or []
    return [
        person.get("name")
        for person in cast[:limit]
        if person.get("name")
    ]


def extract_keywords(details: Dict[str, Any], limit: int) -> List[str]:
    """Extract keyword names."""
    keywords = ((details.get("keywords") or {}).get("keywords")) or []
    return [
        keyword.get("name")
        for keyword in keywords[:limit]
        if keyword.get("name")
    ]


def to_needed_keys(details: Dict[str, Any], cast_limit: int, keyword_limit: int) -> Dict[str, Any]:
    """Keep only fields useful for a richer movie vector."""
    release_date = details.get("release_date") or ""
    year = release_date[:4] if release_date else None

    return {
        "tmdb_id": details.get("id"),
        "title": details.get("title"),
        "original_title": details.get("original_title"),
        "year": int(year) if year and year.isdigit() else None,
        "release_date": release_date or None,
        "genres": [
            genre.get("name")
            for genre in details.get("genres", [])
            if genre.get("name")
        ],
        "overview": details.get("overview") or None,
        "keywords": extract_keywords(details, keyword_limit),
        "director": extract_directors(details),
        "cast": extract_cast(details, cast_limit),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test TMDB enrichment lookup for one movie title.")
    parser.add_argument("title", nargs="?", default="The Bourne Identity", help="Movie title to search.")
    parser.add_argument("--year", help="Optional release year to improve matching.")
    parser.add_argument("--cast-limit", type=int, default=5, help="Number of cast names to keep.")
    parser.add_argument("--keyword-limit", type=int, default=12, help="Number of keywords to keep.")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    result = search_movie(args.title, args.year)
    details = get_movie_details(result["id"])
    needed = to_needed_keys(details, args.cast_limit, args.keyword_limit)

    print(json.dumps(needed, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
