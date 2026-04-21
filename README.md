# Movie CRS - Conversational Movie Recommender

A multi-turn movie recommender built on the LLM-Redial movie dataset.

The API exposes one recommendation endpoint backed by two recommender designs:

- `rag`: a classic retrieval-augmented generation pipeline.
- `agent`: a LangGraph tool-calling agent.

Both paths share the same dataset loader, Chroma vector index, prompt templates,
and server-side conversation memory keyed by `user_id`.

```text
POST /recommend
  request:  {"query": "...", "user_id": "...", "model_type": "rag" | "agent"}
  server:   resolve history by user_id
            select RAG or Agent recommender
            retrieve/search movies when needed
            call OpenAI chat model
            append turn to memory
  response: {"response_text": "...", "recommendations": [...], ...}
```

## Dataset

The dataset lives under [data/llm_redial](data/llm_redial).

| File | Count | Purpose |
|---|---:|---|
| `item_map.json` | 9,687 titles | Movie catalog. Loaded into `MovieDataLoader.movies`, embedded by Sentence Transformers, and indexed into ChromaDB. |
| `user_ids.json` | 3,131 ids | Known-user registry. A request `user_id` may match one of these ids or may be any stable client-provided id. |
| `final_data.jsonl` | 3,131 records | Per-user structured history used for dataset-side personalization. |
| `Conversation.txt` | 232K lines | Raw dialogues. Startup samples a few snippets for prompt tone examples. |

Data limitations are reflected in the implementation:

- `year` is not populated in the catalog, so year filtering is not exposed.
- `rating` is not populated in the catalog, so rating filtering is not exposed.
- `genre` is inferred from title keywords and used as the only lightweight filter.

## Project Layout

```text
.
|-- app/
|   |-- main.py              FastAPI app, startup lifecycle, endpoints, memory store
|   |-- schemas.py           Pydantic request/response models
|   `-- config.py            Env-driven settings
|-- models/
|   |-- rag/
|   |   |-- recommender.py   RAG coordinator class
|   |   |-- filters.py       Query filter extraction
|   |   |-- retrieval.py     Async vector retrieval helper
|   |   |-- parser.py        Recommendation response parser
|   |   `-- utils.py         Profile/history/message utilities
|   `-- agent/
|       |-- recommender.py   Agent coordinator class
|       |-- graph.py         LangGraph construction
|       |-- tools.py         Tool definitions
|       |-- messages.py      Prompt/message assembly
|       `-- state.py         AgentState definition
|-- prompts/
|   `-- templates.py         Shared RAG and Agent prompt templates
|-- utils/
|   `-- vector_store.py      ChromaDB + SentenceTransformer wrapper
|-- data/
|   |-- loader.py            LLM-Redial loader and user profile helpers
|   |-- llm_redial/          Source dataset files
|   `-- vector_store/        Generated Chroma index, ignored by git
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- .env.example
`-- .gitignore
```

The old top-level recommender modules were removed. Import the recommenders
from their packages:

```python
from models.rag import RAGRecommender
from models.agent import AgentRecommender
```

## Architecture

```text
Client
  |
  | POST /recommend
  v
FastAPI app
  |
  |-- validates RecommendationRequest
  |-- resolves app.state.conversations[user_id]
  |-- selects recommender by model_type
  |
  |-- rag:
  |     extract filters -> vector search -> prompt -> OpenAI
  |
  `-- agent:
        prompt -> LangGraph agent -> optional tools -> OpenAI
  |
  |-- appends user and assistant messages to memory
  `-- returns RecommendationResponse
```

Shared infrastructure:

- [data/loader.py](data/loader.py): loads movies, user ids, conversation records, and few-shot examples.
- [utils/vector_store.py](utils/vector_store.py): owns Chroma persistence and SentenceTransformer embeddings.
- [prompts/templates.py](prompts/templates.py): central prompt surface for both recommender paths.
- [app/main.py](app/main.py): owns FastAPI startup, global recommender instances, and in-process session memory.

## Request Flow

### 1. Validate

[app/main.py](app/main.py) receives JSON and FastAPI validates it against
`RecommendationRequest` in [app/schemas.py](app/schemas.py).

`user_id` is required and must be non-empty. Missing or empty values return
HTTP `422`.

### 2. Resolve History

The server owns conversation memory:

```python
conversations = app.state.conversations
history = conversations.setdefault(request.user_id, [])
```

The request schema still accepts a `history` field for compatibility, but the
endpoint currently ignores client-supplied history and uses the server memory
keyed by `user_id`.

### 3. Generate With RAG

[models/rag/recommender.py](models/rag/recommender.py) coordinates the RAG path:

1. Build a dataset-side user profile block when `user_id` exists in LLM-Redial.
2. Extract a lightweight genre filter from the query.
3. Retrieve candidate movies from Chroma via [models/rag/retrieval.py](models/rag/retrieval.py).
4. Format conversation history and retrieved movies into the shared RAG prompt.
5. Call OpenAI chat completions.
6. Parse markdown-style recommendations with [models/rag/parser.py](models/rag/parser.py).

### 4. Generate With Agent

[models/agent/recommender.py](models/agent/recommender.py) coordinates a
structured LangGraph pipeline. `classify_intent` (LLM, temperature=0) routes
each turn to one of two branches:

- **Chit-chat / clarify / closing** → `chat_reply` node → END.
- **Recommend** → `extract_preferences` → `retrieve` → `rank_score` → `explain` → END.

Stage details:

1. [models/agent/intent.py](models/agent/intent.py) — LLM picks `recommend | chat | clarify | closing`.
2. [models/agent/nodes.py](models/agent/nodes.py) — `extract_preferences` pulls filters via [models/rag/filters.py](models/rag/filters.py) and user history via the loader; `retrieve` hits Chroma through the vector store.
3. [models/agent/ranking.py](models/agent/ranking.py) — heuristic `score_candidates`: similarity + history affinity (liked director/genre overlap) − negative signal (disliked titles) + recency.
4. `explain` calls the main LLM with the ranked shortlist to produce `**Title** - reason` recommendations.
5. Parse recommendations with the shared RAG parser.

The graph is assembled in [models/agent/graph.py](models/agent/graph.py) from nodes built in [models/agent/nodes.py](models/agent/nodes.py).

### 5. Persist and Respond

After generation, the API appends the turn:

```python
history.append(Message(role="user", content=request.query))
history.append(Message(role="assistant", content=response_text))
```

The response shape is:

```json
{
  "response_text": "...",
  "recommendations": [
    {
      "title": "...",
      "confidence": 0.8,
      "reason": "..."
    }
  ],
  "model_used": "rag",
  "processing_time_ms": 1843.12
}
```

If the user is only chit-chatting or closing the conversation, the assistant may
return a natural response with `recommendations: []`.

## Recommenders

| Capability | RAG | Agent |
|---|---|---|
| Main file | [models/rag/recommender.py](models/rag/recommender.py) | [models/agent/recommender.py](models/agent/recommender.py) |
| Flow | Retrieve first, then generate | LLM decides whether to call tools |
| Retrieval | Always before the LLM call | On demand through tools |
| Personalization | User profile block in prompt | `get_user_history` tool |
| Latency | Lower and bounded | Variable, depends on tool calls |
| Best fit | Direct recommendation requests | Exploratory or multi-constraint requests |

Both recommenders use the same prompt mode design:

| Mode | Trigger | Behavior |
|---|---|---|
| A: Chit-chat | Greetings, reactions, opinions, small talk | 1-3 natural sentences, no recommendations |
| B: Recommendation | Explicit suggestions, mood, genre, actor, "what should I watch" | 1-3 catalog picks in `**Title** - reason` format |
| C: Clarify | Truly ambiguous recommendation request | One short clarifying question |
| D: Closing | Thanks, goodbye, "will do" | One warm one-liner |

## Session Memory

Conversation memory is stored in process:

```python
app.state.conversations: Dict[str, List[Message]] = {}
```

Known trade-offs:

- Memory is lost on restart.
- History growth is unbounded.
- There is no per-user write lock.
- There is no reset endpoint.

For production, replace this with Redis, SQLite, or another durable session
store and add request coordination for the same `user_id`.

## Running

### Docker

```bash
cp .env.example .env
# Fill OPENAI_API_KEY in .env

docker compose up -d --force-recreate
docker logs -f movie-crs-api
```

Docker Compose defaults `LLM_MODEL` to `gpt-4o-mini` when it is not set.

### Local Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Fill OPENAI_API_KEY in .env

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

First boot indexes the 9,687-title catalog into `data/vector_store/`. Later
boots reuse the persisted Chroma index.

## Optional TMDB Enrichment

The base LLM-Redial catalog is sparse, so the app can optionally enrich movies
with TMDB metadata. Enrichment adds real genres, overview, keywords, director,
cast, release date, and year.

Set one TMDB credential in `.env`:

```env
TMDB_ACCESS_TOKEN=...
# or
TMDB_API_KEY=...
```

Test one movie:

```bash
python scripts/test_tmdb_lookup.py "The Bourne Identity" --year 2002
```

Enrich the whole catalog (parallel, ~5–10 min for ~9.7k titles on a fast
network; resumable — re-run anytime to pick up where it stopped):

```bash
python scripts/enrich_tmdb_catalog.py --workers 16 --retry-misses
```

Useful flags:

- `--workers N` — concurrent TMDB workers (default `16`). Each worker makes 2
  sequential calls per movie; 16 workers ≈ 32 in-flight requests.
- `--clean-titles` / `--no-clean-titles` — strip retail suffixes like `VHS`,
  `DVD`, `[Blu-ray]`, `Special Edition` before searching TMDB. On by default;
  falls back to the raw title if the cleaned search returns nothing.
- `--retry-misses` — re-run rows previously saved with `error`.
- `--save-every 25` — checkpoint to disk every N completed items (atomic
  write, safe to Ctrl+C).
- `--sleep S` — per-worker sleep after each movie (default `0.0`). Retries on
  429/5xx are handled by the built-in exponential backoff.

The generated file is:

```text
data/llm_redial/tmdb_enriched_movies.json
```

The loader automatically merges that file when present. After enrichment,
delete/rebuild `data/vector_store/` or the Docker `vector-store-data` volume so
Chroma reindexes movies with the richer text.

## API

### `GET /`

Returns service metadata and endpoint names.

### `GET /health`

Example:

```json
{
  "status": "healthy",
  "vector_store": {
    "total_movies": 9687,
    "backend": "chromadb"
  },
  "llm_model": "gpt-4o-mini"
}
```

### `POST /recommend`

Request:

```json
{
  "query": "suggest a horror movie",
  "user_id": "demo-alice",
  "model_type": "rag",
  "max_recommendations": 5
}
```

Fields:

| Field | Required | Default | Notes |
|---|---:|---|---|
| `query` | Yes | N/A | Current user message. |
| `user_id` | Yes | N/A | Stable id for server-side memory and optional dataset profile lookup. |
| `model_type` | No | `rag` | Must be `rag` or `agent`. |
| `max_recommendations` | No | `5` | Integer from 1 to 10. RAG uses this to size retrieval/context. |
| `history` | No | `[]` | Accepted by schema but ignored by endpoint. Server memory is used instead. |

Response:

```json
{
  "response_text": "**The Shining** - A tense, atmospheric horror pick.",
  "recommendations": [
    {
      "title": "The Shining",
      "confidence": 0.8,
      "reason": "Recommended based on your preferences"
    }
  ],
  "model_used": "rag",
  "processing_time_ms": 1843.12
}
```

## Example Requests

New user with continuity:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query":"suggest a horror movie","user_id":"demo-alice","model_type":"rag"}'

curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query":"something less gory","user_id":"demo-alice","model_type":"rag"}'
```

Known LLM-Redial user:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query":"recommend a comedy","user_id":"A30Q8X8B1S3GGT","model_type":"rag"}'
```

Agent path:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query":"a thriller with a strong female lead","user_id":"demo-bob","model_type":"agent"}'
```

Chit-chat:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query":"Hi! I just watched The Screaming Skull. Ever seen it?","user_id":"demo-alice","model_type":"rag"}'
```

Validation failure:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query":"anything"}'
```

## Configuration

All settings are read from environment variables or `.env`. See
[.env.example](.env.example).

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | N/A | Required for generation. |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model. |
| `TEMPERATURE` | `0.7` | LLM sampling temperature. |
| `MAX_TOKENS` | `350` | Response length cap. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model. |
| `VECTOR_STORE_PATH` | `./data/vector_store` | Chroma persistence path. |
| `API_HOST` | `0.0.0.0` | Uvicorn host. |
| `API_PORT` | `8000` | Uvicorn port. |
| `LOG_LEVEL` | `INFO` | Python logging level. |

## Logging

Logging is configured in [app/main.py](app/main.py) with console output and a
file handler writing to `app.log`.

Docker Compose bind-mounts `./app.log:/app/app.log`, so host-side log tailing
works while the container runs.

Typical RAG log flow:

```text
[/recommend] IN user_id='demo-alice' model=rag query='...' prior_turns=0 history_len=0
[RAG] start user_id='demo-alice' history_turns=0 max_recs=5
[RAG] extracted filters: {'genre': 'horror'}
Retrieved N movies for query: ... (filters=...)
[RAG] calling LLM model=gpt-4o-mini messages=2 candidates=5
[RAG] LLM done tokens_in=812 tokens_out=144 response_len=298
[/recommend] OUT user_id='demo-alice' took=1843ms parsed=3 history_len=2 response_preview='...'
```

The agent path also logs graph/tool activity such as:

```text
[Agent] start ...
[Agent] node=agent messages_in=N
[Agent.tool] search_movies ...
[Agent] done steps=N
```

## Git Hygiene

[.gitignore](.gitignore) ignores:

- `.env` and local env variants, while keeping `.env.example`.
- Python caches and virtual environments.
- test, lint, and coverage caches.
- build artifacts.
- logs such as `app.log`.
- generated Chroma data under `data/vector_store/`.
- local editor and OS files.

The source dataset under `data/llm_redial/` remains trackable.

## Future Work

- Persist session memory in Redis or SQLite.
- Add a reset/history endpoint.
- Add structured recommendation output instead of regex parsing.
- Enrich the catalog with TMDB or IMDb metadata.
- Reintroduce year/rating filters after metadata enrichment.
- Add evaluation on held-out LLM-Redial conversations.
- Add thumbs up/down feedback for online user profiles.
