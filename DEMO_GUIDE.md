# Movie CRS Demo Guide

Use this document as your interview walkthrough script. The goal is to show the
flow from a user question entering the recommender API to the final reply.

## 1. Demo Goal

One-sentence opening:

> This project is a conversational movie recommender built on the LLM-Redial
> dataset. It supports two approaches: a RAG recommender and a LangGraph
> tool-calling agent, both exposed through the same FastAPI endpoint.

What to demonstrate:

- API receives a user query.
- Server resolves conversation memory by `user_id`.
- System chooses `rag` or `agent`.
- RAG path retrieves movies first, then calls the LLM.
- Agent path lets the LLM call tools when needed.
- Response is returned as natural text plus parsed recommendations.

Fast spoken flow:

```text
Request comes to /recommend -> schema validation -> get user history from
server memory -> select RAG or Agent -> retrieve/use tools -> build prompt ->
OpenAI returns answer -> clean/parse recommendations -> append memory -> JSON
response.
```

## 2. Files To Keep Open

Open these files before the interview:

1. [app/main.py](app/main.py)
   - FastAPI app.
   - Startup lifecycle.
   - `/recommend` endpoint.
   - Conversation memory.

2. [app/schemas.py](app/schemas.py)
   - Request and response models.
   - `RecommendationRequest`.
   - `RecommendationResponse`.

3. [models/rag/recommender.py](models/rag/recommender.py)
   - RAG flow coordinator.
   - User profile block.
   - Filter extraction.
   - Retrieval.
   - Prompt formatting.
   - OpenAI call.

4. [models/agent/recommender.py](models/agent/recommender.py)
   - Agent flow coordinator.
   - Drives the structured LangGraph pipeline.

5. [models/agent/graph.py](models/agent/graph.py)
   - Pipeline: `classify_intent` → `{chat_reply | extract_preferences → retrieve → rank_score → explain}`.
   - Conditional routing off the intent label.

6. [models/agent/nodes.py](models/agent/nodes.py)
   - Node closures: classify_intent, extract_preferences, retrieve, rank_score, explain, chat_reply.
   - [models/agent/intent.py](models/agent/intent.py): LLM intent classifier (temperature=0).
   - [models/agent/ranking.py](models/agent/ranking.py): heuristic scorer (similarity + history affinity + recency − negative signal).

7. [utils/vector_store.py](utils/vector_store.py)
   - ChromaDB and SentenceTransformer retrieval.

8. [data/loader.py](data/loader.py)
   - Dataset loading.
   - User history lookup.

9. [prompts/templates.py](prompts/templates.py)
   - RAG and Agent prompts.
   - Conversation modes.

10. [conversation_live_api_transcript.txt](conversation_live_api_transcript.txt)
    - Actual live API transcript from the demo test run.

## 3. Start The Service

Use Docker for the demo:

```bash
docker compose up -d --build --force-recreate
```

Check health:

```bash
curl http://localhost:8000/health
```

Expected shape:

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

Optional log tail:

```bash
docker logs -f movie-crs-api
```

## 4. Live Demo Script

### Step A: Show Startup Flow

Open [app/main.py](app/main.py), function `lifespan`.

Explain:

1. The service loads LLM-Redial files from `data/llm_redial`.
2. `MovieDataLoader.load_movies()` creates the movie catalog.
3. `MovieVectorStore` loads or creates a ChromaDB collection.
4. If the vector store is empty, movies are indexed.
5. A few real conversation snippets are loaded for prompt style.
6. Both recommenders are created:
   - `RAGRecommender`
   - `AgentRecommender`
7. `app.state.conversations` is initialized as in-memory session storage.

Key line to point at:

```python
app.state.conversations: Dict[str, List[Message]] = {}
```

Say:

> This is how multi-turn memory works in the current demo version. In
> production, I would replace this with Redis or a database.

### Step B: Show Request Schema

Open [app/schemas.py](app/schemas.py).

Explain request fields:

- `query`: current user message.
- `user_id`: required; used for memory and optional dataset profile.
- `model_type`: `rag` or `agent`.
- `max_recommendations`: capped from 1 to 10.

Explain response fields:

- `response_text`: final assistant message.
- `recommendations`: parsed list of titles from the assistant text.
- `model_used`: selected pipeline.
- `processing_time_ms`: API latency.

### Step C: Run RAG Request

Run:

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Can you recommend something exciting but not too silly? I liked The Bourne Identity.\",\"user_id\":\"demo-flow-rag\",\"model_type\":\"rag\"}"
```

Then show the flow in [app/main.py](app/main.py):

1. FastAPI validates the request.
2. The handler chooses the recommender:

```python
if request.model_type == "rag":
    recommender = rag_recommender
```

3. The handler resolves memory:

```python
history = conversations.setdefault(request.user_id, [])
```

4. It calls:

```python
response_text = await recommender.generate_recommendation(...)
```

Now open [models/rag/recommender.py](models/rag/recommender.py).

Explain RAG internals:

1. Build known-user profile if `user_id` exists in dataset.
2. Add recent likes to retrieval query as a boost.
3. Extract filters like `horror`, `comedy`, `thriller`.
4. Retrieve movie candidates from ChromaDB.
5. Format retrieved movies into prompt context.
6. Call OpenAI.
7. Strip accidental mode labels.
8. Return text.

Open [utils/vector_store.py](utils/vector_store.py).

Explain:

> Without enrichment, each movie is represented by title and inferred genres.
> When `data/llm_redial/tmdb_enriched_movies.json` exists, the loader adds real
> TMDB genres, overview, keywords, director, cast, release date, and year. The
> vector store embeds those richer fields, and ChromaDB returns nearest
> neighbors for the user query.

Open [prompts/templates.py](prompts/templates.py).

Explain:

> The prompt has modes. It can chit-chat, recommend, clarify, or close the
> conversation. It only uses `**Title** - reason` when making recommendations.

Open [models/rag/parser.py](models/rag/parser.py).

Explain:

> After generation, I parse only markdown-bold recommendation lines. This avoids
> treating normal prose or hyphenated words as movie recommendations.

### Step D: Show Multi-Turn Memory

Run a second request with the same `user_id`:

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Can you make it darker but still not cheap horror?\",\"user_id\":\"demo-flow-rag\",\"model_type\":\"rag\"}"
```

Explain:

> The client did not send history. The server reused the conversation list
> stored under `demo-flow-rag`, so the second answer can use the previous turn.

Point at:

```python
history.append(Message(role="user", content=request.query))
history.append(Message(role="assistant", content=response_text))
```

### Step E: Run Agent Request

Run:

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"I want something suspenseful, but avoid Apollo 18 or Primer.\",\"user_id\":\"AQP1VPK16SVWM\",\"model_type\":\"agent\"}"
```

Open [models/agent/recommender.py](models/agent/recommender.py).

Explain:

> The agent path builds messages and invokes a LangGraph graph. Unlike RAG, it
> does not always retrieve first. The LLM decides whether to call tools.

Open [models/agent/graph.py](models/agent/graph.py).

Explain graph flow:

```text
classify_intent -> (recommend) -> extract_preferences -> retrieve -> rank_score -> explain -> END
classify_intent -> (chat | clarify | closing) -> chat_reply -> END
```

Open [models/agent/nodes.py](models/agent/nodes.py).

Explain each node:

- `classify_intent`: LLM (temperature=0) picks one of `recommend | chat | clarify | closing`.
- `extract_preferences`: pulls filters (genre/year/director/actor) and looks up the user's liked/disliked history.
- `retrieve`: semantic search via the vector store with those filters applied.
- `rank_score`: heuristic score from [models/agent/ranking.py](models/agent/ranking.py) — similarity, history affinity, recency, negative signal.
- `explain`: LLM renders the top-ranked candidates as `**Title** - reason`.
- `chat_reply`: LLM-only reply for non-recommend turns; no retrieval, no ranking.

Point out that `AQP1VPK16SVWM` is an old dataset user. `extract_preferences`
picks up their likes/dislikes from the loader automatically.

### Step F: Show Actual Test Transcript

Open [conversation_live_api_transcript.txt](conversation_live_api_transcript.txt).

Say:

> I prepared four live API test conversations: new user with RAG, old user with
> RAG, new user with Agent, and old user with Agent. This file contains only the
> exact response text returned by the live API, formatted as conversations.

Use this file if the live API call is slow during the interview.

## 5. End-To-End Flow Summary

Use this as your verbal summary:

```text
1. User sends POST /recommend.
2. FastAPI validates query, user_id, model_type, and max_recommendations.
3. Server fetches conversation history from app.state.conversations[user_id].
4. If model_type is rag:
   - Build user profile if known.
   - Extract lightweight filters.
   - Search Chroma vector store.
   - Build prompt with history + retrieved catalog movies.
   - Call OpenAI and return response.
5. If model_type is agent:
   - Build messages with history.
   - Run LangGraph.
   - LLM may call tools such as search_movies or get_user_history.
   - Return final assistant response.
6. Server appends user and assistant messages to memory.
7. Server parses bold movie titles into structured recommendations.
8. API returns response_text, recommendations, model_used, and latency.
```

## 6. Why Two Designs?

RAG:

- Simpler and predictable.
- Always retrieves before generation.
- Lower latency.
- Good for direct recommendation requests.

Agent:

- More flexible.
- Can choose tools based on intent.
- Better for multi-constraint or exploratory requests.
- Higher and more variable latency.

Good interview line:

> I included both because they illustrate two common CRS patterns: deterministic
> retrieve-then-generate and agentic tool use. They share the same dataset and
> prompt surface, so behavior is easy to compare.

## 7. Important Design Decisions

### Server-side memory

The client only sends `query` and `user_id`; the API stores history internally.

Trade-off:

- Simple for a demo.
- Lost on restart.
- Should become Redis or DB-backed in production.

### Genre-only filters

The source dataset has sparse metadata. Year and rating are not reliable, so
the system does not expose those filters.

Good line:

> I chose not to fake capabilities the data does not support.

### Prompt modes

The assistant should not recommend on every turn. It can:

- chat naturally,
- recommend,
- ask a clarifying question,
- close the conversation.

### Parser safety

Recommendations are parsed only from markdown-bold movie titles:

```text
**Title** - reason
```

This avoids false positives from normal prose.

## 8. Generic GenAI / ML Questions To Prepare

### What is RAG?

RAG means Retrieval-Augmented Generation. Instead of asking the LLM to answer
only from its parameters, we first retrieve relevant external context and put it
into the prompt. Here, the external context is movie candidates from ChromaDB.

### Why use embeddings?

Embeddings convert text into vectors so semantic similarity can be computed.
For example, "slow-burn horror" can retrieve movies whose text representation
is semantically close even if the exact words differ.

### What is a vector database?

A vector database stores embeddings and supports nearest-neighbor search. This
project uses ChromaDB to store movie embeddings and retrieve similar movies.

### What is the difference between RAG and an agent?

RAG follows a fixed flow: retrieve, prompt, generate.

An agent can decide actions dynamically, such as whether to call a search tool,
fetch user history, or answer directly.

### What is LangGraph?

LangGraph is a framework for building graph-based LLM workflows. In this
project, it cycles between an LLM node and a tool node until the LLM returns a
final answer.

### What is prompt engineering here?

The prompt defines behavior modes:

- chit-chat,
- recommendation,
- clarification,
- closing.

It also constrains recommendations to catalog movies and asks for a parseable
`**Title** - reason` format.

### What are hallucinations and how do you reduce them?

Hallucinations are unsupported model outputs. This project reduces them by:

- retrieving catalog candidates,
- telling the model to use catalog movies,
- parsing only structured recommendation lines,
- keeping dataset limitations explicit.

More robust production fixes would include strict structured outputs and
post-generation validation against the catalog.

### Why use `user_id`?

It supports two things:

- server-side multi-turn memory,
- dataset-side personalization for known LLM-Redial users.

### How would you improve this for production?

- Persistent memory store.
- Per-user concurrency control.
- Structured JSON output from the LLM.
- Validate recommended titles against the catalog.
- Add TMDB/IMDb metadata.
- Add offline evaluation.
- Add online feedback such as thumbs up/down.

### How would you evaluate this system?

Offline:

- Recall@K against held-out `rec_item` labels.
- Diversity and novelty metrics.
- Response quality checks.

Online:

- click/watch/save rate,
- user satisfaction,
- follow-up acceptance,
- thumbs up/down feedback.

## 9. Known Limitations To Mention Honestly

- The catalog metadata is sparse.
- Genre detection is keyword-based.
- Session memory is in-process.
- Recommendation parsing is heuristic.
- Agent path has higher latency.
- There is no auth/rate limiting.
- Docker currently includes dataset files in the image.

Good closing line:

> The current implementation is demo-ready and intentionally transparent. The
> next step would be hardening: durable memory, structured outputs, catalog
> validation, richer metadata, and evaluation.

## 10. Quick Backup Commands

Health:

```bash
curl http://localhost:8000/health
```

RAG request:

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"suggest a suspenseful thriller\",\"user_id\":\"demo-backup-rag\",\"model_type\":\"rag\"}"
```

Agent request:

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"recommend something eerie but not confusing\",\"user_id\":\"demo-backup-agent\",\"model_type\":\"agent\"}"
```

Known old user request:

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"recommend something suspenseful but avoid Apollo 18\",\"user_id\":\"AQP1VPK16SVWM\",\"model_type\":\"agent\"}"
```

Logs:

```bash
docker logs -f movie-crs-api
```
