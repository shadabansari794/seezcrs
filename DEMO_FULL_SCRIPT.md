# Full Demo Script — 15–20 minutes

> A second-by-second walkthrough you can read like a teleprompter.
> **[Square brackets]** = stage directions. **"Quoted text"** = say it close to verbatim.
> Time-boxed sections so you know where you are and can compress on the fly.
> **Performance & scalability is woven through every section** — this build was designed to be productionizable, not just demoable.

---

## Pre-Demo Checklist (T-5 minutes)

Run **before** the call, not during it.

```bash
# 1. API healthy?
curl -s http://localhost:8000/health | python -m json.tool
# expect: total_movies: 9687
```

- [ ] Streamlit open at `http://localhost:8501`, sidebar **User ID = `demo_user`**
- [ ] One terminal open: `docker logs -f movie-crs-api 2>&1 | grep -E "Tool\]|intent|filters|Reranker"`
- [ ] One file editor open with these tabs ready (in order): `models/agent/graph.py`, `models/agent/nodes.py`, `utils/vector_store.py`, `models/agent/tools.py`
- [ ] `docs/graph_outer.png` open in image viewer
- [ ] `DEMO_TEST_CASES.md` open as your cheat sheet
- [ ] Browser zoom on Streamlit set so reasoning sidebar is fully readable
- [ ] Notifications muted
- [ ] Backup: 30-second screen recording of test 6 saved on desktop

If any check fails, fix before screen-share. Don't debug live.

---

## Section 1 — Opening Hook (1.5 min)

**[Stage: share screen showing Streamlit UI, do NOT type anything yet]**

**Say:**
> "Thanks for having me. I want to walk you through a conversational movie recommender I built end-to-end — backend, **two pipelines (RAG and Agent)**, retrieval, real-time UI, and a design that's meant to scale.
>
> The problem in one sentence: **classic recommenders match on IDs, not intent.** A user who says *'something mind-bending like Nolan but not too heavy'* gets matched to recent clicks, not to that actual sentence. So I built something that takes the sentence seriously — and built it with production traffic in mind from day one.
>
> In the next 18 minutes I'll show you five things, in order:
> 1. What it looks like to talk to it,
> 2. The architecture under the hood,
> 3. A few of the engineering decisions I'd defend,
> 4. **How it's designed to scale to millions of users**, and
> 5. What I'd build next."

**[Pause briefly. The "scale" promise sets up your differentiator.]**

---

## Section 2 — The Data Story (1.5 min)

**[Stage: switch to a terminal showing `ls data/llm_redial/`]**

**Say:**
> "The catalog is the LLM-Redial dataset — 9,687 movie titles, 3,131 user dialogue records. Raw titles are noise: *'The Godfather [Blu-ray]'*, *'Outland VHS'*. Embedding that gives you nothing.
>
> So I wrote an offline enrichment pipeline that strips retail suffixes, hits the TMDB API in parallel — 16 worker threads, full catalog in a few minutes — and writes the result to a JSON file keyed on the original ReDial ID. 7,200 of 9,687 enriched cleanly.
>
> **The key design choice here is amortization.** TMDB calls happen *once*, offline, never on the hot path. At a million users per day, that's a million queries that never touch a third-party API for catalog data. The enrichment artifact is committed; cold starts open it and merge it into memory in milliseconds."

**[Optional: open `data/llm_redial/tmdb_enriched_movies.json`, scroll briefly]**

**Say:**
> "Each merged record carries an overview, top-5 cast, director, genres, and a TMDB keywords list — that keyword list is the secret weapon for embedding semantics like *'cyberpunk dystopia'*."

---

## Section 3 — Architecture Overview (2.5 min)

**[Stage: open `docs/graph_outer.png` — the compiled LangGraph PNG]**

**Say:**
> "The system actually has **two pipelines, not one** — both pickable from the UI. They share most of the infrastructure but differ in how the final response is generated.
>
> **Pipeline 1 — RAG (Context-Aware).** Classic retrieval-augmented generation: rewrite the query, classify intent, extract filters, retrieve from Chroma, rerank, then a single LLM call with one big prompt that contains MODE A/B/C/D logic — the model itself picks chit-chat, recommendation, clarify, or closing inside the prompt. Best for narrative, persona-driven responses. Lower orchestration overhead.
>
> **Pipeline 2 — Agent.** Same retrieval and ranking infrastructure, but the response is built by a LangGraph state machine with discrete nodes per stage and a ReAct sub-agent for research-intent queries that need external tools. More observable, more powerful, slightly higher per-turn cost.
>
> This PNG is the Agent pipeline — the actual compiled graph, generated from the running code. Top-down: enter at `rewrite_query`, `classify_intent` decides one of three paths. **Recommend** goes through prefs → retrieve → rank → explain. **Research** goes into the ReAct sub-agent expanded inline. **Chat / clarify / closing** short-circuit to a single reply node.
>
> That short-circuit matters for scale: **three of five intents skip retrieval entirely.** Telemetry would show maybe 30-40% of turns short-circuit — huge cost saving.
>
> Both pipelines are async — FastAPI plus async OpenAI client plus Chroma offloaded to a thread pool. One worker handles hundreds of concurrent requests because CPU mostly waits on I/O. The graph object is compiled once at startup and reused. The streaming endpoint emits two interleaved channels — `updates` (node-completion events for the reasoning timeline) and `messages` (LLM token chunks filtered to terminal nodes). Same compiled graph drives sync and streaming — no duplicated routing logic."

---

## Section 4 — Live Demo (8 min)

**[Stage: switch to Streamlit. Sidebar User ID = `demo_user`.]**

> "I'll show both pipelines. Two minutes on RAG, then six minutes on Agent. Same query infrastructure, different response strategies."

---

### Part 1 — RAG Pipeline (2 min)

**[Stage: set sidebar Mode = `RAG (Context-Aware)`]**

#### RAG Test 1 — Narrative recommendation (1 min)

**[Type:]** `I'm in the mood for a feel-good comedy from the 90s`

**While streaming, say:**
> "RAG mode is the classical pattern — one big prompt, one LLM call after retrieval. The prompt has MODE A/B/C/D inside it, and the LLM picks the right mode based on the query. For this it picks MODE B — recommendation."

**[Click "Extracted filters" in sidebar]**

> "Filters extracted same way as Agent — regex, no LLM. `genre: comedy`, `year_range: [1990, 1999]`. The retrieval, reranking, user profile injection — all shared infrastructure between RAG and Agent. The only thing that differs is what happens after ranking."

**[Click "Generated response"]**

> "And here's the response — narrative, persona-driven, two flowing sentences, titles in bold. No bullet lists, no robotic openers. The single big prompt enforces all of that."

#### RAG Test 2 — Mode-switching inside the prompt (1 min)

**[Type:]** `Thanks, that was helpful!`

**Say:**
> "Same RAG mode, but watch — the prompt's MODE D logic kicks in for closings."

**[Wait for stream — should be a single warm one-liner]**

> "One sentence, no recommendations. **The mode switching happens inside the prompt itself**, not in the application code — that's the design choice for RAG. Lower orchestration overhead, but harder to debug if the LLM picks the wrong mode. Which brings us to Agent mode, where modes are explicit graph nodes."

---

### Part 2 — Agent Pipeline (6 min)

**[Stage: switch sidebar Mode = `Agent (Tool-based)`]**

> "Same retrieval, same reranker, same user profile. But now intent classification, response generation, and tool use are explicit graph nodes — observable in the sidebar, swappable as code."

#### Agent Test A — Filter extraction + retrieval (1.5 min)

**[Type:]** `I want a dark cerebral sci-fi thriller from the 90s`

**While streaming, expand "Extracted filters":**

> "Genre and year-range, both extracted via regex — **no LLM call**. Microseconds, deterministic, free. At a million users per day this is the difference between a 50ms request and a 500ms request, just on filter extraction alone."

**[Expand "Retrieved candidates"]**

> "Chroma over-fetched the top-20 by similarity, then Python filtered to genre + decade. We over-fetch 4× because metadata arrays are stored comma-joined — post-filtering in Python is more reliable than DB-side filtering, and it's a single Chroma round trip instead of multiple narrowing queries."

**[Expand "Ranked candidates"]**

> "Keyword-overlap reranker — 5 lines of Python, runs in microseconds. **No cross-encoder, no per-candidate inference.** At top-k=20 that's 20 inference calls saved per request. Chroma's dense embeddings already did the heavy lifting; this is just a tiebreaker."

#### Agent Test B — Pronoun follow-up (45s)

**[Type:]** `Something lighter than that`

**[Expand "Rewrote query"]**

> "*'Something lighter than that'* expanded into a standalone query referencing the previous turn. Without this, retrieval would bomb. The rewriter uses the utility LLM — `gpt-4o-mini`, low temperature, low token budget. About 10× cheaper than the main model. Same rewriter is shared between RAG and Agent paths."

#### Agent Test C — Director filter (45s)

**[Type:]** `Any 90s movies directed by Quentin Tarantino?`

**[Expand "Extracted filters"]**

> "Director matching uses a 4,124-name set built once at startup from the merged catalog. Longest-match grep against the lowercased query. O(N) but N is tiny and the set is in memory — no DB call. Same pattern works for actors against the 16,213-name cast set."

#### Agent Test D — Chat short-circuit (45s)

**[Type:]** `What is The Godfather about and who is in it?`

**[Click "Classified intent"]**

> "Classified as `chat`, not `research`. Two steps total — classify, reply. **No retrieval, no tools, no reranker.** Production-wise this is huge: factual questions about famous films are common, and routing them around the expensive pipeline is what keeps the average request cheap."

#### Agent Test E — Research with both tools (the showpiece, 1.5 min)

**[Type:]** `What is the latest Christopher Nolan movie coming out and what is it about?`

**[Switch to docker logs]**

> "Watch the logs."

**[Point at the logs:]**
> "There — `Searching Web for: latest Christopher Nolan movie release 2026` and `Searching TMDB for: Oppenheimer`. **Both tools fired in parallel** — that's a single LangGraph turn doing two tool calls concurrently, not sequentially. Cuts research-path latency roughly in half."

**[Click "Searched the web" in sidebar]**

> "Response synthesizes both sources, names *The Odyssey*, July 17, 2026. The agent picked both tools based on the docstrings — primary routing signal, prompt is secondary."

#### Agent Test F — Date-aware research (45s)

**[Type:]** `Any new movies releasing next month?`

**[Switch to logs]**

> "Search query should contain `May 2026`, not `November 2023`. Today's date is injected into the research prompt at every invocation — without it, the LLM defaults to its training cutoff. Cheap fix, easy regression to introduce."

#### Agent Test G — Closing short-circuit (15s)

**[Type:]** `Thanks, that's enough!`

> "Closing intent. Two steps in the sidebar — classified, replied. Zero retrieval. This is what 30% of production traffic looks like: cheap, fast, done."

---

## Section 5 — Performance & Scalability (3.5 min) — **the centerpiece**

**[Switch to a clean terminal or just stay on Streamlit]**

**Say:**
> "Now I want to walk you through the scale story explicitly, because this was a design constraint, not an afterthought."

### A. What's already production-friendly (1.5 min)

**Say:**
> "Seven things in the current codebase support productionization to millions of users:
>
> **Zero — shared infrastructure across pipelines.** Both RAG and Agent share the same query rewriter, intent classifier, filter extractor, vector store, reranker, and user profile builder. One investment in retrieval quality benefits both modes. Adding a third pipeline tomorrow doesn't double the surface area.
>
> **One — async everywhere.** FastAPI, async OpenAI client, Chroma calls offloaded via `loop.run_in_executor` so they don't block the event loop. One Python worker can handle hundreds of concurrent requests because the CPU is mostly waiting on I/O.
>
> **Two — split LLM tiers.** `gpt-4o-mini` for utility tasks (intent classification, query rewrite) at temperature 0.1, max 300 tokens. `gpt-4o` for user-facing prose with streaming. Roughly **10× cost ratio**. At a million daily turns, the utility split alone saves thousands of dollars.
>
> **Three — intent-based short-circuiting.** Three of five intents skip retrieval entirely. We don't pay Chroma + reranker + main LLM cost for a greeting.
>
> **Four — single graph compilation.** The LangGraph object is compiled once at startup and reused. Per-request cost is just running the graph, not building it.
>
> **Five — amortized expensive operations.** Catalog embedding happens once at index time, persisted to a Docker volume. TMDB enrichment is fully offline. The user query is the only thing embedded at request time — that's one OpenAI embedding call, ~50ms.
>
> **Six — connection pooling and persistent clients.** TMDB calls go through an `httpx.Client` with 32 keepalive connections, retries on 429/5xx. Chroma uses a persistent client that opens once. No per-request connection setup."

### B. Per-request latency budget (45s)

**Say:**
> "Back-of-envelope p50 latency by intent:
>
> | Intent | Path | p50 (rough) |
> |---|---|---|
> | chat / closing | classify + reply | ~1.5–2 s |
> | clarify | classify + reply | ~1.5–2 s |
> | recommend | full RAG pipeline | ~3–5 s |
> | research | rewrite + classify + ReAct (1–2 tool calls) | ~6–10 s |
>
> The time-to-first-token matters more than total — we stream tokens, so a 5-second response feels like a 0.5-second response. **Perceived latency is what users measure, not total wall-clock.**"

### C. What breaks first at scale, and how I'd swap it (1 min)

**Say:**
> "Honestly, at maybe 1,000 RPS sustained you'd hit four bottlenecks in this order:
>
> **First — ChromaDB single-instance.** Chroma's persistent client is great at 10k docs and one machine. Beyond that, swap to **Qdrant or Pinecone** with replicas. The interface is similar; one file changes.
>
> **Second — in-process session dict.** `app.state.conversations` is a Python dict — fine for one container, useless across replicas. Swap for **Redis with a per-user TTL key**. One function changes in `app/main.py`.
>
> **Third — OpenAI rate limits.** At scale you'd queue requests against multiple API keys, or shift to **Bedrock-hosted Claude / Azure-hosted GPT** for higher dedicated quota.
>
> **Fourth — single FastAPI process.** Container becomes the unit of horizontal scale: **k8s with HPA on CPU + custom metric (queue depth)**. The API is already stateless once sessions move to Redis."

### D. Things I'd add for production (45s)

**Say:**
> "Things I deliberately left out for the demo but would add for production:
>
> - **Prompt caching** — both OpenAI and Anthropic support cached system prompts; would cut per-request token cost meaningfully on the explain prompt which has a long stable preamble.
> - **Per-user rate limiting** — Redis token bucket, ten reqs/min per user.
> - **Circuit breakers around tools** — Tavily and TMDB get timeouts and fail-open paths so a slow tool doesn't tank a request.
> - **Result-level caching for hot queries** — same query string + same user_id → cached response for 5 minutes. CDN-style.
> - **Cost tracking** — log token counts per request, per intent, per user. Without this you can't tune anything.
> - **Observability stack** — Prometheus metrics for retrieval hit rate, tool call frequency, intent distribution, p50/p95/p99 by intent. The reasoning sidebar is great for one user; metrics are what you watch in production."

---

## Section 6 — Code Tour (2 min)

**[Switch to the editor. Open `models/agent/graph.py` first.]**

**Say:**
> "Four files, thirty seconds each. Two for the Agent pipeline, one for RAG, one for shared infra."

### File 1 — `models/agent/graph.py`

**[Scroll through the ~30-line file]**

> "Entire Agent graph. Eight `add_node` calls, one conditional edge. Routing as code, not as prompt logic."

### File 2 — `models/agent/nodes.py`

**[Scroll to `extract_preferences_node` and `retrieve_node`]**

> "Each node is a closure over shared dependencies. `extract_preferences` calls regex filter extraction and the user profile builder, returns a partial state dict. LangGraph merges it. Composable, testable."

### File 3 — `models/rag/recommender.py`, `stream_recommendation`

**[Scroll to ~line 158]**

> "RAG pipeline as a single async generator. Same rewrite + classify + filters + retrieval + rerank from shared modules — but the response is one `AsyncOpenAI` streaming call against one big prompt. No graph, no nodes. Half the lines, narrower behavior, but easier to reason about. Dual-mode by design — pick the right tool for the request shape."

### File 4 — `utils/vector_store.py`

**[Scroll to `search` method, ~line 185]**

> "Over-fetch + post-filter, used by **both** pipelines. `top_k * 4` headroom, Python filters for genre / year / director / actor, fallback to unfiltered if everything gets eliminated. The fallback matters in production — '*I have nothing for you*' is a worse failure than '*here's the closest match.*'"

---

## Section 7 — Challenges & Decisions (1.5 min)

**Say:**
> "Three engineering moments worth flagging.
>
> **Catalog sparsity.** Embedding *'Outland VHS'* gets you nothing. The fix wasn't a smarter prompt or a bigger model — it was the offline enrichment pipeline. Boring solution, biggest impact.
>
> **The LLM thought it was 2023.** *'Next month'* searched for November 2023. Fix was injecting `today=date.today().isoformat()` into the research prompt at every invocation, with an explicit rule. Cheap, effective, easy regression to test for.
>
> **Streaming.** I initially used `astream_events(v2)` which doesn't propagate token events from nested ReAct sub-agents. Fix was switching to `graph.astream(stream_mode=['updates','messages'])` — same graph, different iteration contract. Lesson: when the framework gives you two ways, the lower-level one is usually more reliable."

---

## Section 8 — What's Next + Close (1 min)

**Say:**
> "If I had another two days, four things would go on top.
>
> **One — evaluation loop.** Recall@K against held-out ReDial conversations. Right now I trust the demo; a metric loop lets me iterate on prompts without guessing.
>
> **Two — per-user past-conversation retrieval.** Tiny vector index per user over their prior chat snippets. Adds richness without uncanny-valley risk.
>
> **Three — Redis sessions + k8s deploy.** The architecture is already stateless aside from the session dict; this is the unblocker for horizontal scale.
>
> **Four — observability stack.** Prometheus + Grafana on retrieval hit rate, tool latency, intent mix, cost per request. The reasoning sidebar is great for one user; metrics are what you watch in production.
>
> What I'm proud of in this build: it's small, it's complete, it explains itself, and **the pieces compose at scale.** Happy to dig into any layer you want."

**[Stop sharing. Smile. Wait for Q&A.]**

---

## Q&A — Prepared Answers (Reference, do not read)

### Architecture & Design
| Question | Anchor |
|---|---|
| Why two pipelines (RAG + Agent)? | Different request shapes need different machinery. RAG: narrative recommendation with one persona-heavy prompt, lower orchestration cost. Agent: discrete graph nodes, observable, can branch to tools for research-intent. Both share retrieval + rerank + filters + profile — no duplicated infra. |
| When does RAG win, when does Agent win? | RAG wins when the request is "match my taste, narrate it" — low orchestration overhead. Agent wins when intent matters (chat short-circuit, research with tools), when you want explicit observability, or when you'll add more tools/branches over time. |
| Why MODE A/B/C/D inside the RAG prompt? | Lets one LLM call handle chat + recommend + clarify + closing. Lower orchestration overhead than four separate code paths. Trade-off: harder to debug if the LLM picks the wrong mode — which is why Agent mode externalizes the choice as a graph node. |
| Why two LLM models? | Determinism + cost on utility (~10× cheaper); creativity + streaming on user-facing answers. |
| Why fetch_k = top_k × 4? | Headroom for post-filter to still return top_k. Empirical, not magical. |
| Why ChromaDB? | 10k docs, single machine, zero ops. Pinecone/Qdrant when scale demands replicas. |
| Why a keyword reranker not cross-encoder? | Tiebreaker, not re-scorer. Microseconds vs N inference calls. Chroma did the heavy work. |
| Why regex filter extraction? | Determinism, zero token cost, zero p99 risk. LLM extraction is a net loss here. |
| Why ReAct only for research? | Tools are expensive. Recommendation needs retrieval, not tools. Reserves the loop for queries that need fresh data. |
| Why LangGraph? | Same graph for sync + stream; conditional edges with TypedDict state; PNG visualization with sub-graph expansion. |

### Performance & Scale
| Question | Anchor |
|---|---|
| What's your p50 latency? | ~1.5–2 s for chat/closing, ~3–5 s for recommend, ~6–10 s for research. Streaming makes perceived latency much lower. |
| How would you scale to 10M users? | Four swaps in order: Chroma → Qdrant/Pinecone, in-process sessions → Redis, single API key → multi-key + Bedrock, single container → k8s HPA. API is already stateless aside from session dict. |
| What breaks first under load? | Single Chroma instance at ~1k RPS. Then in-process sessions across replicas. Then OpenAI rate limits. |
| What's your cost per query? | Roughly $0.001-0.002 for a recommend turn (mostly the gpt-4o explain call), $0.0001 for chat (utility model only), $0.005-0.01 for research (multiple LLM calls + tools). |
| Where would caching help most? | Three places: prompt cache on the long stable explain preamble, result cache on hot query+user combos for 5 min, embedding cache on common rewrites. |
| How do you handle tool latency? | Today: timeouts via httpx. Production: circuit breakers, fail-open with degraded responses, tool call telemetry. |
| Cold start vs warm start? | Cold: ~10 s — load 9,687 movies + open Chroma. Warm: instant. Chroma index persists on a Docker volume; deploys don't re-embed. |
| Why is the API async? | One worker, many concurrent requests. CPU mostly waits on I/O. Throughput per container scales with concurrency. |

### Engineering Hygiene
| Question | Anchor |
|---|---|
| What about evaluation? | Acknowledged gap. Recall@K against ReDial held-out is the next thing I'd add. |
| What about tests? | Demo script as integration test. Filter extractor is the first unit-test target — pure function, deterministic. |
| What's missing for production? | Prompt caching, Redis sessions, rate limiting, circuit breakers, Prometheus metrics. All deliberate scope cuts, all named in the roadmap. |
| Why a sentinel byte for reasoning? | One stream, one protocol. Doorbell pattern — UI re-reads the log file on each ping. No second connection, no JSON-in-stream parsing. |
| Why personalize through both prompt AND retrieval boost? | Two channels. Prompt tells the LLM what the user likes; retrieval boost biases what comes out of Chroma. Belt and suspenders. |
| Why TMDB *and* IMDb originally? | TMDB for bulk enrichment (real API). IMDb was for runtime tool variety, but IMDb scraping started returning empty — swapped to TMDB for both. |

---

## Time-Box Reference

| Section | Budget | Cut first if behind |
|---|---|---|
| 1. Opening | 1.5 min | — |
| 2. Data story | 1.5 min | trim by 30s |
| 3. Architecture | 2.5 min | trim PNG narration |
| 4a. Live demo (RAG) | 2 min | drop RAG Test 2 (closing) |
| 4b. Live demo (Agent) | 6 min | drop Agent Test G or C |
| 5. **Perf & Scale** | 3.5 min | **never cut — this is the moat** |
| 6. Code tour | 2 min | drop File 3 (RAG) or File 4 |
| 7. Challenges | 1.5 min | drop one story |
| 8. Close | 1 min | — |
| **Total** | **~21 min** | Tight; trim live demo if behind |

If you're at minute 12 and only on section 3, **skip directly to section 4** — and if needed, skip RAG Part 1 in section 4 and start straight on Agent. Live demo (Agent) + perf section are the two non-negotiables.
If something breaks during section 4, **switch to the backup recording, narrate over it**, and continue. Do not debug live.

---

## Two Final Cues

1. **Pause after each section.** Silence feels longer to you than to them. It signals confidence.
2. **When you click into a reasoning step, stop talking for 2 seconds.** Let them read it. Then narrate.

The performance & scalability section is where you separate yourself from candidates who only built a demo. Lean into it.

Good luck.
