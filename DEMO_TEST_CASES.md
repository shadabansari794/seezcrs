# Demo Test Cases

Eleven test cases that walk through every path in the system. Designed for a 5–10 min demo: each case proves something specific. Run them in the order below — each builds on the last.

> **Setup before demo:**
> 1. `docker compose -f docker-compose.yml up -d --build movie-crs-api movie-crs-web`
> 2. Open `http://localhost:8501` in browser (Streamlit UI)
> 3. Open one terminal: `docker logs -f movie-crs-api 2>&1 | grep -E "Tool\]|intent|filters|Reranker"` for live tool/intent logs
> 4. Set sidebar **User ID = `demo_user`** for tests 1–7, then change to a known ReDial id for test 8

---

## Quick Reference

| # | Query | Mode | Intent | What it proves |
|--:|-------|------|--------|----------------|
| 1 | "Recommend me a feel-good comedy" | RAG | recommend | Basic RAG path + persona prompt |
| 2 | "I want a dark cerebral sci-fi thriller from the 90s" | Agent | recommend | Filter extraction (genre + decade) + retrieval |
| 3 | "Something lighter than that" | Agent | recommend | `rewrite_query` resolves "that" |
| 4 | "Any 90s movies directed by Quentin Tarantino?" | Agent | recommend | Director-name filter |
| 5 | "What is The Godfather about and who is in it?" | Agent | chat | Chat short-circuit (LLM uses training knowledge, no tools) |
| 6 | "What is the latest Christopher Nolan movie coming out and what is it about?" | Agent | research | **Both tools fire** — `search_web` + `search_tmdb` |
| 7 | "Any new movies releasing next month?" | Agent | research | `search_web` only, **today's date** injected into prompt |
| 8 | "I'm in the mood for something space-related but with heart" | RAG | recommend | Personalization via dataset-side user history |
| 9 | "recommend something" | Agent | clarify | Clarify short-circuit — asks one question, no retrieval |
| 10 | "Thanks, that's enough for tonight!" | Agent | closing | Closing short-circuit — single warm one-liner |
| 11 | "Hey there!" | Agent | chat | Chit-chat short-circuit |

---

## How to Run

**Streamlit (preferred for demo):** type the query into the chat box at `http://localhost:8501`.

**Curl (for terminal demo / verification):**
```bash
curl -s -X POST http://localhost:8000/recommend/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"<query here>","user_id":"demo_user","model_type":"agent"}' \
  --max-time 120 -o /tmp/out.bin
python -c "print(open('/tmp/out.bin','rb').read().replace(b'\x1e', b'').decode())"
```

(Replace `agent` with `rag` for RAG-mode tests.)

---

## Test 1 — Basic RAG recommendation

**Query:** `Recommend me a feel-good comedy`
**Mode:** `RAG (Context-Aware)`

**Click in sidebar after run — expand:**
- **Searching catalog** → see comedies retrieved.
- **Ranking candidates** → see scores changed the order.
- **Generating response** → preview shows **bolded titles**.

**What to point out:**
> "Mode-A/B/C/D inside the prompt — the LLM picks recommendation mode and returns 1–2 paragraphs with titles in bold. The reasoning timeline shows every step the system took."

---

## Test 2 — Agent with structured filter extraction

**Query:** `I want a dark cerebral sci-fi thriller from the 90s`
**Mode:** `Agent (Tool-based)`

**Expand in sidebar:**
- **Rewrote query** → likely unchanged (already standalone).
- **Classified intent** → `intent: recommend`.
- **Extracted filters** → `{"filters": {"genre": "thriller", "year_range": [1990, 1999]}}`.
- **Retrieved candidates** → titles, all matching genre + decade.
- **Ranked candidates** → scores like `0.58, 0.55, …`.

**What to point out:**
> "Filter extraction is **regex + a known-name set**, not an LLM call — deterministic, zero token cost. ChromaDB over-fetches `top_k * 4` and filters in Python because metadata arrays are stored comma-joined."

---

## Test 3 — Pronoun follow-up (after Test 2)

**Query:** `Something lighter than that`
**Mode:** `Agent`

**Expand:**
- **Rewrote query** → expanded version mentioning the genre/era you just discussed (e.g. *"recommend a lighter movie similar to a 90s sci-fi thriller"*).
- **Retrieved candidates** → not the same as Test 2 — broader / lighter tone.

**What to point out:**
> "*'Something lighter than that'* embedded directly is gibberish to the vector store. The rewrite node uses the last 4 messages as context and produces a standalone query before retrieval. Same rewriter is used by both RAG and Agent paths."

---

## Test 4 — Director filter from known-name set

**Query:** `Any 90s movies directed by Quentin Tarantino?`
**Mode:** `Agent`

**Expand:**
- **Extracted filters** → `{"filters": {"genre":"...", "year_range":[1990,1999], "director":"quentin tarantino"}}`.
- **Retrieved candidates** → Pulp Fiction, Reservoir Dogs, Jackie Brown.

**What to point out:**
> "Director matching uses the `known_directors` set (4,124 names) built from the merged TMDB catalog at startup. We grep the query for the longest known name — *Denis Villeneuve* would also work. No LLM call needed for filter extraction."

---

## Test 5 — Chat short-circuit (factual follow-up)

**Query:** `What is The Godfather about and who is in it?`
**Mode:** `Agent`

**Expand:**
- **Classified intent** → `intent: chat` (NOT `research`).
- **Generated reply** → answer comes from LLM training knowledge.

**Tail logs — should NOT contain `[Tool]`:**
```bash
docker logs --tail 20 movie-crs-api 2>&1 | grep "Tool\]"
```

**What to point out:**
> "Factual questions about well-known movies route to `chat`, not `research` — the LLM answers from training. **No retrieval, no tools, no Chroma call** — saves tokens and latency for trivia. The intent classifier draws this boundary based on time-sensitive markers (*new, upcoming, latest, recent*)."

---

## Test 6 — Research with BOTH tools (the showpiece)

**Query:** `What is the latest Christopher Nolan movie coming out and what is it about?`
**Mode:** `Agent`

**Expand in sidebar:**
- **Classified intent** → `intent: research`.
- **Searched the web** → response preview names the actual upcoming film.

**Tail logs — should see BOTH tools:**
```bash
docker logs --tail 40 movie-crs-api 2>&1 | grep "Tool\]"
# Expect:
#   [Tool] Searching Web for:  latest Christopher Nolan movie release 2026
#   [Tool] Searching TMDB for: Oppenheimer  (or The Odyssey)
```

**What to point out:**
> "Two tools, two specialties. The LLM reads the docstrings (primary routing signal) and the prompt (secondary), then **calls both in parallel** in a ReAct sub-graph. `search_web` for time-sensitive info, `search_tmdb` for established-movie facts. Either could have run alone — the LLM decided to do both."

---

## Test 7 — Research, time-sensitive only

**Query:** `Any new movies releasing next month?`
**Mode:** `Agent`

**Tail logs:**
```bash
docker logs --tail 30 movie-crs-api 2>&1 | grep "Tool\]"
# Expect:
#   [Tool] Searching Web for: movies releasing May 2026 theatrical   ← correct month!
```

**What to point out:**
> "The research prompt injects `today=date.today().isoformat()` at every invocation. Without this, the LLM defaults to its training cutoff and would search for *'November 2023'*. We test this every time — easy regression to introduce."

---

## Test 8 — Personalization via dataset-side history

**Setup:** change sidebar **User ID** to a known ReDial ID, e.g. `A2RGZTOIFUYJG3` (or any from `data/llm_redial/user_ids.json`).

**Query:** `I'm in the mood for something space-related but with heart`
**Mode:** `RAG`

**Expand:**
- **Searching catalog** → see candidates influenced by user's prior likes (look in API logs for `Personalizing RAG for user=...`).
- **Generating response** → may explicitly reference their prior preferences ("*since you liked X earlier…*").

**What to point out:**
> "When a user_id matches the ReDial dataset, we pull their `recent_likes` / `recent_dislikes` / past `rec_items`, inject them as a `USER PROFILE:` block in the prompt, AND concatenate likes onto the embedding query as a `retrieval_boost`. Personalization shapes **both prompt and retrieval**."

Then re-run with **User ID = `guest_user`** — same query, very different recommendations. Side-by-side comparison sells the personalization.

---

## Test 9 — Clarify short-circuit

**Query:** `recommend something`
**Mode:** `Agent`

**Expand:**
- **Classified intent** → `intent: clarify`.
- **Generated reply** → one short clarifying question, no titles.

**What to point out:**
> "*Clarify* and *closing* share the `chat_reply` node — they don't need their own branches because the prompt adapts to the intent variable. Two-line prompts don't deserve two extra graph nodes."

---

## Test 10 — Closing short-circuit

**Query:** `Thanks, that's enough for tonight!`
**Mode:** `Agent`

**Expand:**
- **Classified intent** → `intent: closing`.
- **Generated reply** → single warm one-liner. No retrieval, no Chroma call.

**What to point out:**
> "We don't burn embeddings on goodbye. The graph short-circuits here — sidebar shows `Classified intent` → `Generated reply` and that's it."

---

## Test 11 — Chit-chat opener

**Query:** `Hey there!`
**Mode:** `Agent`

**Expand:**
- **Classified intent** → `intent: chat`.
- **Generated reply** → friendly greeting, no recommendations.

**What to point out:**
> "Same path as test 10. The intent classifier is the *gatekeeper* — three out of five intents skip retrieval entirely."

---

## Closing the Demo — Highlight the Reasoning UI

After all tests, point at the sidebar one more time:

> "Every reasoning step has structured input/output you can click open. Filters, candidate titles with scores, response previews — it's all there. **No more guessing what the agent did**, you can audit every turn."

Click any old step → expand → show the JSON.

---

## Edge-Case Tests (if time allows)

| Query | What it shows |
|-------|---------------|
| `something with Christopher Nolan` (Agent) | Director extraction without genre/year. |
| `like Inception` (Agent, after rec) | Pronoun-free follow-up still rewrites because of *"like"* anchor. |
| `recommend a comedy from the 80s with Eddie Murphy` (Agent) | Three filters at once: genre + decade + actor. |
| Empty string `""` | API rejects with 422 (Pydantic validation). |
| Same query 3× rapid-fire | History keeps growing; assistant references prior turns. |

---

## Failure-Mode Tests (be ready to explain)

| Symptom | Likely cause | What to say |
|---------|--------------|-------------|
| Reasoning sidebar empty | `/app/logs/reasoning.log` not mounted | `docker volume inspect logs-data` |
| `Tool] TMDB search failed: Missing TMDB auth` | Env vars not forwarded | We forward them via `docker-compose.yml`. Check `docker exec movie-crs-api env \| grep TMDB` |
| Streaming response shows literal `\x1e` | Client not stripping sentinels | Streamlit strips them; raw curl needs `tr` or Python decode |
| First request takes 30s+ | Cold ChromaDB load | Mention persistent volume — only happens on cold image |

---

## One-liner Smoke Test (run before every demo)

```bash
curl -s http://localhost:8000/health | python -m json.tool
# expect: total_movies: 9687
```

If that prints `total_movies: 9687`, the system is ready.
