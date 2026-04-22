# Conversational Movie Recommender System: Project Report

## 1. Problem Statement
Traditional recommendation systems (like collaborative filtering or matrix factorization) excel at identifying patterns based on historical user behavior but often fail to capture nuanced, real-time user intents. Users often have complex, multi-faceted requests that are best expressed in natural language (e.g., *"I want a mind-bending sci-fi movie from the 90s that isn't too long"*). 

The objective of this project is to build an **Agentic Conversational Recommender System (CRS)**. This system must be capable of engaging users in a dynamic dialogue, understanding varied and ambiguous natural language queries, reasoning over structured and unstructured metadata, and generating accurate, explainable movie recommendations. 

**Key Requirements:**
- **Context-Aware Recommendations:** Ability to understand narrative queries and find semantically similar movies.
- **Factual Accuracy:** Ability to execute specific database lookups (e.g., querying by a specific director or actor).
- **Explainability:** Providing real-time insight into the AI's reasoning process so users understand *why* a movie is recommended.
- **Interactive UI:** A real-time chat interface capable of streaming responses.

---

## 2. Data Analysis & Preparation

The foundation of the recommendation engine relies heavily on high-quality metadata because LLMs hallucinate unless strongly grounded with factual contexts. The initial baseline utilized the widely recognized **ReDial (Recommendation Dialogues)** dataset.

### The ReDial Dataset Analysis & Addressing Sparsity
The raw LLM-ReDial dataset consists of:
- **`item_map.json`**: An index of **9,687 movie titles**.
- **`user_ids.json` / `final_data.jsonl`**: Real conversational histories across **3,131 users**.
- **`Conversation.txt`**: Over **232,000 lines** of transcribed multi-turn recommendation dialogues.

**Data Limitations:**
Analysis of the raw ReDial `item_map.json` revealed severe metadata sparsity. The catalog primarily held raw string titles containing retail junk tags (e.g., `"The Godfather [Blu-ray]"`, `"Outland VHS"`, `"Special Edition"`). Crucial recommendation fields such as *release year*, *rating*, *director*, *cast*, and thematic *overviews* were missing entirely. Relying solely on inferred genres from noisy titles resulted in poor vector embeddings and heavily hallucinated recommendation rationales from the LLM. 

### The TMDB Enrichment Pipeline
To solve the sparsity problem, a robust data-enrichment pipeline (`scripts/enrich_tmdb_catalog.py`) was developed to aggressively supplement the LLM-ReDial dataset by interacting with the **TMDB (The Movie Database) API**:
1. **Title Cleaning & Fallbacks:** A Regex-based cleaning layer aggressively strips formatting suffixes (`VHS`, `DVD`, `Blu-ray`, `Director's Cut`) from the 9.6k source titles before searching. A fallback mechanism preserves the raw title search if the cleaned search fails.
2. **Metadata Acquisition:** The pipeline queries the TMDB API to pull `overview` (synopsis), `cast` (up to 5 top-billed actors), `director`, `year/release_date`, and granular TMDB `keywords` & `genres`. 
3. **Structured Storage:** Extracted data is merged into a secondary mapping file (`data/llm_redial/tmdb_enriched_movies.json`), keyed securely by the original Redial element IDs to retain downstream mapping consistency.

### Embedding & Vectorization Strategy
Once the dataset was enriched with the TMDB metadata, the movie records transitioned from sparse keyword tokens into rich textual representations. 
- The textual representations—combining `Title`, `Year`, `Genres`, `Director`, `Cast`, `Keywords`, and the plot `Overview`—were concatenated and chunked.
- These chunks were then converted into dense vector representations using robust embedding models (`sentence-transformers/all-MiniLM-L6-v2`) and indexed into **ChromaDB**. 
- This semantic density enables narrative and conceptual similarity searches, mitigating standard lexical search limitations.

---

## 3. Solution Architecture
The solution was built using a modular, decoupled architecture consisting of a Python-based intelligent backend and a Streamlit frontend. The core AI logic utilizes **LangGraph** to model the reasoning process as a state machine.

### Key Components

#### A. Dual-Mode Routing Pipeline
To handle diverse query types optimally, the system routes queries to one of two specialized pipelines:
1. **RAG Pipeline (Context-Aware):** Ideal for narrative, thematic, or conceptual queries. It retrieves context chunks from a vector database and uses an LLM to synthesize a conversational response.
2. **Agentic Pipeline (Tool-Based):** Ideal for factual or constraint-based queries. It uses a LangGraph ReAct agent connected to external tools (e.g., metadata lookups) to iteratively gather necessary facts before responding.

#### B. Query Rewriting & Reranking
- **Query Rewriter:** Raw user inputs are often noisy. An intermediate LLM step rewrites the user query to maximize retrieval effectiveness.
- **Reranker:** The initial `top-k` documents retrieved by the vector database are scored and re-ordered by a dedicated Reranker model, ensuring that the most contextually relevant movies are heavily prioritized.

#### C. System Observability & Streaming
- Real-time chunk streaming guarantees an interactive user experience with minimal perceived latency.
- Internal reasoning steps are captured and streamed to the UI in parallel. This builds user trust, as they can monitor the database retrievals and tool invocations the agent performs behind the scenes.

---

## 4. Workflows & Examples

### Example 1: Thematic Search (RAG Pipeline)
**User Input:** *"Can you suggest a visually stunning movie about space travel where time moves differently?"*
- **System Reasoning:** Identifies thematic concepts: 'visually stunning', 'space travel', 'time dilation'. Routes to RAG pipeline.
- **Retrieval:** Vector search identifies *Interstellar*, *Arrival*, and *Gravity*.
- **Reranking:** *Interstellar* is boosted to the top rank due to the exact match with the "time moves differently" concept.
- **Response:** System recommends *Interstellar*, explaining the narrative match, alongside *Gravity*.

### Example 2: Factual Search (Agent Pipeline)
**User Input:** *"What are the best movies directed by Christopher Nolan before 2010?"*
- **System Reasoning:** Detects specific constraints (`Director: Christopher Nolan`, `Release Year < 2010`). Routes to Agent pipeline.
- **Tool Invocation:** Agent queries the database with parameters `{'director': 'Christopher Nolan', 'max_year': 2009}`.
- **Tool Output:** Returns metadata for *The Dark Knight*, *The Prestige*, *Batman Begins*, *Memento*, etc.
- **Response:** Agent naturalizes the data into a conversational response and displays the corresponding movie cards.

### Example 3: Contextual Memory
**User Input (Turn 1):** *"I like action movies with Tom Cruise."* (System recommends *Mission: Impossible*, *Edge of Tomorrow*).
**User Input (Turn 2):** *"Find me one of those that has a sci-fi twist."* 
- **System Reasoning:** Resolves the coreference "those" using conversation history. Focuses search on Tom Cruise action movies with sci-fi elements.
- **Response:** Suggests *Edge of Tomorrow* and *Minority Report*.

---

## 5. Continuity and Personalization (Old vs. New Users)

The system intelligently distinguishes between historically known users (from the ReDial dataset) and completely new guests, dynamically adjusting its generation strategy to maximize personalization while gracefully handling "cold start" scenarios.

### Existing ReDial Users (Rich History)
When a known dataset user (e.g., `A30Q8X8B1S3GGT`) connects, the `MovieDataLoader` automatically extracts their historical profile from `final_data.jsonl` and transcribed dialogues in `Conversation.txt`. 
- **Implicit RAG Personalization:** In the RAG pipeline, the system prompt is automatically prepended with a personalized constraint block (e.g., *"This user historically liked [Movie A, Movie B] and disliked [Movie C]"*). This forces the LLM to steer new recommendations away from previously rejected concepts.
- **Explicit Agent Deep-Dives:** The LangGraph agent possesses a dedicated `search_user_history` tool. If the user asks, *"What did you recommend to me last time?"*, the tool retrieves exact transcript chunks of their past interactions, allowing the agent to reason over past relationships and seamlessly resume the conversation.

### New/Guest Users (Cold Start & Session Memory)
For entirely new users interacting via the Streamlit UI, historical data does not exist, requiring robust short-term state management.
- **Server-Side Session State:** The FastAPI backend maintains an active `app.state.conversations` dictionary bound to the custom `user_id`. 
- **Dialogue Continuity:** Each chat turn is appended to this short-term memory block and forwarded on subsequent requests. This allows the LLM to effortlessly resolve coreferences in conversational chains. For instance, if a guest follows up with *"Is there another one like that?"*, the system accurately remembers the preceding context window for both RAG and Agent evaluation scopes.

---

## 6. Conclusion
The delivered application successfully combines modern Agent and RAG paradigms to create an intelligent, responsive, and highly capable movie recommender system. By deeply integrating query rewriting, LLM reasoning, semantic search, and an interactive Streamlit layer, the resulting application solves the limitations of traditional, non-conversational recommendation engines.
