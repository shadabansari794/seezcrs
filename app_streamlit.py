import streamlit as st
import requests
import os
import json
import time
import re
from pathlib import Path

# --- CONFIGURATION ---
API_URL = os.getenv("API_URL", "http://localhost:8000/recommend")
STREAM_URL = API_URL + "/stream"
TRACE_PATH = Path("/app/logs/pipeline_trace.txt")  
REASONING_PATH = Path("/app/logs/reasoning.log")
ENRICHED_DATA_PATH = Path("/app/data/llm_redial/tmdb_enriched_movies.json")

# --- DATA CACHING ---
@st.cache_data
def load_movie_metadata():
    if ENRICHED_DATA_PATH.exists():
        try:
            with open(ENRICHED_DATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both dict and list structures
                movies = data.values() if isinstance(data, dict) else data
                return {m["title"].lower(): m for m in movies if isinstance(m, dict) and "title" in m}
        except Exception as e:
            st.error(f"Error loading metadata: {e}")
    return {}

MOVIE_LOOKUP = load_movie_metadata()

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Movie Expert AI",
    page_icon="🎬",
    layout="wide",
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0b0d11;
        color: #ffffff;
    }
    
    /* Force high visibility on all text */
    p, span, label, div, .stMarkdown, .stChatFloatingInputContainer {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* Sidebar labels and controls */
    section[data-testid="stSidebar"] {
        background-color: #161922 !important;
        border-right: 1px solid #30333d;
    }
    
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
    }

    /* Clear History Button styling with high specificity */
    [data-testid="stSidebar"] div.stButton > button {
        background-color: #ff4b4b !important;
        color: white !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        font-weight: 900 !important;
        width: 100% !important;
        margin-top: 15px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    /* Titles */
    h1, h2, h3 {
        color: #00d2ff !important;
        font-weight: 800 !important;
    }
    h1 { font-size: 2.2rem !important; }
    
    /* Movie Card Styling */
    .movie-card {
        background: #1c202a;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #3d4251;
        margin-bottom: 20px;
        height: 100%;
    }
    .movie-title {
        font-weight: 900;
        color: #00d2ff !important;
        font-size: 1.2rem;
        margin-bottom: 8px;
    }
    
    /* Chat bubbling */
    .stChatMessage {
        background-color: #171b26 !important;
        border: 1px solid #303541 !important;
        margin-bottom: 12px !important;
    }
    
    /* Reasoning Timeline */
    .reasoning-step {
        background: rgba(0, 210, 255, 0.1);
        border-left: 3px solid #00d2ff;
        padding: 10px;
        margin-bottom: 12px;
        border-radius: 6px;
        font-size: 0.95rem;
        color: #00d2ff !important;
        font-weight: 600 !important;
    }
    details.reasoning-step {
        padding: 0;
        overflow: hidden;
    }
    details.reasoning-step > summary {
        list-style: none;
        cursor: pointer;
        padding: 8px 12px;
        color: #00d2ff !important;
        font-weight: 700 !important;
        user-select: none;
    }
    details.reasoning-step > summary::-webkit-details-marker { display: none; }
    details.reasoning-step > summary::before {
        content: "▸";
        display: inline-block;
        margin-right: 6px;
        transition: transform 0.15s ease-in-out;
    }
    details.reasoning-step[open] > summary::before { transform: rotate(90deg); }
    details.reasoning-step .reasoning-ts {
        color: #7a8191 !important;
        font-weight: 500 !important;
        font-size: 0.8rem;
        margin-right: 6px;
    }
    details.reasoning-step pre {
        margin: 0;
        padding: 10px 12px;
        background: rgba(0, 0, 0, 0.35);
        color: #d7f7ff !important;
        font-size: 0.78rem;
        line-height: 1.4;
        white-space: pre-wrap;
        word-break: break-word;
        border-top: 1px solid rgba(0, 210, 255, 0.25);
    }

    /* Movie title highlight inside assistant prose */
    .movie-name {
        color: #ffd166 !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILS ---
REASON_PING = "\x1e"

def parse_recs_simple(text):
    return re.findall(r"\*\*(.*?)\*\*", text)

def highlight_titles(text):
    return re.sub(r"\*\*(.*?)\*\*", r'<span class="movie-name">\1</span>', text)

def get_reasoning_steps():
    """Return a list of reasoning entries as dicts: {ts, step, detail, data}."""
    if not REASONING_PATH.exists():
        return []
    entries = []
    try:
        with open(REASONING_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    entries.append({"ts": "", "step": line, "detail": "", "data": {}})
    except Exception:
        return []
    return entries


def _escape_html(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_reasoning_entry(entry):
    step = _escape_html(entry.get("step", ""))
    ts = _escape_html(entry.get("ts", ""))
    detail = _escape_html(entry.get("detail", "") or "")
    data = entry.get("data") or {}
    body = json.dumps(data, indent=2, ensure_ascii=False, default=str) if data else ""
    body_html = f"<pre>{_escape_html(body)}</pre>" if body else ""
    detail_html = f" — {detail}" if detail else ""
    return (
        f'<details class="reasoning-step">'
        f'<summary><span class="reasoning-ts">{ts}</span>{step}{detail_html}</summary>'
        f"{body_html}"
        f"</details>"
    )

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎬 Movie Expert")
    st.markdown("---")
    
    mode = st.radio(
        "Recommendation Mode",
        ["RAG (Context-Aware)", "Agent (Tool-based)"],
        index=0,
        help="RAG mode uses the narrative expert persona. Agent mode uses tools for factual lookups."
    )
    model_type = "rag" if "RAG" in mode else "agent"
    
    user_id = st.text_input("User ID", value="guest_user", help="Used to persist conversation history.")
    
    if st.button("Clear History"):
        st.session_state.messages = []
        try:
            if REASONING_PATH.exists():
                REASONING_PATH.write_text("", encoding="utf-8")
        except Exception:
            pass
        st.rerun()

    st.markdown("---")
    st.subheader("🕵️ Reasoning Steps")
    reasoning_container = st.empty()
    
    def update_reasoning_ui():
        entries = get_reasoning_steps()
        if entries:
            content = "".join(render_reasoning_entry(e) for e in entries[-10:])
            reasoning_container.markdown(content, unsafe_allow_html=True)
        else:
            reasoning_container.info("Awaiting first interaction...")
    
    update_reasoning_ui()

# --- CHAT INTERFACE ---
st.title("🍿 Conversational Recommender")
st.caption("A premium AI experience for discovering your next favorite movie.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(highlight_titles(message["content"]), unsafe_allow_html=True)
        else:
            st.markdown(message["content"])
        if "recommendations" in message:
            recs = message["recommendations"]
            if recs:
                cols = st.columns(min(len(recs), 4))
                for idx, title in enumerate(recs):
                    with cols[idx % 4]:
                        meta = MOVIE_LOOKUP.get(title.lower(), {})
                        year = meta.get("release_date", "")[:4]
                        rating = meta.get("vote_average", "N/A")
                        st.markdown(f"""
                        <div class="movie-card">
                            <div class="movie-title">{title}</div>
                            <div class="movie-meta" style="color: #888888; font-size: 0.8rem;">{year} • ⭐ {rating}</div>
                        </div>
                        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask for a recommendation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        
        try:
            payload = {
                "query": prompt,
                "user_id": user_id,
                "model_type": model_type
            }
            
            with requests.post(STREAM_URL, json=payload, stream=True, timeout=120) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if not chunk:
                        continue
                    tick_count = chunk.count(REASON_PING)
                    clean = chunk.replace(REASON_PING, "")
                    if tick_count:
                        update_reasoning_ui()
                    if clean:
                        full_response += clean
                        response_container.markdown(
                            highlight_titles(full_response) + "▌",
                            unsafe_allow_html=True,
                        )

            response_container.markdown(highlight_titles(full_response), unsafe_allow_html=True)
            update_reasoning_ui() # Final reasoning catch-up
            
            # Parse recommendations for card display
            found_titles = parse_recs_simple(full_response)
            
            if found_titles:
                st.markdown("---")
                cols = st.columns(min(len(found_titles), 4))
                for idx, title in enumerate(found_titles):
                    with cols[idx % 4]:
                        meta = MOVIE_LOOKUP.get(title.lower(), {})
                        year = meta.get("release_date", "")[:4]
                        rating = meta.get("vote_average", "N/A")
                        st.markdown(f"""
                        <div class="movie-card">
                            <div class="movie-title">{title}</div>
                            <div class="movie-meta" style="color: #888888; font-size: 0.8rem;">{year} • ⭐ {rating}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Update history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "recommendations": found_titles
            })
            
        except Exception as e:
            st.error(f"Streaming Error: {str(e)}")
