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
TRACE_PATH = Path("/app/logs/pipeline_trace.txt")  # Shared volume in Docker

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
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    
    /* Sidebar glassmorphism */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Movie Card Styling */
    .movie-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        height: 100%;
    }
    .movie-title {
        font-weight: bold;
        color: #00d2ff;
        font-size: 1.1rem;
        margin-bottom: 5px;
    }
    .movie-meta {
        font-size: 0.85rem;
        color: #94bbe9;
        margin-bottom: 10px;
    }
    
    /* Chat bubbling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILS ---
def parse_recs_simple(text):
    """Fallback parser for movie titles in bold **Title** format."""
    return re.findall(r"\*\*(.*?)\*\*", text)

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
        st.rerun()

    st.markdown("---")
    st.subheader("🕵️ Reasoning Trace")
    
    # Trace viewer
    if st.checkbox("Show real-time reasoning", value=True):
        trace_container = st.empty()
        if TRACE_PATH.exists():
            with open(TRACE_PATH, "r") as f:
                lines = f.readlines()[-20:] # Last 20 lines
                trace_container.code("".join(lines), language="text")
        else:
            trace_container.info("Awaiting first interaction...")

# --- CHAT INTERFACE ---
st.title("🍿 Conversational Recommender")
st.caption("A premium AI experience for discovering your next favorite movie. Now with real-time streaming.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "recommendations" in message:
            recs = message["recommendations"]
            if recs:
                cols = st.columns(min(len(recs), 4))
                for idx, rec in enumerate(recs):
                    with cols[idx % 4]:
                        title = rec if isinstance(rec, str) else rec.get('title')
                        poster_url = "https://via.placeholder.com/500x750?text=" + title.replace(" ", "+")
                        st.markdown(f"""
                        <div class="movie-card">
                            <img src="{poster_url}" style="width: 100%; border-radius: 8px; margin-bottom: 10px;">
                            <div class="movie-title">{title}</div>
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
            
            # Use requests with stream=True for the new /stream endpoint
            with requests.post(STREAM_URL, json=payload, stream=True, timeout=120) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        response_container.markdown(full_response + "▌")
            
            response_container.markdown(full_response)
            
            # Parse recommendations for card display
            found_titles = parse_recs_simple(full_response)
            
            if found_titles:
                st.markdown("---")
                cols = st.columns(min(len(found_titles), 4))
                for idx, title in enumerate(found_titles):
                    with cols[idx % 4]:
                        poster_url = "https://via.placeholder.com/500x750?text=" + title.replace(" ", "+")
                        st.markdown(f"""
                        <div class="movie-card">
                            <img src="{poster_url}" style="width: 100%; border-radius: 8px; margin-bottom: 10px;">
                            <div class="movie-title">{title}</div>
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
