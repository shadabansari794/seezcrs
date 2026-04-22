import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import settings
from data.loader import MovieDataLoader
from utils.vector_store import MovieVectorStore
from models.agent.recommender import AgentRecommender
from app.schemas import Message

logging.basicConfig(level=logging.INFO)

async def test_agent_streaming():
    print("--- Initializing Agent Recommender ---")
    loader = MovieDataLoader()
    # Use the correct method name
    loader.load_movies()
    
    vector_store = MovieVectorStore()
    recommender = AgentRecommender(vector_store, loader)
    
    query = "Recommend a movie like Inception and explain why."
    history = []
    
    print(f"--- Sending Query: {query} ---")
    print("Tokens: ", end="", flush=True)
    
    full_response = ""
    try:
        async for chunk in recommender.stream_recommendation(query, history):
            print(chunk, end="", flush=True)
            full_response += chunk
    except Exception as e:
        print(f"\n[ERROR] Streaming failed: {e}")
        if hasattr(e, "response"):
            print(f"Response: {e.response}")
        if hasattr(e, "body"):
            print(f"Body: {e.body}")
        import traceback
        traceback.print_exc()

    print("\n--- Final Response Length: ", len(full_response))
    if not full_response:
        print("[!] Warning: EMPTY RESPONSE")

if __name__ == "__main__":
    asyncio.run(test_agent_streaming())
