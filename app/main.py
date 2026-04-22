"""
FastAPI application for movie recommendation system.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import logging
import time
from typing import Dict, List

from app.config import settings
from app.schemas import (
    Message,
    RecommendationRequest,
    RecommendationResponse,
)
from data.loader import MovieDataLoader
from utils.vector_store import MovieVectorStore
from models.rag import RAGRecommender
from models.agent import AgentRecommender
from prompts.templates import PromptTemplates
from utils.reranker import MovieReranker

# Configure logging (console + app.log file)
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# Global instances
vector_store: MovieVectorStore = None
movie_loader: MovieDataLoader = None
rag_recommender: RAGRecommender = None
agent_recommender: AgentRecommender = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Initializes resources on startup and cleans up on shutdown.
    """
    global vector_store, movie_loader, rag_recommender, agent_recommender

    logger.info("Starting up Movie CRS API...")

    # Initialize data loader with actual LLM-Redial dataset
    from pathlib import Path
    data_dir = Path("data/llm_redial")

    # Check if LLM-Redial dataset exists
    if (data_dir / "item_map.json").exists():
        logger.info("Loading actual LLM-Redial dataset...")
        user_ids_file = data_dir / "user_ids.json"
        movie_loader = MovieDataLoader(
            item_map_path=str(data_dir / "item_map.json"),
            conversations_jsonl_path=str(data_dir / "final_data.jsonl"),
            conversations_txt_path=str(data_dir / "Conversation.txt"),
            user_ids_path=str(user_ids_file) if user_ids_file.exists() else None,
            tmdb_enriched_path=str(data_dir / "tmdb_enriched_movies.json"),
        )
    else:
        logger.warning("LLM-Redial dataset not found, using sample data")
        logger.info("To use actual dataset, run: python load_dataset.py")
        movie_loader = MovieDataLoader()

    movies = movie_loader.load_movies()
    logger.info(f"Loaded {len(movies)} movies")

    # Initialize vector store
    vector_store = MovieVectorStore(
        embedding_model=settings.embedding_model,
        persist_directory=settings.vector_store_path
    )

    # Index movies if not already indexed
    stats = vector_store.get_collection_stats()
    if stats["total_movies"] == 0:
        logger.info("Indexing movies into vector store...")
        vector_store.index_movies(movies)
    else:
        logger.info(f"Vector store already has {stats['total_movies']} movies")

    # Build few-shot block from curated examples (to fix repetitive persona issues)
    few_shot_examples = PromptTemplates.format_conversation_examples()
    logger.info("Using curated few-shot dialogue examples for persona consistency")

    # Initialize reranker
    reranker = MovieReranker()

    # Initialize recommenders
    rag_recommender = RAGRecommender(vector_store, movie_loader, reranker=reranker, few_shot_examples=few_shot_examples)
    agent_recommender = AgentRecommender(vector_store, movie_loader, few_shot_examples=few_shot_examples)

    # Per-user conversation memory (in-process; lost on restart)
    app.state.conversations: Dict[str, List[Message]] = {}

    logger.info("Startup complete!")

    yield

    # Cleanup
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Movie CRS API",
    description="Conversational Movie Recommendation System with LLM",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Movie CRS API",
        "version": "1.0.0",
        "endpoints": {
            "recommend": "/recommend",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    stats = vector_store.get_collection_stats()

    return {
        "status": "healthy",
        "vector_store": stats,
        "models": {
            "main": settings.llm_model_main,
            "utility": settings.llm_model_utility
        }
    }


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_movies(request: RecommendationRequest):
    """
    Generate movie recommendations.

    Server holds per-user conversation memory keyed on `user_id`, so the
    client only needs to send `query` + `user_id` on each turn.
    """
    start_time = time.time()

    try:
        # Select recommender based on model type
        if request.model_type == "rag":
            recommender = rag_recommender
        elif request.model_type == "agent":
            recommender = agent_recommender
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type")

        # Resolve server-side history for this user
        conversations = app.state.conversations
        history = conversations.setdefault(request.user_id, [])

        logger.info(
            f"[/recommend] IN user_id={request.user_id!r} model={request.model_type} "
            f"query={request.query!r} prior_turns={len(history) // 2} history_len={len(history)}"
        )

        # Generate recommendation
        response_text = await recommender.generate_recommendation(
            query=request.query,
            history=history,
            max_recommendations=request.max_recommendations,
            user_id=request.user_id,
        )

        # Persist this turn for the next call
        history.append(Message(role="user", content=request.query))
        history.append(Message(role="assistant", content=response_text))

        # Parse recommendations
        recommendations = recommender.parse_recommendations(response_text)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        logger.info(
            f"[/recommend] OUT user_id={request.user_id!r} took={processing_time:.0f}ms "
            f"parsed={len(recommendations)} history_len={len(history)} "
            f"response_preview={response_text[:120]!r}"
        )

        return RecommendationResponse(
            response_text=response_text,
            recommendations=recommendations,
            model_used=request.model_type,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Error in recommend_movies: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/stream")
async def recommend_movies_stream(request: RecommendationRequest):
    """
    Stream movie recommendations token-by-token.
    """
    try:
        # Select recommender
        if request.model_type == "rag":
            recommender = rag_recommender
        elif request.model_type == "agent":
            recommender = agent_recommender
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type")

        # Resolve history
        history = app.state.conversations.setdefault(request.user_id, [])

        async def generate():
            full_response = []
            async for chunk in recommender.stream_recommendation(
                query=request.query,
                history=history,
                max_recommendations=request.max_recommendations,
                user_id=request.user_id,
            ):
                full_response.append(chunk)
                yield chunk

            # After stream finishes, persist to history
            final_text = "".join(full_response)
            history.append(Message(role="user", content=request.query))
            history.append(Message(role="assistant", content=final_text))
            
            logger.info(f"[/recommend/stream] DONE user_id={request.user_id!r} history_len={len(history)}")

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error in recommend_movies_stream: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
