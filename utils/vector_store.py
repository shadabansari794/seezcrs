"""
Vector store utilities for movie retrieval in RAG-based CRS.
"""
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MovieVectorStore:
    """Vector store for movie retrieval using embeddings."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./data/vector_store",
        collection_name: str = "movies"
    ):
        """
        Initialize vector store.
        
        Args:
            embedding_model: Name of the embedding model
            persist_directory: Directory to persist vector store
            collection_name: Name of the collection
        """
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedder = SentenceTransformer(embedding_model)
        else:
            logger.warning("sentence-transformers not available, using mock embeddings")
            self.embedder = None
        
        # Initialize vector store
        if CHROMA_AVAILABLE:
            self._init_chroma()
        else:
            logger.warning("chromadb not available, using in-memory store")
            self.collection = None
            self.in_memory_store: List[Dict[str, Any]] = []
    
    def _init_chroma(self) -> None:
        """Initialize ChromaDB client and collection."""
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Movie embeddings for recommendation"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def _create_movie_text(self, movie: Dict[str, Any]) -> str:
        """
        Create searchable text representation of a movie.
        
        Args:
            movie: Movie dictionary
            
        Returns:
            Text representation for embedding
        """
        parts = [
            f"Title: {movie.get('title', '')}",
            f"Genres: {', '.join(movie.get('genres', []))}",
            f"Description: {movie.get('description', '')}",
            f"Director: {movie.get('director', '')}",
            f"Cast: {', '.join(movie.get('cast', [])[:3])}"  # Top 3 actors
        ]
        return " | ".join(parts)
    
    def index_movies(self, movies: List[Dict[str, Any]]) -> None:
        """
        Index movies into vector store.
        
        Args:
            movies: List of movie dictionaries
        """
        if not movies:
            logger.warning("No movies to index")
            return
        
        logger.info(f"Indexing {len(movies)} movies...")
        
        # Create text representations
        texts = [self._create_movie_text(movie) for movie in movies]
        
        # Create embeddings
        if self.embedder:
            embeddings = self.embedder.encode(texts, show_progress_bar=True)
            embeddings = embeddings.tolist()
        else:
            # Mock embeddings for testing
            import random
            embeddings = [[random.random() for _ in range(384)] for _ in texts]
        
        # Store in vector DB
        if self.collection:
            # ChromaDB storage
            ids = [movie.get("id", f"movie_{i}") for i, movie in enumerate(movies)]
            metadatas = [
                {
                    "title": movie.get("title", ""),
                    "year": str(movie.get("year", "")),
                    "genres": ",".join(movie.get("genres", [])),
                    "rating": str(movie.get("rating", "")),
                }
                for movie in movies
            ]
            
            batch_size = 5000
            for i in range(0, len(movies), batch_size):
                end = i + batch_size
                self.collection.upsert(
                    ids=ids[i:end],
                    embeddings=embeddings[i:end],
                    documents=texts[i:end],
                    metadatas=metadatas[i:end],
                )
            logger.info(f"Indexed {len(movies)} movies to ChromaDB")
        else:
            # In-memory storage
            self.in_memory_store = [
                {"movie": movie, "embedding": emb, "text": text}
                for movie, emb, text in zip(movies, embeddings, texts)
            ]
            logger.info(f"Indexed {len(movies)} movies to in-memory store")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        genre: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with an optional genre filter.

        Genres are stored on each Chroma record as a comma-joined string, so we
        over-fetch from the dense index (top_k * 4) and narrow in Python.

        Args:
            query: Search query
            top_k: Number of results to return after filtering
            genre: Case-insensitive substring match against movie genres

        Returns:
            List of movie dictionaries (up to top_k)
        """
        if self.embedder:
            query_embedding = self.embedder.encode([query])[0].tolist()
        else:
            import random
            query_embedding = [random.random() for _ in range(384)]

        fetch_k = top_k * 4

        if self.collection:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_k,
            )

            candidates: List[Dict[str, Any]] = []
            for i, metadata in enumerate(results["metadatas"][0]):
                candidates.append({
                    "id": results["ids"][0][i],
                    "title": metadata.get("title", ""),
                    "genres": metadata.get("genres", "").split(",") if metadata.get("genres") else [],
                    "distance": results["distances"][0][i],
                })
        else:
            import numpy as np

            candidates = []
            for item in self.in_memory_store:
                emb = np.array(item["embedding"])
                query_emb = np.array(query_embedding)
                similarity = np.dot(emb, query_emb) / (np.linalg.norm(emb) * np.linalg.norm(query_emb))

                movie = item["movie"].copy()
                movie["similarity"] = float(similarity)
                candidates.append(movie)

            candidates.sort(key=lambda x: x["similarity"], reverse=True)
            candidates = candidates[:fetch_k]

        filtered = [m for m in candidates if self._passes_genre_filter(m, genre)]

        if genre and len(filtered) < top_k:
            logger.warning(f"Only {len(filtered)}/{top_k} movies passed genre filter (genre={genre})")

        return filtered[:top_k]

    @staticmethod
    def _passes_genre_filter(movie: Dict[str, Any], genre: Optional[str]) -> bool:
        """Case-insensitive genre substring match. Returns True when no filter is set."""
        if not genre:
            return True
        genres_lower = [g.lower() for g in movie.get("genres") or []]
        return any(genre.lower() in g for g in genres_lower)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.collection:
            count = self.collection.count()
            return {"total_movies": count, "backend": "chromadb"}
        else:
            return {"total_movies": len(self.in_memory_store), "backend": "in-memory"}
