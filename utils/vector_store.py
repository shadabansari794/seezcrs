"""
Vector store utilities for movie retrieval in RAG-based CRS.
"""
from typing import Any, Dict, List, Optional, Tuple
import logging
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OPENAI_EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MovieVectorStore:
    """Vector store for movie retrieval using embeddings."""
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-large",
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
        from app.config import settings
        if OPENAI_EMBEDDINGS_AVAILABLE:
            if not settings.openai_api_key:
                logger.warning("OpenAI API key not set, embeddings will fail if invoked.")
            self.embedder = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=settings.openai_api_key
            )
        else:
            logger.warning("langchain-openai not available, using mock embeddings")
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

        The base LLM-Redial catalog provides title and inferred genres. When
        TMDB enrichment is available, this also includes overview, keywords,
        director, cast, and year to make semantic retrieval much richer.
        
        Args:
            movie: Movie dictionary
            
        Returns:
            Text representation for embedding
        """
        def join_values(value: Any, limit: Optional[int] = None) -> str:
            values = value if isinstance(value, list) else [value]
            values = [str(item) for item in values[:limit] if item]
            return ", ".join(values)

        title = movie.get("title") or ""
        overview = movie.get("overview") or movie.get("description")
        if overview == f"Movie: {title}":
            overview = None

        fields = [
            ("Title", movie.get("title")),
            ("Year", movie.get("year")),
            ("Genres", join_values(movie.get("genres") or [])),
            ("Overview", overview),
            ("Keywords", join_values(movie.get("keywords") or [], limit=12)),
            ("Director", join_values(movie.get("director") or [])),
            ("Cast", join_values(movie.get("cast") or [], limit=5)),
        ]

        parts = [f"{label}: {value}" for label, value in fields if value]
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
            embeddings = self.embedder.embed_documents(texts)
        else:
            # Mock embeddings for testing
            import random
            embeddings = [[random.random() for _ in range(3072)] for _ in texts]
        
        # Store in vector DB
        if self.collection:
            # ChromaDB storage
            ids = [movie.get("id", f"movie_{i}") for i, movie in enumerate(movies)]
            metadatas = [
                {
                    "title": movie.get("title", ""),
                    "year": str(movie.get("year") or ""),
                    "genres": ",".join(movie.get("genres", [])),
                    "overview": movie.get("overview") or "",
                    "keywords": ",".join(movie.get("keywords") or []),
                    "director": ",".join(movie.get("director") or []) if isinstance(movie.get("director"), list) else str(movie.get("director") or ""),
                    "cast": ",".join(movie.get("cast") or []),
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
        year: Optional[int] = None,
        year_range: Optional[Tuple[int, int]] = None,
        director: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with optional metadata filters.

        We over-fetch from the dense index (top_k * 4) and narrow in Python,
        because array-like TMDB fields (genres/director/cast) are stored as
        comma-joined strings in Chroma metadata. Filters are ANDed. If the
        combined filter eliminates every candidate we fall back to the
        unfiltered top-K so the LLM always has something to reason over.

        Args:
            query: Search query
            top_k: Number of results to return after filtering
            genre: Case-insensitive substring match against movie genres
            year: Exact TMDB year match
            year_range: Inclusive (start, end) year range
            director: Case-insensitive full-name substring match in director list
            actor: Case-insensitive full-name substring match in cast list

        Returns:
            List of movie dictionaries (up to top_k)
        """
        if self.embedder:
            query_embedding = self.embedder.embed_query(query)
        else:
            import random
            query_embedding = [random.random() for _ in range(3072)]

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
                    "year": metadata.get("year") or None,
                    "genres": metadata.get("genres", "").split(",") if metadata.get("genres") else [],
                    "overview": metadata.get("overview", ""),
                    "description": metadata.get("overview", ""),
                    "keywords": metadata.get("keywords", "").split(",") if metadata.get("keywords") else [],
                    "director": metadata.get("director", "").split(",") if metadata.get("director") else [],
                    "cast": metadata.get("cast", "").split(",") if metadata.get("cast") else [],
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

        filters_active = any(f is not None for f in (genre, year, year_range, director, actor))
        filtered = [
            m for m in candidates
            if self._passes_genre_filter(m, genre)
            and self._passes_year_filter(m, year, year_range)
            and self._passes_person_filter(m.get("director"), director)
            and self._passes_person_filter(m.get("cast"), actor)
        ]

        if filters_active and not filtered:
            logger.warning(
                "All candidates filtered out (genre=%s year=%s year_range=%s director=%s actor=%s); "
                "falling back to unfiltered top-K",
                genre, year, year_range, director, actor,
            )
            return candidates[:top_k]

        if filters_active and len(filtered) < top_k:
            logger.warning(
                "Only %d/%d movies passed filters (genre=%s year=%s year_range=%s director=%s actor=%s)",
                len(filtered), top_k, genre, year, year_range, director, actor,
            )

        return filtered[:top_k]

    @staticmethod
    def _passes_genre_filter(movie: Dict[str, Any], genre: Optional[str]) -> bool:
        """Case-insensitive genre substring match. Returns True when no filter is set."""
        if not genre:
            return True
        genres_lower = [g.lower() for g in movie.get("genres") or []]
        return any(genre.lower() in g for g in genres_lower)

    @staticmethod
    def _passes_year_filter(
        movie: Dict[str, Any],
        year: Optional[int],
        year_range: Optional[Tuple[int, int]],
    ) -> bool:
        """Year filter. Movies without a known year fail when a filter is active."""
        if year is None and year_range is None:
            return True
        raw = movie.get("year")
        try:
            movie_year = int(raw) if raw not in (None, "") else None
        except (TypeError, ValueError):
            movie_year = None
        if movie_year is None:
            return False
        if year is not None and movie_year != year:
            return False
        if year_range is not None and not (year_range[0] <= movie_year <= year_range[1]):
            return False
        return True

    @staticmethod
    def _passes_person_filter(people: Any, needle: Optional[str]) -> bool:
        """Case-insensitive substring match of ``needle`` against a name list/string."""
        if not needle:
            return True
        if isinstance(people, list):
            names = [str(p).lower() for p in people if p]
        elif isinstance(people, str):
            names = [p.strip().lower() for p in people.split(",") if p.strip()]
        else:
            return False
        needle_lower = needle.lower()
        return any(needle_lower in name for name in names)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.collection:
            count = self.collection.count()
            return {"total_movies": count, "backend": "chromadb"}
        else:
            return {"total_movies": len(self.in_memory_store), "backend": "in-memory"}
