"""
Data loader for LLM-Redial dataset (Movie category).
"""
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MovieDataLoader:
    """Loader for LLM-Redial movie dataset."""
    
    def __init__(
        self,
        item_map_path: Optional[str] = None,
        conversations_jsonl_path: Optional[str] = None,
        conversations_txt_path: Optional[str] = None,
        user_ids_path: Optional[str] = None,
        tmdb_enriched_path: Optional[str] = None,
        data_path: Optional[str] = None  # Backward compatibility
    ):
        """
        Initialize data loader for LLM-Redial dataset.

        Args:
            item_map_path: Path to item_map.json (movie ID -> title mapping)
            conversations_jsonl_path: Path to final_data.jsonl
            conversations_txt_path: Path to Conversation.txt
            user_ids_path: Path to user_ids.json (Amazon user_id -> index)
            tmdb_enriched_path: Optional path to TMDB enrichment JSON
            data_path: Legacy path (for backward compatibility)
        """
        self.item_map_path = item_map_path
        self.conversations_jsonl_path = conversations_jsonl_path
        self.conversations_txt_path = conversations_txt_path
        self.user_ids_path = user_ids_path
        self.tmdb_enriched_path = tmdb_enriched_path
        self.data_path = data_path

        self.movies: List[Dict[str, Any]] = []
        self.conversations: List[Dict[str, Any]] = []
        self.item_map: Dict[str, str] = {}
        self.user_ids: Dict[str, int] = {}
        self.user_data: Dict[str, Dict[str, Any]] = {}
        self.tmdb_enriched_data: Dict[str, Dict[str, Any]] = {}
        self.conversation_transcripts: Dict[str, str] = {}
        self.known_directors: set[str] = set()
        self.known_cast: set[str] = set()
        
    def load_movies(self) -> List[Dict[str, Any]]:
        """
        Load movie catalog from dataset.
        
        Returns:
            List of movie dictionaries with metadata
        """
        # Try loading from LLM-Redial format first
        if self.item_map_path and Path(self.item_map_path).exists():
            return self._load_from_llm_redial()
        # Fall back to legacy format
        elif self.data_path and Path(self.data_path).exists():
            return self._load_from_file()
        else:
            logger.warning("No dataset file provided, using sample data")
            return self._get_sample_movies()
    
    def _load_from_llm_redial(self) -> List[Dict[str, Any]]:
        """
        Load movies from LLM-Redial dataset format.
        
        LLM-Redial structure:
        - item_map.json: {item_id: movie_title}
        - final_data.jsonl: Conversation interactions
        - Conversation.txt: Text conversations
        """
        logger.info(f"Loading LLM-Redial dataset from {self.item_map_path}")
        
        # Load item map
        with open(self.item_map_path, "r", encoding="utf-8") as f:
            self.item_map = json.load(f)
        
        logger.info(f"Loaded {len(self.item_map)} movies from item_map")
        self.tmdb_enriched_data = self._load_tmdb_enrichment()
        
        # Convert to movie list with metadata
        movies = []
        for item_id, title in self.item_map.items():
            # Create movie entry
            movie = {
                "id": item_id,
                "title": title,
                "genres": self._extract_genres_from_title(title),
                "description": f"Movie: {title}",
                "rating": None,
                "year": None,
                "director": None,
                "cast": []
            }
            self._merge_tmdb_metadata(movie, self.tmdb_enriched_data.get(item_id))
            movies.append(movie)
        
        self.movies = movies
        self._build_people_index()

        # Load conversations if available
        if self.conversations_jsonl_path and Path(self.conversations_jsonl_path).exists():
            self._load_conversations()
            self._load_conversation_transcripts()

        # Load user_ids map if available
        if self.user_ids_path and Path(self.user_ids_path).exists():
            self.load_user_ids()

        return movies

    def _load_tmdb_enrichment(self) -> Dict[str, Dict[str, Any]]:
        """Load optional TMDB enrichment keyed by LLM-Redial item id."""
        candidate_paths: List[Path] = []
        if self.tmdb_enriched_path:
            candidate_paths.append(Path(self.tmdb_enriched_path))
        if self.item_map_path:
            candidate_paths.append(Path(self.item_map_path).with_name("tmdb_enriched_movies.json"))

        for path in candidate_paths:
            if not path.exists():
                continue

            logger.info(f"Loading TMDB enrichment from {path}")
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if isinstance(raw, list):
                enriched = {
                    str(item.get("id")): item
                    for item in raw
                    if isinstance(item, dict) and item.get("id")
                }
            else:
                enriched = {
                    str(item_id): item
                    for item_id, item in raw.items()
                    if isinstance(item, dict)
                }

            usable = sum(1 for item in enriched.values() if item.get("tmdb_id"))
            logger.info(f"Loaded TMDB enrichment for {usable}/{len(enriched)} records")
            return enriched

        logger.info("No TMDB enrichment file found; using LLM-Redial title/genre fields only")
        return {}

    def _merge_tmdb_metadata(self, movie: Dict[str, Any], enriched: Optional[Dict[str, Any]]) -> None:
        """Merge useful TMDB fields into a movie record when available."""
        if not enriched or not enriched.get("tmdb_id"):
            return

        movie["tmdb_id"] = enriched.get("tmdb_id")
        movie["tmdb_title"] = enriched.get("title")
        movie["original_title"] = enriched.get("original_title")
        movie["year"] = enriched.get("year")
        movie["release_date"] = enriched.get("release_date")
        movie["genres"] = enriched.get("genres") or movie.get("genres") or []
        movie["overview"] = enriched.get("overview")
        movie["description"] = enriched.get("overview") or movie.get("description")
        movie["keywords"] = enriched.get("keywords") or []
        movie["director"] = enriched.get("director") or []
        movie["cast"] = enriched.get("cast") or []

    def _build_people_index(self) -> None:
        """Collect lowercased full-name sets for director/actor filter extraction."""
        for movie in self.movies:
            for name in movie.get("director") or []:
                if name:
                    self.known_directors.add(name.lower())
            for name in movie.get("cast") or []:
                if name:
                    self.known_cast.add(name.lower())
        logger.info(
            f"People index built: directors={len(self.known_directors)} cast={len(self.known_cast)}"
        )

    def load_user_ids(self) -> Dict[str, int]:
        """Load user_ids.json into self.user_ids."""
        logger.info(f"Loading user_ids from {self.user_ids_path}")
        with open(self.user_ids_path, "r", encoding="utf-8") as f:
            self.user_ids = json.load(f)
        logger.info(f"Loaded {len(self.user_ids)} user ids")
        return self.user_ids
    
    def _extract_genres_from_title(self, title: str) -> List[str]:
        """
        Extract potential genres from movie title.
        
        Args:
            title: Movie title
            
        Returns:
            List of potential genres
        """
        genres = []
        title_lower = title.lower()
        
        # Simple keyword-based genre detection
        genre_keywords = {
            "Action": ["action", "war", "battle", "fight"],
            "Comedy": ["comedy", "funny", "laugh"],
            "Drama": ["drama"],
            "Horror": ["horror", "scary", "terror", "vampire", "zombie"],
            "Sci-Fi": ["sci-fi", "science fiction", "space", "alien"],
            "Romance": ["romance", "love"],
            "Thriller": ["thriller", "suspense"],
            "Western": ["western", "cowboy"],
            "Animation": ["animation", "animated", "cartoon"],
            "Documentary": ["documentary"],
            "Family": ["family", "kids"],
            "Fantasy": ["fantasy", "magic"]
        }
        
        for genre, keywords in genre_keywords.items():
            if any(kw in title_lower for kw in keywords):
                genres.append(genre)
        
        # Default if no genres detected
        if not genres:
            genres = ["General"]
        
        return genres
    
    def _load_conversations(self) -> None:
        """
        Load per-user records from final_data.jsonl.

        Each line is a single-key JSON object:
            {user_id_str: {history_interaction, user_might_like, Conversation}}
        We flatten into self.user_data for O(1) lookup by user_id, and keep
        self.conversations as the raw list for backward compatibility.
        """
        logger.info(f"Loading conversations from {self.conversations_jsonl_path}")

        conversations: List[Dict[str, Any]] = []
        user_data: Dict[str, Dict[str, Any]] = {}

        with open(self.conversations_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                conversations.append(record)
                for user_id_str, payload in record.items():
                    user_data[user_id_str] = payload

        self.conversations = conversations
        self.user_data = user_data
        logger.info(f"Loaded {len(conversations)} conversation records across {len(user_data)} users")

    def _load_conversation_transcripts(self) -> None:
        """Parse Conversation.txt into a dict keyed by conversation ID string."""
        if not self.conversations_txt_path or not Path(self.conversations_txt_path).exists():
            return
        logger.info(f"Loading conversation transcripts from {self.conversations_txt_path}")

        current_id = None
        current_lines = []
        
        with open(self.conversations_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line_stripped = line.strip()
                if line_stripped.isdigit():
                    if current_id is not None and current_lines:
                        self.conversation_transcripts[current_id] = "\n".join(current_lines).strip()
                    current_id = line_stripped
                    current_lines = []
                else:
                    if current_id is not None and line_stripped:
                        current_lines.append(line.rstrip())
        
        if current_id is not None and current_lines:
            self.conversation_transcripts[current_id] = "\n".join(current_lines).strip()
        
        logger.info(f"Loaded {len(self.conversation_transcripts)} conversation transcripts")

    def get_user_history(self, user_id: str, max_items: int = 10) -> Dict[str, Any]:
        """
        Return a compact view of a user's past preferences, resolved to titles.

        Args:
            user_id: Amazon-style user id (key in user_ids.json and final_data.jsonl)
            max_items: Cap on items in each list

        Returns:
            {"recent_likes", "recent_dislikes", "historical_sample"} — titles only.
            Empty dict if the user is unknown.
        """
        payload = self.user_data.get(user_id)
        if not payload:
            return {}

        def to_titles(item_ids: List[Any]) -> List[str]:
            titles: List[str] = []
            for iid in item_ids:
                title = self.item_map.get(str(iid))
                if title:
                    titles.append(title)
            return titles

        recent_likes: List[str] = []
        recent_dislikes: List[str] = []
        full_history = []
        for conv_entry in reversed(payload.get("Conversation", [])):
            # Each entry is wrapped as {"conversation_N": {user_likes, user_dislikes, ...}}
            for conv_key, inner in conv_entry.items():
                recent_likes.extend(to_titles(inner.get("user_likes", [])))
                recent_dislikes.extend(to_titles(inner.get("user_dislikes", [])))
                
                # Build rich history item for the agent tools
                conv_id = conv_key.replace("conversation_", "")
                hist_item = {
                    "id": conv_key,
                    "likes": to_titles(inner.get("user_likes", [])),
                    "dislikes": to_titles(inner.get("user_dislikes", [])),
                    "rec_items": to_titles(inner.get("rec_item", [])),
                    "transcript": self.conversation_transcripts.get(conv_id, "")
                }
                full_history.append(hist_item)
                
            if len(recent_likes) >= max_items and len(recent_dislikes) >= max_items:
                break

        historical_sample = to_titles(payload.get("history_interaction", [])[-max_items:])

        return {
            "recent_likes": recent_likes[:max_items],
            "recent_dislikes": recent_dislikes[:max_items],
            "historical_sample": historical_sample,
            "full_history": full_history,
        }
    
    def get_conversation_examples(self, n: int = 5) -> List[str]:
        """
        Get sample conversations from Conversation.txt.

        Conversations in the LLM-Redial file are separated by a line
        containing only the conversation's numeric ID (e.g. "0", "1", "2"),
        with blank lines between speaker turns within a conversation.

        Args:
            n: Number of examples to return

        Returns:
            List of full dialogue strings (each containing multiple User/Agent turns)
        """
        if not self.conversations_txt_path or not Path(self.conversations_txt_path).exists():
            return []

        with open(self.conversations_txt_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = re.split(r"(?m)^\s*\d+\s*$", content)

        examples = []
        for chunk in chunks:
            dialogue = chunk.strip()
            if dialogue and "User:" in dialogue and "Agent:" in dialogue:
                examples.append(dialogue)
            if len(examples) >= n:
                break

        return examples
    
    def _load_from_file(self) -> List[Dict[str, Any]]:
        """Load movies from legacy dataset file."""
        path = Path(self.data_path)
        
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return self._parse_llm_redial_format(data)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            return df.to_dict("records")
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _parse_llm_redial_format(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse legacy LLM-Redial dataset format.
        
        Expected format:
        {
            "movies": [...],
            "conversations": [...]
        }
        """
        self.movies = data.get("movies", [])
        self.conversations = data.get("conversations", [])
        return self.movies
    
    def _get_sample_movies(self) -> List[Dict[str, Any]]:
        """
        Generate sample movie data for testing.
        
        Returns:
            List of sample movies
        """
        sample_movies = [
            {
                "id": "m1",
                "title": "The Shawshank Redemption",
                "year": 1994,
                "genres": ["Drama"],
                "rating": 9.3,
                "description": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
                "director": "Frank Darabont",
                "cast": ["Tim Robbins", "Morgan Freeman"]
            },
            {
                "id": "m2",
                "title": "The Godfather",
                "year": 1972,
                "genres": ["Crime", "Drama"],
                "rating": 9.2,
                "description": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
                "director": "Francis Ford Coppola",
                "cast": ["Marlon Brando", "Al Pacino"]
            },
            {
                "id": "m3",
                "title": "The Dark Knight",
                "year": 2008,
                "genres": ["Action", "Crime", "Drama"],
                "rating": 9.0,
                "description": "When the menace known as the Joker wreaks havoc on Gotham, Batman must accept one of the greatest tests.",
                "director": "Christopher Nolan",
                "cast": ["Christian Bale", "Heath Ledger"]
            },
            {
                "id": "m4",
                "title": "Inception",
                "year": 2010,
                "genres": ["Action", "Sci-Fi", "Thriller"],
                "rating": 8.8,
                "description": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.",
                "director": "Christopher Nolan",
                "cast": ["Leonardo DiCaprio", "Tom Hardy"]
            },
            {
                "id": "m5",
                "title": "Pulp Fiction",
                "year": 1994,
                "genres": ["Crime", "Drama"],
                "rating": 8.9,
                "description": "The lives of two mob hitmen, a boxer, and a pair of diner bandits intertwine in four tales of violence.",
                "director": "Quentin Tarantino",
                "cast": ["John Travolta", "Samuel L. Jackson"]
            },
            {
                "id": "m6",
                "title": "Forrest Gump",
                "year": 1994,
                "genres": ["Drama", "Romance"],
                "rating": 8.8,
                "description": "The presidencies of Kennedy and Johnson unfold through the perspective of an Alabama man.",
                "director": "Robert Zemeckis",
                "cast": ["Tom Hanks", "Robin Wright"]
            },
            {
                "id": "m7",
                "title": "The Matrix",
                "year": 1999,
                "genres": ["Action", "Sci-Fi"],
                "rating": 8.7,
                "description": "A computer hacker learns about the true nature of reality and his role in the war against its controllers.",
                "director": "Lana Wachowski, Lilly Wachowski",
                "cast": ["Keanu Reeves", "Laurence Fishburne"]
            },
            {
                "id": "m8",
                "title": "Goodfellas",
                "year": 1990,
                "genres": ["Crime", "Drama"],
                "rating": 8.7,
                "description": "The story of Henry Hill and his life in the mob, covering his relationship with his wife and partners.",
                "director": "Martin Scorsese",
                "cast": ["Robert De Niro", "Ray Liotta"]
            },
            {
                "id": "m9",
                "title": "Interstellar",
                "year": 2014,
                "genres": ["Adventure", "Drama", "Sci-Fi"],
                "rating": 8.6,
                "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
                "director": "Christopher Nolan",
                "cast": ["Matthew McConaughey", "Anne Hathaway"]
            },
            {
                "id": "m10",
                "title": "Parasite",
                "year": 2019,
                "genres": ["Drama", "Thriller"],
                "rating": 8.6,
                "description": "A poor family schemes to become employed by a wealthy family and infiltrate their household.",
                "director": "Bong Joon-ho",
                "cast": ["Song Kang-ho", "Lee Sun-kyun"]
            }
        ]
        
        self.movies = sample_movies
        return sample_movies
    
    def get_movie_by_id(self, movie_id: str) -> Optional[Dict[str, Any]]:
        """Get movie by ID."""
        for movie in self.movies:
            if movie.get("id") == movie_id:
                return movie
        return None
    
    def get_movie_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Get movie by title (case-insensitive)."""
        title_lower = title.lower()
        for movie in self.movies:
            if movie.get("title", "").lower() == title_lower:
                return movie
        return None
