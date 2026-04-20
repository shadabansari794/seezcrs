"""
Configuration management using Pydantic settings.
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 350
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_type: str = "chroma"
    vector_store_path: str = "./data/vector_store"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    
    # Performance Configuration
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration dictionary."""
        return {
            "model": self.llm_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


# Global settings instance
settings = Settings()
