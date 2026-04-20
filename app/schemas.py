"""
Pydantic schemas for API request and response validation.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class Message(BaseModel):
    """Single message in conversation history."""
    
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    
    @validator("role")
    def validate_role(cls, v: str) -> str:
        """Validate role is either user or assistant."""
        if v not in ["user", "assistant"]:
            raise ValueError("Role must be 'user' or 'assistant'")
        return v


class MovieMetadata(BaseModel):
    """Movie metadata structure."""
    
    title: str
    year: Optional[int] = None
    genres: List[str] = Field(default_factory=list)
    rating: Optional[float] = None
    description: Optional[str] = None
    
    
class RecommendationRequest(BaseModel):
    """Request schema for movie recommendations."""
    
    query: str = Field(..., description="User's current query/question", min_length=1)
    history: List[Message] = Field(
        default_factory=list,
        description=(
            "Optional conversation history. When omitted, the server uses its in-memory "
            "session store keyed on user_id."
        ),
    )
    model_type: str = Field(
        default="rag",
        description="CRS model type: 'rag' or 'agent'"
    )
    max_recommendations: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of recommendations"
    )
    user_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Required user identifier. Either an LLM-Redial user id (for dataset-side "
            "taste signal) or any stable string the client picks for a new user."
        ),
    )

    @validator("model_type")
    def validate_model_type(cls, v: str) -> str:
        """Validate model type."""
        if v not in ["rag", "agent"]:
            raise ValueError("model_type must be 'rag' or 'agent'")
        return v


class MovieRecommendation(BaseModel):
    """Single movie recommendation."""
    
    title: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str
    metadata: Optional[MovieMetadata] = None


class RecommendationResponse(BaseModel):
    """Response schema for movie recommendations."""

    response_text: str = Field(..., description="Generated conversational response")
    recommendations: List[MovieRecommendation] = Field(default_factory=list)
    model_used: str
    processing_time_ms: float
