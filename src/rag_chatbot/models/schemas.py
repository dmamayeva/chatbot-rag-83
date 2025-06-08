from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

# Pydantic models for the chabot schema
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    mode: str = Field("original", description="RAG mode: 'original' or generated")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_count: int
    metadata: Dict[str, Any] = {}
    timestamp: str
    rate_limit: Dict[str, Any] = {}

class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: str

class SessionStatsResponse(BaseModel):
    active_sessions: int
    sessions: Dict[str, Dict]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"

class RateLimitError(BaseModel):
    detail: str
    retry_after_seconds: float
    rate_limit: Dict[str, Any]