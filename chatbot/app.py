from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
import json
import os
from contextlib import asynccontextmanager
from collections import deque
# LangChain imports
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from rag_pipeline.rag_fusion_pipeline import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models and storage
embedding_model = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))
vector_store_path = os.getenv("VECTOR_STORE_PATH", "./data/faiss_index")

class RateLimiter:
    """Rate limiter for tracking message frequency per session"""
    
    def __init__(self, max_requests: int = 5, time_window_minutes: int = 1):
        self.max_requests = max_requests
        self.time_window = timedelta(minutes=time_window_minutes)
        self.request_history: Dict[str, deque] = {}
    
    def is_allowed(self, session_id: str) -> tuple[bool, Optional[float]]:
        """
        Check if request is allowed for session
        Returns (is_allowed, seconds_until_reset)
        """
        current_time = datetime.now()
        
        # Initialize session if not exists
        if session_id not in self.request_history:
            self.request_history[session_id] = deque()
        
        session_requests = self.request_history[session_id]
        
        # Remove old requests outside time window
        cutoff_time = current_time - self.time_window
        while session_requests and session_requests[0] < cutoff_time:
            session_requests.popleft()
        
        # Check if under limit
        if len(session_requests) < self.max_requests:
            session_requests.append(current_time)
            return True, None
        
        # Calculate seconds until oldest request expires
        oldest_request = session_requests[0]
        seconds_until_reset = (oldest_request + self.time_window - current_time).total_seconds()
        
        return False, max(0, seconds_until_reset)
    
    def cleanup_expired_sessions(self, active_session_ids: set):
        """Remove rate limit data for expired sessions"""
        expired_sessions = set(self.request_history.keys()) - active_session_ids
        for session_id in expired_sessions:
            del self.request_history[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up rate limit data for {len(expired_sessions)} expired sessions")
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get rate limit stats for a session"""
        if session_id not in self.request_history:
            return {
                "requests_in_window": 0,
                "requests_remaining": self.max_requests,
                "window_reset_seconds": 0
            }
        
        current_time = datetime.now()
        session_requests = self.request_history[session_id]
        cutoff_time = current_time - self.time_window
        
        # Count valid requests in current window
        valid_requests = [req for req in session_requests if req >= cutoff_time]
        requests_remaining = max(0, self.max_requests - len(valid_requests))
        
        # Calculate reset time
        window_reset_seconds = 0
        if valid_requests:
            oldest_valid = min(valid_requests)
            window_reset_seconds = (oldest_valid + self.time_window - current_time).total_seconds()
            window_reset_seconds = max(0, window_reset_seconds)
        
        return {
            "requests_in_window": len(valid_requests),
            "requests_remaining": requests_remaining,
            "window_reset_seconds": window_reset_seconds
        }

class SessionManager:
    """Manages chat sessions and conversation memory"""
    
    def __init__(self, session_timeout_minutes: int = 30, max_memory_length: int = 10):
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_memory_length = max_memory_length
    
    def create_session(self) -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "memory": ConversationBufferWindowMemory(
                k=self.max_memory_length,
                return_messages=True
            ),
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "message_count": 0
        }
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        # Check if session has expired
        if datetime.now() - session["last_accessed"] > self.session_timeout:
            self.delete_session(session_id)
            return None
        
        # Update last accessed time
        session["last_accessed"] = datetime.now()
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    def get_active_session_ids(self) -> set:
        """Get set of all active session IDs"""
        return set(self.sessions.keys())
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now()
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if current_time - session["last_accessed"] > self.session_timeout
        ]
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_session_stats(self) -> Dict:
        """Get statistics about active sessions"""
        return {
            "active_sessions": len(self.sessions),
            "sessions": {
                sid: {
                    "created_at": session["created_at"].isoformat(),
                    "last_accessed": session["last_accessed"].isoformat(),
                    "message_count": session["message_count"]
                }
                for sid, session in self.sessions.items()
            }
        }

# Initialize session manager and rate limiter
session_manager = SessionManager()
rate_limiter = RateLimiter(max_requests=5, time_window_minutes=1)

# Pydantic models for API
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

# Security (optional - remove if not needed)
security = HTTPBearer(auto_error=False)

async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Optional API key validation"""
    api_key = os.getenv("API_KEY")
    if api_key and (not credentials or credentials.credentials != api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    logger.info("Starting RAG Chatbot API...")
    # Initialize your embedding model here
    global embedding_model
    # embedding_model = YourEmbeddingModel() # Replace with actual initialization
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    try:
        yield
    finally:
        # Shutdown
        logger.info("Shutting down RAG Chatbot API...")
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Production-ready RAG chatbot with session-based conversation memory and rate limiting",
    version="1.0.0",
    docs_url="/docs", 
    redoc_url="/redoc",
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

# Background task for cleanup
async def periodic_cleanup():
    """Periodically clean up expired sessions and rate limit data"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            session_manager.cleanup_expired_sessions()
            
            # Cleanup rate limiter for expired sessions
            active_sessions = session_manager.get_active_session_ids()
            rate_limiter.cleanup_expired_sessions(active_sessions)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup task error: {str(e)}")

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )

@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session(credentials: HTTPAuthorizationCredentials = Depends(get_api_key)):
    """Create a new chat session"""
    session_id = session_manager.create_session()
    return SessionCreateResponse(
        session_id=session_id,
        created_at=datetime.now().isoformat()
    )

@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(get_api_key)
):
    """Delete a specific session"""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully"}

@app.get("/sessions/stats", response_model=SessionStatsResponse)
async def get_session_stats(credentials: HTTPAuthorizationCredentials = Depends(get_api_key)):
    """Get session statistics"""
    stats = session_manager.get_session_stats()
    return SessionStatsResponse(**stats)

@app.get("/sessions/{session_id}/rate-limit")
async def get_rate_limit_stats(
    session_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(get_api_key)
):
    """Get rate limit statistics for a session"""
    # Check if session exists
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    rate_stats = rate_limiter.get_session_stats(session_id)
    return {
        "session_id": session_id,
        "rate_limit": rate_stats
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(
    chat_request: ChatMessage,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(get_api_key)
):
    """Main chat endpoint with conversation memory and rate limiting"""
    try:
        # Get or create session
        if chat_request.session_id:
            session = session_manager.get_session(chat_request.session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found or expired")
            session_id = chat_request.session_id
        else:
            session_id = session_manager.create_session()
            session = session_manager.get_session(session_id)
        
        # Check rate limit
        is_allowed, retry_after = rate_limiter.is_allowed(session_id)
        rate_limit_stats = rate_limiter.get_session_stats(session_id)
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for session {session_id}")
            raise HTTPException(
                status_code=429,
                detail={
                    "message": "Rate limit exceeded. You can send up to 5 messages per minute.",
                    "retry_after_seconds": retry_after,
                    "rate_limit": rate_limit_stats
                }
            )
        
        # Get conversation memory
        memory = session["memory"]
        
        # Build context from conversation history
        conversation_history = memory.chat_memory.messages
        context_messages = []
        
        # Format conversation history
        for msg in conversation_history:
            if isinstance(msg, HumanMessage):
                context_messages.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_messages.append(f"Assistant: {msg.content}")
        
        # Limit to last N exchanges for context (to avoid token limits)
        max_history = 6  # This will include last 3 exchanges (6 messages)
        if context_messages:
            # Take the last N messages
            recent_messages = context_messages[-max_history:] if len(context_messages) > max_history else context_messages
            chat_context = "\n".join(recent_messages)
            logger.info(f"Using {len(recent_messages)} messages for context")
        else:
            chat_context = ""
            logger.info("No previous conversation history")
        
        # Get RAG response with chat context
        logger.info(f"Processing query for session {session_id}: {chat_request.message}")
        logger.info(f"Chat context length: {len(chat_context)} characters")
        
        answer, meta = rag_fusion_answer(
            user_query=chat_request.message,
            local_index_path=vector_store_path,
            embedding_model=embedding_model,
            mode=chat_request.mode,
            chat_context=chat_context
        )
        
        # Update conversation memory AFTER getting the response
        memory.chat_memory.add_user_message(chat_request.message)
        memory.chat_memory.add_ai_message(answer)
        
        # Update session stats
        session["message_count"] += 1
        session["last_accessed"] = datetime.now()
        
        # Add chat context info to metadata
        meta["chat_context_used"] = len(chat_context) > 0
        meta["chat_context_length"] = len(chat_context)
        meta["conversation_turn"] = session["message_count"]
        
        # Get updated rate limit stats after processing
        updated_rate_limit_stats = rate_limiter.get_session_stats(session_id)
        
        # Prepare response
        response = ChatResponse(
            response=answer,
            session_id=session_id,
            message_count=session["message_count"],
            metadata=meta,
            timestamp=datetime.now().isoformat(),
            rate_limit=updated_rate_limit_stats
        )
        
        logger.info(f"Successfully processed query for session {session_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/sessions/{session_id}/history")
async def get_conversation_history(
    session_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(get_api_key)
):
    """Get conversation history for a session"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    messages = []
    for msg in session["memory"].chat_memory.messages:
        messages.append({
            "type": "human" if isinstance(msg, HumanMessage) else "ai",
            "content": msg.content,
            "timestamp": datetime.now()  # In production, store actual timestamps
        })
    
    # Include rate limit info
    rate_limit_stats = rate_limiter.get_session_stats(session_id)
    
    return {
        "session_id": session_id,
        "message_count": session["message_count"],
        "messages": messages,
        "rate_limit": rate_limit_stats
    }

@app.get("/", include_in_schema=False)
async def serve_webpage():
    """Serve the static HTML page"""
    html_path = "/app/embed/index.html"
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Webpage not found")
    return FileResponse(html_path)

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    
    # Run server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level="info"
    )