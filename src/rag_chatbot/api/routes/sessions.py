# src/rag_chatbot/api/routes/sessions.py
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from langchain.schema import HumanMessage, AIMessage
from src.rag_chatbot.models.schemas import SessionCreateResponse, SessionStatsResponse
from src.rag_chatbot.api.middleware.auth import get_api_key
from src.rag_chatbot.config.settings import settings
from src.rag_chatbot.core.instances import session_manager, rate_limiter

router = APIRouter()

@router.post("/", response_model=SessionCreateResponse)
async def create_session(credentials = Depends(get_api_key)):
    """Create a new chat session"""
    session_id = session_manager.create_session()
    return SessionCreateResponse(
        session_id=session_id,
        created_at=datetime.now().isoformat()
    )

@router.delete("/{session_id}")
async def delete_session(session_id: str, credentials = Depends(get_api_key)):
    """Delete a specific session"""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully"}

# In src/rag_chatbot/api/routes/sessions.py - add this route:
@router.get("/{session_id}/history")
async def get_conversation_history(session_id: str, credentials = Depends(get_api_key)):
    """Get conversation history for a session"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    messages = []
    for msg in session["memory"].chat_memory.messages:
        messages.append({
            "type": "human" if isinstance(msg, HumanMessage) else "ai",
            "content": msg.content,
            "timestamp": datetime.now()
        })
    
    rate_limit_stats = rate_limiter.get_session_stats(session_id)
    
    return {
        "session_id": session_id,
        "message_count": session["message_count"],
        "messages": messages,
        "rate_limit": rate_limit_stats
    }

@router.get("/stats", response_model=SessionStatsResponse)
async def get_session_stats(credentials = Depends(get_api_key)):
    """Get session statistics"""
    stats = session_manager.get_session_stats()
    return SessionStatsResponse(**stats)

@router.get("/{session_id}/rate-limit")
async def get_rate_limit_stats(session_id: str, credentials = Depends(get_api_key)):
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