from typing import Dict, Optional
from datetime import datetime, timedelta
from langchain.memory import ConversationBufferWindowMemory
import uuid
from src.rag_chatbot.utils.logger import logger

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