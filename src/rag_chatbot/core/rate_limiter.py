
from datetime import datetime, timedelta
from typing import Dict, Optional
from collections import deque
from src.rag_chatbot.utils.logger import logger

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