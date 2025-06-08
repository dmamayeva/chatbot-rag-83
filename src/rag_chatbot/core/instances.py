from src.rag_chatbot.core.session_manager import SessionManager
from src.rag_chatbot.core.rate_limiter import RateLimiter
from src.rag_chatbot.core.rag_pipeline import RAGPipeline
from src.rag_chatbot.config.settings import settings

# Create shared instances
session_manager = SessionManager(
    session_timeout_minutes=settings.session_timeout_minutes,
    max_memory_length=settings.max_memory_length
)

rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_requests,
    time_window_minutes=settings.rate_limit_window_minutes
)

rag_pipeline = RAGPipeline()