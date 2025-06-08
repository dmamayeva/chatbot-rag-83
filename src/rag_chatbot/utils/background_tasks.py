import asyncio
from src.rag_chatbot.core.session_manager import SessionManager
from src.rag_chatbot.core.rate_limiter import RateLimiter
from src.rag_chatbot.utils.logger import logger

async def periodic_cleanup(session_manager: SessionManager, rate_limiter: RateLimiter):
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            session_manager.cleanup_expired_sessions()
            active_sessions = session_manager.get_active_session_ids()
            rate_limiter.cleanup_expired_sessions(active_sessions)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup task error: {str(e)}")