from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import asyncio
import os
from fastapi import HTTPException
from src.rag_chatbot.utils.background_tasks import periodic_cleanup
from src.rag_chatbot.api.routes import chat, health, sessions
from src.rag_chatbot.config.settings import settings
from src.rag_chatbot.utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    logger.info("Starting RAG Chatbot API...")
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(
        periodic_cleanup(chat.session_manager, chat.rate_limiter)
    )
    
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

app = FastAPI(
    title="RAG Chatbot API",
    description="Production-ready RAG chatbot with session-based conversation memory and rate limiting",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes with /api/v1 prefix
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["sessions"])

# Include routes without prefix for convenience (optional)
app.include_router(chat.router, prefix="/chat", tags=["chat-direct"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(sessions.router, prefix="/sessions", tags=["sessions-direct"])

# Root endpoint - fixes the 404 error
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/v1/health",
            "chat": "/api/v1/chat",
            "sessions": "/api/v1/sessions",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

# Serve the HTML chat interface
@app.get("/chat", include_in_schema=False)
async def serve_chat_interface():
    """Serve the chat interface"""
    # Check for HTML file in different locations
    possible_paths = [
        "static/index.html",
        "embed/index.html", 
        "templates/index.html",
        "index.html"
    ]
    
    for html_path in possible_paths:
        if os.path.exists(html_path):
            return FileResponse(html_path)
    
    # If no HTML file found, return helpful message
    raise HTTPException(
        status_code=404, 
        detail=f"Chat interface not found. Please save your HTML file as one of: {possible_paths}"
    )