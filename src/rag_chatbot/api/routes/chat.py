# src/rag_chatbot/api/routes/chat.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from datetime import datetime

from src.rag_chatbot.models.schemas import ChatMessage, ChatResponse
from src.rag_chatbot.api.middleware.auth import get_api_key
from src.rag_chatbot.utils.logger import logger
from src.rag_chatbot.config.settings import settings
from src.rag_chatbot.core.instances import session_manager, rate_limiter
from rag_pipeline.rag_fusion_pipeline import rag_fusion_answer

router = APIRouter()


embedding_model = OpenAIEmbeddings(model=settings.embedding_model)

@router.post("/", response_model=ChatResponse)
async def chat(
    chat_request: ChatMessage,
    background_tasks: BackgroundTasks,
    credentials = Depends(get_api_key)
):
    """Main chat endpoint with conversation memory and rate limiting"""
    try:
        logger.info(f"Received chat request: {chat_request.message}")
        
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
        
        # ✅ Fixed: Use proper embedding model object
        answer, meta = rag_fusion_answer(
            user_query=chat_request.message,
            local_index_path=settings.vector_store_path,
            embedding_model=embedding_model,  # Use the model object, not the string
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