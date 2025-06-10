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
# from rag_pipeline.rag_fusion_pipeline import rag_fusion_answer
from src.rag_chatbot.core.instances import rag_pipeline

router = APIRouter()


embedding_model = OpenAIEmbeddings(model=settings.embedding_model)

from fastapi import HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
import os
import mimetypes
from pathlib import Path

@router.post("/", response_model=ChatResponse)
async def chat(
    chat_request: ChatMessage,
    background_tasks: BackgroundTasks,
    credentials = Depends(get_api_key)
):
    """Main chat endpoint with conversation memory, rate limiting, and file serving"""
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
                context_messages.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_messages.append(f"Zaure: {msg.content}")
        
        # Limit to last N exchanges for context (to avoid token limits)
        max_history = settings.max_memory_length # This will include last 3 exchanges (6 messages)
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
        
        # Process query using the RAG pipeline
        answer, meta = rag_pipeline.get_response(
            user_query=chat_request.message
        )
        
        # Check if decision was to retrieve document and file exists
        if (meta.get("decision") == "retrieve_document" and 
            meta.get("success") and 
            meta.get("file_path")):
            
            file_path = meta["file_path"]
            
            # Verify file exists and is readable
            if os.path.exists(file_path) and os.path.isfile(file_path):
                # Check if it's a PDF file
                file_extension = Path(file_path).suffix.lower()
                if file_extension == '.pdf':
                    logger.info(f"Returning PDF file: {file_path}")
                    
                    # Update conversation memory before returning file
                    memory.chat_memory.add_user_message(chat_request.message)
                    memory.chat_memory.add_ai_message(f"Retrieved document: {meta.get('document_name', 'Unknown')}")
                    
                    # Update session stats
                    session["message_count"] += 1
                    session["last_accessed"] = datetime.now()
                    
                    # Safely encode metadata for headers (HTTP headers must be Latin-1)
                    def safe_encode_header(value: str) -> str:
                        """Safely encode string for HTTP headers"""
                        if not value:
                            return "Unknown"
                        try:
                            # Try to encode as Latin-1 first
                            value.encode('latin-1')
                            return value
                        except UnicodeEncodeError:
                            # If it fails, encode to bytes then to base64
                            import base64
                            encoded_bytes = value.encode('utf-8')
                            return base64.b64encode(encoded_bytes).decode('ascii')
                    
                    # Prepare safe headers
                    safe_headers = {
                        "X-Session-ID": session_id,
                        "X-Document-Name-B64": safe_encode_header(meta.get('document_name', 'Unknown')),
                        "X-Match-Score": str(meta.get('match_score', 'N/A')),
                        "X-Match-Type": safe_encode_header(meta.get('match_type', 'N/A')),
                        "X-Document-Name-Original": "true" if meta.get('document_name', '').isascii() else "false"
                    }
                    
                    # Return the PDF file
                    return FileResponse(
                        path=file_path,
                        media_type='application/pdf',
                        filename=os.path.basename(file_path),
                        headers=safe_headers
                    )
                else:
                    logger.warning(f"File is not a PDF: {file_path} (extension: {file_extension})")
                    # Continue with normal response for non-PDF files
            else:
                logger.warning(f"File does not exist or is not readable: {file_path}")
                # Continue with normal response if file doesn't exist
        
        # Normal response flow (not a document retrieval or file doesn't exist)
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
    
# @router.post("/", response_model=ChatResponse)
# async def chat(
#     chat_request: ChatMessage,
#     background_tasks: BackgroundTasks,
#     credentials = Depends(get_api_key)
# ):
#     """Main chat endpoint with conversation memory and rate limiting"""
#     try:
#         logger.info(f"Received chat request: {chat_request.message}")
        
#         # Get or create session
#         if chat_request.session_id:
#             session = session_manager.get_session(chat_request.session_id)
#             if not session:
#                 raise HTTPException(status_code=404, detail="Session not found or expired")
#             session_id = chat_request.session_id
#         else:
#             session_id = session_manager.create_session()
#             session = session_manager.get_session(session_id)
        
#         # Check rate limit
#         is_allowed, retry_after = rate_limiter.is_allowed(session_id)
#         rate_limit_stats = rate_limiter.get_session_stats(session_id)
        
#         if not is_allowed:
#             logger.warning(f"Rate limit exceeded for session {session_id}")
#             raise HTTPException(
#                 status_code=429,
#                 detail={
#                     "message": "Rate limit exceeded. You can send up to 5 messages per minute.",
#                     "retry_after_seconds": retry_after,
#                     "rate_limit": rate_limit_stats
#                 }
#             )
        
#         # Get conversation memory
#         memory = session["memory"]
        
#         # Build context from conversation history
#         conversation_history = memory.chat_memory.messages
#         context_messages = []
        
#         # Format conversation history
#         for msg in conversation_history:
#             if isinstance(msg, HumanMessage):
#                 context_messages.append(f"Human: {msg.content}")
#             elif isinstance(msg, AIMessage):
#                 context_messages.append(f"Assistant: {msg.content}")
        
#         # Limit to last N exchanges for context (to avoid token limits)
#         max_history = 6  # This will include last 3 exchanges (6 messages)
#         if context_messages:
#             # Take the last N messages
#             recent_messages = context_messages[-max_history:] if len(context_messages) > max_history else context_messages
#             chat_context = "\n".join(recent_messages)
#             logger.info(f"Using {len(recent_messages)} messages for context")
#         else:
#             chat_context = ""
#             logger.info("No previous conversation history")
        
#         # Get RAG response with chat context
#         logger.info(f"Processing query for session {session_id}: {chat_request.message}")
#         logger.info(f"Chat context length: {len(chat_context)} characters")
        
#         # ✅ Fixed: Use proper embedding model object
#         # answer, meta = rag_fusion_answer(
#         #     user_query=chat_request.message,
#         #     local_index_path=settings.vector_store_path,
#         #     embedding_model=embedding_model,  # Use the model object, not the string
#         #     mode=chat_request.mode,
#         #     chat_context=chat_context
#         # )
#         answer, meta = rag_pipeline.get_response(user_query=chat_request.message)
#         # Update conversation memory AFTER getting the response
#         memory.chat_memory.add_user_message(chat_request.message)
#         memory.chat_memory.add_ai_message(answer)
        
#         # Update session stats
#         session["message_count"] += 1
#         session["last_accessed"] = datetime.now()
        
#         # Add chat context info to metadata
#         meta["chat_context_used"] = len(chat_context) > 0
#         meta["chat_context_length"] = len(chat_context)
#         meta["conversation_turn"] = session["message_count"]
        
#         # Get updated rate limit stats after processing
#         updated_rate_limit_stats = rate_limiter.get_session_stats(session_id)
        
#         # Prepare response
#         response = ChatResponse(
#             response=answer,
#             session_id=session_id,
#             message_count=session["message_count"],
#             metadata=meta,
#             timestamp=datetime.now().isoformat(),
#             rate_limit=updated_rate_limit_stats
#         )
        
#         logger.info(f"Successfully processed query for session {session_id}")
#         return response
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Chat processing error: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Internal server error")