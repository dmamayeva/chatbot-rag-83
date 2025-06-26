# src/rag_chatbot/api/routes/chat.py (Updated with Analytics)
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from datetime import datetime
import time

from src.rag_chatbot.models.schemas import ChatMessage, ChatResponse
from src.rag_chatbot.api.middleware.auth import get_api_key
from src.rag_chatbot.utils.logger import logger
from src.rag_chatbot.config.settings import settings
from fastapi.responses import FileResponse
from src.rag_chatbot.core.instances import session_manager, rate_limiter, rag_pipeline
from src.rag_chatbot.services.analytics_service import AnalyticsService
from src.rag_chatbot.core.database import get_db
import os
from pathlib import Path

router = APIRouter()
embedding_model = OpenAIEmbeddings(model=settings.embedding_model)

@router.post("/", response_model=ChatResponse)
async def chat(
    chat_request: ChatMessage,
    background_tasks: BackgroundTasks,
    request: Request,
    credentials = Depends(get_api_key),
    db = Depends(get_db)
):
    """Основной эндпоинт чата, включающий: - память чата - rate limiting  - отправкой файлов"""
    start_time = time.time()
    analytics = AnalyticsService(db)
    
    try:
        logger.info(f"Received chat request: {chat_request.message}")
        
        # Создание или получение сессии 
        if chat_request.session_id:
            session = session_manager.get_session(chat_request.session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found or expired")
            session_id = chat_request.session_id
        else:
            session_id = session_manager.create_session()
            session = session_manager.get_session(session_id)
        
        # Проверка Лимита
        is_allowed, retry_after = rate_limiter.is_allowed(session_id)
        rate_limit_stats = rate_limiter.get_session_stats(session_id)
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for session {session_id}")
            
            # Трекинг лимита для аналитики
            analytics.track_rate_limit(session_id, retry_after)
            
            raise HTTPException(
                status_code=429,
                detail={
                    "message": "Rate limit exceeded. You can send up to {settings.rate_limit_requests} messages per {settings.rate_limit_window_minutes} minute.",
                    "retry_after_seconds": retry_after,
                    "rate_limit": rate_limit_stats
                }
            )
        
        # Получение предыдущего разговора и создание контекста для промпта на основе истории 
        memory = session["memory"]
        conversation_history = memory.chat_memory.messages
        context_messages = []
        
        # Форматирование разговора для добавление в промпт
        for msg in conversation_history:
            if isinstance(msg, HumanMessage):
                context_messages.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_messages.append(f"Zaure: {msg.content}")
        
        # Ограничение последними N обменами для контекста (чтобы избежать ограничения по токенам)
        max_history = settings.max_memory_length
        if context_messages:
            recent_messages = context_messages[-max_history:] if len(context_messages) > max_history else context_messages
            chat_context = "\n".join(recent_messages)
            logger.info(f"Using {len(recent_messages)} messages for context")
        else:
            chat_context = ""
            logger.info("No previous conversation history")
        
        # Получение ответа от LLM для контекста
        logger.info(f"Processing query for session {session_id}: {chat_request.message}")
        logger.info(f"Chat context length: {len(chat_context)} characters")
        
        # Обработка запроса с помощью RAG + OpenAI
        answer, meta = rag_pipeline.get_response(
            user_query=chat_request.message
        )
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # пайплайн для получения документов—проверка если система решила достать документ и проверка есть ли он 
        if (meta.get("decision") == "retrieve_document" and 
            meta.get("success") and 
            meta.get("file_path")):
            
            file_path = meta["file_path"]
            
            
            if os.path.exists(file_path) and os.path.isfile(file_path):
                file_extension = Path(file_path).suffix.lower()
                if file_extension == '.pdf':
                    logger.info(f"Returning PDF file: {file_path}")
                    
                    # обновление памяти чата
                    memory.chat_memory.add_user_message(chat_request.message)
                    memory.chat_memory.add_ai_message(f"Retrieved document: {meta.get('document_name', 'Unknown')}")
                    
                    # обновление сессии
                    session["message_count"] += 1
                    session["last_accessed"] = datetime.now()
                    
                    # трекинг аналитики
                    background_tasks.add_task(
                        analytics.track_conversation,
                        session_id,
                        chat_request.message,
                        f"Document retrieved: {meta.get('document_name', 'Unknown')}",
                        meta,
                        response_time_ms
                    )
                    
                    # трекинг скачивания документов
                    background_tasks.add_task(
                        analytics.track_document_download,
                        session_id,
                        meta.get('document_name', 'Unknown'),
                        meta.get('file_size_mb', 0)
                    )
                    
                    # Безопасное кодирование метаданных для заголовков (HTTP-заголовки должны быть в кодировке Latin-1)
                    def safe_encode_header(value: str) -> str:
                        if not value:
                            return "Unknown"
                        try:
                            value.encode('latin-1')
                            return value
                        except UnicodeEncodeError:
                            import base64
                            encoded_bytes = value.encode('utf-8')
                            return base64.b64encode(encoded_bytes).decode('ascii')
                    
                    
                    safe_headers = {
                        "X-Session-ID": session_id,
                        "X-Document-Name-B64": safe_encode_header(meta.get('document_name', 'Unknown')),
                        "X-Match-Score": str(meta.get('match_score', 'N/A')),
                        "X-Match-Type": safe_encode_header(meta.get('match_type', 'N/A')),
                        "X-Document-Name-Original": "true" if meta.get('document_name', '').isascii() else "false"
                    }
                    
                    # PDF-файл
                    return FileResponse(
                        path=file_path,
                        media_type='application/pdf',
                        filename=os.path.basename(file_path),
                        headers=safe_headers
                    )
                else:
                    logger.warning(f"File is not a PDF: {file_path} (extension: {file_extension})")
            else:
                logger.warning(f"File does not exist or is not readable: {file_path}")

        # Обычный ответ (без документа)
        # Обновление памяти после получения разговора 
        memory.chat_memory.add_user_message(chat_request.message)
        memory.chat_memory.add_ai_message(answer)
        
        
        session["message_count"] += 1
        session["last_accessed"] = datetime.now()
        
        # Добавить контекст для метаданных
        meta["chat_context_used"] = len(chat_context) > 0
        meta["chat_context_length"] = len(chat_context)
        meta["conversation_turn"] = session["message_count"]
        meta["response_time_ms"] = response_time_ms
        meta["user_agent"] = request.headers.get("user-agent")
        meta["ip_address"] = request.client.host
        
        # Трекинг аналитики
        background_tasks.add_task(
            analytics.track_conversation,
            session_id,
            chat_request.message,
            answer,
            meta,
            response_time_ms
        )
        
        # Обновление rate limit 
        updated_rate_limit_stats = rate_limiter.get_session_stats(session_id)
        
        # Ответ
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
        # Трекинг аналитики для HTTP ошибок
        if 'session_id' in locals():
            background_tasks.add_task(
                analytics.track_error,
                session_id,
                "HTTPException",
                str(e.detail) if hasattr(e, 'detail') else str(e)
            )
        raise
    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}", exc_info=True)
        
        # Трекинг аналитики ошибок
        if 'session_id' in locals():
            background_tasks.add_task(
                analytics.track_error,
                session_id,
                "InternalError",
                str(e)
            )
        
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/analytics/dashboard")
async def get_analytics_dashboard(
    hours: int = 24,
    credentials = Depends(get_api_key),
    db = Depends(get_db)
):
    """Получение данных для дэшборда аналитики"""
    try:
        analytics = AnalyticsService(db)
        dashboard_data = analytics.get_dashboard_data(hours=hours)
        return dashboard_data
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch analytics data")