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
    """Основной эндпоинт чата, включающий:
    - память чата
    - rate limiting 
    - отправкой файлов
    """
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
            raise HTTPException(
                status_code=429,
                detail={
                    "message": f"Rate limit exceeded. You can send up to {settings.rate_limit_requests} messages per {settings.rate_limit_window_minutes} minute.",
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
                context_messages.append(f"AI-assistent : {msg.content}")
        
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
        
        # Обновление rate limit 
        updated_rate_limit_stats = rate_limiter.get_session_stats(session_id)
        
        # Подготовка ответа
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
    C