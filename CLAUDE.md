# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multilingual RAG (Retrieval-Augmented Generation) chatbot called "Zaure/Ұстаз" developed by JSC "Orleu" for answering questions about educational regulatory documents in Kazakhstan. It supports Russian, Kazakh, and English languages, focusing on teacher certification, qualification categories, and professional knowledge assessment.

## Architecture

### Core RAG System
- **UnifiedRAGAgent** (`rag_pipeline/agent.py`): Main AI agent that decides between document retrieval, RAG search, or direct answers
- **RAG Fusion**: Generates multiple queries for better retrieval coverage  
- **Vector Storage**: FAISS-based semantic search with OpenAI embeddings
- **Document Base**: 28 regulatory PDFs in Russian and Kazakh (`documents/83/`)

### Application Structure
```
src/rag_chatbot/
├── main.py                 # FastAPI entry point
├── config/settings.py      # Centralized configuration
├── api/routes/            # REST endpoints (chat, health, sessions)
├── core/                  # RAG pipeline, LLM config, singletons
├── services/              # Analytics service
└── models/                # Pydantic schemas
```

### Multi-Interface Design
- REST API with OpenAPI docs at `/docs`
- Embedded JavaScript widget (`static/index.html`)
- Streamlit dashboard (`static/st_page.py`)
- Built-in analytics dashboard at `/analytics`

## Technology Stack

- **Backend**: FastAPI + SQLAlchemy + Uvicorn
- **AI/ML**: LangChain + OpenAI GPT-4 + OpenAI embeddings
- **Vector Database**: FAISS
- **Frontend**: HTML/JS widget + Streamlit
- **Session Management**: In-memory with 30-minute timeout
- **Analytics**: Built-in tracking with Prometheus metrics

## Development Commands

```bash
# Start API server
uvicorn src.rag_chatbot.main:app --reload --host 0.0.0.0 --port 8000

# Start Streamlit interface
streamlit run static/st_page.py

# Access interfaces
# API: http://127.0.0.1:8000/docs
# Chat: http://127.0.0.1:8000/chat
# Analytics: http://127.0.0.1:8000/analytics
```

## Configuration1

- Environment variables in `.env.example` (require1s OPENAI_API_KEY)
- Settings managed via `src/rag_chatbot/config/settings.py`
- Document paths and embeddings in `documents/` and vector storage directories

## Key Implementation Notes

### Multilingual Support
All responses must match the user's input language (Russian/Kazakh/English). The agent automatically detects and maintains language consistency.

### RAG Pipeline Flow
1. **Agent Decision**: Determines retrieval strategy (document, RAG, or direct)
2. **Query Processing**: Uses RAG fusion for multiple query variants
3. **Semantic Search**: FAISS vector similarity search
4. **Response Generation**: Context-aware LLM response with conversation memory

### Session Management
- Rate limiting: 5 requests per minute per session
- 30-minute session timeout with conversation memory
- Analytics tracking for all interactions

### Document Processing
- PDF documents stored in `documents/83/` with language variants
- Document mappings in JSON format
- Embeddings cached in FAISS indices for performance
- Direct PDF serving capability via FastAPI endpoints