# Project Documentation: Zaure/Ұстаз RAG Chatbot

## Project Overview

This is a multilingual RAG (Retrieval-Augmented Generation) chatbot called "Zaure/Ұстаз" developed by JSC "Orleu" for answering questions about educational regulatory documents in Kazakhstan. The system supports Russian, Kazakh, and English languages, focusing on teacher certification, qualification categories, and professional knowledge assessment.

## Project Structure

### Core Application (`src/rag_chatbot/`)

#### Entry Point
- **`main.py`**: FastAPI application entry point with comprehensive lifecycle management
  - Configures CORS middleware for cross-origin requests
  - Manages application startup/shutdown with analytics initialization
  - Serves built-in analytics dashboard at `/analytics`
  - Provides multiple API endpoints (health, chat, sessions, metrics)
  - Includes Prometheus-style metrics endpoint for monitoring

#### Configuration (`config/`)
- **`settings.py`**: Centralized configuration management using Pydantic
  - LLM settings (OpenAI API configuration)
  - Vector store paths and document locations
  - Memory and rate limiting parameters
  - Session timeout and conversation memory settings
  - Environment variable loading from `.env` files

#### API Layer (`api/`)

##### Routes (`api/routes/`)
- **`chat.py`**: Main chat endpoint with comprehensive functionality
  - Session-based conversation memory management
  - Rate limiting (5 requests per minute per session)
  - Document retrieval and file serving for PDFs
  - Analytics tracking for all interactions
  - Error handling and background task processing
  - Context-aware response generation

- **`health.py`**: Health check endpoints for system monitoring
  - Basic health status checks
  - Database connectivity verification
  - System component status reporting

- **`sessions.py`**: Session management endpoints
  - Session creation and retrieval
  - Session statistics and active session monitoring
  - Session cleanup and lifecycle management

##### Middleware (`api/middleware/`)
- **`auth.py`**: Authentication middleware for API access control
  - API key validation
  - Request authentication and authorization

#### Core Components (`core/`)
- **`database.py`**: SQLAlchemy database configuration and session management
  - SQLite database initialization for analytics
  - Database health checks and connection management
  - Session factory and dependency injection setup

- **`instances.py`**: Singleton instances for core services
  - Global session manager initialization
  - Rate limiter instance creation
  - RAG pipeline instance management

- **`llm.py`**: Language model configuration and initialization
  - OpenAI GPT-4 model setup with temperature controls
  - Token usage tracking and cost monitoring
  - LLM parameter management

- **`rag_pipeline.py`**: RAG pipeline wrapper and agent integration
  - UnifiedRAGAgent instantiation with configuration
  - Query processing and response generation interface
  - Bridge between FastAPI and RAG components

- **`rate_limiter.py`**: Request rate limiting implementation
  - Per-session rate limiting with sliding window
  - Rate limit statistics and monitoring
  - Configurable limits (5 requests per minute default)

- **`session_manager.py`**: Session lifecycle management
  - In-memory session storage with 30-minute timeout
  - Conversation memory integration with LangChain
  - Session cleanup and garbage collection

#### Data Models (`models/`)
- **`schemas.py`**: Pydantic models for API request/response schemas
  - `ChatMessage`: Input message validation and structure
  - `ChatResponse`: Structured response with metadata
  - `SessionCreateResponse`: Session creation response format
  - Rate limiting and health check response models

- **`analytics.py`**: SQLAlchemy ORM models for analytics database
  - `Session`: User session tracking with metadata
  - `Conversation`: Individual conversation storage and analysis
  - `AnalyticsEvent`: System event tracking and logging
  - `DocumentUsage`: Document access patterns and statistics
  - `QueryAnalytics`: Query frequency and performance metrics
  - `SystemMetrics`: Overall system performance tracking

#### Services (`services/`)
- **`analytics_service.py`**: Analytics data processing and reporting
  - Conversation tracking and storage
  - Dashboard data aggregation and analysis
  - Document usage statistics
  - Performance metrics calculation
  - Error tracking and rate limit monitoring

#### Background Tasks (`tasks/`)
- **`analytics_tasks.py`**: Background task management for analytics
  - Asynchronous analytics data processing
  - Periodic cleanup and maintenance tasks
  - Task queue management and error handling

#### Utilities (`utils/`)
- **`background_tasks.py`**: Background task utilities and periodic cleanup
  - Session cleanup and garbage collection
  - Rate limiter maintenance tasks
  - System maintenance scheduling

- **`logger.py`**: Centralized logging configuration
  - Structured logging with timestamps
  - Log level management and formatting
  - Application-wide logging standards

### RAG Pipeline (`rag_pipeline/`)

#### Core Agent
- **`agent.py`**: UnifiedRAGAgent - The heart of the RAG system
  - **Decision Making**: Intelligent routing between document retrieval, knowledge base search, or direct answers
  - **Document Retrieval**: Semantic search for PDF documents with embedding-based matching
  - **RAG Fusion**: Multiple query generation for enhanced retrieval coverage
  - **Conversation Memory**: Context-aware responses with conversation history
  - **Multilingual Support**: Automatic language detection and response matching
  - **Function Calling**: OpenAI function calling for structured interactions

#### Supporting Components
- **`prompts.py`**: System prompts and query generation templates
  - Multilingual chatbot prompts (Russian, Kazakh, English)
  - Query generation prompts for RAG fusion
  - Decision-making prompts for agent routing

- **`rag_fusion_pipeline.py`**: RAG fusion implementation
  - Multiple query generation for better retrieval
  - Reciprocal rank fusion for document ranking
  - Context-aware answer generation

### Document Management

#### Document Storage (`documents/`)
- **`83/kk/`**: Kazakh language regulatory documents (14 PDF files)
- **`83/rus/`**: Russian language regulatory documents (14 PDF files)
- **`documents.json`**: Document name to file path mappings

#### Vector Storage
- **`faiss_base/`**: FAISS vector index for semantic search
- **`double_db_faiss-16-06-2025/`**: Additional vector storage
- **`data/document_embeddings.npy`**: Cached document embeddings for performance

### Frontend Interfaces (`static/`)

#### Web Interfaces
- **`index.html`**: Embedded JavaScript chat widget
  - Real-time chat interface with session management
  - File download support for retrieved documents
  - Responsive design for multiple device types

- **`st_page.py`**: Streamlit dashboard interface
  - Interactive web interface for chatbot interaction
  - Session management and conversation history
  - Document retrieval and download functionality

- **`streamlit_dashboard.py`**: Advanced Streamlit analytics dashboard
  - Real-time analytics visualization
  - System performance monitoring
  - User interaction statistics

### Configuration Files

#### Environment and Dependencies
- **`requirements.txt`**: Python package dependencies
  - FastAPI, LangChain, OpenAI, SQLAlchemy
  - Analytics and monitoring libraries
  - Vector database and ML dependencies

- **`.env.example`**: Environment variable template
  - OpenAI API key configuration
  - Database connection settings
  - Application configuration parameters

#### Project Metadata
- **`package.json`**: Node.js dependencies for frontend components
- **`CLAUDE.md`**: Development instructions and project context
- **`README.md`**: Project overview and setup instructions

## Key Features

### 1. Intelligent RAG System
- **UnifiedRAGAgent**: Centralized decision-making for query routing
- **Multi-Modal Responses**: Document (pdf) retrieval, knowledge base search, or direct answers
- **Context Awareness**: Conversation memory and context-aware responses

### 2. Multilingual Support
- **Language Detection**: Automatic detection of user input language
- **Consistent Responses**: Responses match user's input language
- **Document Support**: Regulatory documents in Russian and Kazakh

### 3. Session Management
- **In-Memory Sessions**: 30-minute timeout with conversation memory
- **Rate Limiting**: 5 requests per minute per session
- **Analytics Tracking**: Comprehensive interaction logging

### 4. Document Management
- **Semantic Search**: Embedding-based document matching
- **Direct Download**: PDF file serving through API endpoints
- **Usage Analytics**: Document access tracking and statistics

### 5. Real-Time Analytics
- **Conversation Tracking**: Detailed interaction logging
- **Performance Monitoring**: Response times and success rates
- **Dashboard Interface**: Built-in analytics visualization
- **Prometheus Metrics**: Standard monitoring format support

### 6. Multiple Interfaces
- **REST API**: Full-featured API with OpenAPI documentation
- **Chat Widget**: Embeddable JavaScript interface
- **Streamlit Dashboard**: Interactive web interface
- **Analytics Dashboard**: Built-in monitoring interface

## Technical Architecture

### Backend Stack
- **FastAPI**: High-performance async web framework
- **SQLAlchemy**: ORM for analytics database management
- **LangChain**: LLM integration and conversation memory
- **FAISS**: Vector database for semantic search
- **OpenAI**: GPT-4 for language generation and embeddings

### Data Flow
1. **User Query** → API endpoint with session management
2. **Rate Limiting** → Request validation and throttling
3. **UnifiedRAGAgent** → Intelligent query routing and processing
4. **Response Generation** → Context-aware answer generation
5. **Analytics Tracking** → Background task logging
6. **Response Delivery** → Structured response with metadata

### Deployment
- **Production Ready**: Comprehensive error handling and logging
- **Scalable Architecture**: Modular design with singleton pattern
- **Monitoring**: Built-in health checks and metrics
- **Security**: API key authentication and request validation

## API Endpoints Documentation

### Chat Endpoints (`/api/v1/chat` or `/chat`)
- **POST `/`**: Main chat endpoint
  - **Purpose**: Process user messages with RAG pipeline
  - **Features**: Session management, rate limiting, document retrieval, analytics
  - **Request**: `ChatMessage` with message text and optional session_id
  - **Response**: `ChatResponse` with answer, metadata, and session info
  - **Special**: Returns PDF files directly when documents are requested

- **GET `/analytics/dashboard`**: Analytics dashboard data
  - **Purpose**: Retrieve analytics data for monitoring dashboard
  - **Parameters**: `hours` (default: 24) for time range filtering
  - **Response**: Comprehensive analytics including sessions, conversations, costs

### Health Endpoints (`/api/v1/health` or `/health`)
- **GET `/`**: Basic health check
  - **Purpose**: Simple health status verification
  - **Response**: `HealthResponse` with status and timestamp

### Session Endpoints (`/api/v1/sessions` or `/sessions`)
- **POST `/`**: Create new session
  - **Purpose**: Initialize new conversation session
  - **Response**: `SessionCreateResponse` with session_id and creation time

- **DELETE `/{session_id}`**: Delete specific session
  - **Purpose**: Manually terminate and cleanup session
  - **Response**: Success/failure message

- **GET `/{session_id}/history`**: Get conversation history
  - **Purpose**: Retrieve complete conversation history for session
  - **Response**: Messages array with timestamps and rate limit stats

- **GET `/stats`**: Global session statistics
  - **Purpose**: Overview of all active sessions
  - **Response**: `SessionStatsResponse` with active session count and details

- **GET `/{session_id}/rate-limit`**: Rate limit status
  - **Purpose**: Check current rate limiting status for session
  - **Response**: Rate limit statistics and remaining requests

### Additional Endpoints

#### Built-in Interfaces
- **GET `/`**: API information and endpoint directory
- **GET `/analytics`**: Built-in analytics dashboard (HTML)
- **GET `/chat-ui`**: Embedded chat interface (HTML)
- **GET `/health-extended`**: Extended health check with component status
- **GET `/metrics`**: Prometheus-style metrics for monitoring

#### Error Responses
- **429**: Rate limit exceeded (5 requests/minute per session)
- **404**: Session not found or expired
- **500**: Internal server error with analytics tracking
- **401**: Authentication required (API key missing/invalid)

## RAG Pipeline Architecture

### UnifiedRAGAgent Decision Flow
1. **Query Analysis**: Analyze user input with conversation context
2. **Decision Routing**: Choose between three strategies:
   - **Document Retrieval**: When user asks for specific documents/PDFs
   - **Knowledge Base Search**: For regulatory questions requiring document context
   - **Direct Answer**: For off-topic queries (with polite redirection)

### Document Retrieval Process
1. **Exact Match**: Check for direct document name matches
2. **Semantic Search**: Use OpenAI embeddings for similarity matching
3. **File Validation**: Verify document exists and is accessible
4. **Response Generation**: Create structured response with metadata

### Knowledge Base Search (RAG Fusion)
1. **Context Integration**: Include conversation history in query processing
2. **Query Generation**: Create multiple related queries for better coverage
3. **Vector Retrieval**: Search FAISS index with generated queries
4. **Rank Fusion**: Apply reciprocal rank fusion to merge results
5. **Answer Generation**: Generate context-aware response with top documents

### Conversation Memory Management
- **Session-Based**: Each session maintains independent conversation history
- **Window Size**: Configurable memory window (default: 5 previous exchanges)
- **Context Awareness**: Include relevant conversation context in responses
- **Language Consistency**: Maintain user's preferred language throughout session

## Development Commands

```bash
# Start API server
uvicorn src.rag_chatbot.main:app --reload --host 0.0.0.0 --port 8000

# Start Streamlit interface
streamlit run static/st_page.py

# Access interfaces
# API Documentation: http://127.0.0.1:8000/docs
# Chat Interface: http://127.0.0.1:8000/chat-ui
# Analytics Dashboard: http://127.0.0.1:8000/analytics
# Health Check: http://127.0.0.1:8000/health
# Metrics: http://127.0.0.1:8000/metrics
```

This documentation provides a comprehensive overview of the file structure and functionality of the Zaure/Ұстаз RAG chatbot project, designed for educational regulatory document assistance in Kazakhstan.