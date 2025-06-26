from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import os
from datetime import datetime

from src.rag_chatbot.utils.background_tasks import periodic_cleanup
from src.rag_chatbot.api.routes import chat, health, sessions
from src.rag_chatbot.config.settings import settings
from src.rag_chatbot.utils.logger import logger

# Analytics imports
from src.rag_chatbot.core.database import init_database, check_database_health
from src.rag_chatbot.core.instances import analytics_task_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown with analytics"""
    logger.info("Starting RAG Chatbot API with Analytics...")
    
    # Инициализация базы для аналитики
    try:
        init_database()
        logger.info("Analytics database initialized successfully")
        
        # Проверка состояния базы
        if not check_database_health():
            logger.warning("Database health check failed, continuing without analytics")
        else:
            logger.info("Database health check passed")
    except Exception as e:
        logger.error(f"Analytics database initialization failed: {e}")
        logger.warning("Continuing without analytics database")
    
    # Фоновая очистка
    cleanup_task = asyncio.create_task(
        periodic_cleanup(chat.session_manager, chat.rate_limiter)
    )
    
    # Аналитика и планировщик задач
    try:
        analytics_task_manager.start()
        logger.info("Analytics task manager started")
    except Exception as e:
        logger.error(f"Failed to start analytics task manager: {e}")
    
    try:
        yield
    finally:
        # Отключение
        logger.info("Shutting down RAG Chatbot API...")
        
        # Остановка аналитики и планировщика задач
        try:
            analytics_task_manager.stop()
            logger.info("Analytics task manager stopped")
        except Exception as e:
            logger.error(f"Error stopping analytics task manager: {e}")
        
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        
        logger.info("RAG Chatbot API shutdown completed")

app = FastAPI(
    title=settings.app_name,
    description="AI-ассистент для объяснения приказов",
    version=settings.version,
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

# Добавление API рутеров
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["sessions"])

# Добавление рутеров без префиксов
app.include_router(chat.router, prefix="/chat", tags=["chat-direct"])
app.include_router(sessions.router, prefix="/sessions", tags=["sessions-direct"])

# Добавление static папки для дэшборда
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Root endpoint - fixes the 404 error
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Chatbot API with Analytics",
        "version": settings.version,
        "status": "running",
        "features": [
            "RAG-based document retrieval",
            "Conversation memory",
            "Rate limiting",
            "Real-time analytics",
            "Performance monitoring"
        ],
        "endpoints": {
            "health": "/api/v1/health",
            "chat": "/api/v1/chat",
            "sessions": "/api/v1/sessions",
            "analytics": "/analytics",
            "docs": "/docs",
            "redoc": "/redoc",
            "metrics": "/metrics"
        }
    }

# эндпоинт аналитики
@app.get("/analytics", response_class=HTMLResponse, include_in_schema=False)
async def analytics_dashboard():
    """Serve the analytics dashboard"""
    dashboard_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Chatbot Analytics Dashboard</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            .header {
                background: white;
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                text-align: center;
            }
            .header h1 {
                color: #2c3e50;
                font-size: 2.5rem;
                margin-bottom: 10px;
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .controls {
                background: white;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.08);
                display: flex;
                align-items: center;
                gap: 20px;
                flex-wrap: wrap;
            }
            .controls select, .controls button {
                padding: 10px 15px;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                background: white;
                font-size: 14px;
                transition: all 0.3s ease;
            }
            .controls button {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border: none;
                cursor: pointer;
                font-weight: 600;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: white;
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.08);
                transition: transform 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            .stat-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(45deg, #667eea, #764ba2);
            }
            .stat-card:hover { transform: translateY(-5px); }
            .stat-value {
                font-size: 2.5rem;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 5px;
            }
            .stat-label {
                color: #7f8c8d;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .loading { text-align: center; padding: 50px; color: #7f8c8d; font-size: 1.2rem; }
            .error {
                background: #e74c3c;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                text-align: center;
            }
            .status-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-online { background: #27ae60; }
            .status-warning { background: #f39c12; }
            .status-error { background: #e74c3c; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 RAG Chatbot Analytics</h1>
                <p>Real-time monitoring and analytics for Zaure AI Assistant</p>
            </div>

            <div class="controls">
                <label for="timeRange">Time Range:</label>
                <select id="timeRange">
                    <option value="1">Last Hour</option>
                    <option value="6">Last 6 Hours</option>
                    <option value="24" selected>Last 24 Hours</option>
                    <option value="168">Last Week</option>
                </select>
                
                <button onclick="refreshData()" id="refreshBtn">
                    🔄 Refresh Data
                </button>
                
                <div style="margin-left: auto;">
                    <span class="status-indicator status-online"></span>
                    <span>System Status: <span id="systemStatus">Online</span></span>
                </div>
            </div>

            <div id="loadingIndicator" class="loading">
                Loading analytics data...
            </div>

            <div id="errorContainer"></div>

            <div id="dashboardContent" style="display: none;">
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="totalSessions">0</div>
                        <div class="stat-label">Total Sessions</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="totalConversations">0</div>
                        <div class="stat-label">Conversations</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="successRate">0%</div>
                        <div class="stat-label">Success Rate</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="totalCost">$0.00</div>
                        <div class="stat-label">Total Cost</div>
                    </div>
                </div>
                
                <div style="background: white; border-radius: 15px; padding: 25px; box-shadow: 0 5px 20px rgba(0,0,0,0.08);">
                    <h3 style="text-align: center; margin-bottom: 20px; color: #2c3e50;">Analytics data will be loaded from /api/v1/chat/analytics/dashboard</h3>
                    <p style="text-align: center; color: #7f8c8d;">Make sure the analytics database is properly configured and the API endpoint is accessible.</p>
                </div>
            </div>
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', function() {
                loadDashboardData();
                setInterval(loadDashboardData, 30000); // Auto-refresh every 30 seconds
            });

            async function loadDashboardData() {
                try {
                    const timeRange = document.getElementById('timeRange').value;
                    showLoading(true);
                    
                    const response = await fetch(`/api/v1/chat/analytics/dashboard?hours=${timeRange}`);
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const data = await response.json();
                    updateDashboard(data);
                    showLoading(false);
                    clearError();
                    
                } catch (error) {
                    console.error('Error loading dashboard data:', error);
                    showError(`Failed to load analytics data: ${error.message}`);
                    showLoading(false);
                }
            }

            function updateDashboard(data) {
                document.getElementById('totalSessions').textContent = data.overview?.total_sessions || 0;
                document.getElementById('totalConversations').textContent = data.overview?.total_conversations || 0;
                document.getElementById('successRate').textContent = `${data.overview?.success_rate || 0}%`;
                document.getElementById('totalCost').textContent = `$${data.overview?.total_cost || 0}`;
            }

            function refreshData() {
                loadDashboardData();
            }

            function showLoading(show) {
                const loadingIndicator = document.getElementById('loadingIndicator');
                const dashboardContent = document.getElementById('dashboardContent');
                
                if (show) {
                    loadingIndicator.style.display = 'block';
                    dashboardContent.style.display = 'none';
                } else {
                    loadingIndicator.style.display = 'none';
                    dashboardContent.style.display = 'block';
                }
            }

            function showError(message) {
                const errorContainer = document.getElementById('errorContainer');
                errorContainer.innerHTML = `<div class="error">${message}</div>`;
            }

            function clearError() {
                const errorContainer = document.getElementById('errorContainer');
                errorContainer.innerHTML = '';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=dashboard_html)

# Интерфейс
@app.get("/chat-ui", response_class=HTMLResponse, include_in_schema=False)
async def serve_chat_interface():
    """Serve the chat interface compatible with PDF handling and markdown"""
    chat_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI-ассистент "Ұстаз"</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.3.0/marked.min.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                display: flex;
                flex-direction: column;
            }
            .header {
                background: white;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }
            .header h1 {
                color: #2c3e50;
                font-size: 1.8rem;
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 5px;
            }
            .header p {
                color: #7f8c8d;
                font-size: 0.9rem;
            }
            .chat-container {
                flex: 1;
                display: flex;
                flex-direction: column;
                max-width: 1200px;
                margin: 20px auto;
                background: white;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            .messages {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 15px;
                background: #f8f9fa;
            }
            .message {
                max-width: 85%;
                padding: 12px 18px;
                border-radius: 18px;
                word-wrap: break-word;
                line-height: 1.5;
            }
            .user-message {
                align-self: flex-end;
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
            }
            .bot-message {
                align-self: flex-start;
                background: white;
                border: 1px solid #e9ecef;
                color: #2c3e50;
            }
            .pdf-message {
                align-self: flex-start;
                background: white;
                border: 2px solid #667eea;
                border-radius: 15px;
                padding: 20px;
                max-width: 95%;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            .pdf-header {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid #e9ecef;
            }
            .pdf-title {
                font-weight: bold;
                color: #2c3e50;
                font-size: 1.1rem;
            }
            .pdf-metadata {
                display: flex;
                gap: 20px;
                margin-bottom: 15px;
                font-size: 0.9rem;
                color: #7f8c8d;
            }
            .pdf-viewer-container {
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
                margin-bottom: 15px;
            }
            .pdf-viewer {
                width: 100%;
                height: 500px;
                border: none;
            }
            .pdf-actions {
                display: flex;
                gap: 10px;
                align-items: center;
            }
            .download-btn {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                text-decoration: none;
                font-size: 0.9rem;
                display: inline-flex;
                align-items: center;
                gap: 5px;
                transition: transform 0.2s ease;
            }
            .download-btn:hover {
                transform: translateY(-2px);
                text-decoration: none;
                color: white;
            }
            .file-info {
                font-size: 0.8rem;
                color: #7f8c8d;
            }
            .input-container {
                display: flex;
                padding: 20px;
                background: white;
                border-top: 1px solid #e9ecef;
                gap: 10px;
            }
            .message-input {
                flex: 1;
                padding: 12px 18px;
                border: 2px solid #e9ecef;
                border-radius: 25px;
                outline: none;
                font-size: 14px;
                transition: border-color 0.3s ease;
                resize: none;
                min-height: 20px;
                max-height: 100px;
            }
            .message-input:focus {
                border-color: #667eea;
            }
            .send-button {
                padding: 12px 24px;
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-weight: 600;
                transition: transform 0.2s ease;
                min-width: 80px;
            }
            .send-button:hover {
                transform: translateY(-2px);
            }
            .send-button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            .typing-indicator {
                align-self: flex-start;
                padding: 12px 18px;
                background: white;
                border: 1px solid #e9ecef;
                border-radius: 18px;
                color: #7f8c8d;
                font-style: italic;
            }
            .error-message {
                background: #e74c3c;
                color: white;
                text-align: center;
                padding: 10px;
                border-radius: 8px;
                margin: 10px 0;
            }
            .session-info {
                background: #e8f5e8;
                color: #2d5a2d;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 0.8rem;
                text-align: center;
                margin-bottom: 10px;
            }
            .markdown-content {
                line-height: 1.6;
            }
            .markdown-content h1, .markdown-content h2, .markdown-content h3 {
                margin: 10px 0;
                color: #2c3e50;
            }
            .markdown-content p {
                margin: 8px 0;
            }
            .markdown-content ul, .markdown-content ol {
                margin: 8px 0 8px 20px;
            }
            .markdown-content code {
                background: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }
            .markdown-content pre {
                background: #f4f4f4;
                padding: 10px;
                border-radius: 6px;
                overflow-x: auto;
                margin: 10px 0;
            }
            .status-badge {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8rem;
                font-weight: 500;
            }
            .match-exact { background: #d4edda; color: #155724; }
            .match-semantic { background: #d1ecf1; color: #0c5460; }
            .match-partial { background: #fff3cd; color: #856404; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 AI-ассистент "Ұстаз"</h1>
            <p>Разработан АО "Өрлеу"</p>
        </div>

        <div class="chat-container">
            <div class="session-info" id="sessionInfo" style="display: none;">
                Сессия создана успешно
            </div>
            
            <div class="messages" id="messages">
                <div class="bot-message">
                    <div class="markdown-content">Здравствуйте! Я AI-ассистент разработанный АО "Өрлеу". Как я могу помочь вам сегодня?</div>
                </div>
            </div>
            
            <div class="input-container">
                <textarea 
                    class="message-input" 
                    id="messageInput" 
                    placeholder="Введите ваше сообщение..."
                    rows="1"
                    onkeydown="handleKeyDown(event)"
                ></textarea>
                <button class="send-button" id="sendButton" onclick="sendMessage()">
                    Отправить
                </button>
            </div>
        </div>

        <script>
            let currentSessionId = null;
            let isLoading = false;

            // Initialize session on page load
            document.addEventListener('DOMContentLoaded', function() {
                createSession();
                autoResizeTextarea();
            });

            async function createSession() {
                try {
                    const response = await fetch('/api/v1/sessions', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });

                    if (response.ok) {
                        const data = await response.json();
                        currentSessionId = data.session_id;
                        document.getElementById('sessionInfo').style.display = 'block';
                        setTimeout(() => {
                            document.getElementById('sessionInfo').style.display = 'none';
                        }, 3000);
                    } else {
                        showError('Не удалось создать сессию');
                    }
                } catch (error) {
                    console.error('Error creating session:', error);
                    showError('Ошибка подключения к серверу');
                }
            }

            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const sendButton = document.getElementById('sendButton');
                const messagesContainer = document.getElementById('messages');
                
                const message = input.value.trim();
                if (!message || isLoading) return;

                if (!currentSessionId) {
                    showError('Сессия не создана. Обновите страницу.');
                    return;
                }

                // Disable input and show user message
                isLoading = true;
                input.disabled = true;
                sendButton.disabled = true;
                input.value = '';
                resetTextareaHeight();

                // Add user message to chat
                addMessage(message, 'user');

                // Show typing indicator
                const typingIndicator = addTypingIndicator();

                try {
                    const response = await fetch('/api/v1/chat/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            session_id: currentSessionId,
                            mode: "generated"
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    // Check content type
                    const contentType = response.headers.get('content-type') || '';
                    
                    if (contentType.includes('application/pdf')) {
                        // Handle PDF response
                        const pdfData = await response.arrayBuffer();
                        const pdfBlob = new Blob([pdfData], { type: 'application/pdf' });
                        
                        // Extract metadata from headers
                        const isOriginal = response.headers.get('X-Document-Name-Original') === 'true';
                        const documentNameHeader = response.headers.get('X-Document-Name-B64') || 'Unknown Document';
                        const documentName = isOriginal ? documentNameHeader : atob(documentNameHeader);
                        const matchScore = response.headers.get('X-Match-Score') || 'N/A';
                        const matchType = response.headers.get('X-Match-Type') || 'N/A';
                        
                        // Get filename from Content-Disposition
                        let filename = 'document.pdf';
                        const contentDisposition = response.headers.get('content-disposition') || '';
                        if (contentDisposition.includes('filename=')) {
                            filename = contentDisposition.split('filename=')[1].replace(/"/g, '');
                        }

                        typingIndicator.remove();
                        addPdfMessage(pdfBlob, filename, documentName, matchScore, matchType);
                        
                    } else {
                        // Handle JSON response
                        const data = await response.json();
                        
                        // Update session ID if provided
                        if (data.session_id) {
                            currentSessionId = data.session_id;
                        }

                        typingIndicator.remove();
                        addMessage(data.response || 'Извините, не удалось получить ответ.', 'bot');
                    }

                } catch (error) {
                    console.error('Error sending message:', error);
                    typingIndicator.remove();
                    addMessage('Извините, произошла ошибка. Попробуйте еще раз.', 'bot');
                    showError('Ошибка отправки сообщения');
                } finally {
                    // Re-enable input
                    isLoading = false;
                    input.disabled = false;
                    sendButton.disabled = false;
                    input.focus();
                }
            }

            function addMessage(text, sender) {
                const messagesContainer = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                
                if (sender === 'bot') {
                    // Parse markdown for bot messages
                    const markdownContent = document.createElement('div');
                    markdownContent.className = 'markdown-content';
                    markdownContent.innerHTML = marked.parse(text);
                    messageDiv.appendChild(markdownContent);
                } else {
                    messageDiv.textContent = text;
                }
                
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                return messageDiv;
            }

            function addPdfMessage(pdfBlob, filename, documentName, matchScore, matchType) {
                const messagesContainer = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'pdf-message';
                
                // Create PDF viewer URL
                const pdfUrl = URL.createObjectURL(pdfBlob);
                
                // Determine match type styling
                let matchClass = 'match-partial';
                if (matchType.toLowerCase().includes('exact')) matchClass = 'match-exact';
                else if (matchType.toLowerCase().includes('semantic')) matchClass = 'match-semantic';
                
                messageDiv.innerHTML = `
                    <div class="pdf-header">
                        <span style="font-size: 1.5rem;">📄</span>
                        <div>
                            <div class="pdf-metadata">
                            </div>
                        </div>
                    </div>
                    
                    <div class="pdf-viewer-container">
                        <iframe src="${pdfUrl}" class="pdf-viewer" type="application/pdf">
                            <p>Ваш браузер не поддерживает просмотр PDF. 
                               <a href="${pdfUrl}" download="${escapeHtml(filename)}">Скачать файл</a>
                            </p>
                        </iframe>
                    </div>
                    
                    <div class="pdf-actions">
                        <a href="${pdfUrl}" download="${escapeHtml(filename)}" class="download-btn">
                            📥 Скачать ${escapeHtml(filename)}
                        </a>
                        <div class="file-info">
                            Размер: ${(pdfBlob.size / 1024).toFixed(1)} KB
                        </div>
                    </div>
                `;
                
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                return messageDiv;
            }

            function addTypingIndicator() {
                const messagesContainer = document.getElementById('messages');
                const typingDiv = document.createElement('div');
                typingDiv.className = 'typing-indicator';
                typingDiv.textContent = 'Ищу ответ... Пожалуйста, подождите.';
                messagesContainer.appendChild(typingDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                return typingDiv;
            }

            function handleKeyDown(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                } else if (event.key === 'Enter' && event.shiftKey) {
                    // Allow new line with Shift+Enter
                    autoResizeTextarea();
                }
            }

            function autoResizeTextarea() {
                const textarea = document.getElementById('messageInput');
                textarea.style.height = 'auto';
                textarea.style.height = Math.min(textarea.scrollHeight, 100) + 'px';
            }

            function resetTextareaHeight() {
                const textarea = document.getElementById('messageInput');
                textarea.style.height = 'auto';
            }

            function showError(message) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.textContent = message;
                document.body.insertBefore(errorDiv, document.body.firstChild);
                
                setTimeout(() => {
                    errorDiv.remove();
                }, 5000);
            }

            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }

            // Auto-resize textarea on input
            document.getElementById('messageInput').addEventListener('input', autoResizeTextarea);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=chat_html)

# Проверка состояния с аналитикой
@app.get("/health-extended", include_in_schema=False)
async def extended_health_check():
    """Extended health check including analytics status"""
    try:
        # базовая проверка 
        health_response = await health.health_check()
        basic_health = health_response['status'] == 'healthy'
        
        # Статус базы для аналитики
        db_healthy = check_database_health()
        
        # Статур планировщика аналитики
        analytics_running = analytics_task_manager.running if analytics_task_manager else False
        
        status = "healthy"
        if not db_healthy or not analytics_running:
            status = "degraded"
        if not basic_health:
            status = "unhealthy"
        
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "api": "healthy" if basic_health else "unhealthy",
                "database": "connected" if db_healthy else "disconnected", 
                "analytics": "running" if analytics_running else "stopped"
            },
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

# Prometheus-style metrics endpoint
@app.get("/metrics", include_in_schema=False)
async def get_metrics():
    """Prometheus-style metrics endpoint for monitoring"""
    try:
        from src.rag_chatbot.core.database import get_db_session
        from src.rag_chatbot.services.analytics_service import AnalyticsService
        
        with get_db_session() as db:
            analytics = AnalyticsService(db)
            data = analytics.get_dashboard_data(hours=1)
            
            metrics = [
                f"# HELP rag_chatbot_sessions_total Total number of chat sessions",
                f"# TYPE rag_chatbot_sessions_total counter",
                f"rag_chatbot_sessions_total {data.get('overview', {}).get('total_sessions', 0)}",
                f"",
                f"# HELP rag_chatbot_conversations_total Total number of conversations",
                f"# TYPE rag_chatbot_conversations_total counter", 
                f"rag_chatbot_conversations_total {data.get('overview', {}).get('total_conversations', 0)}",
                f"",
                f"# HELP rag_chatbot_success_rate Success rate percentage",
                f"# TYPE rag_chatbot_success_rate gauge",
                f"rag_chatbot_success_rate {data.get('overview', {}).get('success_rate', 0)}",
                f"",
                f"# HELP rag_chatbot_cost_total Total API cost in USD",
                f"# TYPE rag_chatbot_cost_total counter",
                f"rag_chatbot_cost_total {data.get('overview', {}).get('total_cost', 0)}",
                f"",
                f"# HELP rag_chatbot_response_time_avg Average response time in seconds",
                f"# TYPE rag_chatbot_response_time_avg gauge",
                f"rag_chatbot_response_time_avg {data.get('response_times', {}).get('avg', 0):.3f}",
                f"",
                f"# HELP rag_chatbot_rate_limit_events Rate limit events count",
                f"# TYPE rag_chatbot_rate_limit_events counter", 
                f"rag_chatbot_rate_limit_events {data.get('overview', {}).get('rate_limit_events', 0)}"
            ]
            
            return Response(content="\n".join(metrics), media_type="text/plain")
            
    except Exception as e:
        # Вернуть базовые метрики если не доступно
        basic_metrics = [
            f"# HELP rag_chatbot_status API status",
            f"# TYPE rag_chatbot_status gauge",
            f"rag_chatbot_status 1",
            f"",
            f"# Error: Analytics not available - {str(e)}"
        ]
        return Response(content="\n".join(basic_metrics), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=getattr(settings, 'host', '0.0.0.0'),
        port=getattr(settings, 'port', 8000),
        reload=getattr(settings, 'debug', False),
        log_level="info"
    )