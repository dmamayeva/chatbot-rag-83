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
from src.rag_chatbot.tasks.analytics_tasks import analytics_task_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown with analytics"""
    logger.info("Starting RAG Chatbot API with Analytics...")
    
    # Initialize database for analytics
    try:
        init_database()
        logger.info("Analytics database initialized successfully")
        
        # Check database health
        if not check_database_health():
            logger.warning("Database health check failed, continuing without analytics")
        else:
            logger.info("Database health check passed")
    except Exception as e:
        logger.error(f"Analytics database initialization failed: {e}")
        logger.warning("Continuing without analytics database")
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(
        periodic_cleanup(chat.session_manager, chat.rate_limiter)
    )
    
    # Start analytics task manager
    try:
        analytics_task_manager.start()
        logger.info("Analytics task manager started")
    except Exception as e:
        logger.error(f"Failed to start analytics task manager: {e}")
    
    try:
        yield
    finally:
        # Shutdown
        logger.info("Shutting down RAG Chatbot API...")
        
        # Stop analytics task manager
        try:
            analytics_task_manager.stop()
            logger.info("Analytics task manager stopped")
        except Exception as e:
            logger.error(f"Error stopping analytics task manager: {e}")
        
        # Cancel cleanup task
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        
        logger.info("RAG Chatbot API shutdown completed")

app = FastAPI(
    title="RAG Chatbot API with Analytics",
    description="Production-ready RAG chatbot with session-based conversation memory, rate limiting, and comprehensive analytics",
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
app.include_router(sessions.router, prefix="/sessions", tags=["sessions-direct"])

# Mount static files for analytics dashboard
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Root endpoint - fixes the 404 error
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Chatbot API with Analytics",
        "version": "1.0.0",
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

# Analytics Dashboard endpoint
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

# Serve the HTML chat interface
@app.get("/chat-ui", include_in_schema=False)
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

# Health check endpoint with analytics status
@app.get("/health-extended", include_in_schema=False)
async def extended_health_check():
    """Extended health check including analytics status"""
    try:
        # Basic health from your existing health router
        basic_health = True  # You can call your existing health check here
        
        # Analytics database health
        db_healthy = check_database_health()
        
        # Analytics task manager status
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
        # Return basic metrics if analytics is not available
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