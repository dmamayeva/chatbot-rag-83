# src/rag_chatbot/tasks/analytics_tasks.py
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import schedule
import time
import threading

from src.rag_chatbot.core.database import get_db_session
from src.rag_chatbot.services.analytics_service import AnalyticsService
from src.rag_chatbot.utils.logger import logger

class AnalyticsTaskManager:
    """Manages background analytics tasks"""
    
    def __init__(self):
        self.running = False
        self.scheduler_thread = None
    
    def start(self):
        """Start the analytics task scheduler"""
        if self.running:
            return
        
        self.running = True
        
        # Schedule periodic tasks
        schedule.every(5).minutes.do(self._update_system_metrics)
        schedule.every(1).hours.do(self._cleanup_old_sessions)
        schedule.every(1).days.do(self._generate_daily_reports)
        schedule.every(1).weeks.do(self._cleanup_old_analytics)
        
        # Start scheduler in background thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Analytics task manager started")
    
    def stop(self):
        """Stop the analytics task scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Analytics task manager stopped")
    
    def _run_scheduler(self):
        """Run the scheduler in a loop"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
    
    def _update_system_metrics(self):
        """Update system metrics every 5 minutes"""
        try:
            with get_db_session() as db:
                analytics = AnalyticsService(db)
                analytics.update_system_metrics()
            logger.info("System metrics updated")
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def _cleanup_old_sessions(self):
        """Clean up old inactive sessions"""
        try:
            with get_db_session() as db:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                # Mark old sessions as inactive
                from src.rag_chatbot.models.analytics import Session
                old_sessions = db.query(Session).filter(
                    Session.last_accessed < cutoff_time,
                    Session.is_active == True
                ).all()
                
                for session in old_sessions:
                    session.is_active = False
                
                db.commit()
                logger.info(f"Marked {len(old_sessions)} sessions as inactive")
                
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
    
    def _generate_daily_reports(self):
        """Generate daily analytics reports"""
        try:
            with get_db_session() as db:
                analytics = AnalyticsService(db)
                
                # Generate report for yesterday
                yesterday = datetime.utcnow() - timedelta(days=1)
                report_data = analytics.get_dashboard_data(hours=24)
                
                # Here you could save the report to a file, send email, etc.
                logger.info(f"Daily report generated: {report_data.get('overview', {})}")
                
        except Exception as e:
            logger.error(f"Error generating daily reports: {e}")
    
    def _cleanup_old_analytics(self):
        """Clean up analytics data older than 90 days"""
        try:
            with get_db_session() as db:
                cutoff_time = datetime.utcnow() - timedelta(days=90)
                
                from src.rag_chatbot.models.analytics import (
                    Conversation, AnalyticsEvent, SystemMetrics
                )
                
                # Delete old conversations
                old_conversations = db.query(Conversation).filter(
                    Conversation.timestamp < cutoff_time
                ).delete()
                
                # Delete old events
                old_events = db.query(AnalyticsEvent).filter(
                    AnalyticsEvent.timestamp < cutoff_time
                ).delete()
                
                # Keep only daily system metrics (delete hourly ones older than 30 days)
                old_metrics_cutoff = datetime.utcnow() - timedelta(days=30)
                old_metrics = db.query(SystemMetrics).filter(
                    SystemMetrics.timestamp < old_metrics_cutoff
                ).delete()
                
                db.commit()
                logger.info(f"Cleaned up old analytics: {old_conversations} conversations, "
                           f"{old_events} events, {old_metrics} metrics")
                
        except Exception as e:
            logger.error(f"Error cleaning up old analytics: {e}")

# Global instance
analytics_task_manager = AnalyticsTaskManager()

# Alternative: Celery-based tasks (if you're using Celery)
# from celery import Celery
# 
# celery_app = Celery('analytics_tasks')
# 
# @celery_app.task
# def update_system_metrics():
#     """Celery task to update system metrics"""
#     try:
#         with get_db_session() as db:
#             analytics = AnalyticsService(db)
#             analytics.update_system_metrics()
#     except Exception as e:
#         logger.error(f"Celery task error: {e}")
# 
# @celery_app.task
# def cleanup_old_data():
#     """Celery task to cleanup old data"""
#     # Implementation here
#     pass