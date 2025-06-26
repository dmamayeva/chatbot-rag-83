# src/rag_chatbot/services/analytics_service.py
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func, desc, and_
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import hashlib
import json
import psutil
import os
from collections import defaultdict

from src.rag_chatbot.models.analytics import (
    Session, Conversation, AnalyticsEvent, 
    DocumentUsage, QueryAnalytics, SystemMetrics
)
from src.rag_chatbot.utils.logger import logger

class AnalyticsService:
    def __init__(self, db_session: DBSession):
        self.db = db_session
    
    def track_conversation(self, session_id: str, user_message: str, 
                          bot_response: str, metadata: Dict[str, Any],
                          response_time_ms: float = None) -> None:
        """Отслеживание переписки"""
        try:
            # Проверка есть ли сессия
            session = self.db.query(Session).filter(Session.id == session_id).first()
            if not session:
                session = Session(id=session_id, total_messages=0)
                self.db.add(session)
            
            # Обновление данных сессии
            session.last_accessed = datetime.utcnow()
            session.total_messages = (session.total_messages or 0) + 1
            
            # Создание записи переписки
            conversation = Conversation(
                session_id=session_id,
                user_message=user_message,
                bot_response=bot_response,
                response_time_ms=response_time_ms,
                decision_type=metadata.get('decision'),
                success=metadata.get('success', True),
                total_cost=metadata.get('total_cost', 0.0),
                document_retrieved=metadata.get('document_name'),
                document_path=metadata.get('file_path'),
                match_score=metadata.get('match_score'),
                match_type=metadata.get('match_type'),
                queries_generated=metadata.get('queries_used'),
                num_documents_used=metadata.get('num_documents'),
                chat_context_used=metadata.get('chat_context_used', False),
                chat_context_length=metadata.get('chat_context_length', 0),
                conversation_turn=metadata.get('conversation_turn', 1),
                rate_limit_hit=False  # Обновляется если превышен лимит
            )
            
            self.db.add(conversation)
            
            # Отслеживание данных если было решено достать документ 
            if metadata.get('decision') == 'retrieve_document' and metadata.get('success'):
                self._track_document_usage(
                    metadata.get('document_name'),
                    metadata.get('file_path'),
                    metadata.get('match_score')
                )
            
            # Отслеживание аналитики запроса
            self._track_query_analytics(user_message, response_time_ms, metadata.get('success', True))
            
            # Создание ивента аналитики 
            self._track_event(session_id, 'conversation', {
                'decision_type': metadata.get('decision'),
                'success': metadata.get('success', True),
                'response_time_ms': response_time_ms
            })
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error tracking conversation: {e}")
            self.db.rollback()
    
    def track_rate_limit(self, session_id: str, retry_after: int) -> None:
        """Отслеживание событий ограничения"""
        try:
            self._track_event(session_id, 'rate_limit', {
                'retry_after_seconds': retry_after,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Обновление последней переписки для отметки срабатывания ограничения скорости
            last_conv = self.db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).order_by(desc(Conversation.timestamp)).first()
            
            if last_conv:
                last_conv.rate_limit_hit = True
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error tracking rate limit: {e}")
            self.db.rollback()
    
    def track_document_download(self, session_id: str, document_name: str, 
                               file_size_mb: float) -> None:
        """Отслеживание загрузок документов"""
        try:
            self._track_event(session_id, 'document_download', {
                'document_name': document_name,
                'file_size_mb': file_size_mb
            })
            
            # Обновление статистики использования документа
            doc_usage = self.db.query(DocumentUsage).filter(
                DocumentUsage.document_name == document_name
            ).first()
            
            if doc_usage:
                doc_usage.total_downloads += 1
                doc_usage.last_accessed = datetime.utcnow()
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error tracking document download: {e}")
            self.db.rollback()
    
    def track_error(self, session_id: str, error_type: str, error_message: str) -> None:
        """Отслеживание системных ошибок"""
        try:
            self._track_event(session_id, 'error', {
                'error_type': error_type,
                'error_message': error_message
            })
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error tracking error event: {e}")
            self.db.rollback()
    
    def update_system_metrics(self) -> None:
        """Обновление системных метрик"""
        try:
            # Вычисление текущих метрик
            now = datetime.utcnow()
            last_hour = now - timedelta(hours=1)
            
            # Активные сессии (доступ за последний час)
            active_sessions = self.db.query(Session).filter(
                Session.last_accessed >= last_hour
            ).count()
            
            # Общее количество запросов за последний час
            total_requests = self.db.query(Conversation).filter(
                Conversation.timestamp >= last_hour
            ).count()
            
            # Среднее время отклика

            avg_response_time = self.db.query(func.avg(Conversation.response_time_ms)).filter(
                Conversation.timestamp >= last_hour,
                Conversation.response_time_ms.isnot(None)
            ).scalar() or 0.0
            
            # Частота ошибок
            total_conversations = self.db.query(Conversation).filter(
                Conversation.timestamp >= last_hour
            ).count()
            
            failed_conversations = self.db.query(Conversation).filter(
                Conversation.timestamp >= last_hour,
                Conversation.success == False
            ).count()
            
            error_rate = (failed_conversations / total_conversations * 100) if total_conversations > 0 else 0.0
            
            # Общая сумма
            total_cost = self.db.query(func.sum(Conversation.total_cost)).filter(
                Conversation.timestamp >= last_hour
            ).scalar() or 0.0
            
            # Использование системной памяти
            memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Создание записи системных метрик
            metrics = SystemMetrics(
                active_sessions=active_sessions,
                total_requests=total_requests,
                avg_response_time=avg_response_time,
                error_rate=error_rate,
                total_cost=total_cost,
                memory_usage_mb=memory_usage_mb
            )
            
            self.db.add(metrics)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
            self.db.rollback()
    
    def get_dashboard_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Базовые статистики
            total_sessions = self.db.query(Session).filter(
                Session.created_at >= cutoff_time
            ).count()
            
            total_conversations = self.db.query(Conversation).filter(
                Conversation.timestamp >= cutoff_time
            ).count()
            
            # Частота успеха
            successful_conversations = self.db.query(Conversation).filter(
                Conversation.timestamp >= cutoff_time,
                Conversation.success == True
            ).count()
            
            success_rate = (successful_conversations / total_conversations * 100) if total_conversations > 0 else 0.0
            
            # Распределение типов решений
            decision_stats = self.db.query(
                Conversation.decision_type,
                func.count(Conversation.id).label('count')
            ).filter(
                Conversation.timestamp >= cutoff_time
            ).group_by(Conversation.decision_type).all()
            
            # Топ документов
            top_documents = self.db.query(
                DocumentUsage.document_name,
                DocumentUsage.access_count,
                DocumentUsage.total_downloads
            ).order_by(desc(DocumentUsage.access_count)).limit(10).all()
            
            # Топ запросов
            top_queries = self.db.query(
                QueryAnalytics.query_text,
                QueryAnalytics.frequency
            ).order_by(desc(QueryAnalytics.frequency)).limit(10).all()
            
            # # Почасовые тренды переписок (совместимо с SQLite)
            hourly_stats = self.db.query(
                func.strftime('%Y-%m-%d %H:00:00', Conversation.timestamp).label('hour'),
                func.count(Conversation.id).label('count')
            ).filter(
                Conversation.timestamp >= cutoff_time
            ).group_by(func.strftime('%Y-%m-%d %H:00:00', Conversation.timestamp)).order_by('hour').all()
            
           # Статистика времени отклика (совместимо с SQLite — без percentile_cont)
            response_time_stats = self.db.query(
                func.avg(Conversation.response_time_ms).label('avg'),
                func.min(Conversation.response_time_ms).label('min'),
                func.max(Conversation.response_time_ms).label('max')
            ).filter(
                Conversation.timestamp >= cutoff_time,
                Conversation.response_time_ms.isnot(None)
            ).first()
            
            # Анализ стоимости 
            total_cost = self.db.query(func.sum(Conversation.total_cost)).filter(
                Conversation.timestamp >= cutoff_time
            ).scalar() or 0.0
            
            # События ограничения
            rate_limit_events = self.db.query(AnalyticsEvent).filter(
                AnalyticsEvent.timestamp >= cutoff_time,
                AnalyticsEvent.event_type == 'rate_limit'
            ).count()
            
            return {
                'overview': {
                    'total_sessions': total_sessions,
                    'total_conversations': total_conversations,
                    'success_rate': round(success_rate, 2),
                    'total_cost': round(total_cost, 4),
                    'rate_limit_events': rate_limit_events
                },
                'decision_distribution': [
                    {'type': stat[0] or 'unknown', 'count': stat[1]} 
                    for stat in decision_stats
                ],
                'top_documents': [
                    {
                        'name': doc[0], 
                        'access_count': doc[1], 
                        'downloads': doc[2]
                    } for doc in top_documents
                ],
                'top_queries': [
                    {'query': query[0][:100], 'frequency': query[1]} 
                    for query in top_queries
                ],
                'hourly_trends': [
                    {
                        'hour': stat[0] if stat[0] else None, 
                        'count': stat[1]
                    } for stat in hourly_stats
                ],
                'response_times': {
                    'avg': round((response_time_stats.avg or 0) / 1000, 3),
                    'min': round((response_time_stats.min or 0) / 1000, 3),
                    'max': round((response_time_stats.max or 0) / 1000, 3)
                } if response_time_stats else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}
    
    def _track_document_usage(self, document_name: str, document_path: str, 
                             match_score: float) -> None:
        """Внутренний метод для отслеживания использования документа"""
        if not document_name:
            return
            
        doc_usage = self.db.query(DocumentUsage).filter(
            DocumentUsage.document_name == document_name
        ).first()
        
        if doc_usage:
            doc_usage.access_count += 1
            doc_usage.last_accessed = datetime.utcnow()
            # Обновление среднего коэффициента совпадения
            if match_score:
                if doc_usage.avg_match_score:
                    doc_usage.avg_match_score = (doc_usage.avg_match_score + match_score) / 2
                else:
                    doc_usage.avg_match_score = match_score
        else:
            doc_usage = DocumentUsage(
                document_name=document_name,
                document_path=document_path or '',
                access_count=1,
                avg_match_score=match_score
            )
            self.db.add(doc_usage)
    
    def _track_query_analytics(self, query_text: str, response_time: float, 
                              success: bool) -> None:
        """Внутренний метод для отслеживания аналитики запросов"""
        query_hash = hashlib.md5(query_text.lower().encode()).hexdigest()
        
        query_analytics = self.db.query(QueryAnalytics).filter(
            QueryAnalytics.query_hash == query_hash
        ).first()
        
        if query_analytics:
            query_analytics.frequency += 1
            query_analytics.last_used = datetime.utcnow()
            
            # Обновление среднего времени отклика
            if response_time and query_analytics.avg_response_time:
                query_analytics.avg_response_time = (
                    query_analytics.avg_response_time + response_time
                ) / 2
            elif response_time:
                query_analytics.avg_response_time = response_time
            
            # Обновление коэффициента успешных ответов
            total_uses = query_analytics.frequency
            current_successes = query_analytics.success_rate * (total_uses - 1)
            new_successes = current_successes + (1 if success else 0)
            query_analytics.success_rate = new_successes / total_uses
            
        else:
            query_analytics = QueryAnalytics(
                query_text=query_text,
                query_hash=query_hash,
                frequency=1,
                avg_response_time=response_time,
                success_rate=1.0 if success else 0.0
            )
            self.db.add(query_analytics)
    
    def _track_event(self, session_id: str, event_type: str, event_data: Dict) -> None:
        """Внутренний метод для отслеживания аналитических событий"""
        event = AnalyticsEvent(
            session_id=session_id,
            event_type=event_type,
            event_data=event_data
        )
        self.db.add(event)