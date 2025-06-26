# src/rag_chatbot/models/analytics.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    total_messages = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    user_agent = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)
    
    # Отношения
    conversations = relationship("Conversation", back_populates="session")
    analytics_events = relationship("AnalyticsEvent", back_populates="session")

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    response_time_ms = Column(Float, nullable=True)
    
    # Относящиеся к генерации 
    decision_type = Column(String, nullable=True)  # retrieve_document, search_knowledge_base, direct_answer
    success = Column(Boolean, default=True)
    total_cost = Column(Float, default=0.0)
    
    # Относящиеся к выбору документа
    document_retrieved = Column(String, nullable=True)
    document_path = Column(String, nullable=True)
    match_score = Column(Float, nullable=True)
    match_type = Column(String, nullable=True)
    
    # Относящиеся к поиску по тексту 
    queries_generated = Column(JSON, nullable=True)
    num_documents_used = Column(Integer, nullable=True)
    
    # Контекст чата
    chat_context_used = Column(Boolean, default=False)
    chat_context_length = Column(Integer, default=0)
    conversation_turn = Column(Integer, default=1)
    
    # Ограничитель 
    rate_limit_hit = Column(Boolean, default=False)
    
    # Отношения
    session = relationship("Session", back_populates="conversations")

class AnalyticsEvent(Base):
    __tablename__ = "analytics_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    event_type = Column(String, nullable=False)  # query, document_download, error, rate_limit
    event_data = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Отношения
    session = relationship("Session", back_populates="analytics_events")

class DocumentUsage(Base):
    __tablename__ = "document_usage"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_name = Column(String, nullable=False)
    document_path = Column(String, nullable=False)
    access_count = Column(Integer, default=1)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    total_downloads = Column(Integer, default=0)
    avg_match_score = Column(Float, nullable=True)

class QueryAnalytics(Base):
    __tablename__ = "query_analytics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(Text, nullable=False)
    query_hash = Column(String, nullable=False)  # Хэш для поиска дупликатов
    frequency = Column(Integer, default=1)
    avg_response_time = Column(Float, nullable=True)
    success_rate = Column(Float, default=1.0)
    last_used = Column(DateTime, default=datetime.utcnow)
    language_detected = Column(String, nullable=True)

class SystemMetrics(Base):
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    active_sessions = Column(Integer, default=0)
    total_requests = Column(Integer, default=0)
    avg_response_time = Column(Float, nullable=True)
    error_rate = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    vector_store_size = Column(Integer, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)