from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # General settings
    app_name: str = "RAG Chatbot"
    version: str = "1.0.0"
    api_key: str = ""
    
    # OpenAI API settings
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-large-"

    # Vector database settings
    vector_store_path: str = "./data/faiss_index"

    # Chatbot settings
    session_timeout_minutes: int = 30
    max_memory_length: int = 10 # maximum number of messages to remember
    rate_limit_requests: int = 5
    rate_limit_window_minutes: int = 1 # in seconds, 1 minute
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
settings = Settings()