from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # General settings
    app_name: str = "RAG Chatbot"
    version: str = "1.0.0"
    api_key: str = ""
    
    embedding_model: str = "text-embedding-3-large-"

    # LLM settings
    llm_type: str = "OpenAI"
    openai_api_key: str = ""
    llm_params: str = ""

    # Vector database settings
    vector_store_path: str = "./data/faiss_index"
    document_json_path: str = "./data/documents.json"
    document_embeddings_path: str = "./data/document_embeddings.npy"

    # memory settings
    memory_window_size: int = 5 # number of previous messages to remember in the chat context

    # Chatbot settings
    session_timeout_minutes: int = 30
    max_memory_length: int = 10 # maximum number of messages to remember
    rate_limit_requests: int = 5
    rate_limit_window_minutes: int = 1
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
settings = Settings()