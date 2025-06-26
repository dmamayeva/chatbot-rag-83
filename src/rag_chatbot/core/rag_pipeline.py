from rag_pipeline.rag_fusion_pipeline import rag_fusion_answer
from rag_pipeline.agent import *
from langchain_openai import OpenAIEmbeddings
from src.rag_chatbot.config.settings import settings

class RAGPipeline:
    def __init__(self):
        self.agent = create_unified_rag_agent(
            local_index_path = settings.vector_store_path,
            embedding_model = OpenAIEmbeddings(model=settings.embedding_model),
            documents_json_path = settings.document_json_path,
            memory_window_size = settings.memory_window_size,
            document_embeddings_path=settings.document_embeddings_path)
    
    def get_response(self, user_query: str, chat_context: str = "", mode: str = "generated"):
        return  self.agent.process_query(user_query)
    