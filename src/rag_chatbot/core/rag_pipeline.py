from rag_pipeline.rag_fusion_pipeline import rag_fusion_answer
from langchain_openai import OpenAIEmbeddings
from src.rag_chatbot.config.settings import settings

class RAGPipeline:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings(model=settings.embedding_model)
        self.vector_store_path = settings.vector_store_path
    
    async def get_response(self, user_query: str, chat_context: str = "", mode: str = "original"):
        return rag_fusion_answer(
            user_query=user_query,
            local_index_path=self.vector_store_path,
            embedding_model=self.embedding_model,
            mode=mode,
            chat_context=chat_context
        )

