# file used to setup llm connections
from src.rag_chatbot.config.settings import settings
import json

if settings.llm_type == "OpenAI":
    from langchain_openai import ChatOpenAI
    params = json.loads(settings.llm_params)
    llm = ChatOpenAI(**params)
    print("Finished!")