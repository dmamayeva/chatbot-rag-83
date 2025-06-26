import json
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import mimetypes
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
from openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.manager import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from datetime import datetime

from src.rag_chatbot.core.llm import llm
from src.rag_chatbot.utils.logger import logger
from rag_pipeline.prompts import chatbot_prompt, query_generation_prompt


class UnifiedRAGAgent:
    """
    Возможности:
    1. Доставание нужных PDF-документов
    2. Система решений — ответить прямо, искать информацию в базе данных или достать документ
    3. RAG Fusion для генерации дополнительных запросов для поиска (перефразирование запроса пользователя)
    4. Память о разговоре
    """
    
    def __init__(self,
                 local_index_path: str,
                 embedding_model,
                 documents_json_path: str,
                 llm_params: Optional[Dict] = None,
                 openai_embedding_model: str = "text-embedding-3-small",
                 openai_api_key: Optional[str] = None,
                 memory_window_size: int = 10,
                 document_embeddings_path: Optional[str] = "document_embeddings.npy"):
        """
       Инициализация агентов
        Args:
            local_index_path: путь к векторной базе FAISS
            embedding_model: эмбеддинги базы 
            documents_json_path: путь до JSON с названиями и адресами документов
            llm_params: параметры LLM
            openai_embedding_model: OpenAI embedding model для поиска докуметов
            openai_api_key: OpenAI API key
            memory_window_size: окно разговора для добавления в контекст
        """
        self.document_embeddings_path = document_embeddings_path
        self.local_index_path = local_index_path
        self.embedding_model = embedding_model
        self.documents_json_path = documents_json_path
        self.openai_embedding_model = openai_embedding_model
        self.memory_window_size = memory_window_size
        
        self.llm = llm
        # for embeddings: since the FAISS base is using OpenAI embeddings, it's necessary to have this one. 
        # In case you want to use other embeddings—REBUILD vector index
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()
        
        # Инициализация памяти
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.conversation_history = []
        
        self.document_mappings = self._load_document_mappings()
        self._load_or_build_semantic_index()
        
        # Объявление всех инструментов LLM. Для добавления новый инструментов—добавлять сюда. 
        self.function_definitions = [
            {
                "name": "retrieve_document",
                "description": "Retrieve a specific PDF document by name when the user explicitly asks for a document, manual, guide, or file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document_query": {
                            "type": "string",
                            "description": "The user's description or name of the document they want"
                        }
                    },
                    "required": ["document_query"]
                }
            },
            {
                "name": "search_knowledge_base",
                "description": "Search the knowledge base for specific information using RAG",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant information"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["generated"],
                            "description": "Search mode: 'generated' creates multiple related queries"
                        },
                        "num_queries": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Number of queries to generate if using 'generated' mode"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        
        self.decision_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are Zaure (Зауре) — a professional AI legal assistant created by the company Orleu. You help users with questions about regulatory documents in education, especially concerning аттестация педагогов, квалификационные категории, and ОЗП (Педагогтердің білімін бағалау).

Available capabilities:
1. **Document Retrieval** — when the user explicitly asks for a document, PDF, manual, guide, or mentions "download".
   - Available documents: {self._get_document_list_summary()}

2. **Knowledge Base Search** — when the user asks questions requiring regulatory information from documents.

Decision Guidelines:
- Use **Document Retrieval** when: The user explicitly asks for a document/PDF/manual/download.
- Use **Knowledge Base Search** when: The user asks regulatory/procedural questions requiring details from documents.
- If a user asks something unrelated to regulatory documents and it is **not connected to the chat context**, politely reply:
   - In **Russian**: "Извините, я — AI-ассистент разработанный АО Орлеу и помогаю только с вопросами о нормативных документах в сфере образования. Есть ли у вас вопрос по правилам, с которым я могу помочь?"
   - In **Kazakh**: "Кешіріңіз, мен АО «Өрлеу» әзірлеген AI-ассистентпін және тек білім беру саласындағы нормативтік құжаттарға қатысты сұрақтарға жауап беремін. Осы тақырыпта сұрағыңыз бар ма?"
   - In **English**: "Sorry, am an AI assistant developed by JSC "Órleu" and I assist with questions about education regulations. Do you have a question about education rules I can help with?"
- If the query is unrelated on its own but logically connected to the chat context (as a clarification or follow-up), proceed with answering as usual.

Additionally:
- Always answer in the language the user used.
- In Kazakh queries, translate mentions of "ОЗП" to "ПББ" in your answers.
- Maintain a professional, formal tone.

Conversation context so far:
{{chat_context}}

Now analyze the user query and decide whether to retrieve a document, search the knowledge base, answer directly, or politely redirect."""),
            ("user", "{user_query}")
        ])

    def _save_document_embeddings(self):
        """Сохранение базы данных документов (эмбеддингов)"""
        try:
            data = {
                "document_names": self.document_names,
                "document_embeddings": self.document_embeddings
            }
            os.makedirs(os.path.dirname(self.document_embeddings_path), exist_ok=True)
            np.save(self.document_embeddings_path, data)
            logger.info(f"Saved document embeddings to {self.document_embeddings_path}")
        except Exception as e:
            logger.error(f"Failed to save document embeddings: {e}")

    def _load_or_build_semantic_index(self):
        """Подгрузить базу данных документов, если её нет—создать и сохранить """
        if os.path.exists(self.document_embeddings_path):
            try:
                logger.info(f"Loading cached document embeddings from {self.document_embeddings_path}")
                data = np.load(self.document_embeddings_path, allow_pickle=True).item()
                self.document_names = data["document_names"]
                self.document_embeddings = data["document_embeddings"]
                return
            except Exception as e:
                logger.error(f"Failed to load cached embeddings: {e}")
        
        self._build_semantic_index()
        self._save_document_embeddings()

    def _load_document_mappings(self) -> Dict[str, str]:
        """Подгрузить JSON с информацией о документах"""
        try:
            with open(self.documents_json_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
            logger.info(f"Loaded {len(mappings)} document mappings")
            return mappings
        except FileNotFoundError:
            logger.error(f"Document mappings file not found: {self.documents_json_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing document mappings JSON: {e}")
            return {}

    def _get_openai_embedding(self, text: str) -> Optional[List[float]]:
        """Получение эмбеддингов"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.openai_embedding_model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None

    def _build_semantic_index(self):
        """Создание базы для документов"""
        if not self.document_mappings:
            self.document_names = []
            self.document_embeddings = np.array([])
            return
        
        self.document_names = list(self.document_mappings.keys())
        embeddings = []
        
        for doc_name in self.document_names:
            searchable_text = f"{doc_name} {doc_name.lower()} {re.sub(r'[_\\-]', ' ', doc_name.lower())}"
            embedding = self._get_openai_embedding(searchable_text)
            if embedding:
                embeddings.append(embedding)
        
        if embeddings:
            self.document_embeddings = np.array(embeddings)
            logger.info(f"Built semantic index for {len(embeddings)} documents")
        else:
            self.document_embeddings = np.array([])

    def _get_document_list_summary(self) -> str:
        """Информация о доступных документах"""
        if not self.document_mappings:
            return "No documents available"
        return f"{len(self.document_mappings)} documents including: " + ", ".join(list(self.document_mappings.keys())[:5]) + "..."

    def _find_document_semantic(self, query: str) -> Optional[Tuple[str, str, float]]:
        """Поиск документов по запросу пользователя"""
        if len(self.document_embeddings) == 0:
            return None
        
        query_embedding = self._get_openai_embedding(query)
        if not query_embedding:
            return None
        
        query_embedding = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        best_idx = np.argmax(similarities)
        
        if similarities[best_idx] > 0.3:  # порог
            doc_name = self.document_names[best_idx]
            doc_path = self.document_mappings[doc_name]
            if os.path.exists(doc_path):
                return doc_name, doc_path, similarities[best_idx]
        
        return None

    def retrieve_document(self, document_query: str) -> Dict[str, Any]:
        """Достать документ с помощью семантического поиска"""
        logger.info(f"Searching for document: '{document_query}'")
        
        # Попробовать найти по полному названию
        for doc_name, doc_path in self.document_mappings.items():
            if document_query.lower() in doc_name.lower() or doc_name.lower() in document_query.lower():
                if os.path.exists(doc_path):
                    return self._create_document_response(doc_name, doc_path, 1.0, "exact_match")
        
        # Семантический поиск
        result = self._find_document_semantic(document_query)
        if result:
            doc_name, doc_path, score = result
            return self._create_document_response(doc_name, doc_path, score, "semantic_match")
        
        # Нет результата
        return {
            "success": False,
            "message": f"No document found matching '{document_query}'",
            "available_documents": list(self.document_mappings.keys())
        }

    def _create_document_response(self, doc_name: str, doc_path: str, score: float, match_type: str) -> Dict[str, Any]:
        """Стандартный ответ документа"""
        file_stat = os.stat(doc_path)
        file_size_mb = file_stat.st_size / (1024 * 1024)
        
        return {
            "success": True,
            "document_name": doc_name,
            "file_path": doc_path,
            "file_size_mb": round(file_size_mb, 2),
            "match_score": round(score, 3),
            "match_type": match_type,
            "message": f"Document '{doc_name}' found and ready for download."
        }

    def search_knowledge_base(self, query: str, mode: str = "generated", num_queries: int = 3) -> Dict[str, Any]:
        """Поиск по базе данных с генерацией дополнительных запросов"""
        try:
            logger.info(f"Searching knowledge base: '{query}' in {mode} mode")
            
            # Загрузка базы данны 
            vectorstore = FAISS.load_local(
                self.local_index_path,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
            retriever = vectorstore.as_retriever()
            
            # Генерация дополнительных запросов
            if mode == "generated":
                # История для контекста
                context = self._get_conversation_context()
                queries = self._generate_queries(query, num_queries, context)
            else:
                queries = [query]
            
            # Достать документы
            all_docs = []
            for q in queries:
                docs = retriever.invoke(q)
                all_docs.append(docs)
            
            # Повторная оценка результатов 
            fused_docs = self._reciprocal_rank_fusion(all_docs)
            top_docs = fused_docs[:3]  # Toп 3 документа
            
            # Сгенерировать ответ 
            context = self._get_conversation_context()
            answer = self._generate_answer(query, top_docs, context)
            
            return {
                "answer": answer,
                "success": True,
                "queries_used": queries,
                "num_documents": len(top_docs)
            }
            
        except Exception as e:
            logger.error(f"Error in knowledge base search: {str(e)}")
            return {
                "answer": f"Error searching knowledge base: {str(e)}",
                "success": False,
                "error": str(e)
            }

    def _generate_queries(self, original_query: str, n: int, chat_context: str = "") -> List[str]:
        """Генерация дополнительных запросов для поиска по базе данных"""
        prompt = query_generation_prompt
        chain = prompt | self.llm | StrOutputParser()
        
        # Включить предыдущий разговор для контекста
        result = chain.invoke({
            "query": original_query, 
            "n": n,
            "chat_context": chat_context or "No previous conversation context."
        })
        
        queries = [q.strip() for q in result.split("\n") if q.strip()]
        
        # Хотя бы оригинальный запрос 
        if not queries:
            queries = [original_query]
        
        # Логирование дополнительных заппросов
        logger.info(f"Generated {len(queries)} queries with context: {queries}")
        
        return queries[:n]

    def _reciprocal_rank_fusion(self, results: List[List[Any]], k: int = 3) -> List[Any]:
        """Сортировка результатов"""
        fused_scores = {}
        
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_key = doc.page_content[:100]  # Use first 100 chars as key
                if doc_key not in fused_scores:
                    fused_scores[doc_key] = {"score": 0, "doc": doc}
                fused_scores[doc_key]["score"] += 1 / (rank + k)
        
        sorted_docs = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]

    def _generate_answer(self, query: str, documents: List[Any], context: str) -> str:
        """Сгенерировать ответ для генерации """
        prompt = chatbot_prompt
        doc_contents = "\n\n".join([f"Exracted from: {doc.metadata['doc_info']}: {doc.page_content}" for doc in documents])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "query": query,
            "documents": doc_contents,
            "context": context
        })

    def _get_conversation_context(self) -> str:
        """Форматирование разговора для контекста"""
        if not self.conversation_history:
            return "No previous conversation."
        
        context_parts = []
        for turn in self.conversation_history[-self.memory_window_size:]:
            context_parts.append(f"User: {turn['user']}")
            context_parts.append(f"Assistant: {turn['assistant']}")
        
        return "\n".join(context_parts)

    def _update_memory(self, user_query: str, response: str):
        """Обновление памяти разговора"""
        self.conversation_history.append({
            "user": user_query,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Последние N разговоров
        if len(self.conversation_history) > self.memory_window_size:
            self.conversation_history = self.conversation_history[-self.memory_window_size:]
        
        # Обновление памяти
        self.memory.save_context({"input": user_query}, {"answer": response})

    def process_query(self, user_query: str, verbose: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Основная функция для обработки запроса
        Returns:
            Tuple of (answer, metadata)
        """
        metadata = {
            "user_query": user_query,
            "timestamp": datetime.now().isoformat(),
            "decision": None,
            "success": False,
            "total_cost": 0.0
        }
        
        try:
            context = self._get_conversation_context()
            
            chain = self.decision_prompt | self.llm.bind(
                functions=self.function_definitions,
                function_call="auto"
            )
            
            with get_openai_callback() as cb:
                response = chain.invoke({
                    "user_query": user_query,
                    "chat_context": context
                })
                metadata["decision_cost"] = cb.total_cost
                metadata["total_cost"] += cb.total_cost
            
            if hasattr(response, 'additional_kwargs') and 'function_call' in response.additional_kwargs:
                function_call = response.additional_kwargs['function_call']
                function_name = function_call['name']
                function_args = json.loads(function_call['arguments'])
                
                metadata["decision"] = function_name
                metadata["function_args"] = function_args
                
                if function_name == "retrieve_document":
                    # Возрращение документа
                    result = self.retrieve_document(function_args['document_query'])
                    if result['success']:
                        answer = (
                            f"Найден документ: **{result['document_name']}**\n\n"
                            f"Файл: {os.path.basename(result['file_path'])}\n"
                            f"Размер: {result['file_size_mb']} MB\n"
                            # f"Расположение: {result['file_path']}\n"
                            f"Соответствие: {result['match_type']} (score: {result['match_score']})"
                        )
                    else:
                        answer = result['message']
                        if result.get('available_documents'):
                            answer += f"\n\nAvailable documents:\n" + "\n".join(
                                [f"• {doc}" for doc in result['available_documents'][:10]]
                            )
                    metadata.update(result)
                
                elif function_name == "search_knowledge_base":
                    # Поиск по базе знаний
                    result = self.search_knowledge_base(
                        query=function_args.get('query', user_query),
                        mode=function_args.get('mode', 'generated'),
                        num_queries=function_args.get('num_queries', 3)
                    )
                    answer = result['answer']
                    metadata.update(result)
                
                else:
                    answer = "Unknown function called"
            
            else:
                metadata["decision"] = "direct_answer"
                answer = response.content
            
            self._update_memory(user_query, answer)
            metadata["success"] = True
            
            if verbose:
                self._print_verbose_output(metadata)
            
            return answer, metadata
            
        except Exception as e:
            print(e)
            logger.error(f"Error processing query: {str(e)}")
            error_msg = f"I encountered an error: {str(e)}"
            metadata["error"] = str(e)
            metadata["success"] = False
            return error_msg, metadata

    def _print_verbose_output(self, metadata: Dict[str, Any]):
        """Распечатка результатов"""
        print(f"\n{'='*60}")
        print(f"UNIFIED RAG AGENT EXECUTION REPORT")
        print(f"{'='*60}")
        print(f"Query: {metadata['user_query']}")
        print(f"Decision: {metadata['decision']}")
        print(f"Success: {metadata['success']}")
        print(f"Total Cost: ${metadata.get('total_cost', 0):.4f}")
        
        if metadata.get('function_args'):
            print(f"Function Args: {metadata['function_args']}")
        
        if metadata['decision'] == 'search_knowledge_base':
            print(f"Queries Used: {metadata.get('queries_used', [])}")
            print(f"Documents Retrieved: {metadata.get('num_documents', 0)}")
        elif metadata['decision'] == 'retrieve_document':
            print(f"Match Type: {metadata.get('match_type', 'N/A')}")
            print(f"Match Score: {metadata.get('match_score', 'N/A')}")
        
        print(f"Conversation History Length: {len(self.conversation_history)}")
        print(f"{'='*60}\n")

    def chat(self, user_query: str, verbose: bool = False) -> str:
        """Простой интерфейс, который возвращает только ответ"""
        answer, _ = self.process_query(user_query, verbose)
        return answer

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Возвращает историю переписки"""
        return self.conversation_history

    def clear_memory(self):
        """Очистка памяти"""
        self.conversation_history = []
        self.memory.clear()
        logger.info("Conversation memory cleared")

    def list_documents(self) -> str:
        """Показать все доступные документы"""
        if not self.document_mappings:
            return "No documents available."
        
        doc_list = []
        for i, (name, path) in enumerate(self.document_mappings.items(), 1):
            exists = "EXISTS:" if os.path.exists(path) else "NOT:"
            doc_list.append(f"{i}. {exists} {name}")
        
        return f"**Available Documents ({len(self.document_mappings)}):**\n" + "\n".join(doc_list)

def create_unified_rag_agent(
    local_index_path: str,
    embedding_model,
    documents_json_path: str,
    document_embeddings_path: str,
    openai_api_key: Optional[str] = None,
    memory_window_size: int = 10,
) -> UnifiedRAGAgent:
    """
    Создание объекта агента
    
    Args:
        local_index_path: путь до базы FAISS
        embedding_model: модель эмбеддингов для FAISS
        documents_json_path: путь до JSON с названиями и адресами документов
        openai_api_key: OpenAI API key
        memory_window_size: окно разговора для добавления в контекст
    
    Returns:
        UnifiedRAGAgent instance
    """
    return UnifiedRAGAgent(
        local_index_path=local_index_path,
        embedding_model=embedding_model,
        documents_json_path=documents_json_path,
        llm_params={"temperature": 0, "model": "gpt-4o"},
        openai_embedding_model="text-embedding-3-small",
        openai_api_key=openai_api_key,
        memory_window_size=memory_window_size,
        document_embeddings_path=document_embeddings_path
    )