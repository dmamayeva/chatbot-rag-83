import os
import json
import logging
from typing import List, Tuple, Optional, Any, Dict
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.manager import get_openai_callback
from .prompts import query_generation_prompt, summary_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

DEFAULT_PARAMS = {
    "temperature": 0,
    "model": "gpt-4o"
}

def reciprocal_rank_fusion(results: List[List[Any]], k: int = 3) -> List[Tuple[Any, float]]:
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            # Use (page_content, sorted(metadata)) as a unique key
            doc_key = json.dumps({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }, sort_keys=True)
            if doc_key not in fused_scores:
                fused_scores[doc_key] = {"score": 0, "doc": doc}
            fused_scores[doc_key]["score"] += 1 / (rank + k)
    
    reranked_results = [
        (entry["doc"], entry["score"])
        for entry in sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    ]
    return reranked_results

def get_llm(params: Optional[dict] = None):
    # For langchain_openai, most params are keyword args
    return ChatOpenAI(**(params or DEFAULT_PARAMS))

def load_vectorstore(local_index_path: str, embedding_model):
    # Note: langchain_community.vectorstores
    return FAISS.load_local(local_index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)

def generate_queries(query: str, llm, n: int) -> Tuple[List[str], Dict[str, Any]]:
    """Generate queries and return both queries and metadata"""
    prompt = query_generation_prompt
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    
    # Track token usage for query generation
    with get_openai_callback() as cb:
        queries = chain.invoke({"original_query": query})
    
    # Extract metadata
    metadata = {
        "total_tokens": cb.total_tokens,
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "total_cost": cb.total_cost,
        "successful_requests": cb.successful_requests
    }
    
    queries = queries.split("\n")
    filtered_queries = [q.strip() for q in queries if q.strip()][:n]
    return filtered_queries, metadata

def retrieve_documents(queries: List[str], retriever, fusion_fn, top_k: int):
    # Each query: get docs, then fuse
    results = [retriever.invoke(q) for q in queries]
    fused = fusion_fn(results)
    return [doc for doc, _ in fused[:top_k]]

def summarize_answer(query: str, documents: List[str], llm, chat_context: str = None) -> Tuple[str, Dict[str, Any]]:
    """Summarize answer and return both answer and metadata"""
    chain = summary_prompt | llm | StrOutputParser()
    
    # Prepare chat context section for the prompt
    chat_context_section = ""
    if chat_context:
        chat_context_section = f"Previous conversation context:\n{chat_context}\n"
    
    # Track token usage for answer generation
    with get_openai_callback() as cb:
        answer = chain.invoke({
            "user_query": query,
            "documents": "\n\n".join(documents),
            "chat_context_section": chat_context_section
        })
    
    # Extract metadata
    metadata = {
        "total_tokens": cb.total_tokens,
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "total_cost": cb.total_cost,
        "successful_requests": cb.successful_requests
    }
    
    return answer, metadata

def rag_fusion_answer(
    user_query: str,
    local_index_path: str,
    embedding_model,
    mode: str = 'generated',
    num_generated_queries: int = 3,
    top_k: int = 3,
    params: Optional[dict] = None,
    chat_context: Optional[str] = None
) -> Tuple[str, dict]:
    """
    mode: 'original' | 'generated'
    chat_context: (Optional) Full conversation so far, to be given to LLM
    
    Returns:
        Tuple of (answer, metadata) where metadata includes:
        - llm_params: Parameters used for LLM
        - token_usage: Combined token usage from all LLM calls
        - query_generation_usage: Token usage for query generation (if applicable)
        - answer_generation_usage: Token usage for answer generation
        - queries_used: List of queries used for retrieval
        - num_documents_retrieved: Number of documents retrieved
    """
    llm = get_llm(params)
    vectorstore = load_vectorstore(local_index_path, embedding_model)
    retriever = vectorstore.as_retriever()
    
    # Initialize metadata
    metadata = {
        "llm_params": params or DEFAULT_PARAMS,
        "mode": mode,
        "num_generated_queries": num_generated_queries,
        "top_k": top_k,
        "token_usage": {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
            "successful_requests": 0
        },
        "query_generation_usage": None,
        "answer_generation_usage": None,
        "queries_used": [],
        "num_documents_retrieved": 0
    }
    
    if mode == 'original':
        queries = [user_query]
        logger.info("Running in original query mode.")
    elif mode == 'generated':
        queries, query_gen_metadata = generate_queries(user_query, llm, num_generated_queries)
        metadata["query_generation_usage"] = query_gen_metadata
        metadata["token_usage"]["total_tokens"] += query_gen_metadata["total_tokens"]
        metadata["token_usage"]["prompt_tokens"] += query_gen_metadata["prompt_tokens"]
        metadata["token_usage"]["completion_tokens"] += query_gen_metadata["completion_tokens"]
        metadata["token_usage"]["total_cost"] += query_gen_metadata["total_cost"]
        metadata["token_usage"]["successful_requests"] += query_gen_metadata["successful_requests"]
        logger.info(f"Running in query generation mode with {num_generated_queries} queries: {queries}")
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    metadata["queries_used"] = queries
    
    # Retrieve documents
    docs = retrieve_documents(queries, retriever, reciprocal_rank_fusion, top_k)
    metadata["num_documents_retrieved"] = len(docs)
    
    # Extract document content
    top_docs_content = [doc.page_content for doc in docs]
    
    # Generate final answer with metadata tracking
    final_answer, answer_gen_metadata = summarize_answer(
        user_query,
        top_docs_content,
        llm,
        chat_context=chat_context
    )
    
    # Update metadata with answer generation usage
    metadata["answer_generation_usage"] = answer_gen_metadata
    metadata["token_usage"]["total_tokens"] += answer_gen_metadata["total_tokens"]
    metadata["token_usage"]["prompt_tokens"] += answer_gen_metadata["prompt_tokens"]
    metadata["token_usage"]["completion_tokens"] += answer_gen_metadata["completion_tokens"]
    metadata["token_usage"]["total_cost"] += answer_gen_metadata["total_cost"]
    metadata["token_usage"]["successful_requests"] += answer_gen_metadata["successful_requests"]
    
    # Calculate total cost
    metadata["total_price"] = metadata["token_usage"]["total_cost"]
    
    return final_answer, metadata

# Usage examples:
# answer, meta = rag_fusion_answer("What is RAG?", "./faiss_index", embedding_model, mode="original")
# print(f"Answer: {answer}")
# print(f"Total tokens used: {meta['token_usage']['total_tokens']}")
# print(f"Total cost: ${meta['token_usage']['total_cost']:.4f}")

# answer, meta = rag_fusion_answer("What is RAG?", "./faiss_index", embedding_model, mode="generated", num_generated_queries=5)
# print(f"Queries generated: {meta['queries_used']}")
# print(f"Query generation cost: ${meta['query_generation_usage']['total_cost']:.4f}")
# print(f"Answer generation cost: ${meta['answer_generation_usage']['total_cost']:.4f}")