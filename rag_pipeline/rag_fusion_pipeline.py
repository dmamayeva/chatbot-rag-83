import os
import json
import logging
from typing import List, Tuple, Optional, Any
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

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

def generate_queries(query: str, llm, n: int) -> List[str]:
    prompt = query_generation_prompt
    output_parser = StrOutputParser()
    # Using LCEL, build a chain and invoke
    chain = prompt | llm | output_parser
    queries = chain.invoke({"original_query": query})
    queries = queries.split("\n")
    return [q.strip() for q in queries if q.strip()][:n]

def retrieve_documents(queries: List[str], retriever, fusion_fn, top_k: int):
    # Each query: get docs, then fuse
    results = [retriever.invoke(q) for q in queries]
    fused = fusion_fn(results)
    return [doc for doc, _ in fused[:top_k]]

def summarize_answer(query: str, documents: List[str], llm) -> str:
    chain = summary_prompt | llm | StrOutputParser()
    return chain.invoke({
        "user_query": query,
        "documents": "\n\n".join(documents)
    })

def rag_fusion_answer(
    user_query: str,
    local_index_path: str,
    embedding_model,
    mode: str = 'generated',
    num_generated_queries: int = 3,
    top_k: int = 3,
    params: Optional[dict] = None,
    chat_context: Optional[str] = None  # <-- NEW
) -> Tuple[str, dict]:
    """
    mode: 'original' | 'generated'
    chat_context: (Optional) Full conversation so far, to be given to LLM
    """
    llm = get_llm(params)
    vectorstore = load_vectorstore(local_index_path, embedding_model)
    retriever = vectorstore.as_retriever()

    if mode == 'original':
        queries = [user_query]
        logger.info("Running in original query mode.")
    elif mode == 'generated':
        queries = generate_queries(user_query, llm, num_generated_queries)
        logger.info(f"Running in query generation mode with {num_generated_queries} queries: {queries}")
    else:
        raise ValueError(f"Invalid mode: {mode}")

    docs = retrieve_documents(queries, retriever, reciprocal_rank_fusion, top_k)
    top_docs_content = [doc.page_content for doc in docs]

    # Use the chat context as prompt for the final answer
    if chat_context:
        prompt = (
            f"{chat_context}\n\n"
            f"Use the above conversation to answer the current question:\n"
            f"{user_query}\n\n"
            f"Refer to these documents:\n"
            f"{'---'.join(top_docs_content)}"
        )
    else:
        prompt = (
            f"Answer the following question using the provided documents:\n"
            f"{user_query}\n\n"
            f"Documents:\n"
            f"{'---'.join(top_docs_content)}"
        )

    # Pass this prompt to your summarize function
    final_answer = summarize_answer(prompt, [], llm)  # docs already in prompt

    return final_answer, params or DEFAULT_PARAMS

# Usage example:
# answer, meta = rag_fusion_answer("What is RAG?", "./faiss_index", embedding_model, mode="original")
# answer, meta = rag_fusion_answer("What is RAG?", "./faiss_index", embedding_model, mode="generated", num_generated_queries=5)
