from langchain_core.prompts import ChatPromptTemplate

# Query generation prompt — TRANSLATE TO RUSSIAN
query_generation_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that generates multiple search queries based on a single input query.
Generate 3 different search queries that are related to the following input query:

Input query: {original_query}

The search queries should explore different aspects and perspectives of the input query.
If the search query is in Kazakh language, generate queries in Russian. Translate "ПББ (Педагогтің білімін бағалау)" from Kazakh to "ОЗП" in Russian.
Generate queries to search for information in legal documents.
Output the search queries, one per line.
""")

# query_generation_prompt = ChatPromptTemplate.from_template("""
# You are a helpful assistant that generates multiple search queries based on a single input query.
# Generate 3 different search queries that are related to the following input query:

# Input query: {original_query}

# The search queries should explore different aspects and perspectives of the input query.
# Generate queries to search for information in legal documents.
# Output the search queries, one per line.
# """)

# Summary prompt with chat context support
summary_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions based on the provided documents and conversation history.

{chat_context_section}Current question: {user_query}

Relevant documents:
{documents}

Instructions:
1. If there is previous conversation context, consider it when formulating your answer to maintain continuity.
2. Answer the current question based on the provided documents and any relevant conversation history.
3. If the documents don't contain relevant information, say so.
4. Be concise but thorough in your response.
5. Answer the question either in Kazakh or Russian, depending on the language of the question.
6. Translate "ПББ (Педагогтің білімін бағалау)" from Kazakh to "ОЗП" in Russian.

Answer:
""")

# Alternative: More explicit chat context handling
summary_prompt_with_context = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant answering questions based on provided documents. Consider the conversation history when formulating your response."),
    ("human", """
{chat_context}

Current question: {user_query}

Relevant documents:
{documents}

Please answer the question based on the documents and conversation history.
""")
])