from langchain_core.prompts import ChatPromptTemplate

query_generation_prompt = ChatPromptTemplate.from_template(
    """Given the prompt: '{user_query}', generate 5 questions that are better articulated.
    Return in the form of an list.""")

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a legal assistant AI helping summarize answers based on provided legal documents.."),
    ("user", "Original user query: {user_query}\n\nRelevant legal documents:\n{documents}\n\nBased on these, provide a clear, concise answer in Russian. If there is no relevant information, say 'I am sorry, please contact an operator'")
])