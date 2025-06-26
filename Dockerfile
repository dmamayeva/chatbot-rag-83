FROM python:3.12

WORKDIR /app
COPY ./src /app/src
COPY ./rag_pipeline /app/rag_pipeline
COPY ./data /app/data
COPY ./double_db_faiss-16-06-2025 /app/double_db_faiss-16-06-2025
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "chatbot.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]