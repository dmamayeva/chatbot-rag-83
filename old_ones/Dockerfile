FROM python:3.12

WORKDIR /app
COPY ./chatbot /app/chatbot
COPY ./rag_pipeline /app/rag_pipeline
COPY ./data /app/data
COPY requirements.txt /app/requirements.txt
COPY ./embed/page_examp.html /app/embed/index.html
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "chatbot.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]