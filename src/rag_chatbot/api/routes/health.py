from fastapi import APIRouter
from src.rag_chatbot.models.schemas import HealthResponse
from datetime import datetime

router = APIRouter()

@router.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now()
    )