from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from service import answer_question


router = APIRouter()


class Query(BaseModel):
    question: str


@router.post("/ask")
async def ask(query: Query):
    try:
        result = answer_question(query.question)
        return {"answer": result.get("answer")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def root():
    return {"message": "Unified Chatbot API is running", "endpoints": ["POST /ask"]}


