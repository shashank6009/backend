from fastapi import FastAPI
from pydantic import BaseModel
from .inference import pipeline

app = FastAPI(title="Trustworthy LLM API")

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    result = pipeline(query.question)
    return result
