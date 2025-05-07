from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from utils import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

class TextBlock(BaseModel):
    block: str

class AnalysisRequest(BaseModel):
    blocks: List[str]

class NegativeBlockResponse(BaseModel):
    block: str
    negative_words: List[str]

class ErrorResponse(BaseModel):
    status: str
    message: str

@app.post(
    "/analyze",
    response_model=List[NegativeBlockResponse],
    responses={500: {"model": ErrorResponse}}
)
async def analyze_text(request: AnalysisRequest):
    """
    Анализирует текст на наличие негативных слов.
    
    Принимает список текстовых блоков и возвращает те из них,
    в которых найдены негативные слова.
    """
    negative_blocks = []
    
    for block in request.blocks:
        negative_words = list(get_negative_words(block))
        if negative_words:
            negative_blocks.append({
                "block": block,
                "negative_words": negative_words
            })
    
    return negative_blocks

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)