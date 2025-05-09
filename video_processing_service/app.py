from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models import AudioDownloadRequest
from services import AudioProcessor
import os
import requests
from typing import List, Dict, Any
import logging
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

processor = AudioProcessor()
logger = logging.getLogger(__name__)
DELAY = 5
print("Processor was successfully loaded")

def analyze_text_with_api(texts: List[str]) -> List[Dict[str, Any]]:
    """Анализирует тексты через NLP API и возвращает результаты с метками"""
    try:
        payload = {"blocks": texts}
        response = requests.post(
            "http://nlp_words:5001/analyze",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return [
                {
                    "text": text,
                    "label": "not_valid" if len(analysis) > 0 else "valid",
                    "analysis": analysis
                }
                for text, analysis in zip(texts, response.json())
            ]
        return [{"text": text, "label": "valid", "analysis": []} for text in texts]
    except Exception as e:
        logger.error(f"API analysis error: {str(e)}")
        return [{"text": text, "label": "valid", "analysis": []} for text in texts]

@app.post("/transcribe")
async def transcribe_audio(request: AudioDownloadRequest):
    try:
        output_file = "/app/output/downloaded_audio.wav"
        
        if not processor.download_audio(
            url=request.url,
            output_file=output_file,
            max_duration=request.max_duration,
            retries=request.retries
        ):
            raise HTTPException(status_code=400, detail="Failed to download audio")
        
        transcription = processor.transcribe_audio(output_file)
        os.remove(output_file)  

        time.sleep(DELAY)
    
        analysis_results = analyze_text_with_api([transcription])
        text_analysis = analysis_results[0] if analysis_results else {
            "text": transcription,
            "label": "valid",
            "analysis": []
        }
        
        return JSONResponse(content={
            "status": "success",
            "transcription": transcription,
            "analysis": {
                "label": text_analysis["label"],
                "details": text_analysis["analysis"]
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}