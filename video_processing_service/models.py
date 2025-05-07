from pydantic import BaseModel

class AudioDownloadRequest(BaseModel):
    url: str
    max_duration: int = 60
    retries: int = 3