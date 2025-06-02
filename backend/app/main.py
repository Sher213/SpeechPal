from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .speech_recognition_service import SpeechRecognitionService
import os
import tempfile
from typing import Dict, Any

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Allow both Vite and React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize speech recognition service
speech_service = SpeechRecognitionService()

@app.post("/api/analyze-speech")
async def analyze_speech(audio_file: UploadFile = File(...)) -> Dict[str, Any]:
    ext = os.path.splitext(audio_file.filename)[1].lower()
    suffix = ext if ext in ['.wav', '.mp3'] else ext or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        content = await audio_file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    try:
        # Directly analyze speech, do not check WAV header
        result = await speech_service.transcribe_audio(temp_file_path)
        return result
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.unlink(temp_file_path)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}