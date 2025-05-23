from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .speech_recognition_service import SpeechRecognitionService
import os
import tempfile
from typing import Dict, Any
import wave

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
    # Create temporary file with proper WAV format
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        content = await audio_file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Verify and fix WAV format if needed
        with wave.open(temp_file_path, 'rb') as wav_file:
            # If we can open it, it's a valid WAV file
            pass
        
        # Analyze speech
        result = await speech_service.transcribe_audio(temp_file_path)
        return result
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"} 