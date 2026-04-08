import traceback
import torch
import torchaudio
import librosa
import numpy as np
from typing import Dict, Any, List
import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import soundfile as sf
from dotenv import load_dotenv
import re

from transformers import pipeline
from ibm_watson.natural_language_understanding_v1 import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions, SentimentOptions
from ibm_watson import ApiException
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger("SpeechRecognitionService")
logging.basicConfig(level=logging.INFO)

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
SEG_FILE_PATH = "audio.wav"

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

ibm_api_key = os.getenv("IBM_TONE_ANALYZER_API_KEY")
ibm_url = os.getenv("IBM_TONE_ANALYZER_URL")


class SpeechRecognitionService:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.loop = asyncio.get_event_loop()

        # HuggingFace models
        self.hf_tone_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.audio_emotion_classifier = pipeline(
            "audio-classification",
            model="superb/wav2vec2-base-superb-er",
            sampling_rate=16000,
            return_all_scores=True
        )

        # IBM Watson NLU
        authenticator = IAMAuthenticator(ibm_api_key)
        self.nlu = NaturalLanguageUnderstandingV1(
            version="2021-08-01",
            authenticator=authenticator
        )
        self.nlu.set_service_url(ibm_url)

        # Load speech guide for RAG
        self.speech_guide_path = os.path.join(os.path.dirname(__file__), "../speech_guide.txt")
        try:
            with open(self.speech_guide_path, "r", encoding="utf-8") as f:
                self.speech_guide = f.read()
            logger.info("Loaded speech guide for RAG.")
        except Exception as e:
            logger.warning(f"Could not load speech guide: {e}")
            self.speech_guide = ""

    # ===================== HELPERS =====================
    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except:
            return default

    def _retrieve_guide_context(self, segment_text: str, top_n: int = 3) -> str:
        if not self.speech_guide:
            return ""
        lines = self.speech_guide.splitlines()
        words = set(re.findall(r'\w+', segment_text.lower()))
        scored = []
        for line in lines:
            line_words = set(re.findall(r'\w+', line.lower()))
            score = len(words & line_words)
            if score > 0:
                scored.append((score, line))
        top_lines = [line for _, line in sorted(scored, reverse=True)[:top_n]]
        return "\n".join(top_lines)

    # ===================== AUDIO SPLIT =====================
    async def split_audio(self, waveform, sample_rate, segment_length=15.0):
        return await self.loop.run_in_executor(
            self.executor,
            self._split_audio_sync,
            waveform,
            sample_rate,
            segment_length
        )

    def _split_audio_sync(self, waveform, sample_rate, segment_length):
        total_samples = waveform.shape[1]
        segment_samples = int(segment_length * sample_rate)
        segments = []
        for start in range(0, total_samples, segment_samples):
            end = min(start + segment_samples, total_samples)
            seg = waveform[0, start:end].cpu().numpy()
            segments.append(seg)
        return segments

    # ===================== IBM Watson TONE ANALYSIS =====================
    async def _analyze_text_tone_sync(self, text):
        """Run IBM Watson NLU in thread to avoid blocking event loop"""
        def nlu_call():
            try:
                nlu_res = self.nlu.analyze(
                    text=text,
                    features=Features(
                        emotion=EmotionOptions(),
                        sentiment=SentimentOptions()
                    ),
                    language="en"
                ).get_result()

                emotion_scores = nlu_res.get("emotion", {}).get("document", {}).get("emotion", {})
                sentiment_doc = nlu_res.get("sentiment", {}).get("document", {})
                sentiment_scores = {sentiment_doc.get("label", ""): sentiment_doc.get("score", 0.0)}

                return {**emotion_scores, **sentiment_scores}

            except ApiException as e:
                logger.error(
                    "IBM Watson NLU ApiException: Code=%s, Message=%s",
                    e.code, e.message
                )
                return {}
            except Exception as e:
                logger.error("NLU unexpected error: %s", e)
                return {}

        return await self.loop.run_in_executor(self.executor, nlu_call)

    # ===================== MAIN TRANSCRIBE + RATE SEGMENT =====================
    async def _transcribe_and_rate_segment(self, audio_np):
        try:
            # --- Write audio to in-memory buffer ---
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, 16000, format="WAV")
            buffer.seek(0)  # Reset pointer

            # --- Convert buffer to a file on disk ---
            file_path = SEG_FILE_PATH
            with open(file_path, "wb") as f:
                f.write(buffer.getvalue())

            # --- Upload using Gemini FilesAPI ---
            seg_file = client.files.upload(file=SEG_FILE_PATH)

            # --- Gemini transcription using file ---
            with open(file_path, "rb") as f:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=["Transcribe this audio", seg_file]  # send the uploaded file object
                )

            text = response.text or ""

            # --- Compute metrics ---
            duration = len(audio_np) / 16000
            wpm = len(text.split()) / (duration / 60) if duration > 0 else 0

            # --- Parallel async tasks ---
            tone_task = self._analyze_text_tone_sync(text)
            summary_task = self.summarize_content(text)
            clarity_task = self.rate_clarity(text)
            prosody_task = self.extract_prosodic_features(audio_np, 16000)
            emotion_task = self.analyze_audio_emotion(audio_np)

            tone_result, summary, clarity, prosody, emotion = await asyncio.gather(
                tone_task, summary_task, clarity_task, prosody_task, emotion_task
            )

            tone = {k: v for k, v in tone_result.items() if k not in ["positive", "negative", "neutral"]}
            sentiment = {k: v for k, v in tone_result.items() if k in ["positive", "negative", "neutral"]}

            segment = {
                "text": text,
                "metrics": {
                    "duration_sec": duration,
                    "wpm": wpm,
                    "clarity": clarity,
                    **prosody
                },
                "tone": tone,
                "sentiment": sentiment,
                "summary": summary,
                "emotion_audio": emotion
            }

            # --- Build Gemini rating prompt ---
            guide_context = self._retrieve_guide_context(text)
            rag_section = f"\n\n[Speech Guide Context]\n{guide_context}\n" if guide_context else ""
            rating_prompt = f"""
You are a professional speech coach assessing a single speech segment.  
{rag_section}
Text: {text}
Metrics: {segment['metrics']}
Tone: {tone}
Sentiment: {sentiment}
Audio Emotion: {emotion}

Rate the segment 1-10 (poor=1–3, fair=4–6, good=7–8, excellent=9–10)
Respond strictly in format:
---
Rating: <number>
Reason: <brief explanation>
---
"""

            # --- Call Gemini for rating ---
            rating_response = ""
            for chunk in client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=[rating_prompt],
                config=types.GenerateContentConfig(response_mime_type="text/plain")
            ):
                rating_response += chunk.text

            num_match = re.search(r"(?<![\d.])([1-9]|10)(?:\.\d+)?(?![\d.])", rating_response)
            rating_value = float(num_match.group(0)) if num_match else 5.0
            reasoning = rating_response.replace(f"Rating: {str(rating_value)}", "").strip() if num_match else "No reasoning provided."
            segment["rate_reason"] = {"rate": rating_value, "reason": reasoning}

            return segment

        except Exception as e:
            logger.error(f"_transcribe_and_rate_segment failed: {e} {traceback.format_exc()}")
            return {"error": str(e)}

    # ===================== FEATURE COMPRESSION =====================
    def _compress_features(self, segment):
        metrics = segment.get("metrics", {})
        wpm = self._safe_float(metrics.get("wpm"))
        clarity = self._safe_float(metrics.get("clarity"))
        pace = "slow" if wpm < 100 else "fast" if wpm > 160 else "optimal"
        clarity_band = "low" if clarity < 4 else "medium" if clarity < 7 else "high"
        return {
            "pace": pace,
            "clarity": clarity_band,
            "energy": round(self._safe_float(metrics.get("rms_mean")), 4),
            "pitch": round(self._safe_float(metrics.get("pitch_mean")), 2)
        }

    # ===================== OTHER ASYNC HELPERS =====================
    async def summarize_content(self, text):
        return self.summarizer(text, max_length=50, min_length=20)[0]['summary_text']

    async def rate_clarity(self, text):
        fillers = ["um", "uh", "like", "you know"]
        words = text.lower().split()
        ratio = sum(w in fillers for w in words) / len(words) if words else 0
        return max(0.0, 10 - ratio * 50)

    async def extract_prosodic_features(self, audio_np, sr=16000):
        def prosody_sync():
            rms = librosa.feature.rms(y=audio_np)[0]
            f0 = librosa.yin(audio_np, fmin=50, fmax=300)
            pitch_mean = float(np.mean(f0)) if f0.size else 0.0
            return {"rms_mean": float(np.mean(rms)), "pitch_mean": pitch_mean}
        return await self.loop.run_in_executor(self.executor, prosody_sync)

    async def analyze_audio_emotion(self, audio_np):
        return self.audio_emotion_classifier(audio_np)

    # ===================== MAIN PIPELINE =====================
    async def transcribe_audio(self, audio_file_path):
        waveform, sr = torchaudio.load(audio_file_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            sr = 16000

        segments_np = await self.split_audio(waveform, sr)
        tasks = [self._transcribe_and_rate_segment(seg) for seg in segments_np]
        results = await asyncio.gather(*tasks)

        return {"segments": [r for r in results if not isinstance(r, Exception)]}