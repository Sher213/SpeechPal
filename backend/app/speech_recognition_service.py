import traceback
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)
import torch
import torchaudio
import librosa
import numpy as np
from typing import Dict, Any, List
from ibm_watson.natural_language_understanding_v1 import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions, SentimentOptions
from ibm_watson import ApiException
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from dotenv import load_dotenv
import os
import logging
from google import genai
from google.genai import types


load_dotenv()

ibm_api_key = os.getenv("IBM_TONE_ANALYZER_API_KEY")
ibm_url     = os.getenv("IBM_TONE_ANALYZER_URL")
client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

logger = logging.getLogger("SpeechRecognitionService")
logging.basicConfig(level=logging.INFO)

class SpeechRecognitionService:
    def __init__(self):
        logger.info("Initializing SpeechRecognitionService...")
        # Whisper model
        model_name = "openai/whisper-large-v3"
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        logger.info("Loaded Whisper model: %s", model_name)

        # Text emotion pipeline (Hugging Face)
        self.hf_tone_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )

        # Summarization pipeline
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        logger.info("Loaded HuggingFace tone classifier and summarizer.")

        # IBM Watson Natural Language Understanding
        authenticator = IAMAuthenticator(ibm_api_key)
        self.nlu = NaturalLanguageUnderstandingV1(
            version="2021-08-01",
            authenticator=authenticator
        )
        self.nlu.set_service_url(ibm_url)

        logger.info("Loaded IBM Watson NLU.")

        # Audio emotion classifier (HF)
        self.audio_emotion_classifier = pipeline(
            "audio-classification",
            model="superb/wav2vec2-base-superb-er",
            sampling_rate=16000,
            return_all_scores=True
        )

        logger.info("Loaded audio emotion classifier.")

    def split_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        segment_length: float = 15.0
    ) -> List[np.ndarray]:
        logger.info(f"Splitting audio: {waveform.shape[1]/sample_rate:.2f}s into {segment_length}s segments.")
        total_samples = waveform.shape[1]
        segment_samples = int(segment_length * sample_rate)
        segments = []
        for start in range(0, total_samples, segment_samples):
            end = min(start + segment_samples, total_samples)
            seg = waveform[0, start:end].cpu().numpy()
            segments.append(seg)
        logger.info("Splitting complete. Total segments: %d", len(segments))
        return segments

    def transcribe_segment(self, audio_np: np.ndarray) -> Dict[str, Any]:
        logger.info("Transcribing audio segment of length %.2fs", len(audio_np)/16000)
        inputs = self.processor(
            audio_np,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="en",
            task="transcribe"
        )
        generate_kwargs = {"return_timestamps": True}
        if not hasattr(self.model.generation_config, "forced_decoder_ids") or self.model.generation_config.forced_decoder_ids is None:
            generate_kwargs["forced_decoder_ids"] = forced_decoder_ids
        outputs = self.model.generate(
            inputs,
            **generate_kwargs,
            return_dict_in_generate=True
        )
        transcription = self.processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True
        )[0]
        logger.info("Transcription: %s", transcription)
        # Use processor.decode to get segments with timestamps
        segments = self.processor.decode(outputs.sequences[0], skip_special_tokens=True, return_timestamps=True)
        logger.info("Transcription with timestamps complete.")
        return {"text": transcription, "timestamps": segments}

    def analyze_text_tone(self, text: str) -> Dict[str, float]:
        logger.info("Analyzing text tone for: %s", text[:60])
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

            logger.info("Tone scores: %s", {**emotion_scores, **sentiment_scores})
            return {**emotion_scores, **sentiment_scores}

        except ApiException as e:
            logger.error(
                "IBM Watson NLU ApiException: Code=%s, Message=%s, X-global-transaction-id=%s",
                e.code, e.message, e.http_response.headers.get("X-global-transaction-id") if e.http_response else "N/A"
            )
            logger.debug("Full response: %s", e.http_response.text if e.http_response else "No response body.")
            logger.debug("Traceback:\n%s", traceback.format_exc())

        except Exception as e:
            logger.error("Unexpected error: %s", str(e))
            logger.debug("Traceback:\n%s", traceback.format_exc())

        # Fallback to Hugging Face
        try:
            hf_res = self.hf_tone_classifier(text)
            logger.info("Fallback to Hugging Face tone scores: %s", hf_res)
            return {item['label']: float(item['score']) for item in hf_res[0]}
        except Exception as hf_e:
            logger.error("Hugging Face fallback failed: %s", str(hf_e))
            logger.debug("Traceback:\n%s", traceback.format_exc())
            return {}

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        logger.info("Analyzing sentiment for: %s", text[:60])
        # Optional: local HF fallback
        res = self.sentiment_analyzer(text)[0]
        logger.info("Sentiment: %s", res)
        return {res['label']: float(res['score'])}

    def summarize_content(self, text: str) -> str:
        logger.info("Summarizing content for: %s", text[:60])
        summary = self.summarizer(text, max_length=50, min_length=20, do_sample=False)[0]
        logger.info("Summary: %s", summary['summary_text'])
        return summary['summary_text']

    def rate_clarity(self, text: str) -> float:
        logger.info("Rating clarity for: %s", text[:60])
        fillers = ["um", "uh", "like", "you know", "so", "actually"]
        words = text.lower().split()
        filler_count = sum(w in fillers for w in words)
        ratio = filler_count / len(words) if words else 0
        clarity_score = max(0.0, 10.0 - ratio * 50)
        logger.info("Clarity score: %.2f", clarity_score)
        return clarity_score

    def extract_prosodic_features(
        self,
        audio_np: np.ndarray,
        sr: int = 16000
    ) -> Dict[str, float]:
        logger.info("Extracting prosodic features.")
        rms = librosa.feature.rms(y=audio_np)[0]
        pitches, magnitudes = librosa.piptrack(y=audio_np, sr=sr)
        pitch_vals = pitches[magnitudes > np.median(magnitudes)]
        pitch_mean = float(np.mean(pitch_vals)) if pitch_vals.size else 0.0
        logger.info("RMS mean: %.4f, Pitch mean: %.2f", float(np.mean(rms)), pitch_mean)
        return {"rms_mean": float(np.mean(rms)), "pitch_mean": pitch_mean}

    def analyze_audio_emotion(self, audio_np: np.ndarray) -> List[Dict[str, float]]:
        logger.info("Analyzing audio emotion.")
        res = self.audio_emotion_classifier(audio_np)
        logger.info("Audio emotion: %s", res)
        return [{item['label']: float(item['score'])} for item in res]
    
    def rate_segment_with_gemini(self, segment: Dict[str, Any]) -> float:
        logger.info("Calling Gemini to rate segment.")
        model = "gemini-2.0-flash"

        user_prompt = f"""
You are a professional speech coach. Your job is to assess short speech segments from a communication skills training app.

Evaluate each segment and assign a rating from 1 to 10 based on:
- Clarity: How easy it is to understand the speaker
- Fluency: Flow of words without stammering or excessive pauses
- Tone: Appropriateness and modulation of the speaker's tone
- Emotional Expressiveness: Variability and strength of emotional delivery

Use this scale:

- **1–3 (Poor)**: Unclear, hard to follow, disfluent, monotone, many filler words.
- **4–6 (Fair)**: Understandable but missing strong fluency or emotional expression. Some filler words or flat tone.
- **7–8 (Good)**: Clear, structured, good fluency, some expressiveness and tone variation.
- **9–10 (Excellent)**: Confident, fluent, expressive, emotionally engaging, and well-paced.

**Do not default to a 6 or 7.** Choose a score that aligns closely with the rubric. Be honest and constructive.

Respond strictly in this format:

---
Rating: <a number from 1 to 10>

Reason: <1–3 sentence explanation of the rating based on the segment qualities>
---

Here is the segment for evaluation:

Text:
{segment['text']}

Objective Speech Metrics:
{segment['metrics']}

Detected Emotional Tone (Text):
{segment['tone']}

Sentiment:
{segment['sentiment']}

Audio Emotion Scores:
{segment['emotion_audio']}
"""

        contents = [
            # types.Content(role="system", parts=[types.Part.from_text(text=system_prompt)]), NOT SUPPORTED
            types.Content(role="user",   parts=[types.Part.from_text(text=user_prompt)]),
        ]

        config = types.GenerateContentConfig(response_mime_type="text/plain")
        full_response = ""

        try:
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            ):
                full_response += chunk.text

            # pull out the first numeric token
            match = next(
                (
                    float(tok)
                    for tok in full_response.split()
                    if tok.replace(".", "", 1).isdigit()
                ),
                None
            )

            reasoning = full_response.replace(str(match), "").strip() if match is not None else "No reasoning provided."
            
            logger.info("Gemini response: %s", full_response)

            return {'rate': float(max(1.0, min(10.0, match))), 'reason': reasoning}

        except Exception as e:
            logger.warning("Gemini rating failed: %s", e)
            return 5.0

    async def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        logger.info(f"Transcribing audio file: {audio_file_path}")
        # Use torchaudio to load both wav and mp3 files
        ext = os.path.splitext(audio_file_path)[1].lower()
        if ext == '.mp3':
            waveform, sr = torchaudio.load(audio_file_path, format='mp3')
        else:
            waveform, sr = torchaudio.load(audio_file_path)
        if waveform.shape[0] > 1:
            logger.info("Averaging stereo to mono.")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != 16000:
            logger.info(f"Resampling from {sr} to 16000 Hz.")
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            sr = 16000
        logger.info(f"Loaded audio file: {audio_file_path}, Sample rate: {sr}, Channels: {waveform.shape[0]}")
        segments = self.split_audio(waveform, sr)
        logger.info(f"Split into {len(segments)} segments.")
        results = []
        for seg in segments:
            logger.info("Processing segment of length %.2fs", len(seg)/sr)
            trans = self.transcribe_segment(seg)
            text = trans['text']
            duration = len(seg) / sr
            wpm = len(text.split()) / (duration / 60)
            logger.info(f"Segment text: {text}")
            logger.info(f"Segment duration: {duration:.2f}s, WPM: {wpm:.2f}")
            tone = self.analyze_text_tone(text)
            logger.info(f"Segment tone: {tone}")
            sentiment = 0.5  # placeholder or extract from tone
            logger.info(f"Segment sentiment: {sentiment}")
            summary = self.summarize_content(text)
            logger.info(f"Segment summary: {summary}")
            clarity = self.rate_clarity(text)
            logger.info(f"Segment clarity: {clarity}")
            prosody = self.extract_prosodic_features(seg, sr)
            logger.info(f"Segment prosody: {prosody}")
            audio_emotion = self.analyze_audio_emotion(seg)
            logger.info(f"Segment audio emotion: {audio_emotion}")
            segment_data = {
                "timestamps": trans["timestamps"],
                "text": text,
                "metrics": {
                    "duration_sec": duration,
                    "wpm": wpm,
                    "clarity": clarity,
                    **prosody
                },
                "tone": tone,
                "sentiment": sentiment,
                "emotion_audio": audio_emotion,
                "summary": summary
            }
            logger.info(f"Segment data before Gemini: {segment_data}")
            segment_rate_reason = self.rate_segment_with_gemini(segment_data)
            excerpt = text[:80] + "..." if len(text) > 80 else text
            segment_data.update({"rate_reason": segment_rate_reason, "excerpt": excerpt})
            logger.info(f"Segment data after Gemini: {segment_data}")
            results.append(segment_data)
        logger.info("All segments processed. Returning results.")
        return {"segments": results}
