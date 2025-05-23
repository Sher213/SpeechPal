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
        """
        Use IBM Watson NLU to get emotion and sentiment scores.
        """
        nlu_res = self.nlu.analyze(
            text=text,
            features=Features(
                emotion=EmotionOptions(),
                sentiment=SentimentOptions()
            ),
            language="en" # Specify the language
        ).get_result()
        emotion_scores = nlu_res.get("emotion", {}).get("document", {}).get("emotion", {})
        sentiment_doc = nlu_res.get("sentiment", {}).get("document", {})
        sentiment_scores = {sentiment_doc.get("label", ""): sentiment_doc.get("score", 0.0)}
        logger.info("Tone scores: %s", {**emotion_scores, **sentiment_scores})
        return {**emotion_scores, **sentiment_scores}

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

        # 1) system prompt to set behavior, 2) user prompt with the segment data
        system_prompt = (
            "You are a professional speech coach. "
            "Rate the following speech segment on a scale of 1 to 10 based on clarity, tone, "
            "fluency, and emotional expressiveness. "
            "Respond with only a single numeric score."
        )
        user_prompt = f"""
**You are a professional speech coach.**
**Rate the following speech segment on a scale of 1 to 10 based on clarity, tone, fluency, and emotional expressiveness.**
**Respond with only a single numeric score.**

Text:
{segment['text']}

Metrics:
{segment['metrics']}

Tone:
{segment['tone']}

Sentiment:
{segment['sentiment']}

Audio Emotion:
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
            return float(max(1.0, min(10.0, match))) if match is not None else 5.0

        except Exception as e:
            logger.warning("Gemini rating failed: %s", e)
            return 5.0

    async def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        logger.info(f"Transcribing audio file: {audio_file_path}")
        waveform, sr = torchaudio.load(audio_file_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != 16000:
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

            tone = self.analyze_text_tone(text)
            sentiment = 0.5  # placeholder or extract from tone
            summary = self.summarize_content(text)
            clarity = self.rate_clarity(text)
            prosody = self.extract_prosodic_features(seg, sr)
            audio_emotion = self.analyze_audio_emotion(seg)

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

            rating = self.rate_segment_with_gemini(segment_data)
            excerpt = text[:80] + "..." if len(text) > 80 else text
            segment_data.update({"rating": rating, "excerpt": excerpt})
            results.append(segment_data)

        return {"segments": results}
