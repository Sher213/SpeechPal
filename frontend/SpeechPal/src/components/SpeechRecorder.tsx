import React, { useRef, useState, useEffect } from 'react';
import axios from 'axios';
import {
  Box,
  Button,
  Typography,
  Paper,
  CircularProgress,
  Card,
  CardContent,
  Input,
} from '@mui/material';
import ErrorBoundary from './ErrorBoundary'; // Assuming you have this component
import { webmBlobToWavBlob } from './webmToWav'; // Assuming you have this utility
import { RatingsChart } from './RatingsChart';

interface SegmentAnalysis {
  timestamps: string | Array<{ start: number; end: number; text: string }>;
  text: string;
  metrics: {
    duration_sec: number;
    wpm: number;
    clarity: number;
    rms_mean: number;
    pitch_mean: number;
  };
  tone: Record<string, number>;
  sentiment: Record<string, number> | number;
  emotion_audio: Array<Record<string, number>>;
  summary: string;
  rate_reason: { rate: number; reason: string } | null;
  excerpt: string;
}

interface SpeechAnalysis {
  segments: SegmentAnalysis[];
}

const SpeechRecorderContent: React.FC = () => {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<SpeechAnalysis | null>(null);
  // Add a key to force RatingsChart remount when analysis changes
  // This is crucial for ensuring the old Chart.js instance is destroyed.
  const [ratingsChartKey, setRatingsChartKey] = useState<string>("");

  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const startRecording = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: 16000, sampleSize: 16 },
      });

      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 1024;
      sourceRef.current.connect(analyserRef.current);

      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      recorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error(err);
      setError('Microphone access failed.');
    }
  };

  const stopRecording = async () => {
    if (!mediaRecorderRef.current) return;
    setIsRecording(false);
    setIsLoading(true);
    setError(null);

    // Close audio context and clear references on stop recording
    audioContextRef.current?.close();
    audioContextRef.current = null;
    analyserRef.current = null;
    sourceRef.current = null;

    await new Promise<void>((resolve) => {
      mediaRecorderRef.current!.onstop = async () => {
        try {
          const blob = new Blob(audioChunksRef.current, { type: 'audio/webm;codecs=opus' });
          const url = URL.createObjectURL(blob);
          setAudioUrl(url);

          const wav = await webmBlobToWavBlob(blob);
          await uploadAudioFile(wav);
        } catch (err: any) {
          console.error(err);
          setError(err.message || 'Analysis failed.');
        } finally {
          setIsLoading(false);
          resolve();
        }
      };
      mediaRecorderRef.current!.stop();
    });
  };

  const uploadAudioFile = async (file: Blob) => {
    setIsLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append('audio_file', file, 'upload.wav');
      const res = await axios.post<SpeechAnalysis>('http://localhost:8000/api/analyze-speech', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setAnalysis(res.data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Upload failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const arrayBuffer = await file.arrayBuffer();
      const blob = new Blob([arrayBuffer]);
      await uploadAudioFile(blob);
    }
  };

  // This useEffect updates the key for RatingsChart, forcing it to remount
  // when the analysis data changes. This triggers the cleanup in RatingsChart.
  useEffect(() => {
    if (analysis && analysis.segments) {
      setRatingsChartKey(JSON.stringify(analysis.segments.map(seg => seg.rate_reason?.rate)));
    }
  }, [analysis]); // Dependency on 'analysis' ensures key updates when new analysis is set

  return (
    <Box sx={{ maxWidth: 900, mx: 'auto', p: 3 }}>
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom>Speech Coach</Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap', mb: 2 }}>
          <Button
            variant="contained"
            color={isRecording ? 'error' : 'primary'}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isLoading}
          >
            {isRecording ? 'Stop Recording' : 'Start Recording'}
          </Button>

          <label htmlFor="file-upload">
            <Input
              id="file-upload"
              type="file"
              inputProps={{ accept: 'audio/*' }}
              onChange={handleFileChange}
              disabled={isLoading}
              style={{ display: 'none' }}
            />
            <Button variant="outlined" component="span" disabled={isLoading}>Upload Audio</Button>
          </label>
        </Box>

        {error && <Typography color="error" sx={{ mb: 2 }}>{error}</Typography>}
        {isLoading && <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}><CircularProgress /></Box>}
        {audioUrl && <Box sx={{ mb: 2 }}><audio src={audioUrl} controls /></Box>}

        {analysis && (
          <Box>
            <Typography variant="h6" gutterBottom>Analysis Results</Typography>
            {/* Ratings Chart for segments with rate_reason */}
            {analysis.segments.some(seg => seg.rate_reason) && (
              <Box sx={{ my: 4 }}>
                <RatingsChart
                  // The key prop is essential. When ratingsChartKey changes,
                  // React unmounts the old RatingsChart component and mounts a new one.
                  // This triggers the useEffect cleanup in RatingsChart to destroy the old chart.
                  key={ratingsChartKey}
                  segments={analysis.segments
                    .filter(seg => seg.rate_reason)
                    .map(seg => ({
                      rating: seg.rate_reason!.rate,
                      excerpt: seg.excerpt
                    }))}
                />
              </Box>
            )}
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              {analysis.segments.map((seg, idx) => (
                <Box key={idx} sx={{ flex: '1 1 45%', minWidth: 300, maxWidth: '48%' }}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle1">Segment {idx + 1}</Typography>
                      <Typography variant="body2" paragraph><strong>Text:</strong> {seg.text}</Typography>
                      <Typography variant="body2"><strong>WPM:</strong> {seg.metrics.wpm.toFixed(1)}</Typography>
                      <Typography variant="body2"><strong>Clarity:</strong> {seg.metrics.clarity.toFixed(1)}/10</Typography>
                      <Typography variant="body2"><strong>Duration:</strong> {seg.metrics.duration_sec.toFixed(1)}s</Typography>
                      <Typography variant="body2"><strong>RMS:</strong> {seg.metrics.rms_mean.toFixed(2)}</Typography>
                      <Typography variant="body2" gutterBottom><strong>Pitch:</strong> {seg.metrics.pitch_mean.toFixed(1)}</Typography>
                      <Typography variant="body2"><strong>Summary:</strong> {seg.summary}</Typography>
                      {seg.rate_reason && (
                        <Box sx={{ my: 2, p: 2, bgcolor: '#f0f4ff', borderRadius: 2 }}>
                          <Typography variant="body2" sx={{ fontWeight: 600, color: 'primary.main' }}>
                            <strong>AI Rating:</strong> {seg.rate_reason.rate} / 10
                          </Typography>
                          <Typography variant="body2" sx={{ mt: 1 }}>
                            <strong>Reason:</strong> {seg.rate_reason.reason}
                          </Typography>
                        </Box>
                      )}
                      <Typography variant="body2" sx={{ mt: 1 }}><strong>Text Tone:</strong></Typography>
                      <Box component="ul" sx={{ pl: 2, mb: 1 }}>
                        {Object.entries(seg.tone).map(([tone, score]) => (
                          <li key={tone}>{tone}: {score.toFixed(2)}</li>
                        ))}
                      </Box>
                      <Typography variant="body2"><strong>Sentiment:</strong> {typeof seg.sentiment === 'object'
                        ? Object.entries(seg.sentiment).map(([lab, sc]) => `${lab}(${sc.toFixed(2)})`).join(', ')
                        : seg.sentiment.toFixed(2)}</Typography>
                      <Typography variant="body2" sx={{ mt: 1 }}><strong>Audio Emotion:</strong></Typography>
                      <Box component="ul" sx={{ pl: 2 }}>
                        {seg.emotion_audio.map((emo, i) => (
                          <li key={i}>{Object.entries(emo).map(([lab, sc]) => `${lab}: ${sc.toFixed(2)}`).join(', ')}</li>
                        ))}
                      </Box>
                    </CardContent>
                  </Card>
                </Box>
              ))}
            </Box>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

const SpeechRecorder: React.FC = () => (
  <ErrorBoundary>
    <SpeechRecorderContent />
  </ErrorBoundary>
);

export default SpeechRecorder;