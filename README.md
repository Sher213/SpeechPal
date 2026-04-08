# 🎙️ SpeechPal --- Real-Time AI Speech Coach

SpeechPal is an AI-powered speech coaching application that analyzes how
you speak --- not just what you say.

It provides real-time insights on clarity, tone, pacing, and sentiment,
helping users improve communication for interviews, presentations, sales
calls, and more.

------------------------------------------------------------------------

## 🚀 Features

### 🎧 Real-Time Speech Processing

-   Live audio recording from the browser\
-   Seamless audio pipeline (buffer → WAV → processing)

### 🧠 AI-Powered Analysis

-   Speech-to-text transcription using Whisper\
-   Sentiment and tone detection\
-   Context-aware feedback generation

### 📊 Speech Metrics & Insights

-   Speech rate (words per minute)\
-   Clarity and delivery scoring\
-   Segment-level analysis of speech patterns

### 📈 Interactive Visualization

-   Dynamic charts for speech metrics\
-   Visual breakdown of tone and pacing\
-   Real-time feedback interface

### 🎨 Modern UI

-   Built with Material-UI for a clean, responsive experience\
-   Smooth user interaction and intuitive controls

------------------------------------------------------------------------

## 🏗️ Architecture Overview

    Frontend (React + TypeScript)
            ↓
    Audio Recording (Browser)
            ↓
    Backend API (FastAPI)
            ↓
    Audio Processing Pipeline
       • Format conversion
       • Whisper transcription
       • Sentiment analysis
       • Metric extraction
            ↓
    Results + Visualization

------------------------------------------------------------------------

## 🛠️ Tech Stack

### Backend

-   FastAPI\
-   Python\
-   PyTorch\
-   Whisper (Speech-to-Text)

### Frontend

-   React\
-   TypeScript\
-   Material-UI\
-   D3.js

### Other

-   Axios (API communication)

------------------------------------------------------------------------

## ⚙️ Setup Instructions

### 1. Clone the Repository

``` bash
git clone https://github.com/yourusername/speechpal.git
cd speechpal
```

### 2. Backend Setup

``` bash
cd backend
python -m venv venv
```

#### Activate Environment

**Windows**

``` bash
venv\Scripts\activate
```

**Mac/Linux**

``` bash
source venv/bin/activate
```

#### Install Dependencies

``` bash
pip install -r requirements.txt
```

#### Run Server

``` bash
uvicorn app.main:app --reload
```

Backend runs at: http://localhost:8000

------------------------------------------------------------------------

### 3. Frontend Setup

``` bash
cd frontend
npm install
npm start
```

Frontend runs at: http://localhost:3000

------------------------------------------------------------------------

## 🧪 Usage

1.  Open the app in your browser\
2.  Click **"Start Recording"**\
3.  Speak naturally into your microphone\
4.  Click **"Stop Recording"**\
5.  View analysis results

------------------------------------------------------------------------

## 🔌 API Endpoints

### Analyze Speech

POST /api/analyze-speech

### Health Check

GET /api/health

------------------------------------------------------------------------

## 💡 Use Cases

-   Public speaking coaching\
-   Interview preparation\
-   Sales call optimization\
-   Presentation practice\
-   Communication training

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Real-time (streaming) feedback\
-   Speaker emotion detection\
-   Personalized coaching suggestions\
-   Mobile app integration

------------------------------------------------------------------------

## 📌 Summary

SpeechPal goes beyond transcription --- it turns speech into actionable
feedback.

**From speech → to insight → to improvement**
