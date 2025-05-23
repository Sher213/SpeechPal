# Speech Coach App

A real-time speech analysis application that provides feedback on speech quality, tone, clarity, and content using Whisper AI and sentiment analysis.

## Features

- Real-time speech recording
- Speech-to-text transcription using Whisper AI
- Sentiment analysis
- Speech rate analysis
- Interactive visualization of speech metrics
- Modern Material-UI interface

## Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

## Setup

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Start the backend server:
```bash
uvicorn app.main:app --reload
```

The backend server will run on http://localhost:8000

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The frontend application will run on http://localhost:3000

## Usage

1. Open http://localhost:3000 in your web browser
2. Click "Start Recording" to begin recording your speech
3. Speak into your microphone
4. Click "Stop Recording" when finished
5. View the analysis results and visualization

## API Endpoints

- POST /api/analyze-speech: Analyze speech from audio file
- GET /api/health: Health check endpoint

## Technologies Used

- Backend:
  - FastAPI
  - Whisper AI
  - PyTorch
  - Python

- Frontend:
  - React
  - TypeScript
  - Material-UI
  - D3.js
  - Axios 