# A.N.I.E - Autonomous Neural Intelligence Engine

A unified AI assistant supporting **text chat** and **voice interaction** modes with web search capabilities.

## Architecture

```
A.N.I.E/
├── backend/           # Python FastAPI server + AI agent
├── frontend/          # Web interface (HTML/CSS/JS)
└── services/          # External services (SearXNG)
```

## Quick Start

### 1. Start SearXNG (Web Search)
```bash
cd services/searxng
docker-compose up -d
```

### 2. Start Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### 3. Access the App
Open http://localhost:8000 in your browser.

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description |
|----------|-------------|
| `LIVEKIT_URL` | LiveKit server URL (voice mode) |
| `DEEPGRAM_API_KEY` | Deepgram STT API key |
| `CARTESIA_API_KEY` | Cartesia TTS API key |
| `SEARXNG_SECRET_KEY` | SearXNG instance key |

## Features

- 💬 **Text Chat** - Web-based chat with streaming responses
- 🎤 **Voice Mode** - Real-time voice interaction via LiveKit
- 🔍 **Web Search** - Integrated SearXNG for up-to-date information
- ⚡ **Streaming** - Real-time response streaming
- 🔧 **Tool Calling** - Automatic web search when needed

## Tech Stack

- **Backend**: FastAPI, Ollama, httpx
- **Frontend**: Vanilla HTML/CSS/JS
- **Voice**: LiveKit, Deepgram (STT), Cartesia (TTS)
- **Search**: SearXNG
