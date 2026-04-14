# A.N.I.E Backend - Unified Voice & Text AI Assistant

**Autonomous Neural Intelligence Engine**

## Overview

A.N.I.E is a unified AI assistant that supports both **text chat** and **voice interaction** modes using a single codebase.

## Features

- **Dual Mode Operation**: Run as web chat API or voice agent
- **Tool Calling**: Integrated web search via SearXNG
- **Streaming Support**: Real-time responses in both text and voice
- **Session Management**: Automatic cleanup of inactive sessions
- **Voice Processing**: STT (Deepgram) + TTS (Cartesia) with LiveKit

## Architecture

```
backend/
├── main.py           # FastAPI server entrypoint
├── agent.py          # Unified OllamaAssistant with voice & text support
├── config.py         # Configuration settings
├── tools.py          # Web search tool (SearXNG)
├── prompts.py        # System prompts for both modes
├── chat_storage.py   # Chat history management
├── database.py       # Database utilities
└── requirements.txt  # Python dependencies
```

## Installation

### Prerequisites

- Python 3.10+
- Ollama with `yasserrmd/qwen2.5-7b-instruct-1m:latest` model
- SearXNG instance running (for web search)
- LiveKit server (for voice mode)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the `Agent/` directory:

```env
# SearXNG
SEARXNG_URL=http://localhost:8080
SEARXNG_SECRET_KEY=

# LiveKit (for voice mode)
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

# Deepgram (STT for voice)
DEEPGRAM_API_KEY=your_deepgram_key

# Cartesia (TTS for voice)
CARTESIA_API_KEY=your_cartesia_key
```

## Usage

### 1. Text Chat Mode (FastAPI Web Server)

Run the assistant as a web API:

```bash
cd Agent
python main.py
```

The server will start on `http://localhost:8000`

**🌐 Access in Browser:**

Once the server is running, open your browser and navigate to:

```
http://localhost:8000
```

The frontend interface will be automatically served and you can interact with the assistant directly in your browser!

**API Endpoints:**
- `POST /api/chat` - Send messages (supports streaming)
- `POST /api/clear` - Clear conversation history
- `GET /` - Serves the frontend interface

**Example Request (API):**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the latest NEC 2023 requirements?",
    "session_id": "user123",
    "stream": true
  }'
```

### 2. Voice Agent Mode (LiveKit)

Run the assistant as a voice agent:

```bash
cd Agent
python main.py --voice
```

Or directly with LiveKit CLI:

```bash
cd Agent
python agent.py --voice
```

### 3. CLI Testing Mode

Run the assistant in terminal for quick testing:

```bash
cd Agent
python agent.py
```

Commands:
- `quit` - Exit
- `clear` - Reset conversation
- `debug` - Show conversation history

## Configuration

### Model Selection

Edit [config.py](config.py:11-15):

```python
TEXT_MODEL = "yasserrmd/qwen2.5-7b-instruct-1m:latest"
VOICE_MODEL = "llama3.2:1b"  # Can change to same as TEXT_MODEL
ASSISTANT_MODEL = TEXT_MODEL  # Default model
```

### GPU Configuration

**Text Mode** (High creativity):
```python
TEXT_GPU_CONFIG = {
    "temperature": 1.3,
    "top_p": 0.98,
    "num_ctx": 12288,
    # ... more settings
}
```

**Voice Mode** (Faster, focused):
```python
VOICE_GPU_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.9,
    "num_ctx": 8192,
    # ... more settings
}
```

## Tool Calling

The assistant can search the web when needed:

**User:** "What are the latest solar panel efficiency ratings?"

**Assistant:** `[TOOL: search_web(query="latest solar panel efficiency 2025", time_range="month")]`

**System:** Executes search and provides results

**Assistant:** Returns comprehensive answer based on search results

### Time Ranges
- `day` - Past 24 hours
- `week` - Past 7 days
- `month` - Past 30 days
- `year` - Past 365 days

## API Reference

### OllamaAssistant Class

```python
from agent import OllamaAssistant

# Initialize
assistant = OllamaAssistant(
    model="yasserrmd/qwen2.5-7b-instruct-1m:latest",
    base_url="http://localhost:11434",
    mode="text"  # or "voice"
)

# Send message (non-streaming)
response = await assistant.send_message("Hello")

# Send message (streaming)
async for chunk in assistant.send_message_stream("Hello"):
    if "content" in chunk:
        print(chunk["content"], end="")

# Voice mode streaming
async for token in assistant.send_message_voice("Hello"):
    # Tokens are cleaned for TTS
    pass
```

## Deployment

### Production Web Server

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY Agent/ /app/
RUN pip install -r requirements.txt

# For text mode
CMD ["python", "main.py"]

# For voice mode
# CMD ["python", "main.py", "--voice"]
```

## Troubleshooting

### LiveKit not available
If you see "LiveKit not available", install voice dependencies:
```bash
pip install livekit-agents livekit-plugins-deepgram livekit-plugins-cartesia
```

### Model not found
Pull the Ollama model:
```bash
ollama pull yasserrmd/qwen2.5-7b-instruct-1m:latest
```

### SearXNG connection error
Ensure SearXNG is running and accessible at the URL in `.env`

## License

Proprietary - Solaroot Engineering Services Private Limited

## Support

For issues or questions, contact the development team.
