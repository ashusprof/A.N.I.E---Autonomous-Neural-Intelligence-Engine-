import uuid
import logging
import json
import asyncio
import sys
from asyncio import Lock
from pathlib import Path

# Handle imports for both package and script execution
try:
    from .agent import OllamaAssistant
    from .config import (
        ASSISTANT_MODEL,
        OLLAMA_BASE_URL,
        assistant_conversations,
        ASSISTANT_CONVERSATION_TIMEOUT,
    )
except ImportError:
    # If relative imports fail, try absolute imports (when running as script)
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.agent import OllamaAssistant
    from backend.config import (
        ASSISTANT_MODEL,
        OLLAMA_BASE_URL,
        assistant_conversations,
        ASSISTANT_CONVERSATION_TIMEOUT,
    )

from fastapi import FastAPI, HTTPException, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import httpx

# Try to import LiveKit for voice mode
try:
    from livekit import agents
    try:
        from .agent import entrypoint
    except ImportError:
        from backend.agent import entrypoint
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    logging.warning("LiveKit not available. Voice mode will be disabled.")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="A.N.I.E. API",
    description="API for A.N.I.E. – Autonomous Neural Intelligence Engine",
    version="2.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session Management ---
session_last_access = {}
session_lock = Lock()

async def cleanup_inactive_sessions():
    """Periodically cleans up old, inactive sessions to free up memory."""
    while True:
        await asyncio.sleep(60)
        async with session_lock:
            current_time = asyncio.get_running_loop().time()
            inactive_sessions = [
                sid for sid, last_access in session_last_access.items()
                if current_time - last_access > ASSISTANT_CONVERSATION_TIMEOUT
            ]

            for sid in inactive_sessions:
                if sid in assistant_conversations:
                    del assistant_conversations[sid]
                if sid in session_last_access:
                    del session_last_access[sid]
                logger.info(f"Cleaned up inactive session: {sid}")

# --- API Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = Field(None, description="Unique ID for the conversation session.")
    stream: bool = Field(True, description="Whether to stream the response.")
    mode: str = Field("text", description="Conversation mode: 'text' or 'voice'.")

class ChatResponse(BaseModel):
    reply: str
    session_id: str

class ClearRequest(BaseModel):
    session_id: str

class ClearResponse(BaseModel):
    message: str

# --- Chat Endpoint ---
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest = Body(...)):
    """
    Handles a chat message from the user, manages conversation state, and
    returns the assistant's response, with support for streaming.
    """
    session_id = request.session_id or str(uuid.uuid4())
    user_message = request.message

    # Get or create an assistant instance for the session
    if session_id not in assistant_conversations:
        mode = request.mode if hasattr(request, 'mode') else "text"
        logger.info(f"Creating new assistant for session_id: {session_id}, mode: {mode}")
        assistant_conversations[session_id] = OllamaAssistant(
            model=ASSISTANT_MODEL, 
            base_url=OLLAMA_BASE_URL,
            mode=mode
        )
    else:
        # If switching modes, update the assistant mode
        assistant = assistant_conversations[session_id]
        requested_mode = request.mode if hasattr(request, 'mode') else "text"
        if requested_mode != assistant.mode:
            logger.info(f"Switching assistant mode from {assistant.mode} to {requested_mode} for session {session_id}")
            # Preserve conversation history
            conversation_history = assistant.conversation.copy()
            # Create new assistant with new mode
            assistant_conversations[session_id] = OllamaAssistant(
                model=ASSISTANT_MODEL,
                base_url=OLLAMA_BASE_URL,
                mode=requested_mode
            )
            # Restore conversation history (new system message is already set)
            new_assistant = assistant_conversations[session_id]
            # Keep user/assistant messages, skip old system message
            for msg in conversation_history[1:]:  # Skip old system message
                if msg["role"] in ["user", "assistant"]:
                    new_assistant.conversation.append(msg)
    
    # Update last access time for the session
    async with session_lock:
        session_last_access[session_id] = asyncio.get_running_loop().time()

    assistant = assistant_conversations[session_id]
    
    # Log mode information
    mode_info = f" (mode: {assistant.mode})" if hasattr(assistant, 'mode') else ""
    logger.info(f"[{session_id}] User: {user_message}{mode_info}")

    try:
        async def stream_generator():
            """Generator for streaming the response chunks."""
            # First, send the session_id
            initial_data = {"session_id": session_id}
            yield f"data: {json.dumps(initial_data)}\n\n"

            # Then, stream the assistant's reply
            full_response = ""
            import re
            tool_pattern = re.compile(r'\[TOOL:.*?\]', flags=re.DOTALL | re.IGNORECASE)
            
            async for data_chunk in assistant.send_message_stream(user_message):
                response_data = data_chunk
                if "content" in response_data:
                    content = response_data["content"]
                    
                    # Skip chunks that are just tool calls
                    if content.strip().startswith("[TOOL:"):
                        continue
                    
                    # Filter tool calls from content
                    cleaned_content = tool_pattern.sub('', content)
                    
                    if cleaned_content.strip():
                        response_data["content"] = cleaned_content
                        full_response += cleaned_content
                    else:
                        continue
                        
                yield f"data: {json.dumps(response_data)}\n\n"

            logger.info(f"[{session_id}] Assistant: {full_response[:100]}...")

        if request.stream:
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            # Non-streaming fallback
            assistant_reply = await assistant.send_message(user_message)
            logger.info(f"[{session_id}] Assistant: {assistant_reply[:100]}...")
            return ChatResponse(reply=assistant_reply, session_id=session_id)

    except Exception as e:
        logger.error(f"Error during chat for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")

# --- Clear History Endpoint ---
@app.post("/api/clear", response_model=ClearResponse)
async def clear_history(request: ClearRequest = Body(...)):
    """Clears the conversation history for a given session_id."""
    session_id = request.session_id
    if session_id in assistant_conversations:
        async with session_lock:
            if session_id in assistant_conversations:
                del assistant_conversations[session_id]
            if session_id in session_last_access:
                del session_last_access[session_id]
            logger.info(f"Cleared conversation for session_id: {session_id}")
            return ClearResponse(message=f"Conversation for session {session_id} cleared.")
    raise HTTPException(status_code=404, detail="Session not found.")

# --- Voice Mode Endpoints ---
class VoiceChatRequest(BaseModel):
    message: str
    session_id: str | None = Field(None, description="Unique ID for the conversation session.")
    stream: bool = Field(True, description="Whether to stream the response.")

@app.post("/api/voice/chat")
async def voice_chat_endpoint(request: VoiceChatRequest = Body(...)):
    """Voice chat endpoint - delegates to main chat with voice mode"""
    chat_request = ChatRequest(
        message=request.message,
        session_id=request.session_id,
        stream=request.stream,
        mode="voice"
    )
    return await chat_endpoint(chat_request)

class TTSSystemRequest(BaseModel):
    text: str
    voice: str = Field("default", description="Voice ID for TTS")

@app.post("/api/voice/tts")
async def text_to_speech_endpoint(request: TTSSystemRequest = Body(...)):
    """Convert text to speech audio using Cartesia TTS with FEMALE voice"""
    try:
        import base64
        import os
        
        CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
        if not CARTESIA_API_KEY:
            logger.error("CARTESIA_API_KEY not found in environment")
            raise HTTPException(status_code=500, detail="CARTESIA_API_KEY not configured")
        
        FEMALE_VOICE_ID = "faf0731e-dfb9-4cfc-8119-259a79b27e12"
        
        logger.info("=" * 60)
        logger.info("TTS REQUEST START")
        logger.info(f"Text: '{request.text[:80]}...'")
        logger.info(f"Voice ID (FEMALE): {FEMALE_VOICE_ID}")
        logger.info("=" * 60)
        
        url = "https://api.cartesia.ai/tts/bytes"
        headers = {
            "X-API-Key": CARTESIA_API_KEY,
            "Cartesia-Version": "2024-06-10",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model_id": "sonic-2",
            "transcript": request.text,
            "voice": {
                "mode": "id",
                "id": FEMALE_VOICE_ID
            },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": 24000
            },
            "language": "en"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                logger.info("Sending request to Cartesia...")
                response = await client.post(url, headers=headers, json=payload)
                
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"Cartesia error: {error_text}")
                    raise HTTPException(status_code=500, detail=f"Cartesia error: {error_text}")
                
                audio_data = response.content
                audio_size = len(audio_data)
                logger.info(f"Received audio: {audio_size} bytes ({audio_size/1024:.2f} KB)")
                
                if audio_size == 0:
                    raise HTTPException(status_code=500, detail="No audio data received")
                
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                logger.info("✓ TTS SUCCESS - FEMALE VOICE")
                logger.info("=" * 60)
                
                return {
                    "status": "success",
                    "audio": audio_base64,
                    "format": "pcm_s16le",
                    "sample_rate": 24000,
                    "voice_id": FEMALE_VOICE_ID,
                    "voice_name": "Anie (Female)",
                    "text": request.text
                }
                
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                logger.error(f"Cartesia HTTP error: {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)
                
            except httpx.RequestError as e:
                logger.error(f"Request error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Connection error: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected TTS error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

# --- Deepgram WebSocket for Real-time Speech-to-Text ---
@app.websocket("/api/voice/stt")
async def deepgram_stt_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time speech-to-text using Deepgram"""
    await websocket.accept()
    logger.info("Deepgram STT WebSocket connection established")
    
    try:
        from backend.config import DEEPGRAM_API_KEY
        
        if not DEEPGRAM_API_KEY:
            await websocket.send_json({"error": "DEEPGRAM_API_KEY not configured"})
            await websocket.close()
            return
        
        deepgram_url = (
            "wss://api.deepgram.com/v1/listen?"
            "encoding=linear16&"
            "sample_rate=16000&"
            "channels=1&"
            "model=nova-2&"
            "language=en-US&"
            "punctuate=true&"
            "interim_results=false&"
            "endpointing=300"
        )
        
        from websockets.asyncio.client import connect
        async with connect(
            deepgram_url,
            additional_headers=[("Authorization", f"Token {DEEPGRAM_API_KEY}")]
        ) as deepgram_ws:
            logger.info("Connected to Deepgram WebSocket")
            
            async def forward_audio():
                try:
                    while True:
                        data = await websocket.receive()
                        
                        if "bytes" in data:
                            await deepgram_ws.send(data["bytes"])
                        elif "text" in data:
                            message = json.loads(data["text"])
                            if message.get("type") == "close":
                                logger.info("Client requested close")
                                break
                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                except Exception as e:
                    logger.error(f"Error forwarding audio: {e}")
            
            async def forward_transcriptions():
                try:
                    async for message in deepgram_ws:
                        result = json.loads(message)
                        
                        if result.get("type") == "Results":
                            channel = result.get("channel", {})
                            alternatives = channel.get("alternatives", [])
                            
                            if alternatives:
                                transcript = alternatives[0].get("transcript", "")
                                is_final = result.get("is_final", False)
                                confidence = alternatives[0].get("confidence", 0)
                                
                                if is_final and transcript.strip():
                                    logger.info(f"Deepgram final: '{transcript}' (confidence: {confidence:.2f})")
                                    await websocket.send_json({
                                        "type": "transcript",
                                        "transcript": transcript,
                                        "is_final": True,
                                        "confidence": confidence
                                    })
                        
                        elif result.get("type") == "Metadata":
                            logger.info(f"Deepgram metadata: {result}")
                        
                except Exception as e:
                    logger.error(f"Error receiving from Deepgram: {e}")
            
            await asyncio.gather(
                forward_audio(),
                forward_transcriptions()
            )
    
    except Exception as e:
        logger.error(f"Deepgram WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass
        logger.info("Deepgram STT WebSocket connection closed")

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_inactive_sessions())
    logger.info("Started background thread for session cleanup.")

# --- Root Route ---
@app.get("/")
async def root():
    """Serve the main chat interface"""
    chat_page = Path(__file__).parent.parent / "frontend" / "fullscreen.html"
    if chat_page.exists():
        return FileResponse(chat_page)
    return {"message": "A.N.I.E. API is running"}

# --- Static Files (must be after all routes) ---
# This is mounted at the end so API routes take precedence
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    # Mount at root so CSS, JS, images are accessible directly
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    logger.info(f"Serving static files from: {frontend_dir}")

# --- Main Entrypoint ---
if __name__ == "__main__":
    # Check if running as voice agent or web API
    if "--voice" in sys.argv and LIVEKIT_AVAILABLE:
        logger.info("Starting in VOICE mode (LiveKit agent)")
        agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
    else:
        logger.info("Starting in TEXT mode (FastAPI web server)")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
