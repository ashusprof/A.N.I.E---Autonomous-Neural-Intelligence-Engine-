import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"

# =============================================================================
# MULTI-MODEL CONFIGURATION
# =============================================================================

# Model definitions with priority and capabilities
# Names match: ollama list
MODELS = {
    "deepseek-v3": {
        "name": "deepseek-v3.1:671b-cloud",  # Correct name from ollama list
        "priority": 1,
        "best_for": ["reasoning", "coding", "analysis", "general"],
        "description": "Best reasoning + coding overall",
        "timeout": 300,
    },
    "qwen3-coder": {
        "name": "qwen3-coder:480b-cloud",
        "priority": 2,
        "best_for": ["coding", "debugging", "code-review"],
        "description": "Best for coders/developers",
        "timeout": 240,
    },
    "qwen3-vl": {
        "name": "qwen3-vl:235b-cloud",
        "priority": 3,
        "best_for": ["vision", "coding", "image-analysis"],
        "description": "Vision + coding",
        "timeout": 240,
    },
    "gpt-oss-120b": {
        "name": "gpt-oss:120b-cloud",
        "priority": 4,
        "best_for": ["general", "writing", "analysis"],
        "description": "General strong model",
        "timeout": 180,
    },
    "gpt-oss-20b": {
        "name": "gpt-oss:20b-cloud",
        "priority": 5,
        "best_for": ["general", "quick-tasks"],
        "description": "Medium tasks",
        "timeout": 120,
    },
    "minimax-m2": {
        "name": "minimax-m2:cloud",
        "priority": 6,
        "best_for": ["chat", "general"],
        "description": "Normal chat",
        "timeout": 120,
    },
    "glm-4.6": {
        "name": "glm-4.6:cloud",
        "priority": 7,
        "best_for": ["chat", "general"],
        "description": "Fallback model",
        "timeout": 120,
    },
}

# Primary model for all requests
PRIMARY_MODEL = "deepseek-v3"

# Fallback chain - used when primary model fails or hits rate limit
FALLBACK_CHAIN = [
    "qwen3-coder",
    "gpt-oss-120b", 
    "gpt-oss-20b",
    "minimax-m2",
    "glm-4.6"
]

# Get the actual model name for Ollama
def get_model_name(model_key: str) -> str:
    """Get the Ollama model name from model key"""
    if model_key in MODELS:
        return MODELS[model_key]["name"]
    return model_key

# Primary model for assistant
ASSISTANT_MODEL = get_model_name(PRIMARY_MODEL)

# =============================================================================
# API KEYS & EXTERNAL SERVICES
# =============================================================================

# SearXNG Configuration
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")
SEARXNG_SECRET_KEY = os.getenv("SEARXNG_SECRET_KEY", "")

# LiveKit Configuration (for voice agent)
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Deepgram Configuration (STT for voice)
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Cartesia Configuration (TTS for voice)
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
CARTESIA_VOICE_ID = "faf0731e-dfb9-4cfc-8119-259a79b27e12"  # Female voice

# =============================================================================
# GPU CONFIGURATION
# =============================================================================

# GPU Configuration for Text Chat (High creativity)
TEXT_GPU_CONFIG = {
    "num_gpu": 99,
    "main_gpu": 0,
    "num_thread": 8,
    "num_ctx": 12288,
    "temperature": 1.3,
    "top_p": 0.98,
    "top_k": 60,
    "repeat_penalty": 1.15,
    "num_batch": 1024,
    "f16_kv": True,
    "use_mmap": True,
    "use_mlock": True,
    "low_vram": False,
    "rope_freq_base": 10000.0,
    "rope_freq_scale": 1.0,
    "numa": False,
    "seed": -1,
    "tfs_z": 0.95
}

# GPU Configuration for Voice (Faster, more focused)
VOICE_GPU_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_ctx": 8192,
    "repeat_penalty": 1.1,
}

# Default GPU config
ASSISTANT_GPU_CONFIG = TEXT_GPU_CONFIG

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

# Multi-user conversation storage
assistant_conversations = {}
assistant_user_sessions = {}
MAX_ASSISTANT_CONVERSATION_HISTORY = 25
ASSISTANT_CONVERSATION_TIMEOUT = 1800  # 30 minutes

# Chat History Storage
CHAT_HISTORY_DIR = os.path.join(os.path.dirname(__file__), "chat_history")
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
