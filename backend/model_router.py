"""
Model Router Module
Handles intelligent model selection, query classification, and automatic fallback.
"""
import re
import logging
from typing import Optional, List, Tuple, Dict, Any

try:
    from .config import MODELS, PRIMARY_MODEL, FALLBACK_CHAIN, get_model_name
except ImportError:
    from config import MODELS, PRIMARY_MODEL, FALLBACK_CHAIN, get_model_name

logger = logging.getLogger(__name__)


# =============================================================================
# QUERY CLASSIFICATION
# =============================================================================

# Keywords for detecting query types
CODING_KEYWORDS = [
    "code", "coding", "program", "function", "class", "debug", "error",
    "python", "javascript", "java", "c++", "rust", "go", "typescript",
    "api", "algorithm", "data structure", "bug", "fix", "implement",
    "syntax", "compile", "runtime", "exception", "refactor", "optimize",
    "html", "css", "react", "vue", "node", "express", "django", "flask",
    "sql", "database", "query", "script", "terminal", "command", "git",
]

REASONING_KEYWORDS = [
    "explain", "why", "how", "analyze", "compare", "evaluate", "reason",
    "think", "logic", "problem", "solve", "calculate", "math", "proof",
    "deduce", "infer", "conclude", "strategy", "plan", "decision",
]

VISION_KEYWORDS = [
    "image", "picture", "photo", "screenshot", "diagram", "chart",
    "graph", "visual", "see", "look", "show me", "drawing",
]


def classify_query(query: str) -> str:
    """
    Classify the query type based on keywords.
    
    Returns:
        Query type: "coding", "reasoning", "vision", or "general"
    """
    query_lower = query.lower()
    
    # Check for vision tasks
    if any(keyword in query_lower for keyword in VISION_KEYWORDS):
        return "vision"
    
    # Check for coding tasks
    coding_score = sum(1 for kw in CODING_KEYWORDS if kw in query_lower)
    if coding_score >= 2:
        return "coding"
    
    # Check for reasoning tasks
    reasoning_score = sum(1 for kw in REASONING_KEYWORDS if kw in query_lower)
    if reasoning_score >= 2:
        return "reasoning"
    
    # Default to general
    return "general"


def get_best_model_for_task(query_type: str) -> str:
    """
    Get the best model for a specific task type.
    
    Args:
        query_type: Type of query (coding, reasoning, vision, general)
        
    Returns:
        Model key for the best suited model
    """
    # For most cases, DeepSeek-V3 is the best choice
    if query_type == "coding":
        # DeepSeek is best, but Qwen3-Coder is also excellent
        return PRIMARY_MODEL
    elif query_type == "vision":
        # Qwen3-VL is the vision model
        return "qwen3-vl"
    elif query_type == "reasoning":
        # DeepSeek excels at reasoning
        return PRIMARY_MODEL
    else:
        # General queries - use primary
        return PRIMARY_MODEL


# =============================================================================
# FALLBACK MANAGEMENT
# =============================================================================

class ModelRouter:
    """
    Manages model selection and automatic fallback on failures.
    """
    
    def __init__(self):
        self.current_model = PRIMARY_MODEL
        self.fallback_index = 0
        self.failed_models = set()
        self.rate_limited_models = {}  # model -> timestamp when rate limit expires
        
    def get_current_model(self) -> Tuple[str, str]:
        """
        Get the current model to use.
        
        Returns:
            Tuple of (model_key, model_name)
        """
        model_name = get_model_name(self.current_model)
        logger.info(f"🤖 Using model: {self.current_model} ({model_name})")
        return self.current_model, model_name
    
    def get_model_timeout(self) -> float:
        """Get the timeout for current model"""
        if self.current_model in MODELS:
            return MODELS[self.current_model].get("timeout", 180)
        return 180
    
    def select_model_for_query(self, query: str) -> Tuple[str, str]:
        """
        Select the best model for a given query.
        
        Args:
            query: User's query
            
        Returns:
            Tuple of (model_key, model_name)
        """
        query_type = classify_query(query)
        best_model = get_best_model_for_task(query_type)
        
        # Check if best model is available
        if best_model not in self.failed_models:
            self.current_model = best_model
        else:
            # Fall back to primary or next available
            self._move_to_next_available()
        
        logger.info(f"📋 Query type: {query_type} → Selected: {self.current_model}")
        return self.get_current_model()
    
    def handle_error(self, error: Exception) -> Tuple[bool, str, str]:
        """
        Handle an error and determine if we should retry with fallback.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Tuple of (should_retry, new_model_key, new_model_name)
        """
        error_str = str(error).lower()
        
        # Detect rate limiting or model unavailable
        is_rate_limit = any(x in error_str for x in [
            "429", "rate limit", "too many requests", 
            "quota exceeded", "capacity"
        ])
        
        is_model_error = any(x in error_str for x in [
            "model not found", "model does not exist",
            "connection refused", "timeout", "503", "502", "404", "not found"
        ])
        
        if is_rate_limit or is_model_error:
            logger.warning(f"⚠️ Model {self.current_model} failed: {error}")
            self.failed_models.add(self.current_model)
            
            # Try to move to fallback
            if self._move_to_next_available():
                model_key, model_name = self.get_current_model()
                logger.info(f"🔄 Falling back to: {model_key}")
                return True, model_key, model_name
            else:
                logger.error("❌ All models exhausted!")
                return False, "", ""
        
        # For other errors, don't retry
        return False, "", ""
    
    def _move_to_next_available(self) -> bool:
        """
        Move to the next available model in the fallback chain.
        
        Returns:
            True if a fallback model is available, False otherwise
        """
        for model_key in FALLBACK_CHAIN:
            if model_key not in self.failed_models:
                self.current_model = model_key
                return True
        
        return False
    
    def reset(self):
        """Reset the router to primary model"""
        self.current_model = PRIMARY_MODEL
        self.failed_models.clear()
        self.fallback_index = 0
        logger.info("🔄 Model router reset to primary model")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current router status"""
        return {
            "current_model": self.current_model,
            "model_name": get_model_name(self.current_model),
            "failed_models": list(self.failed_models),
            "available_fallbacks": [m for m in FALLBACK_CHAIN if m not in self.failed_models]
        }


# Global router instance
model_router = ModelRouter()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_model_for_query(query: str) -> Tuple[str, str]:
    """
    Convenience function to get the best model for a query.
    
    Returns:
        Tuple of (model_key, model_name)
    """
    return model_router.select_model_for_query(query)


def handle_model_error(error: Exception) -> Tuple[bool, str, str]:
    """
    Convenience function to handle errors and get fallback.
    
    Returns:
        Tuple of (should_retry, model_key, model_name)
    """
    return model_router.handle_error(error)


def reset_model_router():
    """Reset the model router to default state"""
    model_router.reset()
