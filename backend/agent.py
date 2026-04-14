import asyncio
import re
import json
import logging
import sys
import os
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv
import httpx

# Handle imports for both package and script execution
try:
    from .config import ASSISTANT_MODEL, OLLAMA_BASE_URL, ASSISTANT_GPU_CONFIG, TEXT_GPU_CONFIG, VOICE_GPU_CONFIG, CARTESIA_VOICE_ID, MODELS, FALLBACK_CHAIN
    from .prompts import AGENT_INSTRUCTION, TEXT_SESSION_INSTRUCTION, VOICE_SESSION_INSTRUCTION
    from .tools import search_web
    from .model_router import model_router, classify_query, get_model_name
except ImportError:
    # If relative imports fail, try absolute imports (when running as script)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.config import ASSISTANT_MODEL, OLLAMA_BASE_URL, ASSISTANT_GPU_CONFIG, TEXT_GPU_CONFIG, VOICE_GPU_CONFIG, CARTESIA_VOICE_ID, MODELS, FALLBACK_CHAIN
    from backend.prompts import AGENT_INSTRUCTION, TEXT_SESSION_INSTRUCTION, VOICE_SESSION_INSTRUCTION
    from backend.tools import search_web
    from backend.model_router import model_router, classify_query, get_model_name

# LiveKit imports (optional, only needed for voice mode)
try:
    from livekit import agents
    from livekit.agents import Agent, AgentSession, JobContext, RoomInputOptions
    from livekit.plugins import cartesia, deepgram, noise_cancellation, openai
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    logging.warning("LiveKit not available. Voice mode will be disabled.")

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Utility: clean text before sending to TTS ---
def clean_for_tts(text: str) -> str:
    """Remove special tokens and tool call markers before TTS"""
    replacements = [
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|start_header_id|>user<|end_header_id|>",
        "<|start_header_id|>system<|end_header_id|>",
    ]
    for r in replacements:
        text = text.replace(r, "")

    # Remove tool call markers
    text = re.sub(r'\[TOOL:.*?\]', '', text, flags=re.DOTALL)

    return text.strip()


class OllamaAssistant:
    """Unified Ollama assistant supporting both text and voice modes with tool calling"""

    def __init__(self, model: str, base_url: str, mode: str = "text"):
        """
        Initialize the assistant.

        Args:
            model: Ollama model name
            base_url: Ollama base URL
            mode: "text" or "voice" - affects response style and GPU config
        """
        self.model = model
        self.base_url = base_url
        self.mode = mode
        # Voice mode needs fewer iterations to avoid multiple responses
        self.max_iterations = 3 if mode == "voice" else 10
        
        # Multi-model support
        self.current_model_key = None  # Track which model key we're using
        self.retry_count = 0
        self.max_retries = 3  # Max retries before falling back

        # Select GPU config based on mode
        self.gpu_config = VOICE_GPU_CONFIG if mode == "voice" else TEXT_GPU_CONFIG

        # Get current date and time
        now = datetime.now()
        current_datetime = now.strftime("%A, %B %d, %Y at %I:%M %p")
        current_date_short = now.strftime("%Y-%m-%d")

        # Tool instruction with examples
        tool_instruction = f"""
## TOOL INSTRUCTIONS

1.  **Web Search**: When you need up-to-date information, recent data, or facts about the world, you **MUST** use the web search tool.
2.  **Format**: To use the tool, you **MUST** respond with **ONLY** the following syntax:
    `[TOOL: search_web(query="your search query")]`
3.  **DO NOT** provide any text or explanation before or after the tool command. Your entire response must be the tool command.
4.  **Today's Date**: {current_date_short}. Use this to decide if a search is needed.
5.  **Multiple Searches**: For complex questions, you can make multiple searches to get comprehensive information. Each search should focus on a different aspect of the question.
6.  **Comprehensive Research**: When the user asks about a topic that might have multiple aspects, perspectives, or recent developments, make multiple searches to ensure you have complete information.

**Time Range (Optional):**
You can add a `time_range` parameter to your query for recent information.
`[TOOL: search_web(query="latest NEC code changes", time_range="month")]`
Available ranges: "day", "week", "month", "year".

**Examples:**
- [TOOL: search_web(query="latest NEC code changes", time_range="month")]
- [TOOL: search_web(query="solar AHJ requirements 2025", time_range="year")]
- For "Tell me about company X", you might search: "company X overview", "company X recent news", "company X products"
"""

        # Add mode-specific instructions
        mode_instruction = ""
        if mode == "voice":
            mode_instruction = "\n**VOICE MODE**: Keep all responses under 60 words for clarity and natural speech."

        # Build system message
        system_content = f"""Current Date and Time: {current_datetime}
Current Date (Short): {current_date_short}

{AGENT_INSTRUCTION}

{tool_instruction}

{mode_instruction}"""

        self.conversation: List[Dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

    def _format_messages_for_chat(self) -> List[Dict[str, str]]:
        """Format conversation for Ollama chat API"""
        return self.conversation

    async def _call_ollama_chat_stream_single(self, messages: List[Dict[str, str]], model: str):
        """Call Ollama's /api/chat endpoint and stream the response for a single model."""
        timeout = model_router.get_model_timeout()
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": self.gpu_config
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        if not chunk.get("done"):
                            content = chunk.get("message", {}).get("content", "")
                            yield content

    async def _call_ollama_chat_stream(self, messages: List[Dict[str, str]]):
        """Call Ollama's /api/chat endpoint and stream the response with fallback support."""
        # Get model from router
        model_key, model_name = model_router.get_current_model()
        logger.info(f"🤖 Streaming with model: {model_name}")
        
        try:
            async for chunk in self._call_ollama_chat_stream_single(messages, model_name):
                yield chunk
        except (httpx.TimeoutException, httpx.HTTPError) as e:
            logger.warning(f"⚠️ Model {model_name} failed: {e}")
            
            # Try fallback
            should_retry, new_key, new_model = model_router.handle_error(e)
            if should_retry:
                logger.info(f"🔄 Retrying with fallback model: {new_model}")
                try:
                    async for chunk in self._call_ollama_chat_stream_single(messages, new_model):
                        yield chunk
                    return
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
            
            yield "I'm having trouble connecting to the AI models. Please try again."
        except Exception as e:
            logger.error(f"Unexpected streaming error: {e}", exc_info=True)
            yield "An unexpected error occurred."

    async def _call_ollama_chat_single(self, messages: List[Dict[str, str]], model: str) -> str:
        """Call Ollama's /api/chat endpoint for a single model."""
        timeout = model_router.get_model_timeout()
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": self.gpu_config
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            message = result.get("message", {})
            return message.get("content", "").strip()

    async def _call_ollama_chat(self, messages: List[Dict[str, str]]) -> str:
        """Call Ollama's /api/chat endpoint with automatic fallback on failure."""
        # Get model from router
        model_key, model_name = model_router.get_current_model()
        logger.info(f"🤖 Calling model: {model_name}")
        
        try:
            return await self._call_ollama_chat_single(messages, model_name)
        except (httpx.TimeoutException, httpx.HTTPError) as e:
            logger.warning(f"⚠️ Model {model_name} failed: {e}")
            
            # Try fallback chain
            should_retry, new_key, new_model = model_router.handle_error(e)
            if should_retry:
                logger.info(f"🔄 Retrying with fallback model: {new_model}")
                try:
                    return await self._call_ollama_chat_single(messages, new_model)
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
                    
                    # Try one more fallback
                    should_retry2, new_key2, new_model2 = model_router.handle_error(fallback_error)
                    if should_retry2:
                        logger.info(f"🔄 Second fallback: {new_model2}")
                        try:
                            return await self._call_ollama_chat_single(messages, new_model2)
                        except Exception as e3:
                            logger.error(f"All fallbacks failed: {e3}")
            
            return "I'm having trouble connecting to the AI models. Please try again later."
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return "An unexpected error occurred."


    def _detect_tool_call(self, text: str) -> tuple[bool, str | None, str | None]:
        """
        Detect if the response contains a tool call.
        Returns (is_tool_call, query, time_range)
        """
        # Pattern with time_range
        pattern_with_time = r'\[TOOL:\s*search_web\s*\(\s*query\s*=\s*["\']([^"\']+)["\']\s*,\s*time_range\s*=\s*["\']([^"\']+)["\']\s*\)\s*\]'
        match = re.search(pattern_with_time, text, re.IGNORECASE | re.DOTALL)
        if match:
            query = match.group(1).strip()
            time_range = match.group(2).strip()
            logger.info(f"✓ Tool call detected - query: {query}, time_range: {time_range}")
            return True, query, time_range

        # Pattern without time_range
        pattern_no_time = r'\[TOOL:\s*search_web\s*\(\s*query\s*=\s*["\']([^"\']+)["\']\s*\)\s*\]'
        match = re.search(pattern_no_time, text, re.IGNORECASE | re.DOTALL)
        if match:
            query = match.group(1).strip()
            logger.info(f"✓ Tool call detected - query: {query}, time_range: None")
            return True, query, None

        return False, None, None

    def _detect_multiple_tool_calls(self, text: str) -> List[tuple[str, str | None]]:
        """
        Detect all tool calls in the model response.
        Returns: List of (query, time_range) tuples
        """
        queries = []
        seen_queries = set()  # Track seen queries to avoid duplicates
        
        # Pattern with time_range
        pattern_with_time = r'\[TOOL:\s*search_web\s*\(\s*query\s*=\s*["\']([^"\']+)["\']\s*,\s*time_range\s*=\s*["\']([^"\']+)["\']\s*\)\s*\]'
        for match in re.finditer(pattern_with_time, text, re.IGNORECASE | re.DOTALL):
            query = match.group(1).strip()
            time_range = match.group(2).strip()
            if query not in seen_queries:
                queries.append((query, time_range))
                seen_queries.add(query)
                logger.info(f"✓ Multiple tool call detected - query: {query}, time_range: {time_range}")

        # Pattern without time_range - check for all matches
        pattern_no_time = r'\[TOOL:\s*search_web\s*\(\s*query\s*=\s*["\']([^"\']+)["\']\s*\)\s*\]'
        for match in re.finditer(pattern_no_time, text, re.IGNORECASE | re.DOTALL):
            query = match.group(1).strip()
            # Only add if we haven't seen this query already
            if query not in seen_queries:
                queries.append((query, None))
                seen_queries.add(query)
                logger.info(f"✓ Multiple tool call detected - query: {query}, time_range: None")

        return queries

    async def send_message_stream(self, message: str):
        """
        Send a message and stream the response, handling tool calls.
        Yields status updates and content chunks.
        """
        self.conversation.append({"role": "user", "content": message})
        yield {"status": "Thinking..."}

        iteration = 0
        full_response = ""

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Stream Iteration {iteration}/{self.max_iterations}")

            messages = self._format_messages_for_chat()

            # Use the non-streaming call first to detect tool usage
            model_response_for_tool_check = await self._call_ollama_chat(messages)
            logger.info(f"Model response for tool check: {model_response_for_tool_check[:200]}...")

            # Check for multiple tool calls first
            multiple_queries = self._detect_multiple_tool_calls(model_response_for_tool_check)
            
            if len(multiple_queries) > 1:
                # Multiple searches detected - execute them in parallel
                yield {"status": f"Performing {len(multiple_queries)} comprehensive searches..."}
                logger.info(f"🔍 Executing {len(multiple_queries)} parallel web searches")
                
                # Execute all searches in parallel
                search_tasks = []
                for query, time_range in multiple_queries:
                    search_tasks.append(search_web(query=query, max_results=30, time_range=time_range))
                
                # Wait for all searches to complete
                all_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                
                # Combine all results
                combined_results = []
                for idx, (result, (query, time_range)) in enumerate(zip(all_results, multiple_queries), 1):
                    if isinstance(result, Exception):
                        logger.error(f"Search {idx} failed: {result}")
                        combined_results.append(f"Search {idx} for '{query}' failed: {str(result)}")
                    else:
                        combined_results.append(f"=== Search {idx}: '{query}' ===\n{result}\n")
                
                combined_search_results = "\n\n".join(combined_results)
                
                # After getting all search results, add a stronger prompt to provide the final answer
                is_last_iteration = iteration >= self.max_iterations - 1
                if is_last_iteration:
                    tool_result_msg = f"""Here are comprehensive web search results from {len(multiple_queries)} different searches:\n\n{combined_search_results}\n\nIMPORTANT: You must now provide a complete, comprehensive answer to the user's question using ALL the information from these searches. Synthesize the information from all sources. Do NOT make another tool call. Provide your final response directly."""
                else:
                    tool_result_msg = f"""Here are comprehensive web search results from {len(multiple_queries)} different searches:\n\n{combined_search_results}\n\nPlease use this comprehensive information to answer the user's question. Synthesize information from all searches. If you have enough information, provide your answer now. Only make another search if absolutely necessary for missing critical information."""
                self.conversation.append({"role": "user", "content": tool_result_msg})
                continue  # Loop again to generate final response
            
            # Single tool call
            is_tool_call, query, time_range = self._detect_tool_call(model_response_for_tool_check)

            if is_tool_call and query:
                # Don't add tool call to conversation - it's internal only, not for display
                # self.conversation.append({"role": "assistant", "content": model_response_for_tool_check})

                yield {"status": f"Searching the web for: '{query}'"}
                logger.info(f"🔍 Executing web search: '{query}'")
                search_results = await search_web(query=query, max_results=30, time_range=time_range)

                # After getting search results, add a stronger prompt to provide the final answer
                # If this is the last iteration, be more explicit
                is_last_iteration = iteration >= self.max_iterations - 1
                if is_last_iteration:
                    tool_result_msg = f"""Here are the web search results for "{query}":\n\n{search_results}\n\nIMPORTANT: You must now provide a complete answer to the user's question using the information above. Do NOT make another tool call. Provide your final response directly."""
                else:
                    tool_result_msg = f"""Here are the web search results for "{query}":\n\n{search_results}\n\nPlease use this information to answer the user's question. If you have enough information, provide your answer now. Only make another search if absolutely necessary."""
                self.conversation.append({"role": "user", "content": tool_result_msg})
                continue  # Loop again to generate final response
            else:
                # No tool call detected - this should be the final response
                # Clean any tool calls from the response
                cleaned_response = re.sub(r'\[TOOL:.*?\]', '', model_response_for_tool_check, flags=re.DOTALL | re.IGNORECASE).strip()
                
                # Check if we have meaningful content after cleaning
                if cleaned_response and len(cleaned_response) > 10:
                    # We have a good response, stream it
                    yield {"status": "Generating response..."}
                    async for chunk in self._call_ollama_chat_stream(messages):
                        full_response += chunk
                        # Filter out tool calls from chunks before sending to frontend
                        cleaned_chunk = re.sub(r'\[TOOL:.*?\]', '', chunk, flags=re.DOTALL | re.IGNORECASE)
                        if cleaned_chunk.strip():  # Only yield if there's content after cleaning
                            yield {"content": cleaned_chunk}

                    # Clean tool calls from final response before saving
                    cleaned_full_response = re.sub(r'\[TOOL:.*?\]', '', full_response, flags=re.DOTALL | re.IGNORECASE).strip()
                    if cleaned_full_response:
                        self.conversation.append({"role": "assistant", "content": cleaned_full_response})
                    return  # End the generator
                else:
                    # Response was empty or only contained tool calls
                    # This shouldn't happen, but if it does, force a response
                    logger.warning(f"Empty response after cleaning at iteration {iteration}")
                    if iteration < self.max_iterations:
                        # Add explicit instruction to respond
                        self.conversation.append({
                            "role": "user", 
                            "content": "Please provide a complete answer to my question. Do not use any tools, just answer based on what you know or what you've already searched."
                        })
                        continue
                    else:
                        # Last iteration, provide fallback
                        yield {"content": "I apologize, but I'm having trouble formulating a response. Could you please rephrase your question?"}

        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        # Try to provide a helpful response even if we hit max iterations
        # Check if we have any search results in the conversation
        has_search_results = any("web search results" in str(msg.get("content", "")).lower() 
                                for msg in self.conversation if isinstance(msg, dict))
        
        if has_search_results:
            # Force a final response using the search results we have
            final_prompt = "Based on the search results provided earlier, please give a concise answer to the user's question. Summarize what you found."
            self.conversation.append({"role": "user", "content": final_prompt})
            messages = self._format_messages_for_chat()
            try:
                final_response = await self._call_ollama_chat(messages)
                if final_response and len(final_response.strip()) > 20:
                    # Clean tool calls from response
                    cleaned_final_response = re.sub(r'\[TOOL:.*?\]', '', final_response, flags=re.DOTALL | re.IGNORECASE).strip()
                    if cleaned_final_response:
                        # Stream the final response
                        async for chunk in self._call_ollama_chat_stream(messages):
                            # Filter tool calls from streaming chunks
                            cleaned_chunk = re.sub(r'\[TOOL:.*?\]', '', chunk, flags=re.DOTALL | re.IGNORECASE)
                            if cleaned_chunk.strip():
                                yield {"content": cleaned_chunk}
                        self.conversation.append({"role": "assistant", "content": cleaned_final_response})
                        return
            except Exception as e:
                logger.error(f"Error generating final response: {e}")
        
        yield {"content": "I'm having trouble processing your request. Please try rephrasing or being more specific."}

    async def send_message(self, message: str) -> str:
        """
        Send a message, handle tool calls iteratively, and return the final response.
        Used for non-streaming scenarios.
        """
        self.conversation.append({"role": "user", "content": message})

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.max_iterations}")

            # Get model's response
            messages = self._format_messages_for_chat()
            model_response = await self._call_ollama_chat(messages)

            logger.info(f"Model response: {model_response[:200]}...")

            # Check for multiple tool calls first
            multiple_queries = self._detect_multiple_tool_calls(model_response)
            
            if len(multiple_queries) > 1:
                # Multiple searches detected - execute them in parallel
                logger.info(f"🔍 Executing {len(multiple_queries)} parallel web searches")
                
                # Execute all searches in parallel
                search_tasks = []
                for query, time_range in multiple_queries:
                    search_tasks.append(search_web(query=query, max_results=30, time_range=time_range))
                
                # Wait for all searches to complete
                all_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                
                # Combine all results
                combined_results = []
                for idx, (result, (query, time_range)) in enumerate(zip(all_results, multiple_queries), 1):
                    if isinstance(result, Exception):
                        logger.error(f"Search {idx} failed: {result}")
                        combined_results.append(f"Search {idx} for '{query}' failed: {str(result)}")
                    else:
                        combined_results.append(f"=== Search {idx}: '{query}' ===\n{result}\n")
                
                combined_search_results = "\n\n".join(combined_results)
                
                # After getting all search results, add a stronger prompt to provide the final answer
                is_last_iteration = iteration >= self.max_iterations - 1
                if is_last_iteration:
                    tool_result_msg = f"""Here are comprehensive web search results from {len(multiple_queries)} different searches:\n\n{combined_search_results}\n\nIMPORTANT: You must now provide a complete, comprehensive answer to the user's question using ALL the information from these searches. Synthesize the information from all sources. Do NOT make another tool call. Provide your final response directly."""
                else:
                    tool_result_msg = f"""Here are comprehensive web search results from {len(multiple_queries)} different searches:\n\n{combined_search_results}\n\nPlease use this comprehensive information to answer the user's question. Synthesize information from all searches. If you have enough information, provide your answer now. Only make another search if absolutely necessary for missing critical information."""
                self.conversation.append({"role": "user", "content": tool_result_msg})
                continue  # Loop again to generate final response
            
            # Single tool call
            is_tool_call, query, time_range = self._detect_tool_call(model_response)

            if is_tool_call and query:
                # Don't add tool call to conversation - it's internal only, not for display
                # self.conversation.append({"role": "assistant", "content": model_response})

                # Execute search
                logger.info(f"🔍 Executing web search: '{query}'")
                search_results = await search_web(query=query, max_results=30, time_range=time_range)

                # After getting search results, add a stronger prompt to provide the final answer
                # If this is the last iteration, be more explicit
                is_last_iteration = iteration >= self.max_iterations - 1
                if is_last_iteration:
                    tool_result_msg = f"""Here are the web search results for "{query}":

{search_results}

IMPORTANT: You must now provide a complete answer to the user's question using the information above. Do NOT make another tool call. Provide your final response directly."""
                else:
                    tool_result_msg = f"""Here are the web search results for "{query}":

{search_results}

Please use this information to answer the user's question. If you have enough information, provide your answer now. Only make another search if absolutely necessary."""

                self.conversation.append({"role": "user", "content": tool_result_msg})
                logger.info(f"✓ Search completed, asking model to use results")

                # Continue loop to get final answer
                continue
            else:
                # No tool call - this is the final response
                # Clean tool calls from response before returning
                cleaned_response = re.sub(r'\[TOOL:.*?\]', '', model_response, flags=re.DOTALL | re.IGNORECASE).strip()
                if cleaned_response:
                    self.conversation.append({"role": "assistant", "content": cleaned_response})
                    logger.info(f"✓ Final response generated")
                    return cleaned_response
                else:
                    # If response was only a tool call, continue to next iteration
                    continue

        # Max iterations reached
        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        # Try to provide a helpful response even if we hit max iterations
        # Check if we have any search results in the conversation
        has_search_results = any("web search results" in str(msg.get("content", "")).lower() 
                                for msg in self.conversation if isinstance(msg, dict))
        
        if has_search_results:
            # Force a final response using the search results we have
            final_prompt = "Based on the search results provided earlier, please give a concise answer to the user's question. Summarize what you found."
            self.conversation.append({"role": "user", "content": final_prompt})
            messages = self._format_messages_for_chat()
            try:
                final_response = await self._call_ollama_chat(messages)
                if final_response and len(final_response.strip()) > 20:
                    # Clean tool calls from response
                    cleaned_response = re.sub(r'\[TOOL:.*?\]', '', final_response, flags=re.DOTALL | re.IGNORECASE).strip()
                    if cleaned_response:
                        self.conversation.append({"role": "assistant", "content": cleaned_response})
                        return cleaned_response
            except Exception as e:
                logger.error(f"Error generating final response: {e}")
        
        final_msg = "I'm having trouble processing your request. Please try rephrasing or being more specific."
        self.conversation.append({"role": "assistant", "content": final_msg})
        return final_msg

    async def send_message_voice(self, message: str):
        """
        Send a message for voice mode, handling tool calls and streaming tokens.
        Yields individual tokens for TTS processing.
        """
        self.conversation.append({"role": "user", "content": message})

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Voice Iteration {iteration}/{self.max_iterations}")

            messages = self._format_messages_for_chat()

            # Stream the response and accumulate it
            model_response_stream = self._call_ollama_chat_stream(messages)
            full_model_response = ""
            tokens_to_yield = []

            async for token in model_response_stream:
                full_model_response += token
                tokens_to_yield.append(token)

            # Check for multiple tool calls first
            multiple_queries = self._detect_multiple_tool_calls(full_model_response)
            
            if len(multiple_queries) > 1:
                # Multiple searches detected - execute them in parallel
                logger.info(f"🔍 Executing {len(multiple_queries)} parallel web searches")
                
                # Execute all searches in parallel
                search_tasks = []
                for query, time_range in multiple_queries:
                    search_tasks.append(search_web(query=query, max_results=30, time_range=time_range))
                
                # Wait for all searches to complete
                all_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                
                # Combine all results
                combined_results = []
                for idx, (result, (query, time_range)) in enumerate(zip(all_results, multiple_queries), 1):
                    if isinstance(result, Exception):
                        logger.error(f"Search {idx} failed: {result}")
                        combined_results.append(f"Search {idx} for '{query}' failed: {str(result)}")
                    else:
                        combined_results.append(f"=== Search {idx}: '{query}' ===\n{result}\n")
                
                combined_search_results = "\n\n".join(combined_results)
                
                tool_result_msg = f"""Here are comprehensive web search results from {len(multiple_queries)} different searches:\n\n{combined_search_results}\n\nPlease use this comprehensive information to answer the user's question. Synthesize information from all searches."""
                self.conversation.append({"role": "user", "content": tool_result_msg})
                logger.info(f"✓ {len(multiple_queries)} searches completed, asking model to use results")
                continue
            
            # Single tool call
            is_tool_call, query, time_range = self._detect_tool_call(full_model_response)

            if is_tool_call and query:
                # Don't add tool call to conversation - it's internal only, not for display
                # self.conversation.append({"role": "assistant", "content": full_model_response})

                logger.info(f"🔍 Executing web search: '{query}'")
                search_results = await search_web(query=query, max_results=30, time_range=time_range)

                tool_result_msg = f"""Here are the web search results for "{query}":\n\n{search_results}\n\nPlease use this information to answer the user's question."""

                self.conversation.append({"role": "user", "content": tool_result_msg})
                logger.info(f"✓ Search completed, asking model to use results")
                continue
            else:
                # No tool call, yield the collected tokens
                # Clean tool calls from response before saving and yielding
                cleaned_response = re.sub(r'\[TOOL:.*?\]', '', full_model_response, flags=re.DOTALL | re.IGNORECASE).strip()
                if cleaned_response:
                    for token in tokens_to_yield:
                        clean_token = clean_for_tts(token)
                        # Also remove tool calls from individual tokens
                        clean_token = re.sub(r'\[TOOL:.*?\]', '', clean_token, flags=re.DOTALL | re.IGNORECASE)
                        if clean_token.strip():
                            yield clean_token
                    self.conversation.append({"role": "assistant", "content": cleaned_response})
                    logger.info(f"✓ Final response generated")
                    return
                else:
                    # If response was only a tool call, continue to next iteration
                    continue

        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        final_msg = "I'm having trouble processing your request. Please try rephrasing."
        self.conversation.append({"role": "assistant", "content": final_msg})
        yield final_msg

    def clear_conversation(self):
        """Reset conversation history"""
        system_msg = self.conversation[0]
        self.conversation = [system_msg]
        logger.info("Conversation history cleared")

    def show_conversation_history(self):
        """Debug: Show conversation history"""
        print("\n=== CONVERSATION HISTORY ===")
        for i, msg in enumerate(self.conversation):
            role = msg["role"]
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            print(f"{i}. [{role}]: {content}")
        print("=== END ===\n")


# --- LiveKit Voice Agent (only available if livekit is installed) ---
if LIVEKIT_AVAILABLE:
    class Assistant(Agent):
        """LiveKit Agent with Ollama backend for voice interactions"""

        def __init__(self) -> None:
            # Initialize the OllamaAssistant in voice mode
            self.ollama_assistant = OllamaAssistant(
                model=ASSISTANT_MODEL,
                base_url=OLLAMA_BASE_URL,
                mode="voice"
            )

            super().__init__(
                instructions=AGENT_INSTRUCTION,
                llm=openai.LLM.with_ollama(
                    model=ASSISTANT_MODEL,
                    base_url=f"{OLLAMA_BASE_URL}/v1",
                    temperature=VOICE_GPU_CONFIG.get("temperature", 0.7),
                    top_p=VOICE_GPU_CONFIG.get("top_p", 0.9),
                ),
            )

        async def on_response(self, response, ctx: JobContext):
            """Clean the text before TTS sees it"""
            if response and response.text:
                response.text = clean_for_tts(response.text)
            return await super().on_response(response, ctx)


    async def entrypoint(ctx: JobContext):
        """LiveKit agent entrypoint for voice mode"""
        # Connect to the LiveKit Room first
        await ctx.connect()

        # Configure STT, LLM, and TTS
        session = AgentSession(
            stt=deepgram.STT(language="multi"),
            llm=openai.LLM.with_ollama(
                model=ASSISTANT_MODEL,
                base_url=f"{OLLAMA_BASE_URL}/v1",
            ),
            tts=cartesia.TTS(voice=CARTESIA_VOICE_ID),
        )

        # Start the session
        await session.start(
            room=ctx.room,
            agent=Assistant(),
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )

        # Generate the initial greeting
        await session.generate_reply(
            instructions=VOICE_SESSION_INSTRUCTION,
        )


# --- CLI Interface for Testing ---
async def main():
    """CLI interface for testing the assistant in text mode"""
    assistant = OllamaAssistant(model=ASSISTANT_MODEL, base_url=OLLAMA_BASE_URL, mode="text")

    print("=== A.N.I.E. Assistant (type 'quit' to exit, 'clear' to reset, 'debug' to show history) ===\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                assistant.clear_conversation()
                print("Conversation cleared.\n")
                continue

            if user_input.lower() == "debug":
                assistant.show_conversation_history()
                continue

            reply = await assistant.send_message(user_input)
            print(f"\nAssistant: {reply}\n")

        except KeyboardInterrupt:
            print("\n\nExiting chat.")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            print(f"Error: {e}\n")


if __name__ == "__main__":
    # Check if running as voice agent or text CLI
    import sys
    if "--voice" in sys.argv and LIVEKIT_AVAILABLE:
        agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
    else:
        asyncio.run(main())
