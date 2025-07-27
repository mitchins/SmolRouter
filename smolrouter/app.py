import os
import json
import logging
import re
import time
import yaml
from typing import AsyncIterator, Dict, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx

# Import database functionality
from smolrouter.database import (init_database, RequestLog, get_recent_logs, get_log_stats, get_inflight_requests,
                     estimate_tokens_from_request, extract_tokens_from_openai_response, estimate_token_count)

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-rerouter")

app = FastAPI(
    title="OpenAI Model Rerouter",
    description="Allows software with hard-coded model IDs to use whatever you desire",
)

# Templates for web UI
script_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(script_dir, "..", "templates")
templates = Jinja2Templates(directory=templates_dir)

# Configuration via environment variables
DEFAULT_UPSTREAM = os.getenv("DEFAULT_UPSTREAM", "http://localhost:8000")
LISTEN_HOST = os.getenv("LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "1234"))
RAW_MODEL_MAP = os.getenv("MODEL_MAP", "{}")
ROUTES_CONFIG = os.getenv("ROUTES_CONFIG", "routes.yaml")

# Feature flags
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "false").lower() in ("1", "true", "yes")
STRIP_THINKING = os.getenv("STRIP_THINKING", "true").lower() in ("1", "true", "yes")
STRIP_JSON_MARKDOWN = os.getenv("STRIP_JSON_MARKDOWN", "false").lower() in ("1", "true", "yes")
ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "true").lower() in ("1", "true", "yes")

# Timeout configuration
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "3000.0"))

def validate_url(url: str, name: str) -> str:
    """Validate and normalize a URL, providing helpful error messages."""
    if not url:
        raise ValueError(f"{name} cannot be empty")
    
    # Handle common mistakes
    if url.startswith("http://http://") or url.startswith("https://https://"):
        logger.warning(f"{name} contains duplicate protocol, fixing: {url}")
        url = url.split("://", 1)[1]  # Remove first protocol
        if not url.startswith("http"):
            url = "http://" + url
    
    # Parse and validate
    try:
        parsed = urlparse(url)
        
        # If no scheme or scheme looks like a hostname, add http://
        if not parsed.scheme or (parsed.scheme and not parsed.netloc):
            logger.warning(f"{name} missing protocol, adding http://: {url}")
            url = "http://" + url
            parsed = urlparse(url)
        
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"{name} must use http or https protocol, got: {parsed.scheme}")
        
        if not parsed.netloc:
            raise ValueError(f"{name} missing hostname: {url}")
            
        return url
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        raise ValueError(f"Invalid {name}: {url} - {e}")


# Load model mapping (simple exact or regex patterns)
try:
    MODEL_MAP = json.loads(RAW_MODEL_MAP)
except json.JSONDecodeError as e:
    logger.error(f"Failed to parse MODEL_MAP: {e}")
    MODEL_MAP = {}

# Load routing configuration
def load_routes_config() -> Dict:
    """Load routing configuration from YAML or JSON file.
    
    Expected format:
    routes:
      - match:
          source_host: "10.0.1.5"  # Optional: match by source IP/host
          model: "gpt-4"           # Optional: match by model name (supports regex)
        route:
          upstream: "http://gpu-server:8000"  # Required: target upstream
          model: "llama3-70b"                 # Optional: override model name
    """
    try:
        if not os.path.exists(ROUTES_CONFIG):
            logger.info(f"No routes config file found at {ROUTES_CONFIG}, using default routing")
            return {"routes": []}
            
        with open(ROUTES_CONFIG, 'r') as f:
            if ROUTES_CONFIG.endswith('.json'):
                config = json.load(f)
            else:  # Assume YAML
                config = yaml.safe_load(f)
                
        # Validate config structure
        if not isinstance(config, dict) or 'routes' not in config:
            logger.error(f"Invalid routes config: missing 'routes' key")
            return {"routes": []}
            
        if not isinstance(config['routes'], list):
            logger.error(f"Invalid routes config: 'routes' must be a list")
            return {"routes": []}
            
        logger.info(f"Loaded {len(config['routes'])} routing rules from {ROUTES_CONFIG}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load routes config from {ROUTES_CONFIG}: {e}")
        return {"routes": []}

ROUTES_CONFIG_DATA = load_routes_config()

def find_route(source_host: str, model: str) -> Tuple[str, Optional[str]]:
    """Find the best matching route for a request.
    
    Args:
        source_host: Source IP address of the request
        model: Original model name from the request
        
    Returns:
        Tuple of (upstream_url, model_override) where model_override is None if no override
    """
    for route in ROUTES_CONFIG_DATA.get('routes', []):
        match_criteria = route.get('match', {})
        route_config = route.get('route', {})
        
        # Check if this route matches
        matches = True
        
        # Check source host match (if specified)
        if 'source_host' in match_criteria:
            expected_host = match_criteria['source_host']
            if source_host != expected_host:
                matches = False
                
        # Check model match (if specified) - supports regex
        if matches and 'model' in match_criteria:
            model_pattern = match_criteria['model']
            if model_pattern.startswith('/') and model_pattern.endswith('/'):
                # Regex pattern
                pattern = model_pattern[1:-1]  # Remove slashes
                if not re.search(pattern, model):
                    matches = False
            else:
                # Exact match
                if model != model_pattern:
                    matches = False
        
        if matches:
            upstream = route_config.get('upstream')
            model_override = route_config.get('model')
            
            if upstream:
                logger.debug(f"Route matched: {source_host}/{model} -> {upstream}" +
                           (f" (model: {model_override})" if model_override else ""))
                return upstream, model_override
    
    # No specific route found, use default
    logger.debug(f"No specific route found for {source_host}/{model}, using default upstream")
    return DEFAULT_UPSTREAM, None

# Validate URLs on startup
try:
    DEFAULT_UPSTREAM = validate_url(DEFAULT_UPSTREAM, "DEFAULT_UPSTREAM")
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    logger.error("Please check your environment variables and restart")
    exit(1)

# Log configuration at startup
logger.info(f"SmolRouter starting...")
logger.info(f"DEFAULT_UPSTREAM: {DEFAULT_UPSTREAM}")
logger.info(f"MODEL_MAP: {MODEL_MAP}")
logger.info(f"ROUTES_CONFIG: {ROUTES_CONFIG} ({len(ROUTES_CONFIG_DATA.get('routes', []))} rules)")
logger.info(f"STRIP_THINKING: {STRIP_THINKING}")
logger.info(f"STRIP_JSON_MARKDOWN: {STRIP_JSON_MARKDOWN}")
logger.info(f"DISABLE_THINKING: {DISABLE_THINKING}")
logger.info(f"ENABLE_LOGGING: {ENABLE_LOGGING}")
logger.info(f"REQUEST_TIMEOUT: {REQUEST_TIMEOUT}s")
logger.info(f"Listening on {LISTEN_HOST}:{LISTEN_PORT}")

# Initialize database if logging is enabled
if ENABLE_LOGGING:
    try:
        init_database()
    except Exception as e:
        logger.error(f"Failed to initialize logging database: {e}")
        logger.warning("Request logging will be disabled")
        ENABLE_LOGGING = False


def rewrite_model(model: str) -> str:
    """Rewrite model names using exact matches or regex patterns.
    
    Args:
        model: Original model name
        
    Returns:
        Rewritten model name or original if no match found
    """
    # Check for exact match first
    if model in MODEL_MAP:
        return MODEL_MAP[model]
    
    # Check regex patterns (keys starting and ending with /)
    for pattern, target in MODEL_MAP.items():
        if pattern.startswith("/") and pattern.endswith("/"):
            regex_pattern = pattern.strip("/")
            match = re.match(regex_pattern, model)
            if match:
                return match.expand(target)
    
    # Return original model if no mapping found
    return model


def strip_think_chain_from_text(text: str) -> str:
    """Remove <think>...</think> blocks from text and normalize whitespace.
    
    Args:
        text: Input text that may contain think chains
        
    Returns:
        Cleaned text with think chains removed and whitespace normalized
    """
    # Remove any <think>...</think> blocks (including tags)
    result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Normalize whitespace
    result = re.sub(r'\s+', ' ', result)
    
    # Remove space before punctuation
    result = re.sub(r'\s+([.!?,:;])', r'\1', result)
    
    return result.strip()


def strip_json_markdown_from_text(text: str) -> str:
    """Extract JSON from markdown code blocks, converting markdown-fenced JSON to pure JSON.
    
    This function finds JSON code blocks like:
    ```json
    {
      "key": "value"
    }
    ```
    
    And extracts just the JSON content, removing the markdown formatting.
    
    Args:
        text: Input text that may contain JSON markdown blocks
        
    Returns:
        Text with JSON markdown blocks replaced by pure JSON content
    """
    # Pattern to match ```json...content...``` blocks (with or without newlines)
    pattern = r'```json\s*(.*?)\s*```'
    
    def extract_json_content(match):
        json_content = match.group(1).strip()
        # Clean up the JSON content - remove extra whitespace but preserve structure  
        lines = json_content.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:  # Only keep non-empty lines
                cleaned_lines.append(stripped)
        
        # Join with single spaces to create compact JSON
        return ' '.join(cleaned_lines)
    
    # Replace all JSON markdown blocks with their content
    result = re.sub(pattern, extract_json_content, text, flags=re.DOTALL)
    
    return result.strip()


def start_request_log(request: Request, service_type: str, upstream_url: str, original_model: str = None, mapped_model: str = None):
    """Create initial log entry for inflight tracking"""
    if not ENABLE_LOGGING:
        return None
    
    try:
        # Get client IP
        source_ip = request.client.host if request.client else "unknown"
        
        # Create initial log entry (inflight - no completed_at)
        log_entry = RequestLog.create(
            source_ip=source_ip,
            method=request.method,
            path=request.url.path,
            service_type=service_type,
            upstream_url=upstream_url,
            original_model=original_model,
            mapped_model=mapped_model
        )
        return log_entry
    except Exception as e:
        logger.error(f"Failed to start request log: {e}")
        return None

def complete_request_log(log_entry, start_time: float, response_data: dict, 
                        request_body: bytes = None, response_body: bytes = None):
    """Complete the log entry when request finishes"""
    if not ENABLE_LOGGING or not log_entry:
        return
    
    try:
        # Calculate metrics
        duration_ms = int((time.time() - start_time) * 1000)
        request_size = len(request_body) if request_body else 0
        response_size = len(response_body) if response_body else 0
        
        # Calculate token counts for performance analytics
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        try:
            # Try to extract accurate token counts from OpenAI usage data first
            if response_data.get('usage'):
                prompt_tokens, completion_tokens, total_tokens = extract_tokens_from_openai_response(response_data)
            else:
                # Fall back to estimation if usage data not available
                if request_body:
                    try:
                        request_data = json.loads(request_body.decode('utf-8'))
                        prompt_tokens = estimate_tokens_from_request(request_data)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        prompt_tokens = estimate_token_count(str(request_body.decode('utf-8', errors='ignore')))
                
                if response_body:
                    try:
                        response_text = response_body.decode('utf-8')
                        # Try to parse as JSON and extract content
                        try:
                            response_json = json.loads(response_text)
                            if response_json.get('response'):  # Ollama format
                                completion_tokens = estimate_token_count(response_json['response'])
                            elif response_json.get('choices'):  # OpenAI format
                                content = ""
                                for choice in response_json['choices']:
                                    if choice.get('message', {}).get('content'):
                                        content += choice['message']['content']
                                    elif choice.get('text'):
                                        content += choice['text']
                                completion_tokens = estimate_token_count(content)
                        except json.JSONDecodeError:
                            # If not JSON, estimate from raw text
                            completion_tokens = estimate_token_count(response_text)
                    except UnicodeDecodeError:
                        completion_tokens = 0
                
                total_tokens = prompt_tokens + completion_tokens
        
        except Exception as e:
            logger.debug(f"Failed to calculate token counts: {e}")
            # Keep defaults of 0 if calculation fails
        
        # Update log entry with completion data
        log_entry.duration_ms = duration_ms
        log_entry.request_size = request_size
        log_entry.response_size = response_size
        log_entry.status_code = response_data.get('status_code')
        log_entry.request_body = request_body
        log_entry.response_body = response_body
        log_entry.error_message = response_data.get('error_message')
        log_entry.completed_at = datetime.now()
        
        # Store token counts for performance analytics
        log_entry.prompt_tokens = prompt_tokens
        log_entry.completion_tokens = completion_tokens
        log_entry.total_tokens = total_tokens
        
        log_entry.save()
        logger.debug(f"Request completed with {prompt_tokens} prompt tokens, {completion_tokens} completion tokens")
    except Exception as e:
        logger.error(f"Failed to complete request log: {e}")


async def proxy_request(path: str, request: Request) -> StreamingResponse:
    start_time = time.time()
    original_model = None
    mapped_model = None
    
    # Get source IP for routing
    source_ip = request.client.host if request.client else "unknown"
    
    # Read and mutate JSON body
    try:
        payload = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse request JSON: {e}")
        # Use default upstream for error logging
        log_entry = start_request_log(request, "openai", DEFAULT_UPSTREAM, original_model, mapped_model)
        complete_request_log(log_entry, start_time, {"status_code": 400, "error_message": "Invalid JSON in request body"})
        return JSONResponse(
            content={"error": "Invalid JSON in request body"},
            status_code=400
        )
    
    # Extract model for routing
    if "model" in payload:
        original_model = payload["model"]
        mapped_model = rewrite_model(original_model)
        if mapped_model != original_model:
            logger.info(f"Rewriting model '{original_model}' -> '{mapped_model}'")
        payload["model"] = mapped_model
    
    # Find the best route for this request
    upstream_url, route_model_override = find_route(source_ip, original_model or "unknown")
    
    # Apply route-specific model override if specified
    final_model = route_model_override or mapped_model
    if route_model_override and route_model_override != mapped_model:
        logger.info(f"Route override: model '{mapped_model}' -> '{route_model_override}'")
        payload["model"] = route_model_override
    
    # Start logging with the determined upstream URL
    log_entry = start_request_log(request, "openai", upstream_url, original_model, final_model)

    # Update log entry with model info
    if log_entry:
        try:
            log_entry.original_model = original_model
            log_entry.mapped_model = final_model
            log_entry.save()
        except Exception as e:
            logger.error(f"Failed to update log entry: {e}")

    # If disabling thinking, append suffix to request content rather than model name
    if DISABLE_THINKING:
        logger.info("Disabling thinking by appending '/no_think' marker to content")
        if "messages" in payload and isinstance(payload["messages"], list):
            payload["messages"].append({"role": "system", "content": "/no_think"})
        elif "prompt" in payload and isinstance(payload["prompt"], str):
            payload["prompt"] = payload["prompt"].rstrip() + " /no_think"

    # Forward headers (keep Authorization)
    headers = {k: v for k, v in request.headers.items() if k.lower() in ["authorization", "openai-organization"]}

    url = f"{upstream_url}{path}"
    logger.debug(f"Proxying request to: {url}")
    
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            # Non-streaming case: forward and return JSON directly
            if not payload.get("stream"):
                resp = await client.post(url, json=payload, headers=headers)
                response_headers = {k: v for k, v in resp.headers.items() if k.lower() not in ["content-length", "transfer-encoding"]}
                data = resp.json()
                logger.debug(f"Downstream non-stream response data: {json.dumps(data)}")
                
                # Strip thinking chains and JSON markdown if enabled
                if STRIP_THINKING or STRIP_JSON_MARKDOWN:
                    for choice in data.get("choices", []):
                        if "message" in choice and isinstance(choice["message"].get("content"), str):
                            content = choice["message"]["content"]
                            if STRIP_THINKING:
                                content = strip_think_chain_from_text(content)
                            if STRIP_JSON_MARKDOWN:
                                content = strip_json_markdown_from_text(content)
                            choice["message"]["content"] = content
                        elif isinstance(choice.get("text"), str):
                            text = choice["text"]
                            if STRIP_THINKING:
                                text = strip_think_chain_from_text(text)
                            if STRIP_JSON_MARKDOWN:
                                text = strip_json_markdown_from_text(text)
                            choice["text"] = text
                    logger.debug(f"Cleaned non-stream response data: {json.dumps(data)}")
                
                response = JSONResponse(
                    content=data,
                    status_code=resp.status_code,
                    headers=response_headers,
                )
                
                # Complete the logging
                complete_request_log(log_entry, start_time, {"status_code": resp.status_code})
                
                return response
            else:
                async with client.stream("POST", url, json=payload, headers=headers) as upstream:
                    async def openai_streaming_response_generator() -> AsyncIterator[bytes]:
                        buffer = ""
                        async for chunk in upstream.aiter_bytes():
                            buffer += chunk.decode('utf-8')
                            try:
                                while True:
                                    eol = buffer.find("\n\n")
                                    if eol == -1:
                                        break

                                    message = buffer[:eol].strip()
                                    buffer = buffer[eol+4:]

                                    if message.startswith("data:"):
                                        json_data = message[len("data:"):].strip()
                                        if json_data == "[DONE]":
                                            yield b"data: [DONE]\n\n"
                                            return

                                        try:
                                            data = json.loads(json_data)
                                            if STRIP_THINKING or STRIP_JSON_MARKDOWN:
                                                if data.get("choices"):
                                                    if "delta" in data["choices"][0] and isinstance(data["choices"][0]["delta"].get("content"), str):
                                                        content = data["choices"][0]["delta"]["content"]
                                                        if STRIP_THINKING:
                                                            content = strip_think_chain_from_text(content)
                                                        if STRIP_JSON_MARKDOWN:
                                                            content = strip_json_markdown_from_text(content)
                                                        data["choices"][0]["delta"]["content"] = content
                                                    elif isinstance(data["choices"][0].get("text"), str):
                                                        text = data["choices"][0]["text"]
                                                        if STRIP_THINKING:
                                                            text = strip_think_chain_from_text(text)
                                                        if STRIP_JSON_MARKDOWN:
                                                            text = strip_json_markdown_from_text(text)
                                                        data["choices"][0]["text"] = text
                                            yield f"data: {json.dumps(data)}\n\n".encode('utf-8')
                                        except json.JSONDecodeError:
                                            logger.warning(f"Could not decode JSON from SSE: {json_data!r}")
                                            continue
                            except Exception as e:
                                logger.error(f"Error processing OpenAI stream: {e}")
                                break

                    response_headers = {k: v for k, v in upstream.headers.items() if k.lower() != "content-length"}
                    
                    # Complete streaming request logging
                    complete_request_log(log_entry, start_time, {"status_code": upstream.status_code})
                    
                    return StreamingResponse(
                        openai_streaming_response_generator(),
                        status_code=upstream.status_code,
                        headers=response_headers,
                        media_type="text/event-stream"
                    )
    except httpx.ConnectError as e:
        logger.error(f"Connection error to upstream {url}: {e}")
        
        # Complete logging with error
        complete_request_log(log_entry, start_time, {"status_code": 502, "error_message": str(e)})
        
        return JSONResponse(
            content={
                "error": "upstream_connection_failed",
                "message": f"Could not connect to upstream server at {upstream_url}",
                "details": str(e)
            },
            status_code=502
        )
    except httpx.TimeoutException as e:
        logger.error(f"Timeout error to upstream {url}: {e}")
        complete_request_log(log_entry, start_time, {"status_code": 504, "error_message": str(e)})
        return JSONResponse(
            content={
                "error": "upstream_timeout",
                "message": f"Upstream server at {upstream_url} did not respond in time",
                "details": str(e)
            },
            status_code=504
        )
    except Exception as e:
        logger.error(f"Unexpected error proxying to {url}: {e}")
        complete_request_log(log_entry, start_time, {"status_code": 500, "error_message": str(e)})
        return JSONResponse(
            content={
                "error": "proxy_error",
                "message": "An unexpected error occurred while proxying the request",
                "details": str(e)
            },
            status_code=500
        )


async def proxy_ollama_request(path: str, request: Request) -> StreamingResponse:
    start_time = time.time()
    original_model = None
    mapped_model = None
    
    # Get source IP for routing
    source_ip = request.client.host if request.client else "unknown"
    
    # Read and mutate JSON body
    try:
        ollama_payload = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse Ollama request JSON: {e}")
        # Use default upstream for error logging
        log_entry = start_request_log(request, "ollama", DEFAULT_UPSTREAM, original_model, mapped_model)
        complete_request_log(log_entry, start_time, {"status_code": 400, "error_message": "Invalid JSON in request body"})
        return JSONResponse(
            content={"error": "Invalid JSON in request body"},
            status_code=400
        )
    
    logger.info(f"Received Ollama request to {path}: {ollama_payload}")

    # Determine if it's a chat or generate endpoint
    is_chat_endpoint = "/chat" in path

    # Transform Ollama request to OpenAI format
    original_model = ollama_payload["model"]
    mapped_model = rewrite_model(original_model)
    
    # Find the best route for this request
    upstream_url, route_model_override = find_route(source_ip, original_model)
    
    # Apply route-specific model override if specified
    final_model = route_model_override or mapped_model
    if route_model_override and route_model_override != mapped_model:
        logger.info(f"Route override: model '{mapped_model}' -> '{route_model_override}'")
    
    openai_payload = {}
    openai_payload["model"] = final_model
    openai_payload["stream"] = ollama_payload.get("stream", False)
    
    # Start logging with the determined upstream URL
    log_entry = start_request_log(request, "ollama", upstream_url, original_model, final_model)
    
    # Update log entry with model info
    if log_entry:
        try:
            log_entry.original_model = original_model
            log_entry.mapped_model = final_model
            log_entry.save()
        except Exception as e:
            logger.error(f"Failed to update Ollama log entry: {e}")

    if is_chat_endpoint:
        openai_payload["messages"] = ollama_payload["messages"]
    else:  # /api/generate
        openai_payload["messages"] = [{"role": "user", "content": ollama_payload["prompt"]}]

    # If disabling thinking, append suffix to request content rather than model name
    if DISABLE_THINKING:
        logger.info("Disabling thinking by appending '/no_think' marker to content")
        openai_payload["messages"].append({"role": "system", "content": "/no_think"})

    # Forward headers (keep Authorization)
    headers = {k: v for k, v in request.headers.items() if k.lower() in ["authorization", "openai-organization"]}

    url = f"{upstream_url}/v1/chat/completions"
    logger.debug(f"Proxying Ollama request to OpenAI endpoint: {url}")
    
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            if not openai_payload.get("stream"):
                resp = await client.post(url, json=openai_payload, headers=headers)
                response_headers = {k: v for k, v in resp.headers.items() if k.lower() not in ["content-length", "transfer-encoding"]}
                openai_data = resp.json()
                logger.debug(f"Downstream non-stream OpenAI response data: {json.dumps(openai_data)}")

                # Extract content from OpenAI response
                ollama_response_content = ""
                if openai_data.get("choices"):
                    choice = openai_data["choices"][0]
                    if "message" in choice and isinstance(choice["message"].get("content"), str):
                        ollama_response_content = choice["message"]["content"]
                    elif isinstance(choice.get("text"), str):
                        ollama_response_content = choice["text"]

                # Strip thinking chains and JSON markdown if enabled
                if STRIP_THINKING:
                    ollama_response_content = strip_think_chain_from_text(ollama_response_content)
                if STRIP_JSON_MARKDOWN:
                    logger.debug(f"Original content before JSON markdown stripping: {repr(ollama_response_content)}")
                    ollama_response_content = strip_json_markdown_from_text(ollama_response_content)
                    logger.debug(f"Content after JSON markdown stripping: {repr(ollama_response_content)}")

                # Transform to Ollama response format
                ollama_response = {
                    "model": ollama_payload["model"],
                    "created_at": openai_data.get("created", ""),
                    "response": ollama_response_content,
                    "done": True,
                    "done_reason": "stop",
                }
                logger.debug(f"Transformed non-stream Ollama response: {json.dumps(ollama_response)}")
                logger.info(f"Final Ollama response content: {repr(ollama_response.get('response', ''))}")
                
                # Complete logging for successful response
                request_body_bytes = json.dumps(ollama_payload).encode('utf-8')
                response_body_bytes = json.dumps(ollama_response).encode('utf-8')
                # Include OpenAI usage data for accurate token counting
                response_data = {"status_code": resp.status_code, "usage": openai_data.get("usage")}
                complete_request_log(log_entry, start_time, response_data, 
                                   request_body=request_body_bytes, response_body=response_body_bytes)
                
                return JSONResponse(
                    content=ollama_response,
                    status_code=resp.status_code,
                    headers=response_headers,
                )
            else:
                async with client.stream("POST", url, json=openai_payload, headers=headers) as upstream:
                    async def ollama_streaming_response_generator() -> AsyncIterator[bytes]:
                        buffer = ""
                        async for chunk in upstream.aiter_bytes():
                            buffer += chunk.decode('utf-8')
                            try:
                                while True:
                                    # Find the end of an SSE message
                                    eol = buffer.find("\n\n")
                                    if eol == -1:
                                        break

                                    message = buffer[:eol].strip()
                                    buffer = buffer[eol+4:] # +4 for \n\n

                                    if message.startswith("data:"):
                                        json_data = message[len("data:"):].strip()
                                        if json_data == "[DONE]":
                                            # Send final done message in Ollama format
                                            final_ollama_chunk = {
                                                "model": ollama_payload["model"],
                                                "created_at": datetime.now().isoformat(),
                                                "response": "",
                                                "done": True,
                                                "done_reason": "stop",
                                            }
                                            yield json.dumps(final_ollama_chunk).encode('utf-8') + b'\n'
                                            return

                                        try:
                                            data = json.loads(json_data)
                                            content = ""
                                            if data.get("choices"):
                                                if "delta" in data["choices"][0] and isinstance(data["choices"][0]["delta"].get("content"), str):
                                                    content = data["choices"][0]["delta"]["content"]
                                                elif isinstance(data["choices"][0].get("text"), str):
                                                    content = data["choices"][0]["text"]

                                            if STRIP_THINKING:
                                                content = strip_think_chain_from_text(content)
                                            if STRIP_JSON_MARKDOWN:
                                                logger.debug(f"Streaming: Original content before JSON markdown stripping: {repr(content)}")
                                                content = strip_json_markdown_from_text(content)
                                                logger.debug(f"Streaming: Content after JSON markdown stripping: {repr(content)}")

                                            # Transform to Ollama streaming format
                                            ollama_chunk = {
                                                "model": ollama_payload["model"],
                                                "created_at": data.get("created", ""),
                                                "response": content,
                                                "done": False,
                                            }
                                            
                                            # Add finish reason if present
                                            if data.get("choices") and data["choices"][0].get("finish_reason"):
                                                ollama_chunk["done_reason"] = data["choices"][0]["finish_reason"]
                                                
                                            yield json.dumps(ollama_chunk).encode('utf-8') + b'\n'
                                        except json.JSONDecodeError:
                                            logger.warning(f"Could not decode JSON from SSE: {json_data!r}")
                                            continue
                            except Exception as e:
                                logger.error(f"Error processing stream: {e}")
                                # Yield an error message or re-raise
                                break

                    response_headers = {k: v for k, v in upstream.headers.items() if k.lower() != "content-length"}
                    # Complete logging for streaming response
                    complete_request_log(log_entry, start_time, {"status_code": upstream.status_code})
                    
                    return StreamingResponse(
                        ollama_streaming_response_generator(),
                        status_code=upstream.status_code,
                        headers=response_headers,
                        media_type="application/x-ndjson"
                    )
    except httpx.ConnectError as e:
        logger.error(f"Connection error to upstream {url}: {e}")
        complete_request_log(log_entry, start_time, {"status_code": 502, "error_message": str(e)})
        return JSONResponse(
            content={
                "error": "upstream_connection_failed",
                "message": f"Could not connect to upstream server at {upstream_url}",
                "details": str(e)
            },
            status_code=502
        )
    except httpx.TimeoutException as e:
        logger.error(f"Timeout error to upstream {url}: {e}")
        complete_request_log(log_entry, start_time, {"status_code": 504, "error_message": str(e)})
        return JSONResponse(
            content={
                "error": "upstream_timeout", 
                "message": f"Upstream server at {upstream_url} did not respond in time",
                "details": str(e)
            },
            status_code=504
        )
    except Exception as e:
        logger.error(f"Unexpected error proxying Ollama request to {url}: {e}")
        complete_request_log(log_entry, start_time, {"status_code": 500, "error_message": str(e)})
        return JSONResponse(
            content={
                "error": "proxy_error",
                "message": "An unexpected error occurred while proxying the request",
                "details": str(e)
            },
            status_code=500
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await proxy_request("/v1/chat/completions", request)


@app.post("/v1/completions")
async def completions(request: Request):
    return await proxy_request("/v1/completions", request)


@app.get("/v1/models")
async def list_models(request: Request):
    # Simply proxy model listing
    headers = {k: v for k, v in request.headers.items() if k.lower() in ["authorization"]}
    url = f"{DEFAULT_UPSTREAM}/v1/models"
    logger.debug(f"Proxying models request to: {url}")
    
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            upstream = await client.get(url, headers=headers)
            data = upstream.json()
        # (Optional) rewrite IDs in data.get("data", []) here
        return JSONResponse(content=data, status_code=upstream.status_code)
    except httpx.ConnectError as e:
        logger.error(f"Connection error to upstream {url}: {e}")
        return JSONResponse(
            content={
                "error": "upstream_connection_failed",
                "message": f"Could not connect to upstream server at {DEFAULT_UPSTREAM}"
            },
            status_code=502
        )
    except Exception as e:
        logger.error(f"Error listing models from {url}: {e}")
        return JSONResponse(
            content={
                "error": "models_error",
                "message": "Failed to retrieve models from upstream"
            },
            status_code=500
        )


@app.post("/api/generate")
async def ollama_generate(request: Request):
    return await proxy_ollama_request("/api/generate", request)

@app.post("/api/chat")
async def ollama_chat(request: Request):
    return await proxy_ollama_request("/api/chat", request)

@app.get("/api/tags")
async def ollama_list_models(request: Request):
    """Convert OpenAI /v1/models to Ollama /api/tags format"""
    headers = {k: v for k, v in request.headers.items() if k.lower() in ["authorization"]}
    url = f"{DEFAULT_UPSTREAM}/v1/models"
    logger.debug(f"Converting OpenAI models from {url} to Ollama tags format")
    
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            upstream = await client.get(url, headers=headers)
            openai_data = upstream.json()
            
            # Convert OpenAI format to Ollama format
            ollama_models = []
            for model in openai_data.get("data", []):
                ollama_models.append({
                    "name": model.get("id", "unknown"),
                    "modified_at": "2024-01-01T00:00:00Z",  # Mock timestamp
                    "size": 4000000000,  # Mock size (4GB)
                    "digest": "sha256:mock_digest"  # Mock digest
                })
            
            ollama_response = {"models": ollama_models}
            logger.debug(f"Converted {len(ollama_models)} models to Ollama format")
            
            return JSONResponse(content=ollama_response, status_code=upstream.status_code)
            
    except httpx.ConnectError as e:
        logger.error(f"Connection error to upstream {url}: {e}")
        return JSONResponse(
            content={
                "error": "upstream_connection_failed", 
                "message": f"Could not connect to upstream server at {DEFAULT_UPSTREAM}"
            },
            status_code=502
        )
    except Exception as e:
        logger.error(f"Error converting models to Ollama format: {e}")
        return JSONResponse(content={"error": "conversion_error"}, status_code=500)


# Web UI Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard showing request logs"""
    try:
        logs = get_recent_logs(limit=100)
        stats = get_log_stats()
        return templates.TemplateResponse(request, "index.html", {
            "logs": logs,
            "stats": stats
        })
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return HTMLResponse(
            content=f"<h1>Error</h1><p>Failed to load dashboard: {e}</p>",
            status_code=500
        )

@app.get("/performance", response_class=HTMLResponse)
async def performance_dashboard(request: Request):
    """Performance analytics dashboard with scatter plots"""
    try:
        return templates.TemplateResponse(request, "performance.html", {
            "title": "Performance Analytics"
        })
    except Exception as e:
        logger.error(f"Error loading performance dashboard: {e}")
        return HTMLResponse(content="<h1>Error loading performance dashboard</h1>", status_code=500)

@app.get("/api/logs")
async def api_logs(limit: int = 100, service_type: str = None):
    """API endpoint for getting logs as JSON"""
    try:
        logs = get_recent_logs(limit=limit, service_type=service_type)
        return [
            {
                "id": log.id,
                "timestamp": log.timestamp.isoformat(),
                "source_ip": log.source_ip,
                "method": log.method,
                "path": log.path,
                "service_type": log.service_type,
                "original_model": log.original_model,
                "mapped_model": log.mapped_model,
                "duration_ms": log.duration_ms,
                "request_size": log.request_size,
                "response_size": log.response_size,
                "status_code": log.status_code,
                "error_message": log.error_message,
                "completed_at": log.completed_at.isoformat() if log.completed_at else None,
                "is_inflight": log.completed_at is None
            }
            for log in logs
        ]
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return JSONResponse(
            content={"error": "Failed to get logs"},
            status_code=500
        )

@app.get("/api/stats")
async def api_stats():
    """API endpoint for getting statistics"""
    try:
        return get_log_stats()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return JSONResponse(
            content={"error": "Failed to get stats"},
            status_code=500
        )

@app.get("/api/inflight")
async def api_inflight():
    """API endpoint for getting currently inflight requests"""
    try:
        inflight = get_inflight_requests()
        return [
            {
                "id": log.id,
                "timestamp": log.timestamp.isoformat(),
                "source_ip": log.source_ip,
                "method": log.method,
                "path": log.path,
                "service_type": log.service_type,
                "original_model": log.original_model,
                "mapped_model": log.mapped_model,
                "elapsed_ms": int((datetime.now() - log.timestamp).total_seconds() * 1000)
            }
            for log in inflight
        ]
    except Exception as e:
        logger.error(f"Error getting inflight requests: {e}")
        return JSONResponse(
            content={"error": "Failed to get inflight requests"},
            status_code=500
        )


@app.get("/api/performance")
async def api_performance(
    limit: int = 1000,
    hours: int = 24,
    model: str = None,
    service_type: str = None
):
    """Get performance analytics data for scatter plot visualization.
    
    Returns data points with prompt_tokens (x-axis) vs duration_ms (y-axis),
    grouped by model and endpoint for performance analysis.
    
    Args:
        limit: Maximum number of data points to return (default: 1000)
        hours: Number of hours to look back (default: 24)
        model: Filter by specific model name (optional)
        service_type: Filter by service type: 'openai' or 'ollama' (optional)
    """
    try:
        from datetime import timedelta
        
        # Build query for completed requests with token data
        query = RequestLog.select().where(
            RequestLog.completed_at.is_null(False),  # Only completed requests
            RequestLog.prompt_tokens.is_null(False),  # Must have token data
            RequestLog.duration_ms.is_null(False),   # Must have duration data
            RequestLog.timestamp >= datetime.now() - timedelta(hours=hours)
        )
        
        # Apply filters
        if model:
            query = query.where(RequestLog.mapped_model == model)
        if service_type:
            query = query.where(RequestLog.service_type == service_type)
        
        # Order by timestamp desc and limit
        query = query.order_by(RequestLog.timestamp.desc()).limit(limit)
        
        # Format data for scatter plot
        data_points = []
        for log in query:
            data_points.append({
                "id": log.id,
                "timestamp": log.timestamp.isoformat(),
                "prompt_tokens": log.prompt_tokens,
                "completion_tokens": log.completion_tokens, 
                "total_tokens": log.total_tokens,
                "duration_ms": log.duration_ms,
                "model": log.mapped_model or log.original_model,
                "original_model": log.original_model,
                "mapped_model": log.mapped_model,
                "service_type": log.service_type,
                "path": log.path,
                "status_code": log.status_code,
                "request_size": log.request_size,
                "response_size": log.response_size,
            })
        
        return {
            "data_points": data_points,
            "meta": {
                "total_points": len(data_points),
                "hours_back": hours,
                "filters": {
                    "model": model,
                    "service_type": service_type
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance data: {e}")
        return JSONResponse(content={"error": "Failed to get performance data"}, status_code=500)


@app.get("/request/{request_id}", response_class=HTMLResponse)
async def request_detail(request_id: int, request: Request):
    """Detailed view of a specific request"""
    try:
        log_entry = RequestLog.get_by_id(request_id)
        
        return templates.TemplateResponse(request, "request_detail.html", {
            "log": log_entry,
            "request_body_str": log_entry.request_body.decode('utf-8') if log_entry.request_body else None,
            "response_body_str": log_entry.response_body.decode('utf-8') if log_entry.response_body else None,
        })
    except RequestLog.DoesNotExist:
        return HTMLResponse(
            content="<h1>Request Not Found</h1><p>The requested log entry does not exist.</p>",
            status_code=404
        )
    except Exception as e:
        logger.error(f"Error rendering request detail: {e}")
        return HTMLResponse(
            content=f"<h1>Error</h1><p>Failed to load request details: {e}</p>",
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=LISTEN_HOST, port=LISTEN_PORT)