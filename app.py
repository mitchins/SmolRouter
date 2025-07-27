import os
import json
import logging
import re
import time
from typing import AsyncIterator
from datetime import datetime
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx

# Import database functionality
from database import init_database, RequestLog, get_recent_logs, get_log_stats

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-rerouter")

app = FastAPI(
    title="OpenAI Model Rerouter",
    description="Allows software with hard-coded model IDs to use whatever you desire",
)

# Templates for web UI
templates = Jinja2Templates(directory="templates")

# Configuration via environment variables
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://localhost:8000")
OLLAMA_UPSTREAM_URL = os.getenv("OLLAMA_UPSTREAM_URL", "http://localhost:11434")
LISTEN_HOST = os.getenv("LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "1234"))
RAW_MODEL_MAP = os.getenv("MODEL_MAP", "{}")

# Feature flags
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "false").lower() in ("1", "true", "yes")
STRIP_THINKING = os.getenv("STRIP_THINKING", "true").lower() in ("1", "true", "yes")
ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "true").lower() in ("1", "true", "yes")

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

# Validate URLs on startup
try:
    UPSTREAM_URL = validate_url(UPSTREAM_URL, "UPSTREAM_URL")
    OLLAMA_UPSTREAM_URL = validate_url(OLLAMA_UPSTREAM_URL, "OLLAMA_UPSTREAM_URL")
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    logger.error("Please check your environment variables and restart")
    exit(1)

# Log configuration at startup
logger.info(f"OpenAI Model Rerouter starting...")
logger.info(f"UPSTREAM_URL: {UPSTREAM_URL}")
logger.info(f"OLLAMA_UPSTREAM_URL: {OLLAMA_UPSTREAM_URL}")
logger.info(f"MODEL_MAP: {MODEL_MAP}")
logger.info(f"STRIP_THINKING: {STRIP_THINKING}")
logger.info(f"DISABLE_THINKING: {DISABLE_THINKING}")
logger.info(f"ENABLE_LOGGING: {ENABLE_LOGGING}")
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


def log_request(request: Request, response_data: dict, start_time: float, service_type: str, 
                original_model: str = None, mapped_model: str = None, 
                request_body: bytes = None, response_body: bytes = None):
    """Log request details to database"""
    if not ENABLE_LOGGING:
        return
    
    try:
        # Get client IP
        source_ip = request.client.host if request.client else "unknown"
        
        # Calculate metrics
        duration_ms = int((time.time() - start_time) * 1000)
        request_size = len(request_body) if request_body else 0
        response_size = len(response_body) if response_body else 0
        
        # Create log entry
        RequestLog.create(
            source_ip=source_ip,
            method=request.method,
            path=request.url.path,
            service_type=service_type,
            upstream_url=UPSTREAM_URL,
            original_model=original_model,
            mapped_model=mapped_model,
            duration_ms=duration_ms,
            request_size=request_size,
            response_size=response_size,
            status_code=response_data.get('status_code'),
            request_body=request_body,
            response_body=response_body,
            error_message=response_data.get('error_message')
        )
    except Exception as e:
        logger.error(f"Failed to log request: {e}")


async def proxy_request(path: str, request: Request) -> StreamingResponse:
    start_time = time.time()
    original_model = None
    mapped_model = None
    
    # Read and mutate JSON body
    try:
        payload = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse request JSON: {e}")
        return JSONResponse(
            content={"error": "Invalid JSON in request body"},
            status_code=400
        )
    
    if "model" in payload:
        original_model = payload["model"]
        mapped_model = rewrite_model(original_model)
        if mapped_model != original_model:
            logger.info(f"Rewriting model '{original_model}' -> '{mapped_model}'")
        payload["model"] = mapped_model

    # If disabling thinking, append suffix to request content rather than model name
    if DISABLE_THINKING:
        logger.info("Disabling thinking by appending '/no_think' marker to content")
        if "messages" in payload and isinstance(payload["messages"], list):
            payload["messages"].append({"role": "system", "content": "/no_think"})
        elif "prompt" in payload and isinstance(payload["prompt"], str):
            payload["prompt"] = payload["prompt"].rstrip() + " /no_think"

    # Forward headers (keep Authorization)
    headers = {k: v for k, v in request.headers.items() if k.lower() in ["authorization", "openai-organization"]}

    url = f"{UPSTREAM_URL}{path}"
    logger.debug(f"Proxying request to: {url}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Non-streaming case: forward and return JSON directly
            if not payload.get("stream"):
                resp = await client.post(url, json=payload, headers=headers)
                response_headers = {k: v for k, v in resp.headers.items() if k.lower() not in ["content-length", "transfer-encoding"]}
                data = resp.json()
                logger.debug(f"Downstream non-stream response data: {json.dumps(data)}")
                
                # Strip thinking chains if enabled
                if STRIP_THINKING:
                    for choice in data.get("choices", []):
                        if "message" in choice and isinstance(choice["message"].get("content"), str):
                            choice["message"]["content"] = strip_think_chain_from_text(choice["message"]["content"])
                        elif isinstance(choice.get("text"), str):
                            choice["text"] = strip_think_chain_from_text(choice["text"])
                    logger.debug(f"Cleaned non-stream response data: {json.dumps(data)}")
                
                response = JSONResponse(
                    content=data,
                    status_code=resp.status_code,
                    headers=response_headers,
                )
                
                # Log the request
                if ENABLE_LOGGING:
                    log_request(request, {"status_code": resp.status_code}, 
                               start_time, "openai", original_model, mapped_model)
                
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
                                            if STRIP_THINKING:
                                                if data.get("choices"):
                                                    if "delta" in data["choices"][0] and isinstance(data["choices"][0]["delta"].get("content"), str):
                                                        data["choices"][0]["delta"]["content"] = strip_think_chain_from_text(data["choices"][0]["delta"]["content"])
                                                    elif isinstance(data["choices"][0].get("text"), str):
                                                        data["choices"][0]["text"] = strip_think_chain_from_text(data["choices"][0]["text"])
                                            yield f"data: {json.dumps(data)}\n\n".encode('utf-8')
                                        except json.JSONDecodeError:
                                            logger.warning(f"Could not decode JSON from SSE: {json_data!r}")
                                            continue
                            except Exception as e:
                                logger.error(f"Error processing OpenAI stream: {e}")
                                break

                    response_headers = {k: v for k, v in upstream.headers.items() if k.lower() != "content-length"}
                    
                    # Log streaming request
                    if ENABLE_LOGGING:
                        log_request(request, {"status_code": upstream.status_code}, 
                                   start_time, "openai", original_model, mapped_model)
                    
                    return StreamingResponse(
                        openai_streaming_response_generator(),
                        status_code=upstream.status_code,
                        headers=response_headers,
                        media_type="text/event-stream"
                    )
    except httpx.ConnectError as e:
        logger.error(f"Connection error to upstream {url}: {e}")
        
        # Log the error
        if ENABLE_LOGGING:
            log_request(request, {"status_code": 502, "error_message": str(e)}, 
                       start_time, "openai", original_model, mapped_model)
        
        return JSONResponse(
            content={
                "error": "upstream_connection_failed",
                "message": f"Could not connect to upstream server at {UPSTREAM_URL}",
                "details": str(e)
            },
            status_code=502
        )
    except httpx.TimeoutException as e:
        logger.error(f"Timeout error to upstream {url}: {e}")
        return JSONResponse(
            content={
                "error": "upstream_timeout",
                "message": f"Upstream server at {UPSTREAM_URL} did not respond in time",
                "details": str(e)
            },
            status_code=504
        )
    except Exception as e:
        logger.error(f"Unexpected error proxying to {url}: {e}")
        return JSONResponse(
            content={
                "error": "proxy_error",
                "message": "An unexpected error occurred while proxying the request",
                "details": str(e)
            },
            status_code=500
        )


async def proxy_ollama_request(path: str, request: Request) -> StreamingResponse:
    # Read and mutate JSON body
    try:
        ollama_payload = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse Ollama request JSON: {e}")
        return JSONResponse(
            content={"error": "Invalid JSON in request body"},
            status_code=400
        )
    
    logger.info(f"Received Ollama request to {path}: {ollama_payload}")

    # Determine if it's a chat or generate endpoint
    is_chat_endpoint = "/chat" in path

    # Transform Ollama request to OpenAI format
    openai_payload = {}
    openai_payload["model"] = rewrite_model(ollama_payload["model"])
    openai_payload["stream"] = ollama_payload.get("stream", False)

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

    url = f"{UPSTREAM_URL}/v1/chat/completions"
    logger.debug(f"Proxying Ollama request to OpenAI endpoint: {url}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
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

                # Strip thinking chains if enabled
                if STRIP_THINKING:
                    ollama_response_content = strip_think_chain_from_text(ollama_response_content)

                # Transform to Ollama response format
                ollama_response = {
                    "model": ollama_payload["model"],
                    "created_at": openai_data.get("created", ""),
                    "response": ollama_response_content,
                    "done": True,
                    "done_reason": "stop",
                }
                logger.debug(f"Transformed non-stream Ollama response: {json.dumps(ollama_response)}")
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
                                                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

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
                    return StreamingResponse(
                        ollama_streaming_response_generator(),
                        status_code=upstream.status_code,
                        headers=response_headers,
                        media_type="application/x-ndjson"
                    )
    except httpx.ConnectError as e:
        logger.error(f"Connection error to upstream {url}: {e}")
        return JSONResponse(
            content={
                "error": "upstream_connection_failed",
                "message": f"Could not connect to upstream server at {UPSTREAM_URL}",
                "details": str(e)
            },
            status_code=502
        )
    except httpx.TimeoutException as e:
        logger.error(f"Timeout error to upstream {url}: {e}")
        return JSONResponse(
            content={
                "error": "upstream_timeout", 
                "message": f"Upstream server at {UPSTREAM_URL} did not respond in time",
                "details": str(e)
            },
            status_code=504
        )
    except Exception as e:
        logger.error(f"Unexpected error proxying Ollama request to {url}: {e}")
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
    url = f"{UPSTREAM_URL}/v1/models"
    logger.debug(f"Proxying models request to: {url}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            upstream = await client.get(url, headers=headers)
            data = upstream.json()
        # (Optional) rewrite IDs in data.get("data", []) here
        return JSONResponse(content=data, status_code=upstream.status_code)
    except httpx.ConnectError as e:
        logger.error(f"Connection error to upstream {url}: {e}")
        return JSONResponse(
            content={
                "error": "upstream_connection_failed",
                "message": f"Could not connect to upstream server at {UPSTREAM_URL}"
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
    headers = {k: v for k, v in request.headers.items() if k.lower() in ["authorization"]}
    url = f"{OLLAMA_UPSTREAM_URL}/api/tags"
    logger.debug(f"Proxying Ollama tags request to: {url}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            upstream = await client.get(url, headers=headers)
            data = upstream.json()
        return JSONResponse(content=data, status_code=upstream.status_code)
    except httpx.ConnectError as e:
        logger.error(f"Connection error to Ollama upstream {url}: {e}")
        return JSONResponse(
            content={
                "error": "ollama_connection_failed",
                "message": f"Could not connect to Ollama server at {OLLAMA_UPSTREAM_URL}"
            },
            status_code=502
        )
    except Exception as e:
        logger.error(f"Error listing Ollama models from {url}: {e}")
        return JSONResponse(
            content={
                "error": "ollama_tags_error",
                "message": "Failed to retrieve models from Ollama upstream"
            },
            status_code=500
        )


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
                "error_message": log.error_message
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=LISTEN_HOST, port=LISTEN_PORT)