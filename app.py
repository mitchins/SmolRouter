import os
import json
import logging
import re
from typing import AsyncIterator
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-rerouter")

app = FastAPI(
    title="OpenAI Model Rerouter",
    description="Allows software with hard-coded model IDs to use whatever you desire",
)

# Configuration via environment variables
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://localhost:8000")
OLLAMA_UPSTREAM_URL = os.getenv("OLLAMA_UPSTREAM_URL", "http://localhost:11434")
LISTEN_HOST = os.getenv("LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "1234"))
RAW_MODEL_MAP = os.getenv("MODEL_MAP", "{}")

# Feature flags
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "false").lower() in ("1", "true", "yes")
STRIP_THINKING = os.getenv("STRIP_THINKING", "true").lower() in ("1", "true", "yes")

# Load model mapping (simple exact or regex patterns)
try:
    MODEL_MAP = json.loads(RAW_MODEL_MAP)
except json.JSONDecodeError as e:
    logger.error(f"Failed to parse MODEL_MAP: {e}")
    MODEL_MAP = {}


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
    # Remove any <think>...</think> blocks (including tags) and normalize whitespace
    result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Clean up extra whitespace, including space before punctuation
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s+([.!?,:;])', r'\1', result)
    return result.strip()


async def proxy_request(path: str, request: Request) -> StreamingResponse:
    # Read and mutate JSON body
    payload = await request.json()
    if "model" in payload:
        original = payload["model"]
        new_model = rewrite_model(original)
        if new_model != original:
            logger.info(f"Rewriting model '{original}' -> '{new_model}'")
        payload["model"] = new_model

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
    async with httpx.AsyncClient(timeout=None) as client:
        # Non-streaming case: forward and return JSON directly
        if not payload.get("stream"):
            resp = await client.post(url, json=payload, headers=headers)
            response_headers = {k: v for k, v in resp.headers.items() if k.lower() not in ["content-length", "transfer-encoding"]}
            data = resp.json()
            logger.debug(f"Downstream non-stream response data: {json.dumps(data)}")
            if STRIP_THINKING:
                # Strip empty think chains from each choice
                for choice in data.get("choices", []):
                    if "message" in choice and isinstance(choice["message"].get("content"), str):
                        choice["message"]["content"] = strip_think_chain_from_text(choice["message"]["content"])
                    elif isinstance(choice.get("text"), str):
                        choice["text"] = strip_think_chain_from_text(choice["text"])
                logger.debug(f"Cleaned non-stream response data: {json.dumps(data)}")
            return JSONResponse(
                content=data,
                status_code=resp.status_code,
                headers=response_headers,
            )
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
                return StreamingResponse(
                    openai_streaming_response_generator(),
                    status_code=upstream.status_code,
                    headers=response_headers,
                    media_type="text/event-stream"
                )


async def proxy_ollama_request(path: str, request: Request) -> StreamingResponse:
    # Read and mutate JSON body
    ollama_payload = await request.json()
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
    async with httpx.AsyncClient(timeout=None) as client:
        if not openai_payload.get("stream"):
            resp = await client.post(url, json=openai_payload, headers=headers)
            response_headers = {k: v for k, v in resp.headers.items() if k.lower() not in ["content-length", "transfer-encoding"]}
            openai_data = resp.json()
            logger.debug(f"Downstream non-stream OpenAI response data: {json.dumps(openai_data)}")

            ollama_response_content = ""
            if openai_data.get("choices"):
                if "message" in openai_data["choices"][0] and isinstance(openai_data["choices"][0]["message"].get("content"), str):
                    ollama_response_content = openai_data["choices"][0]["message"]["content"]
                elif isinstance(openai_data["choices"][0].get("text"), str):
                    ollama_response_content = openai_data["choices"][0]["text"]

            if STRIP_THINKING:
                ollama_response_content = strip_think_chain_from_text(ollama_response_content)

            ollama_response = {
                "model": ollama_payload["model"],
                "created_at": openai_data.get("created", ""),
                "response": ollama_response_content,
                "done": True,
                "done_reason": "stop", # Defaulting to stop for now
                # Add other fields as needed, e.g., context, total_duration, etc.
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
                                        # Send a final done message in Ollama format
                                        final_ollama_chunk = {
                                            "model": ollama_payload["model"],
                                            "created_at": datetime.now().isoformat(), # Use current time for final chunk
                                            "response": "",
                                            "done": True,
                                            "done_reason": "stop", # Assuming stop for now
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

                                        # Ollama streaming response format
                                        ollama_chunk = {
                                            "model": ollama_payload["model"],
                                            "created_at": data.get("created", ""), # OpenAI streaming response has 'created' field
                                            "response": content,
                                            "done": False,
                                        }
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
                    media_type="application/x-ndjson" # Newline delimited JSON
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
    async with httpx.AsyncClient() as client:
        upstream = await client.get(f"{UPSTREAM_URL}/v1/models", headers=headers)
        data = upstream.json()
    # (Optional) rewrite IDs in data.get("data", []) here
    return JSONResponse(content=data, status_code=upstream.status_code)


@app.post("/api/generate")
async def ollama_generate(request: Request):
    return await proxy_ollama_request("/api/generate", request)

@app.post("/api/chat")
async def ollama_chat(request: Request):
    return await proxy_ollama_request("/api/chat", request)

@app.get("/api/tags")
async def ollama_list_models(request: Request):
    headers = {k: v for k, v in request.headers.items() if k.lower() in ["authorization"]}
    async with httpx.AsyncClient() as client:
        upstream = await client.get(f"{OLLAMA_UPSTREAM_URL}/api/tags", headers=headers)
        data = upstream.json()
    return JSONResponse(content=data, status_code=upstream.status_code)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=LISTEN_HOST, port=LISTEN_PORT)
