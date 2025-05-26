import os
import json
import logging
import re

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

# Configuration via environment
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://localhost:8000")
LISTEN_HOST = os.getenv("LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "1234"))
RAW_MODEL_MAP = os.getenv("MODEL_MAP", "{}")

# Load model mapping (simple exact or regex patterns)
try:
    MODEL_MAP = json.loads(RAW_MODEL_MAP)
except json.JSONDecodeError as e:
    logger.error(f"Failed to parse MODEL_MAP: {e}")
    MODEL_MAP = {}


def rewrite_model(model: str) -> str:
    # Exact match
    if model in MODEL_MAP:
        return MODEL_MAP[model]
    # Regex mappings: keys starting and ending with /
    for pattern, target in MODEL_MAP.items():
        if pattern.startswith("/") and pattern.endswith("/"):
            if re.match(pattern.strip("/"), model):
                return target
    return model


async def proxy_request(path: str, request: Request) -> StreamingResponse:
    # Read and mutate JSON body
    payload = await request.json()
    if "model" in payload:
        original = payload["model"]
        new_model = rewrite_model(original)
        if new_model != original:
            logger.info(f"Rewriting model '{original}' -> '{new_model}'")
        payload["model"] = new_model

    # Forward headers (keep Authorization)
    headers = {k: v for k, v in request.headers.items() if k.lower() in ["authorization", "openai-organization"]}

    url = f"{UPSTREAM_URL}{path}"
    async with httpx.AsyncClient(timeout=None) as client:
        upstream = await client.post(url, json=payload, headers=headers, stream=True)
        return StreamingResponse(
            upstream.aiter_bytes(),
            status_code=upstream.status_code,
            headers=upstream.headers,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=LISTEN_HOST, port=LISTEN_PORT)
