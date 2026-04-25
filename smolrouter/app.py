import os
import json
import logging
import re
import time
import hashlib
import inspect
import yaml
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional, Tuple
from datetime import datetime, timezone
from urllib.parse import urlparse

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx
from contextlib import asynccontextmanager

# Import database functionality
from smolrouter.database import (
    RequestLog,
    get_recent_logs,
    get_log_stats,
    get_inflight_requests,
    estimate_tokens_from_request,
    extract_tokens_from_openai_response,
    estimate_token_count,
)
from smolrouter.dashboard_filters import DashboardFilterError, filter_request_logs, parse_dashboard_filter_query
from smolrouter.storage import init_blob_storage
from smolrouter.auth import create_auth_middleware, setup_rate_limiting, verify_request_auth
from smolrouter.security import init_webui_security, get_webui_security
from smolrouter.container import initialize_container
from smolrouter.config_paths import normalize_provider_file_references, resolve_routes_config_path

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-rerouter")


async def _initialize_lua_scripting() -> None:
    from smolrouter.database import RedisApiKeyQuota

    await RedisApiKeyQuota.initialize_lua_script()


async def _initialize_request_logging_system() -> None:
    global ENABLE_LOGGING

    if not ENABLE_LOGGING:
        return

    from smolrouter.database import init_database, start_background_cleanup

    try:
        await init_database()
        start_background_cleanup()
    except Exception as e:
        logger.error(f"Failed to initialize logging database: {e}")
        logger.warning("Request logging will be disabled")
        ENABLE_LOGGING = False


async def _stop_proxy_health_monitor(provider: Any) -> None:
    stop_monitor = getattr(provider, "stop_proxy_health_monitor", None)
    if not callable(stop_monitor):
        return

    stop_result = stop_monitor()
    if inspect.isawaitable(stop_result):
        await stop_result


async def _shutdown_proxy_health_monitors(active_container: Any) -> None:
    if not active_container:
        return

    try:
        for provider in active_container.get_providers():
            await _stop_proxy_health_monitor(provider)
    except Exception as e:
        logger.warning(f"Failed to stop proxy health monitor cleanly: {e}")


def _stop_logging_cleanup_if_enabled() -> None:
    if not ENABLE_LOGGING:
        return

    from smolrouter.database import stop_background_cleanup

    stop_background_cleanup()


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """FastAPI lifespan handler to replace deprecated on_event startup/shutdown."""
    global ENABLE_LOGGING

    # CRITICAL: Initialize Lua script FIRST - server won't start if this fails
    logger.info("🔧 Initializing critical systems (lifespan)...")
    try:
        await _initialize_lua_scripting()
    except Exception as e:
        logger.critical(f"❌ FATAL: Cannot start server - Lua script initialization failed: {e}")
        raise

    await _initialize_request_logging_system()

    # Initialize new architecture container
    await init_new_architecture()
    logger.info("✅ All critical systems initialized (lifespan)")

    # Yield to run the app
    try:
        yield
    finally:
        await _shutdown_proxy_health_monitors(container)
        _stop_logging_cleanup_if_enabled()


app = FastAPI(
    title="OpenAI Model Rerouter",
    description="Allows software with hard-coded model IDs to use whatever you desire",
    lifespan=app_lifespan,
)


@app.middleware("http")
async def disable_cache_for_html_and_json(request: Request, call_next):
    response = await call_next(request)

    content_type = response.headers.get("content-type", "")
    if request.method == "GET" and (
        content_type.startswith("text/html") or content_type.startswith("application/json")
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response

# Setup rate limiting
setup_rate_limiting(app)

# Add JWT authentication middleware if enabled
jwt_secret = os.getenv("JWT_SECRET")
if jwt_secret:
    app.add_middleware(create_auth_middleware())
    logger.info("JWT authentication middleware enabled")

# Templates for web UI
script_dir = os.path.dirname(os.path.abspath(__file__))
# Prefer package-internal templates; fallback to top-level (included via MANIFEST)
pkg_templates_dir = os.path.join(script_dir, "templates")
fallback_templates_dir = os.path.join(script_dir, "..", "templates")
templates_dir = pkg_templates_dir if os.path.isdir(pkg_templates_dir) else fallback_templates_dir
templates = Jinja2Templates(directory=templates_dir)

# Configuration via environment variables
DEFAULT_UPSTREAM = os.getenv("DEFAULT_UPSTREAM", "http://localhost:8000")
LISTEN_HOST = os.getenv("LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "1234"))
RAW_MODEL_MAP = os.getenv("MODEL_MAP", "{}")
ROUTES_CONFIG = str(resolve_routes_config_path(os.getenv("ROUTES_CONFIG")))

# Feature flags
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "false").lower() in ("1", "true", "yes")
STRIP_THINKING = os.getenv("STRIP_THINKING", "false").lower() in ("1", "true", "yes")
STRIP_JSON_MARKDOWN = os.getenv("STRIP_JSON_MARKDOWN", "false").lower() in ("1", "true", "yes")

INVALID_JSON_REQUEST_ERROR = "Invalid JSON in request body"
EXCESSIVE_WHITESPACE_PATTERN = r"\s{2,}"
ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "true").lower() in ("1", "true", "yes")

# Timeout configuration
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "3000.0"))
DASHBOARD_FILTER_SCAN_LIMIT = int(os.getenv("DASHBOARD_FILTER_SCAN_LIMIT", "1000"))


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
        routes_config_path = resolve_routes_config_path(os.getenv("ROUTES_CONFIG"))

        if not routes_config_path.exists():
            logger.info(f"No routes config file found at {ROUTES_CONFIG}, using default routing")
            return {"routes": []}

        with open(routes_config_path, "r") as f:
            if ROUTES_CONFIG.endswith(".json"):
                config = json.load(f)
            else:  # Assume YAML
                config = yaml.safe_load(f)

        config = normalize_provider_file_references(config or {}, routes_config_path)

        # Validate config structure
        if not isinstance(config, dict) or "routes" not in config:
            logger.error("Invalid routes config: missing 'routes' key")
            return {"routes": []}

        if not isinstance(config["routes"], list):
            logger.error("Invalid routes config: 'routes' must be a list")
            return {"routes": []}

        logger.info(f"Loaded {len(config['routes'])} routing rules from {ROUTES_CONFIG}")
        return config

    except Exception as e:
        logger.error(f"Failed to load routes config from {ROUTES_CONFIG}: {e}")
        return {"routes": []}


ROUTES_CONFIG_DATA = load_routes_config()


def _matches_model_pattern(model_pattern: Any, model: str) -> bool:
    """Check whether a model name matches an exact or slash-delimited regex pattern."""
    if model_pattern is None:
        return True

    if not isinstance(model_pattern, str):
        return model == model_pattern

    if model_pattern.startswith("/") and model_pattern.endswith("/"):
        return re.search(model_pattern[1:-1], model) is not None

    return model == model_pattern


def _route_matches_request(match_criteria: Dict[str, Any], source_host: str, model: str) -> bool:
    """Check whether the configured route criteria match the current request."""
    expected_host = match_criteria.get("source_host")
    if expected_host is not None and source_host != expected_host:
        return False

    return _matches_model_pattern(match_criteria.get("model"), model)


def find_route(source_host: str, model: str) -> Tuple[str, Optional[str]]:
    """Find the best matching route for a request.

    Args:
        source_host: Source IP address of the request
        model: Original model name from the request

    Returns:
        Tuple of (upstream_url, model_override) where model_override is None if no override
    """
    for route in ROUTES_CONFIG_DATA.get("routes", []):
        match_criteria = route.get("match", {})
        route_config = route.get("route", {})

        if not _route_matches_request(match_criteria, source_host, model):
            continue

        upstream = route_config.get("upstream")
        if not upstream:
            continue

        model_override = route_config.get("model")
        logger.debug(
            f"Route matched: {source_host}/{model} -> {upstream}"
            + (f" (model: {model_override})" if model_override else "")
        )
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
logger.info("SmolRouter starting...")
logger.info(f"DEFAULT_UPSTREAM: {DEFAULT_UPSTREAM}")
logger.info(f"MODEL_MAP: {MODEL_MAP}")
logger.info(f"ROUTES_CONFIG: {ROUTES_CONFIG} ({len(ROUTES_CONFIG_DATA.get('routes', []))} rules)")
logger.info(f"STRIP_THINKING: {STRIP_THINKING}")
logger.info(f"STRIP_JSON_MARKDOWN: {STRIP_JSON_MARKDOWN}")
logger.info(f"DISABLE_THINKING: {DISABLE_THINKING}")
logger.info(f"ENABLE_LOGGING: {ENABLE_LOGGING}")
logger.info(f"REQUEST_TIMEOUT: {REQUEST_TIMEOUT}s")
logger.info(f"Listening on {LISTEN_HOST}:{LISTEN_PORT}")

# Initialize blob storage if logging is enabled
if ENABLE_LOGGING:
    try:
        init_blob_storage()
    except Exception as e:
        logger.error(f"Failed to initialize blob storage: {e}")
        logger.warning("Request logging will be disabled")
        ENABLE_LOGGING = False

# Initialize WebUI security
init_webui_security()

# Initialize new architecture container
container = None


async def init_new_architecture():
    """Initialize the new architecture container"""
    global container
    try:
        container = await initialize_container()
        for provider in container.get_providers():
            start_monitor = getattr(provider, "start_proxy_health_monitor", None)
            if callable(start_monitor):
                try:
                    start_monitor()
                except Exception as provider_error:
                    logger.warning(
                        f"Failed to start proxy health monitor for {provider.get_provider_id()}: {provider_error}"
                    )
        logger.info("New SmolRouter architecture initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize new architecture: {e}")
        logger.warning("Falling back to legacy architecture only")


## Removed deprecated on_event handlers in favor of lifespan


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


def should_strip_thinking_for_provider(provider_type: str, provider_url: str) -> bool:
    """Determine if thinking chains should be stripped for this provider.

    Only strip for OpenAI-compatible self-hosted models that might emit thinking chains.
    Never strip for cloud providers like Google GenAI, Anthropic, OpenAI cloud.

    Args:
        provider_type: Type of provider (openai, google-genai, anthropic, ollama)
        provider_url: The provider's base URL

    Returns:
        True if thinking chains should be stripped, False otherwise
    """
    # Never strip from cloud providers - they don't use thinking tags
    cloud_providers = ["google-genai", "anthropic"]
    if provider_type in cloud_providers:
        return False

    # For OpenAI-compatible providers, only strip from self-hosted instances
    if provider_type == "openai":
        cloud_openai_urls = [
            "https://api.openai.com",
            "https://oai.azure.com",  # Azure OpenAI
            "https://openai.azure.com",
        ]
        return not any(provider_url.startswith(url) for url in cloud_openai_urls)

    # For Ollama, always strip (local models often use thinking tags)
    if provider_type == "ollama":
        return True

    return False


def strip_think_chain_from_text(text: str, provider_type: Optional[str] = None, provider_url: Optional[str] = None) -> str:
    """Remove thinking chain blocks from text using simple string operations.

    Supports multiple thinking tag formats:
    - <think>...</think> (Qwen, DeepSeek-R1)
    - [think]...[/think] (SmolLM, some local models)
    - <thinking>...</thinking> (general XML format)
    - <reasoning>...</reasoning> (alternative format)

    Only removes content, preserves all whitespace and formatting.

    Args:
        text: Input text that may contain thinking chains
        provider_type: Optional provider type for conditional processing
        provider_url: Optional provider URL for conditional processing

    Returns:
        Text with thinking chains removed, all formatting preserved
    """
    # If provider info is available, check if we should strip
    if provider_type and provider_url:
        if not should_strip_thinking_for_provider(provider_type, provider_url):
            return text

    result = text

    # Define thinking tag pairs to remove
    thinking_tags = [
        ("<think>", "</think>"),  # Qwen, DeepSeek-R1
        ("[think]", "[/think]"),  # SmolLM, bracket format
        ("<thinking>", "</thinking>"),  # XML style
        ("<reasoning>", "</reasoning>"),  # Alternative format
    ]

    # Remove each type of thinking block
    for start_tag, end_tag in thinking_tags:
        while True:
            start = result.find(start_tag)
            if start == -1:
                break
            end = result.find(end_tag, start)
            if end == -1:
                # Handle unclosed tags - remove from start_tag to end of text
                result = result[:start]
                break
            # Remove the entire block including tags
            result = result[:start] + result[end + len(end_tag) :]

    # Preserve original spacing and newlines, but avoid stray spaces before punctuation
    try:
        result = re.sub(r"\s+([,.!?])", r"\1", result)
    except Exception:
        pass
    return result


@dataclass
class RoutedRequestResult:
    data: Any
    status_code: int
    upstream_used: str
    metadata: Any = None
    is_streaming: bool = False


def _extract_model_from_provider_url(model_name: str) -> str:
    """Extract the actual model name from provider URL format"""
    # For gemini-* and claude-* models, use as-is
    return model_name


JSON_CODE_BLOCK_START_MARKER = "```json"
JSON_CODE_BLOCK_END_MARKER = "```"
JSON_INLINE_MARKER = "[json]"


def _normalize_json_markdown_content(json_content: Any) -> str:
    """Normalize JSON extracted from markdown fences into a compact single line."""
    if hasattr(json_content, "group"):
        json_content = json_content.group(1)

    if not json_content:
        return ""

    normalized_content = str(json_content).strip()
    if not normalized_content:
        return ""

    if "\n" not in normalized_content:
        return re.sub(r"\s+", " ", normalized_content).strip()

    return " ".join(line.strip() for line in normalized_content.splitlines() if line.strip())


def _replace_json_code_blocks(text: str) -> str:
    """Replace markdown JSON code blocks with their normalized JSON payload."""
    result = text
    while True:
        start_idx = result.find(JSON_CODE_BLOCK_START_MARKER)
        if start_idx == -1:
            return result

        content_start = start_idx + len(JSON_CODE_BLOCK_START_MARKER)
        end_idx = result.find(JSON_CODE_BLOCK_END_MARKER, content_start)
        if end_idx == -1:
            return result

        cleaned = _normalize_json_markdown_content(result[content_start:end_idx])
        result = result[:start_idx] + cleaned + result[end_idx + len(JSON_CODE_BLOCK_END_MARKER) :]


def _replace_json_marker_blocks(text: str) -> str:
    """Replace [json]... [json] segments with their normalized JSON payload."""
    parts = text.split(JSON_INLINE_MARKER)
    if len(parts) <= 2:
        return text

    return "".join(
        _normalize_json_markdown_content(part) if index % 2 else part for index, part in enumerate(parts)
    )


def strip_json_markdown_from_text(text: str) -> str:
    """Extract JSON from markdown code blocks, converting markdown-fenced JSON to pure JSON.

    This function finds JSON code blocks in multiple formats:

    Format 1 (backticks):
    ```json
    {
      "key": "value"
    }
    ```

    Format 2 (square brackets):
    [json] { "key": "value" } [json]

    And extracts just the JSON content, removing the markdown formatting.

    Args:
        text: Input text that may contain JSON markdown blocks

    Returns:
        Text with JSON markdown blocks replaced by pure JSON content
    """
    result = _replace_json_code_blocks(text)
    result = _replace_json_marker_blocks(result)
    return result.strip()


async def start_request_log(
    request: Request,
    service_type: str,
    upstream_url: str,
    original_model: Optional[str] = None,
    mapped_model: Optional[str] = None,
    auth_payload: Optional[Dict[str, Any]] = None,
    request_body: Optional[bytes] = None,
):
    """Create initial log entry for inflight tracking"""
    if not ENABLE_LOGGING:
        return None

    try:
        # Get client IP
        source_ip = request.client.host if request.client else "unknown"

        # Generate unique request ID for traceability
        request_id = str(uuid.uuid4())

        # Extract user agent
        user_agent = request.headers.get("user-agent", "unknown")

        # Extract authenticated user if available
        auth_user = None
        if auth_payload:
            auth_user = auth_payload.get("sub") or auth_payload.get("user") or auth_payload.get("username")

        # Calculate request size if body provided
        request_size = len(request_body) if request_body else 0

        # Compute request body hash for duplicate detection
        request_body_hash = None
        if request_body:
            try:
                request_body_hash = hashlib.sha256(request_body).hexdigest()
            except Exception:
                request_body_hash = None

        # Create initial log entry (inflight - no completed_at)
        # Use async Redis logging for hot path performance
        log_entry = await RequestLog.create(
            source_ip=source_ip,
            method=request.method,
            path=request.url.path,
            service_type=service_type,
            upstream_url=upstream_url,
            original_model=original_model,
            mapped_model=mapped_model,
            request_id=request_id,
            user_agent=user_agent,
            auth_user=auth_user,
            request_size=request_size,
            request_body_hash=request_body_hash,
        )

        # Store request body asynchronously in blob storage (avoid blocking hot path)
        if request_body:
            from smolrouter.storage import get_blob_storage
            from smolrouter.redis_backend import RedisRequestLog

            blob_storage = get_blob_storage()

            async def _store_request_body_async():
                try:
                    key = await asyncio.to_thread(blob_storage.store, request_body, content_type="application/json")
                    await RedisRequestLog.update_request_body_key(request_id=request_id, request_body_key=key)
                except Exception as e:
                    logger.error(f"Failed to store request body asynchronously: {e}")

            import asyncio

            asyncio.create_task(_store_request_body_async())

        # Log request start with traceability info (reduced verbosity)
        logger.debug(
            f"[{request_id}] Request started: {request.method} {request.url.path} from {source_ip} "
            f"(user: {auth_user or 'anonymous'}, model: {original_model})"
        )

        # Broadcast new request event (fire and forget)
        import asyncio

        try:
            logger.debug(f"Broadcasting new_request event for request {request_id}")
            asyncio.create_task(broadcast_request_event("new_request", log_entry))
        except Exception as e:
            logger.error(f"Failed to broadcast new request event: {e}")

        return log_entry
    except Exception as e:
        logger.error(f"Failed to start request log: {e}")
        return None


def _complete_lb_request(lb_instance, start_time: float, success: bool):
    """Helper to complete load balancer request tracking.

    Calls end_request on the load balancer instance to decrement active_requests.
    This is critical for proper load balancing - without it, active_requests
    keeps incrementing and never decrements, causing load imbalance.
    """
    import asyncio
    from smolrouter.load_balancer import model_load_balancer

    try:
        response_time = time.time() - start_time
        # Schedule the async end_request call
        try:
            asyncio.get_running_loop()
            asyncio.create_task(model_load_balancer.end_request(lb_instance, response_time, success))
        except RuntimeError:
            # No running loop - create one for this call
            asyncio.run(model_load_balancer.end_request(lb_instance, response_time, success))
        logger.debug(f"Load balancer: completed request for {lb_instance.model_id} (success={success})")
    except Exception as e:
        logger.error(f"Failed to complete load balancer request tracking: {e}")


def _complete_lb_request_from_metadata(metadata: Any, start_time: float, status_code: int) -> None:
    """Complete load balancer tracking when request metadata includes an LB instance."""
    lb_instance = getattr(metadata, "lb_instance", None) if metadata else None
    if lb_instance:
        _complete_lb_request(lb_instance, start_time, status_code < 400)


def _estimate_prompt_tokens_from_request_body(request_body: Optional[bytes]) -> int:
    """Estimate prompt tokens from the original request body."""
    if not request_body:
        return 0

    try:
        request_data = json.loads(request_body.decode("utf-8"))
        return estimate_tokens_from_request(request_data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return estimate_token_count(request_body.decode("utf-8", errors="ignore"))


def _extract_completion_text(response_json: Dict[str, Any]) -> str:
    """Extract response text from OpenAI-style choices for token estimation."""
    content_parts = []
    for choice in response_json.get("choices", []):
        message_content = choice.get("message", {}).get("content")
        if message_content:
            content_parts.append(message_content)
            continue

        text_content = choice.get("text")
        if text_content:
            content_parts.append(text_content)

    return "".join(content_parts)


def _estimate_completion_tokens_from_response_body(response_body: Optional[bytes]) -> int:
    """Estimate completion tokens from a serialized upstream response body."""
    if not response_body:
        return 0

    try:
        response_text = response_body.decode("utf-8")
    except UnicodeDecodeError:
        return 0

    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError:
        return estimate_token_count(response_text)

    if response_json.get("response"):
        return estimate_token_count(response_json["response"])

    if response_json.get("choices"):
        return estimate_token_count(_extract_completion_text(response_json))

    return 0


def _calculate_token_counts(
    response_data: Dict[str, Any],
    request_body: Optional[bytes],
    response_body: Optional[bytes],
) -> Tuple[int, int, int]:
    """Derive prompt/completion token counts from usage data or body estimation."""
    try:
        if response_data.get("usage"):
            return extract_tokens_from_openai_response(response_data)

        prompt_tokens = _estimate_prompt_tokens_from_request_body(request_body)
        completion_tokens = _estimate_completion_tokens_from_response_body(response_body)
        return prompt_tokens, completion_tokens, prompt_tokens + completion_tokens
    except Exception as e:
        logger.debug(f"Failed to calculate token counts: {e}")
        return 0, 0, 0


def _update_completed_log_entry(
    log_entry: Any,
    duration_ms: int,
    request_size: int,
    response_size: int,
    status_code: Any,
    error_message: Optional[str],
    request_body: Optional[bytes],
    response_body: Optional[bytes],
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> None:
    """Apply completion metrics to the log entry before persisting it."""
    log_entry.duration_ms = duration_ms
    log_entry.request_size = request_size
    log_entry.response_size = response_size
    log_entry.status_code = status_code
    if request_body:
        log_entry.set_request_body(request_body)
    if response_body:
        log_entry.set_response_body(response_body)
    log_entry.error_message = error_message
    log_entry.completed_at = datetime.now()
    log_entry.prompt_tokens = prompt_tokens
    log_entry.completion_tokens = completion_tokens
    log_entry.total_tokens = total_tokens


def _broadcast_request_completion(log_entry: Any) -> str:
    """Broadcast a request completion event in a fire-and-forget fashion."""
    request_id = getattr(log_entry, "request_id", "unknown")

    import asyncio

    try:
        logger.debug(f"Broadcasting request_completed event for request {request_id}")
        asyncio.create_task(broadcast_request_event("request_completed", log_entry))
    except Exception as e:
        logger.error(f"Failed to broadcast request completion event: {e}")

    return request_id


def complete_request_log(
    log_entry,
    start_time: float,
    response_data: dict,
    request_body: Optional[bytes] = None,
    response_body: Optional[bytes] = None,
    metadata=None,
):
    """Complete the log entry when request finishes.

    Args:
        log_entry: The log entry to complete
        start_time: Request start time
        response_data: Dict with status_code, error_message etc
        request_body: Request body bytes (optional)
        response_body: Response body bytes (optional)
        metadata: RequestMetadata with lb_instance for load balancer tracking (optional)
    """
    status_code = response_data.get("status_code", 200)

    if not ENABLE_LOGGING or not log_entry:
        # Still need to complete load balancer tracking even if logging disabled
        _complete_lb_request_from_metadata(metadata, start_time, status_code)
        return

    try:
        duration_ms = int((time.time() - start_time) * 1000)
        request_size = len(request_body) if request_body else 0
        response_size = len(response_body) if response_body else 0

        prompt_tokens, completion_tokens, total_tokens = _calculate_token_counts(
            response_data,
            request_body,
            response_body,
        )

        _update_completed_log_entry(
            log_entry,
            duration_ms,
            request_size,
            response_size,
            status_code,
            response_data.get("error_message"),
            request_body,
            response_body,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

        log_entry.save()

        request_id = _broadcast_request_completion(log_entry)
        logger.debug(
            f"[{request_id}] Request completed: {duration_ms}ms, {status_code} status, "
            f"{prompt_tokens} prompt tokens, {completion_tokens} completion tokens, upstream: {log_entry.upstream_url}"
        )

        error_message = response_data.get("error_message")
        if error_message:
            logger.warning(f"[{request_id}] Request had error: {error_message}")

        _complete_lb_request_from_metadata(metadata, start_time, status_code)
    except Exception as e:
        logger.error(f"Failed to complete request log: {e}")


def _get_request_source_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _get_request_auth_payload(request: Request) -> Optional[Dict[str, Any]]:
    try:
        return verify_request_auth(request)
    except Exception:
        return None


async def _parse_openai_request_payload(
    request: Request,
    start_time: float,
    auth_payload: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[bytes], Optional[JSONResponse]]:
    try:
        payload = await request.json()
        return payload, json.dumps(payload).encode("utf-8"), None
    except Exception as e:
        logger.error(f"Failed to parse request JSON: {e}")
        log_entry = await start_request_log(request, "openai", DEFAULT_UPSTREAM, None, None, auth_payload, None)
        complete_request_log(log_entry, start_time, {"status_code": 400, "error_message": INVALID_JSON_REQUEST_ERROR})
        return None, None, JSONResponse(content={"error": INVALID_JSON_REQUEST_ERROR}, status_code=400)


def _apply_openai_model_mapping(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    original_model: Optional[str] = payload.get("model")
    if original_model is None:
        return None, None

    mapped_model = rewrite_model(original_model)
    if mapped_model != original_model:
        logger.debug(f"Rewriting model '{original_model}' -> '{mapped_model}'")
    payload["model"] = mapped_model
    return original_model, mapped_model


def _update_log_entry_models(log_entry: Any, original_model: Optional[str], mapped_model: Optional[str]) -> None:
    if not log_entry:
        return

    try:
        log_entry.original_model = original_model
        log_entry.mapped_model = mapped_model
        log_entry.save()
    except Exception as e:
        logger.error(f"Failed to update log entry: {e}")


async def _start_openai_request_log(
    request: Request,
    original_model: Optional[str],
    mapped_model: Optional[str],
    auth_payload: Optional[Dict[str, Any]],
    request_body_bytes: Optional[bytes],
):
    log_entry = await start_request_log(
        request,
        "openai",
        "pending",
        original_model,
        mapped_model,
        auth_payload,
        request_body_bytes,
    )
    _update_log_entry_models(log_entry, original_model, mapped_model)
    return log_entry


def _apply_disable_thinking_request_marker(payload: Dict[str, Any]) -> None:
    if not DISABLE_THINKING:
        return

    logger.debug("Disabling thinking by appending '/no_think' marker to content")
    if "messages" in payload and isinstance(payload["messages"], list):
        payload["messages"].append({"role": "system", "content": "/no_think"})
    elif "prompt" in payload and isinstance(payload["prompt"], str):
        payload["prompt"] = payload["prompt"].rstrip() + " /no_think"


def _build_openai_forward_headers(request: Request) -> Dict[str, str]:
    return {k: v for k, v in request.headers.items() if k.lower() in ["authorization", "openai-organization"]}


def _is_legacy_proxy_mode() -> bool:
    return (
        os.getenv("USE_LEGACY_PROXY", "false").lower() in ("1", "true", "yes")
        or os.getenv("APP_ENV", "dev").lower() == "test"
    )


async def _get_active_container(legacy_proxy: bool):
    global container
    if legacy_proxy:
        return None

    if container is None:
        await init_new_architecture()

    return container


async def _execute_legacy_proxy_request(
    path: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
) -> RoutedRequestResult:
    url = f"{DEFAULT_UPSTREAM}{path}"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(url, json=payload, headers=headers)
        return RoutedRequestResult(resp.json(), resp.status_code, DEFAULT_UPSTREAM)


async def _execute_container_proxy_request(
    active_container: Any,
    source_ip: str,
    actual_model: str,
    payload: Dict[str, Any],
    path: str,
    headers: Dict[str, str],
    is_streaming: bool,
) -> RoutedRequestResult:
    if is_streaming:
        data, status_code, upstream_used, metadata = await active_container.route_streaming_request(
            source_ip, actual_model, payload, path, headers, REQUEST_TIMEOUT
        )
        return RoutedRequestResult(data, status_code, upstream_used, metadata, is_streaming=True)

    data, status_code, upstream_used, metadata = await active_container.route_request(
        source_ip, actual_model, payload, path, headers, REQUEST_TIMEOUT
    )
    return RoutedRequestResult(data, status_code, upstream_used, metadata)


async def _route_openai_request(
    source_ip: str,
    model_name: str,
    payload: Dict[str, Any],
    path: str,
    headers: Dict[str, str],
    is_streaming: bool,
    legacy_proxy: bool,
) -> RoutedRequestResult:
    if legacy_proxy:
        return await _execute_legacy_proxy_request(path, payload, headers)

    active_container = await _get_active_container(False)
    if active_container is None:
        raise RuntimeError("Provider architecture not available")

    actual_model = _extract_model_from_provider_url(model_name)
    payload["model"] = actual_model

    if not is_streaming:
        return await _execute_container_proxy_request(active_container, source_ip, actual_model, payload, path, headers, False)

    try:
        return await _execute_container_proxy_request(active_container, source_ip, actual_model, payload, path, headers, True)
    except Exception as e:
        logger.warning(f"Streaming not supported by provider architecture, falling back to non-streaming: {e}")
        try:
            return await _execute_container_proxy_request(active_container, source_ip, actual_model, payload, path, headers, False)
        except Exception as fallback_error:
            logger.error(f"Both streaming and non-streaming failed: {fallback_error}")
            raise fallback_error


def _update_log_entry_provider_metadata(log_entry: Any, upstream_used: str, metadata: Any) -> None:
    if not log_entry:
        return

    try:
        log_entry.upstream_url = upstream_used
        if metadata:
            log_entry.api_key_suffix = metadata.api_key_suffix
            log_entry.proxy_used = metadata.proxy_used
            log_entry.provider_id = metadata.provider_id
            log_entry.api_key_index = metadata.api_key_index
            log_entry.api_key_total = metadata.api_key_total
    except Exception as e:
        logger.error(f"Failed to update log entry with metadata: {e}")


def _serialize_json_bytes(data: Any) -> Optional[bytes]:
    return json.dumps(data).encode("utf-8") if data is not None else None


def _error_response(
    log_entry: Any,
    start_time: float,
    status_code: int,
    error_message: str,
    request_body_bytes: Optional[bytes],
    response_payload: Optional[Dict[str, Any]] = None,
    metadata: Any = None,
) -> JSONResponse:
    payload = response_payload or {"error": "provider_architecture_failed", "message": error_message}
    complete_request_log(
        log_entry,
        start_time,
        {"status_code": status_code, "error_message": error_message},
        request_body=request_body_bytes,
        response_body=_serialize_json_bytes(payload) if response_payload is not None else None,
        metadata=metadata,
    )
    return JSONResponse(content=payload, status_code=status_code)


def _normalize_response_text(text: str) -> str:
    if STRIP_THINKING:
        text = strip_think_chain_from_text(text)
        text = re.sub(EXCESSIVE_WHITESPACE_PATTERN, " ", text)
    if STRIP_JSON_MARKDOWN:
        text = strip_json_markdown_from_text(text)
    return text


def _normalize_openai_choice(choice: Dict[str, Any]) -> None:
    if "message" in choice and isinstance(choice["message"].get("content"), str):
        choice["message"]["content"] = _normalize_response_text(choice["message"]["content"])
    elif isinstance(choice.get("text"), str):
        choice["text"] = _normalize_response_text(choice["text"])


def _normalize_openai_response_content(data: Dict[str, Any]) -> None:
    if not (STRIP_THINKING or STRIP_JSON_MARKDOWN):
        return

    for choice in data.get("choices", []):
        _normalize_openai_choice(choice)


def _build_request_tracking_headers(log_entry: Any) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if log_entry and hasattr(log_entry, "request_id"):
        headers["x-smolrouter-uuid"] = log_entry.request_id
    return headers


async def proxy_request(path: str, request: Request):
    start_time = time.time()
    source_ip = _get_request_source_ip(request)
    auth_payload = _get_request_auth_payload(request)
    payload, request_body_bytes, error_response = await _parse_openai_request_payload(request, start_time, auth_payload)
    if error_response is not None:
        return error_response
    if payload is None:
        return JSONResponse(content={"error": INVALID_JSON_REQUEST_ERROR}, status_code=400)

    original_model, mapped_model = _apply_openai_model_mapping(payload)
    log_entry = await _start_openai_request_log(request, original_model, mapped_model, auth_payload, request_body_bytes)
    _apply_disable_thinking_request_marker(payload)

    headers = _build_openai_forward_headers(request)
    is_streaming = bool(payload.get("stream", False))
    legacy_proxy = _is_legacy_proxy_mode()
    model_name = mapped_model or original_model or "unknown"

    try:
        route_result = await _route_openai_request(source_ip, model_name, payload, path, headers, is_streaming, legacy_proxy)
    except Exception as e:
        logger.error(f"Provider architecture failed: {e}")
        return _error_response(log_entry, start_time, 503, str(e), request_body_bytes)

    _update_log_entry_provider_metadata(log_entry, route_result.upstream_used, route_result.metadata)

    if route_result.is_streaming:
        complete_request_log(
            log_entry,
            start_time,
            {"status_code": route_result.status_code},
            request_body=request_body_bytes,
            metadata=route_result.metadata,
        )
        return route_result.data

    if route_result.status_code >= 400:
        return _error_response(
            log_entry,
            start_time,
            route_result.status_code,
            str(route_result.data),
            request_body_bytes,
            response_payload=route_result.data,
            metadata=route_result.metadata,
        )

    logger.debug(f"Provider architecture response data: {json.dumps(route_result.data) if route_result.data else 'None'}")
    _normalize_openai_response_content(route_result.data)

    response_body_bytes = _serialize_json_bytes(route_result.data)
    complete_request_log(
        log_entry,
        start_time,
        {"status_code": route_result.status_code},
        request_body=request_body_bytes,
        response_body=response_body_bytes,
        metadata=route_result.metadata,
    )

    return JSONResponse(
        content=route_result.data,
        status_code=route_result.status_code,
        headers=_build_request_tracking_headers(log_entry),
    )


async def _parse_ollama_request_payload(
    request: Request,
    start_time: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[bytes], Optional[JSONResponse]]:
    try:
        payload = await request.json()
        return payload, json.dumps(payload).encode("utf-8"), None
    except Exception as e:
        logger.error(f"Failed to parse Ollama request JSON: {e}")
        log_entry = await start_request_log(request, "ollama", DEFAULT_UPSTREAM, None, None, None, None)
        complete_request_log(log_entry, start_time, {"status_code": 400, "error_message": INVALID_JSON_REQUEST_ERROR})
        return None, None, JSONResponse(content={"error": INVALID_JSON_REQUEST_ERROR}, status_code=400)


def _build_ollama_openai_payload(
    path: str,
    source_ip: str,
    ollama_payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, str, str]:
    is_chat_endpoint = "/chat" in path
    original_model = ollama_payload["model"]
    mapped_model = rewrite_model(original_model)
    upstream_url, route_model_override = find_route(source_ip, original_model)
    final_model = route_model_override or mapped_model

    if route_model_override and route_model_override != mapped_model:
        logger.debug(f"Route override: model '{mapped_model}' -> '{route_model_override}'")

    openai_payload = {
        "model": final_model,
        "stream": ollama_payload.get("stream", False),
        "messages": ollama_payload["messages"] if is_chat_endpoint else [{"role": "user", "content": ollama_payload["prompt"]}],
    }

    if DISABLE_THINKING:
        logger.debug("Disabling thinking by appending '/no_think' marker to content")
        openai_payload["messages"].append({"role": "system", "content": "/no_think"})

    return openai_payload, upstream_url, original_model, final_model


def _extract_openai_choice_content(choice: Dict[str, Any], streaming: bool = False) -> str:
    if streaming and "delta" in choice and isinstance(choice["delta"].get("content"), str):
        return choice["delta"]["content"]
    if "message" in choice and isinstance(choice["message"].get("content"), str):
        return choice["message"]["content"]
    if isinstance(choice.get("text"), str):
        return choice["text"]
    return ""


def _process_ollama_response_content(content: str, normalize_whitespace: bool, log_prefix: str = "") -> str:
    if STRIP_THINKING:
        content = strip_think_chain_from_text(content)
        if normalize_whitespace:
            content = re.sub(EXCESSIVE_WHITESPACE_PATTERN, " ", content)

    if STRIP_JSON_MARKDOWN:
        prefix = f"{log_prefix}: " if log_prefix else ""
        logger.debug(f"{prefix}Original content before JSON markdown stripping: {repr(content)}")
        content = strip_json_markdown_from_text(content)
        logger.debug(f"{prefix}Content after JSON markdown stripping: {repr(content)}")

    return content


def _build_ollama_response(ollama_model: str, openai_data: Dict[str, Any]) -> Dict[str, Any]:
    choice = openai_data.get("choices", [{}])[0] if openai_data.get("choices") else {}
    ollama_response_content = _process_ollama_response_content(
        _extract_openai_choice_content(choice),
        normalize_whitespace=True,
    )
    ollama_response = {
        "model": ollama_model,
        "created_at": openai_data.get("created", ""),
        "response": ollama_response_content,
        "done": True,
        "done_reason": "stop",
    }
    logger.debug(f"Transformed non-stream Ollama response: {json.dumps(ollama_response)}")
    logger.debug(f"Final Ollama response content: {repr(ollama_response.get('response', ''))}")
    return ollama_response


def _ollama_done_chunk(ollama_model: str) -> bytes:
    return (
        json.dumps(
            {
                "model": ollama_model,
                "created_at": datetime.now().isoformat(),
                "response": "",
                "done": True,
                "done_reason": "stop",
            }
        ).encode("utf-8")
        + b"\n"
    )


def _convert_openai_stream_message(ollama_model: str, json_data: str) -> Tuple[Optional[bytes], bool]:
    if json_data == "[DONE]":
        return _ollama_done_chunk(ollama_model), True

    try:
        data = json.loads(json_data)
    except json.JSONDecodeError:
        logger.warning(f"Could not decode JSON from SSE: {json_data!r}")
        return None, False

    choice = data.get("choices", [{}])[0] if data.get("choices") else {}
    content = _process_ollama_response_content(
        _extract_openai_choice_content(choice, streaming=True),
        normalize_whitespace=False,
        log_prefix="Streaming",
    )

    ollama_chunk = {
        "model": ollama_model,
        "created_at": data.get("created", ""),
        "response": content,
        "done": False,
    }
    if choice.get("finish_reason"):
        ollama_chunk["done_reason"] = choice["finish_reason"]

    return json.dumps(ollama_chunk).encode("utf-8") + b"\n", False


def _split_next_sse_message(buffer: str) -> Tuple[Optional[str], str]:
    eol = buffer.find("\n\n")
    if eol == -1:
        return None, buffer

    return buffer[:eol].strip(), buffer[eol + 4 :]


def _extract_sse_data_payload(message: str) -> Optional[str]:
    if not message.startswith("data:"):
        return None

    return message[len("data:") :].strip()


def _consume_ollama_sse_buffer(buffer: str, ollama_model: str) -> Tuple[list[bytes], str, bool]:
    emitted_chunks: list[bytes] = []

    while True:
        message, buffer = _split_next_sse_message(buffer)
        if message is None:
            return emitted_chunks, buffer, False

        json_data = _extract_sse_data_payload(message)
        if json_data is None:
            continue

        chunk_bytes, is_done = _convert_openai_stream_message(ollama_model, json_data)
        if chunk_bytes is not None:
            emitted_chunks.append(chunk_bytes)
        if is_done:
            return emitted_chunks, buffer, True


async def _ollama_streaming_response_generator(upstream: Any, ollama_model: str) -> AsyncIterator[bytes]:
    buffer = ""
    async for chunk in upstream.aiter_bytes():
        buffer += chunk.decode("utf-8")
        try:
            emitted_chunks, buffer, is_done = _consume_ollama_sse_buffer(buffer, ollama_model)
            for emitted_chunk in emitted_chunks:
                yield emitted_chunk
            if is_done:
                return
        except Exception as e:
            logger.error(f"Error processing stream: {e}")
            break


async def _proxy_ollama_non_streaming(
    client: Any,
    url: str,
    openai_payload: Dict[str, Any],
    headers: Dict[str, str],
    ollama_payload: Dict[str, Any],
    log_entry: Any,
    start_time: float,
    request_body_bytes: Optional[bytes],
) -> JSONResponse:
    resp = await client.post(url, json=openai_payload, headers=headers)
    response_headers = {k: v for k, v in resp.headers.items() if k.lower() not in ["content-length", "transfer-encoding"]}
    openai_data = resp.json()
    logger.debug(f"Downstream non-stream OpenAI response data: {json.dumps(openai_data)}")

    ollama_response = _build_ollama_response(ollama_payload["model"], openai_data)
    complete_request_log(
        log_entry,
        start_time,
        {"status_code": resp.status_code, "usage": openai_data.get("usage")},
        request_body=request_body_bytes,
        response_body=_serialize_json_bytes(ollama_response),
    )

    return JSONResponse(content=ollama_response, status_code=resp.status_code, headers=response_headers)


async def _proxy_ollama_streaming(
    client: Any,
    url: str,
    openai_payload: Dict[str, Any],
    headers: Dict[str, str],
    ollama_payload: Dict[str, Any],
    log_entry: Any,
    start_time: float,
    request_body_bytes: Optional[bytes],
) -> StreamingResponse:
    async with client.stream("POST", url, json=openai_payload, headers=headers) as upstream:
        response_headers = {k: v for k, v in upstream.headers.items() if k.lower() != "content-length"}
        complete_request_log(log_entry, start_time, {"status_code": upstream.status_code}, request_body=request_body_bytes)
        return StreamingResponse(
            _ollama_streaming_response_generator(upstream, ollama_payload["model"]),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type="application/x-ndjson",
        )


async def proxy_ollama_request(path: str, request: Request) -> JSONResponse | StreamingResponse:
    start_time = time.time()
    source_ip = _get_request_source_ip(request)
    ollama_payload, request_body_bytes, error_response = await _parse_ollama_request_payload(request, start_time)
    if error_response is not None:
        return error_response
    if ollama_payload is None:
        return JSONResponse(content={"error": INVALID_JSON_REQUEST_ERROR}, status_code=400)

    logger.debug(f"Received Ollama request to {path}: {ollama_payload}")
    openai_payload, upstream_url, original_model, final_model = _build_ollama_openai_payload(path, source_ip, ollama_payload)

    log_entry = await start_request_log(
        request,
        "ollama",
        upstream_url,
        original_model,
        final_model,
        None,
        request_body_bytes,
    )
    _update_log_entry_models(log_entry, original_model, final_model)

    headers = _build_openai_forward_headers(request)
    url = f"{upstream_url}/v1/chat/completions"
    logger.debug(f"Proxying Ollama request to OpenAI endpoint: {url}")

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            if openai_payload.get("stream"):
                return await _proxy_ollama_streaming(
                    client,
                    url,
                    openai_payload,
                    headers,
                    ollama_payload,
                    log_entry,
                    start_time,
                    request_body_bytes,
                )

            return await _proxy_ollama_non_streaming(
                client,
                url,
                openai_payload,
                headers,
                ollama_payload,
                log_entry,
                start_time,
                request_body_bytes,
            )
    except httpx.ConnectError as e:
        logger.error(f"Connection error to upstream {url}: {e}")
        return _error_response(
            log_entry,
            start_time,
            502,
            str(e),
            request_body_bytes,
            response_payload={
                "error": "upstream_connection_failed",
                "message": f"Could not connect to upstream server at {upstream_url}",
                "details": str(e),
            },
        )
    except httpx.TimeoutException as e:
        logger.error(f"Timeout error to upstream {url}: {e}")
        return _error_response(
            log_entry,
            start_time,
            504,
            str(e),
            request_body_bytes,
            response_payload={
                "error": "upstream_timeout",
                "message": f"Upstream server at {upstream_url} did not respond in time",
                "details": str(e),
            },
        )
    except Exception as e:
        logger.error(f"Unexpected error proxying Ollama request to {url}: {e}")
        return _error_response(
            log_entry,
            start_time,
            500,
            str(e),
            request_body_bytes,
            response_payload={
                "error": "proxy_error",
                "message": "An unexpected error occurred while proxying the request",
                "details": str(e),
            },
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await proxy_request("/v1/chat/completions", request)


@app.post("/v1/completions")
async def completions(request: Request):
    return await proxy_request("/v1/completions", request)


@app.post("/v1/responses")
async def responses(request: Request):
    return await proxy_request("/v1/responses", request)


@app.get("/v1/models")
async def list_models(request: Request):
    """List available models with aggregation from multiple providers"""
    global container

    # Initialize container if not already done
    if container is None:
        await init_new_architecture()

    # Try new architecture first
    if container is not None:
        try:
            # Get client context
            source_ip = request.client.host if request.client else "unknown"
            auth_payload = None
            try:
                auth_payload = verify_request_auth(request)
            except Exception:
                pass  # Continue without auth

            client_context = container.create_client_context(
                ip=source_ip, auth_payload=auth_payload, headers=dict(request.headers)
            )

            # Get mediator and fetch models
            mediator = await container.get_mediator()
            models = await mediator.get_available_models(client_context)

            # Convert to OpenAI format
            openai_models = []
            for model in models:
                openai_models.append(
                    {
                        "id": model.display_name,  # Use display name like "llama3-70b [fast-kitten]"
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": model.provider_id,
                        "permission": [],
                        "root": model.name,
                        "parent": None,
                    }
                )

            response_data = {"object": "list", "data": openai_models}

            logger.debug(f"Served {len(openai_models)} aggregated models to {source_ip}")
            return JSONResponse(content=response_data, status_code=200)

        except Exception as e:
            logger.error(f"Error in new architecture model listing: {e}")
            # Fall through to legacy behavior

    # Fallback to legacy single-upstream behavior
    logger.warning("Using legacy model listing (single upstream)")
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
                "message": f"Could not connect to upstream server at {DEFAULT_UPSTREAM}",
            },
            status_code=502,
        )
    except Exception as e:
        logger.error(f"Error listing models from {url}: {e}")
        return JSONResponse(
            content={"error": "models_error", "message": "Failed to retrieve models from upstream"}, status_code=500
        )


@app.post("/api/generate")
async def ollama_generate(request: Request):
    return await proxy_ollama_request("/api/generate", request)


@app.post("/api/chat")
async def ollama_chat(request: Request):
    return await proxy_ollama_request("/api/chat", request)


@app.get("/api/tags")
async def ollama_list_models(request: Request):
    """List available models in Ollama /api/tags format with aggregation"""
    global container

    # Initialize container if not already done
    if container is None:
        await init_new_architecture()

    # Try new architecture first
    if container is not None:
        try:
            # Get client context
            source_ip = request.client.host if request.client else "unknown"
            auth_payload = None
            try:
                auth_payload = verify_request_auth(request)
            except Exception:
                pass  # Continue without auth

            client_context = container.create_client_context(
                ip=source_ip, auth_payload=auth_payload, headers=dict(request.headers)
            )

            # Get mediator and fetch models
            mediator = await container.get_mediator()
            models = await mediator.get_available_models(client_context)

            # Convert to Ollama format
            ollama_models = []
            for model in models:
                # Use metadata if available, otherwise provide defaults
                size = model.metadata.get("size", 4000000000)  # Default 4GB
                modified_at = model.metadata.get("modified_at", "2024-01-01T00:00:00Z")
                digest = model.metadata.get("digest", "sha256:mock_digest")

                ollama_models.append(
                    {
                        "name": model.display_name,  # Use display name like "llama3-70b [fast-kitten]"
                        "modified_at": modified_at,
                        "size": size,
                        "digest": digest,
                    }
                )

            ollama_response = {"models": ollama_models}
            logger.debug(f"Served {len(ollama_models)} aggregated models in Ollama format to {source_ip}")

            return JSONResponse(content=ollama_response, status_code=200)

        except Exception as e:
            logger.error(f"Error in new architecture Ollama model listing: {e}")
            # Fall through to legacy behavior

    # Fallback to legacy behavior
    logger.warning("Using legacy Ollama model listing (single upstream)")
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
                ollama_models.append(
                    {
                        "name": model.get("id", "unknown"),
                        "modified_at": "2024-01-01T00:00:00Z",  # Mock timestamp
                        "size": 4000000000,  # Mock size (4GB)
                        "digest": "sha256:mock_digest",  # Mock digest
                    }
                )

            ollama_response = {"models": ollama_models}
            logger.debug(f"Converted {len(ollama_models)} models to Ollama format")

            return JSONResponse(content=ollama_response, status_code=upstream.status_code)

    except httpx.ConnectError as e:
        logger.error(f"Connection error to upstream {url}: {e}")
        return JSONResponse(
            content={
                "error": "upstream_connection_failed",
                "message": f"Could not connect to upstream server at {DEFAULT_UPSTREAM}",
            },
            status_code=502,
        )
    except Exception as e:
        logger.error(f"Error converting models to Ollama format: {e}")
        return JSONResponse(content={"error": "conversion_error"}, status_code=500)


# Web UI Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dynamic dashboard with real-time updates"""
    # Check WebUI access security
    get_webui_security().check_webui_access(request)

    try:
        return templates.TemplateResponse(request, "index.html", {"current_page": "dashboard"})
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        # Graceful fallback if templates are unavailable in installed package
        return HTMLResponse(
            content=(
                "<!DOCTYPE html><html><head><meta charset='utf-8'><title>SmolRouter</title></head>"
                "<body style='font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;'>"
                "<div style='max-width: 760px; margin: 40px auto; padding: 24px; border: 1px solid #eee; border-radius: 8px;'>"
                "<h1>SmolRouter</h1>"
                "<p>The Web UI templates are not available. The service is running.</p>"
                "<p>If you installed from PyPI, ensure templates are included or upgrade to a version that packages them.</p>"
                "</div></body></html>"
            ),
            status_code=200,
        )


@app.get("/performance", response_class=HTMLResponse)
async def performance_dashboard(request: Request):
    """Performance analytics dashboard with scatter plots"""
    # Check WebUI access security
    get_webui_security().check_webui_access(request)

    try:
        return templates.TemplateResponse(
            request, "performance.html", {"title": "Performance Analytics", "current_page": "performance"}
        )
    except Exception as e:
        logger.error(f"Error loading performance dashboard: {e}")
        return HTMLResponse(content="<h1>Error loading performance dashboard</h1>", status_code=500)


def _invalid_dashboard_filter_response(error: DashboardFilterError) -> JSONResponse:
    return JSONResponse(
        content={
            "error": "invalid_filter",
            "message": str(error),
            "invalid_terms": error.invalid_terms,
        },
        status_code=422,
    )


def _timestamp_now_for_log(timestamp: Optional[datetime]) -> datetime:
    if isinstance(timestamp, datetime) and timestamp.tzinfo is not None:
        return datetime.now(timestamp.tzinfo)
    return datetime.now()


def _serialize_request_log(log_entry) -> dict[str, Any]:
    timestamp = getattr(log_entry, "timestamp", None)
    status_code = getattr(log_entry, "status_code", None)
    duration_ms = getattr(log_entry, "duration_ms", None)

    if duration_ms is None and status_code in (None, "pending") and isinstance(timestamp, datetime):
        elapsed_seconds = (_timestamp_now_for_log(timestamp) - timestamp).total_seconds()
        duration_ms = max(int(elapsed_seconds * 1000), 0)

    completed_at = getattr(log_entry, "completed_at", None)
    provider_id = getattr(log_entry, "provider_id", None) or None
    upstream_url = getattr(log_entry, "upstream_url", None)
    normalized_status = None if status_code == "pending" else status_code

    return {
        "id": getattr(log_entry, "id", None),
        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else None,
        "source_ip": getattr(log_entry, "source_ip", None),
        "method": getattr(log_entry, "method", None),
        "path": getattr(log_entry, "path", None),
        "service_type": getattr(log_entry, "service_type", None),
        "provider_id": provider_id,
        "original_model": getattr(log_entry, "original_model", None),
        "mapped_model": getattr(log_entry, "mapped_model", None),
        "duration_ms": duration_ms,
        "request_size": getattr(log_entry, "request_size", 0) or 0,
        "response_size": getattr(log_entry, "response_size", 0) or 0,
        "status_code": normalized_status,
        "error_message": getattr(log_entry, "error_message", None),
        "completed_at": completed_at.isoformat() if isinstance(completed_at, datetime) else None,
        "is_inflight": completed_at is None,
        "upstream": upstream_url,
        "upstream_url": upstream_url,
        "is_duplicate": getattr(log_entry, "is_duplicate", False),
        "duplicate_count": getattr(log_entry, "duplicate_count", 0),
        "api_key_suffix": getattr(log_entry, "api_key_suffix", None),
        "proxy_used": getattr(log_entry, "proxy_used", None),
        "api_key_index": getattr(log_entry, "api_key_index", None),
        "api_key_total": getattr(log_entry, "api_key_total", None),
    }


async def _get_dashboard_logs(limit: int = 100, q: str | None = None, service_type: str | None = None):
    parsed_query = parse_dashboard_filter_query(q)
    requires_scan = parsed_query.active or bool(service_type)
    scan_limit = max(limit, DASHBOARD_FILTER_SCAN_LIMIT) if requires_scan else limit
    logs = await get_recent_logs(limit=scan_limit)

    if service_type:
        service_name = service_type.casefold()
        logs = [log for log in logs if str(getattr(log, "service_type", "")).casefold() == service_name]

    if parsed_query.active:
        logs = filter_request_logs(logs, parsed_query)

    return logs[:limit], parsed_query


@app.get("/api/logs")
async def api_logs(limit: int = 100, service_type: str | None = None, q: str | None = None):
    """API endpoint for getting logs as JSON"""
    try:
        logs, _ = await _get_dashboard_logs(limit=limit, q=q, service_type=service_type)
        return [_serialize_request_log(log) for log in logs]
    except DashboardFilterError as e:
        return _invalid_dashboard_filter_response(e)
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return JSONResponse(content={"error": "Failed to get logs"}, status_code=500)


@app.get("/api/stats")
async def api_stats():
    """API endpoint for getting statistics"""
    try:
        return await get_log_stats()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return JSONResponse(content={"error": "Failed to get stats"}, status_code=500)


@app.get("/api/dashboard")
async def api_dashboard(limit: int = 100, q: str | None = None):
    """Combined API endpoint for dashboard data (logs + stats)"""
    try:
        logs, parsed_query = await _get_dashboard_logs(limit=limit, q=q)
        stats = await get_log_stats()
        formatted_logs = [_serialize_request_log(log) for log in logs]

        return {
            "logs": formatted_logs,
            "stats": stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filter": parsed_query.to_meta(matched_count=len(formatted_logs), limit=limit),
        }
    except DashboardFilterError as e:
        return _invalid_dashboard_filter_response(e)
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        return JSONResponse(content={"error": f"Failed to get dashboard data. {e}"}, status_code=500)


@app.get("/api/inflight")
async def api_inflight():
    """API endpoint for getting currently inflight requests"""
    try:
        inflight = await get_inflight_requests()
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
                "elapsed_ms": int((datetime.now(timezone.utc) - log.timestamp).total_seconds() * 1000),
            }
            for log in inflight
        ]
    except Exception as e:
        logger.error(f"Error getting inflight requests: {e}")
        return JSONResponse(content={"error": "Failed to get inflight requests"}, status_code=500)


@app.get("/api/load-balancer")
async def api_load_balancer():
    """API endpoint for load balancer statistics and configuration"""
    try:
        from smolrouter.load_balancer import model_load_balancer

        stats = model_load_balancer.get_stats()
        model_groups = model_load_balancer.get_model_groups()
        host_stats = model_load_balancer.get_host_stats()

        return {
            "enabled": True,
            "distribution_strategy": model_load_balancer.default_distribution_strategy.value,
            "stats": stats,
            "model_groups": model_groups,
            "host_stats": host_stats,
            "summary": {
                "total_model_groups": len(model_groups),
                "total_instances": sum(len(instances) for instances in model_groups.values()),
                "active_hosts": len(host_stats),
                "total_requests": stats.get("total_requests", 0),
                "success_rate": stats.get("success_rate", 0),
            },
        }
    except Exception as e:
        logger.error(f"Error getting load balancer stats: {e}")
        return JSONResponse(content={"error": "Failed to get load balancer stats"}, status_code=500)


@app.get("/api/performance")
async def api_performance(limit: int = 1000, hours: int = 24, model: str | None = None, service_type: str | None = None):
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
            RequestLog.duration_ms.is_null(False),  # Must have duration data
            RequestLog.timestamp >= datetime.now() - timedelta(hours=hours),
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
            data_points.append(
                {
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
                }
            )

        return {
            "data_points": data_points,
            "meta": {
                "total_points": len(data_points),
                "hours_back": hours,
                "filters": {"model": model, "service_type": service_type},
            },
        }

    except Exception as e:
        logger.error(f"Failed to get performance data: {e}")
        return JSONResponse(content={"error": "Failed to get performance data"}, status_code=500)


@app.get("/api/google-genai/stats")
async def api_google_genai_stats():
    """API endpoint for getting Google GenAI API key statistics"""
    global container

    if container is None:
        await init_new_architecture()

    if container is not None:
        try:
            providers = container.get_providers()
            google_providers = [p for p in providers if p.get_provider_type() == "google-genai"]

            if not google_providers:
                return {"error": "No Google GenAI providers configured"}

            all_stats = {}
            for provider in google_providers:
                provider_stats = await provider.get_api_key_stats()
                # Filter out _rate_limiter from api_keys for frontend compatibility
                api_keys_only = {k: v for k, v in provider_stats.items() if k != "_rate_limiter"}
                # Wrap in api_keys structure for frontend compatibility
                all_stats[provider.get_provider_id()] = {
                    "api_keys": api_keys_only,
                    "summary": {
                        "total_keys": len(api_keys_only),
                    },
                }

            return {
                "providers": all_stats,
                "summary": {
                    "total_providers": len(google_providers),
                    "total_keys": sum(
                        len([k for k in stats.get("api_keys", {}).keys() if k != "_rate_limiter"])
                        for stats in all_stats.values()
                    ),
                    "timezone": "US/Pacific (Google's reset time)",
                },
            }

        except Exception as e:
            logger.error(f"Error getting Google GenAI stats: {e}")
            return JSONResponse(
                content={"error": "Failed to get Google GenAI stats", "details": str(e)}, status_code=500
            )

    return {"error": "New architecture not available"}


@app.get("/api/anthropic/stats")
async def api_anthropic_stats():
    """API endpoint for getting Anthropic API statistics"""
    global container

    if container is None:
        await init_new_architecture()

    if container is not None:
        try:
            providers = container.get_providers()
            anthropic_providers = [p for p in providers if p.get_provider_type() == "anthropic"]

            if not anthropic_providers:
                return {"error": "No Anthropic providers configured"}

            all_stats = {}
            for provider in anthropic_providers:
                provider_stats = provider.get_api_key_stats()
                all_stats[provider.get_provider_id()] = provider_stats

            return {
                "providers": all_stats,
                "summary": {
                    "total_providers": len(anthropic_providers),
                    "passthrough_mode": True,  # Anthropic uses passthrough keys
                    "note": "Keys are passed through from client requests, with fallback to configured keys",
                },
            }

        except Exception as e:
            logger.error(f"Error getting Anthropic stats: {e}")
            return JSONResponse(content={"error": "Failed to get Anthropic stats", "details": str(e)}, status_code=500)

    return {"error": "New architecture not available"}


@app.get("/api/upstreams")
async def api_upstreams():
    """API endpoint for getting upstream provider information"""
    global container

    # Initialize container if not already done
    if container is None:
        await init_new_architecture()

    # Try new architecture first
    if container is not None:
        try:
            mediator = await container.get_mediator()

            # Get provider health (backward compatibility)
            provider_health = await mediator.get_provider_health()

            # Get detailed provider health information
            detailed_health = await mediator.get_provider_health_detailed()

            # Get architecture stats
            stats = await mediator.get_mediator_stats()

            # Get providers info
            providers = container.get_providers()

            upstreams = []
            for provider in providers:
                provider_id = provider.get_provider_id()
                is_healthy = provider_health.get(provider_id, False)
                health_detail = detailed_health.get(provider_id, {})

                # Get models from this provider
                try:
                    client_context = container.create_client_context(ip="127.0.0.1")
                    provider_models = await mediator.get_models_by_provider(provider_id, client_context)
                    model_count = len(provider_models)
                    models = [
                        {
                            "id": model.id,
                            "name": model.name,
                            "display_name": model.display_name,
                            "aliases": model.aliases,
                        }
                        for model in provider_models[:10]  # Limit to first 10 for UI
                    ]
                except Exception as e:
                    logger.debug(f"Failed to get models for provider {provider_id}: {e}")
                    model_count = 0
                    models = []

                upstream_info = {
                    "id": provider_id,
                    "name": provider_id,
                    "type": provider.get_provider_type(),
                    "endpoint": provider.get_endpoint(),
                    "healthy": is_healthy,
                    "status": health_detail.get("status", "unknown"),
                    "last_checked": health_detail.get("last_checked"),
                    "last_checked_ago": health_detail.get("last_checked_ago", "never"),
                    "last_healthy": health_detail.get("last_healthy"),
                    "model_count": model_count,
                    "models": models,
                    "priority": getattr(provider.config, "priority", 999),
                    "enabled": getattr(provider.config, "enabled", True),
                }
                upstreams.append(upstream_info)

            # Sort by priority
            upstreams.sort(key=lambda x: x["priority"])

            # Get cache stats
            cache_stats = stats.get("aggregation", {}).get("cache_stats", {})

            # Get load balancer stats
            load_balancer_data = {}
            try:
                from smolrouter.load_balancer import model_load_balancer

                lb_stats = model_load_balancer.get_stats()
                lb_model_groups = model_load_balancer.get_model_groups()
                lb_host_stats = model_load_balancer.get_host_stats()

                load_balancer_data = {
                    "enabled": True,
                    "distribution_strategy": model_load_balancer.default_distribution_strategy.value,
                    "total_requests": lb_stats.get("total_requests", 0),
                    "success_rate": lb_stats.get("success_rate", 0),
                    "model_groups": lb_model_groups,
                    "host_stats": lb_host_stats,
                    "instances_by_model": lb_stats.get("instances", {}),
                    "summary": {
                        "total_model_groups": len(lb_model_groups),
                        "total_instances": sum(len(instances) for instances in lb_model_groups.values()),
                        "active_hosts": len(lb_host_stats),
                    },
                }
            except Exception as e:
                logger.error(f"Failed to get load balancer stats: {e}")
                load_balancer_data = {"enabled": False, "error": str(e)}

            return {
                "upstreams": upstreams,
                "summary": {
                    "total_providers": len(upstreams),
                    "healthy_providers": sum(1 for u in upstreams if u["healthy"]),
                    "total_models": sum(u["model_count"] for u in upstreams),
                    "cache_enabled": len(cache_stats) > 0,
                    "cache_entries": cache_stats.get("total_entries", 0),
                },
                "cache_stats": cache_stats,
                "load_balancer": load_balancer_data,
            }

        except Exception as e:
            logger.error(f"Error getting upstream data: {e}")
            return JSONResponse(content={"error": "Failed to get upstream data", "details": str(e)}, status_code=500)

    # Fallback to legacy data
    return {
        "upstreams": [
            {
                "id": "default",
                "name": "Default Upstream",
                "type": "openai",
                "endpoint": DEFAULT_UPSTREAM,
                "healthy": True,  # Assume healthy
                "model_count": "Unknown",
                "models": [],
                "priority": 0,
                "enabled": True,
            }
        ],
        "summary": {
            "total_providers": 1,
            "healthy_providers": 1,
            "total_models": "Unknown",
            "cache_enabled": False,
            "cache_entries": 0,
        },
        "cache_stats": {},
    }


@app.get("/upstreams")
async def upstreams_data():
    """Synchronous API endpoint for upstream providers data"""
    # Return the same data as the API endpoint
    return await api_upstreams()


@app.get("/upstreams-ui", response_class=HTMLResponse)
async def upstreams_dashboard(request: Request):
    """Redirect to unified providers page"""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/providers", status_code=301)


@app.get("/providers", response_class=HTMLResponse)
async def providers_dashboard(request: Request):
    """Unified providers management page"""
    # Check WebUI access security
    get_webui_security().check_webui_access(request)
    try:
        return templates.TemplateResponse(
            request, "providers.html", {"title": "Provider Management", "current_page": "providers"}
        )
    except Exception as e:
        logger.error(f"Error loading providers dashboard: {e}")
        return HTMLResponse(content="<h1>Error loading providers dashboard</h1>", status_code=500)


@app.get("/google-genai", response_class=HTMLResponse)
async def google_genai_dashboard(request: Request):
    """Redirect to unified providers page"""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/providers", status_code=301)


def _proxy_config_to_url(proxy_config: Any) -> Optional[str]:
    if proxy_config is None:
        return None
    if hasattr(proxy_config, "to_httpx_proxy"):
        return proxy_config.to_httpx_proxy()
    return str(proxy_config)


def _mask_proxy_url(proxy_url: Optional[str]) -> Optional[str]:
    if not proxy_url:
        return None

    parsed = urlparse(proxy_url)
    if not parsed.scheme or not parsed.hostname:
        return proxy_url

    netloc = parsed.hostname
    if parsed.port:
        netloc += f":{parsed.port}"
    return parsed._replace(netloc=netloc).geturl()


def _build_proxy_entry(
    label: str,
    *,
    kind: str,
    url: str,
    status: str = "unknown",
    model_name: Optional[str] = None,
    pool_index: Optional[int] = None,
    selected_next: bool = False,
    health: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    health = health or {}
    return {
        "label": label,
        "kind": kind,
        "url": url,
        "status": status,
        "model_name": model_name,
        "pool_index": pool_index,
        "selected_next": selected_next,
        "last_checked_at": health.get("last_checked_at"),
        "last_success_at": health.get("last_success_at"),
        "last_failure_at": health.get("last_failure_at"),
        "last_error": health.get("last_error"),
        "failure_count": health.get("failure_count", 0),
        "success_count": health.get("success_count", 0),
        "cooldown_remaining_seconds": health.get("cooldown_remaining_seconds", 0),
    }


def _summarize_proxy_entries(entries: list[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "entry_count": len(entries),
        "proxy_count": sum(1 for entry in entries if entry["status"] != "direct"),
        "pool_entry_count": sum(1 for entry in entries if entry["kind"] in {"pool", "direct"}),
        "direct_entry_count": sum(1 for entry in entries if entry["status"] == "direct"),
        "healthy_count": sum(1 for entry in entries if entry["status"] == "healthy"),
        "unhealthy_count": sum(1 for entry in entries if entry["status"] == "unhealthy"),
        "unknown_count": sum(1 for entry in entries if entry["status"] == "unknown"),
    }


def _build_generic_default_proxy_entry(provider_config: Any) -> Optional[Dict[str, Any]]:
    default_proxy_url = _proxy_config_to_url(getattr(provider_config, "proxy_config", None))
    if not default_proxy_url:
        return None

    return _build_proxy_entry(
        "Default",
        kind="default",
        url=_mask_proxy_url(default_proxy_url) or default_proxy_url,
    )


def _build_generic_model_override_entries(provider_config: Any) -> list[Dict[str, Any]]:
    entries = []
    per_model_proxy = getattr(provider_config, "per_model_proxy", {}) or {}
    for model_name, proxy_config in sorted(per_model_proxy.items()):
        proxy_url = _proxy_config_to_url(proxy_config)
        if not proxy_url:
            continue
        entries.append(
            _build_proxy_entry(
                model_name,
                kind="override",
                url=_mask_proxy_url(proxy_url) or proxy_url,
                model_name=model_name,
            )
        )
    return entries


def _build_generic_pool_entries(provider: Any, provider_config: Any) -> list[Dict[str, Any]]:
    entries = []
    next_pool_index = getattr(provider, "_proxy_pool_index", None)
    proxy_pool = getattr(provider_config, "proxy_pool", None) or []

    for idx, proxy_config in enumerate(proxy_pool):
        if proxy_config is None:
            entries.append(
                _build_proxy_entry(
                    f"Pool #{idx + 1}",
                    kind="direct",
                    url="direct",
                    status="direct",
                    pool_index=idx + 1,
                    selected_next=next_pool_index == idx,
                )
            )
            continue

        proxy_url = _proxy_config_to_url(proxy_config)
        if not proxy_url:
            continue
        entries.append(
            _build_proxy_entry(
                f"Pool #{idx + 1}",
                kind="pool",
                url=_mask_proxy_url(proxy_url) or proxy_url,
                pool_index=idx + 1,
                selected_next=next_pool_index == idx,
            )
        )

    return entries


def _build_generic_provider_proxy_diagnostics(provider: Any) -> Dict[str, Any]:
    provider_config = getattr(provider, "config", None)
    if provider_config is None:
        return {
            "provider_id": provider.get_provider_id(),
            "provider_type": provider.get_provider_type(),
            "configured": False,
            "pool_enabled": False,
            "monitor_running": False,
            "health_check_interval_seconds": None,
            "failure_cooldown_seconds": None,
            "next_pool_index": None,
            "default_proxy": None,
            "model_overrides": [],
            "pool_entries": [],
            "summary": _summarize_proxy_entries([]),
        }

    default_proxy = _build_generic_default_proxy_entry(provider_config)
    model_overrides = _build_generic_model_override_entries(provider_config)
    next_pool_index = getattr(provider, "_proxy_pool_index", None)
    proxy_pool = getattr(provider_config, "proxy_pool", None) or []
    pool_entries = _build_generic_pool_entries(provider, provider_config)

    all_entries = [entry for entry in [default_proxy] if entry] + model_overrides + pool_entries
    pool_enabled = bool(getattr(provider_config, "proxy_pool_enabled", False) and proxy_pool)
    return {
        "provider_id": provider.get_provider_id(),
        "provider_type": provider.get_provider_type(),
        "configured": bool(default_proxy or model_overrides or pool_enabled),
        "pool_enabled": pool_enabled,
        "monitor_running": False,
        "health_check_interval_seconds": None,
        "failure_cooldown_seconds": None,
        "next_pool_index": next_pool_index + 1 if next_pool_index is not None and proxy_pool else None,
        "default_proxy": default_proxy,
        "model_overrides": model_overrides,
        "pool_entries": pool_entries,
        "summary": _summarize_proxy_entries(all_entries),
    }


def _build_proxy_configuration_report(providers_list: list[Any]) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    proxy_providers = []

    for provider in providers_list:
        try:
            if hasattr(provider, "get_proxy_diagnostics") and callable(provider.get_proxy_diagnostics):
                diagnostics = provider.get_proxy_diagnostics()
            else:
                diagnostics = _build_generic_provider_proxy_diagnostics(provider)

            if diagnostics.get("configured"):
                proxy_providers.append(diagnostics)
        except Exception as e:
            logger.warning(f"Could not analyze proxy configuration for {provider.get_provider_id()}: {e}")

    summary = {
        "configured": bool(proxy_providers),
        "configured_provider_count": len(proxy_providers),
        "pool_provider_count": sum(1 for provider in proxy_providers if provider.get("pool_enabled")),
        "proxy_count": sum(provider["summary"].get("proxy_count", 0) for provider in proxy_providers),
        "healthy_count": sum(provider["summary"].get("healthy_count", 0) for provider in proxy_providers),
        "unhealthy_count": sum(provider["summary"].get("unhealthy_count", 0) for provider in proxy_providers),
        "unknown_count": sum(provider["summary"].get("unknown_count", 0) for provider in proxy_providers),
        "direct_entry_count": sum(provider["summary"].get("direct_entry_count", 0) for provider in proxy_providers),
    }

    return proxy_providers, summary


@app.get("/system", response_class=HTMLResponse)
async def system_dashboard(request: Request):
    """Web UI for viewing system configuration and tuneables"""
    # Check WebUI access security
    get_webui_security().check_webui_access(request)

    try:
        import platform
        import sys
        import os
        from smolrouter.storage import get_blob_storage

        # Gather all system settings and configuration
        blob_storage = get_blob_storage()
        settings = {
            "request_timeout": REQUEST_TIMEOUT,
            "default_upstream": DEFAULT_UPSTREAM,
            "strip_thinking": STRIP_THINKING,
            "strip_json_markdown": STRIP_JSON_MARKDOWN,
            "disable_thinking": DISABLE_THINKING,
            "enable_logging": ENABLE_LOGGING,
            "blob_storage_type": type(blob_storage).__name__,
            "blob_storage_path": str(getattr(blob_storage, "base_path", "")),
        }

        # Initialize container if not already done
        global container
        if container is None:
            await init_new_architecture()

        # Provider configurations
        providers = []
        if container:
            try:
                container_providers = container.get_providers()
                for provider in container_providers:
                    providers.append(
                        {
                            "name": provider.get_provider_id(),
                            "type": provider.get_provider_type(),
                            "url": provider.get_endpoint(),
                            "timeout": getattr(provider.config, "timeout", "N/A"),
                            "enabled": getattr(provider.config, "enabled", True),
                            "priority": getattr(provider.config, "priority", 0),
                            "has_api_key": bool(
                                getattr(provider.config, "api_key", None) or getattr(provider.config, "api_keys", None)
                            ),
                        }
                    )
            except Exception as e:
                logger.warning(f"Could not load provider configurations: {e}")

        # Security settings
        jwt_secret = os.getenv("JWT_SECRET")
        security = {
            "jwt_enabled": bool(jwt_secret),
            "webui_policy": "AUTH_WHEN_PROXIED",  # This is the current policy
            "rate_limiting_enabled": True,  # Rate limiting is enabled by default
            "ip_restrictions_count": 0,  # Placeholder - would need to check actual rules
            "model_restrictions_count": 0,  # Placeholder
            "user_restrictions_count": 0,  # Placeholder
        }

        # Load balancing configuration and stats
        load_balancing = None
        try:
            from smolrouter.load_balancer import model_load_balancer

            lb_stats = model_load_balancer.get_stats()
            lb_model_groups = model_load_balancer.get_model_groups()
            lb_host_stats = model_load_balancer.get_host_stats()

            load_balancing = {
                "enabled": True,
                "distribution_strategy": model_load_balancer.default_distribution_strategy.value,
                "total_requests": lb_stats.get("total_requests", 0),
                "successful_requests": lb_stats.get("successful_requests", 0),
                "failed_requests": lb_stats.get("failed_requests", 0),
                "success_rate": f"{lb_stats.get('success_rate', 0) * 100:.1f}%",
                "model_groups": len(lb_model_groups),
                "total_instances": sum(len(instances) for instances in lb_model_groups.values()),
                "active_hosts": len(lb_host_stats),
                "hosts": lb_host_stats,
                "instances_by_model": lb_stats.get("instances", {}),
            }
            logger.debug(f"Load balancer stats: {load_balancing}")
        except Exception as e:
            logger.error(f"Failed to load load balancer stats: {e}")
            load_balancing = {
                "enabled": False,
                "error": str(e),
                "distribution_strategy": "unknown",
                "total_requests": 0,
                "model_groups": 0,
                "total_instances": 0,
                "active_hosts": 0,
                "success_rate": "0%",
            }

        # Proxy configuration analysis
        proxy_providers = []
        proxy_summary = {
            "configured": False,
            "configured_provider_count": 0,
            "pool_provider_count": 0,
            "proxy_count": 0,
            "healthy_count": 0,
            "unhealthy_count": 0,
            "unknown_count": 0,
            "direct_entry_count": 0,
        }
        if container:
            try:
                providers_list = container.get_providers()
                proxy_providers, proxy_summary = _build_proxy_configuration_report(providers_list)

            except Exception as e:
                logger.warning(f"Could not analyze proxy configurations: {e}")

        # Routing configuration
        routing = {
            "strategy_type": "smart" if container else "unknown",
            "aliases_count": 0,  # Would need to check actual aliases
            "routes_count": 0,  # Would need to check routes.yaml
            "health_check_interval": 60,  # Default health check interval
            "cache_ttl": 300,  # Default cache TTL
            "auto_refresh": True,
            "load_balancing": load_balancing,
            "proxy_providers": proxy_providers,
            "proxy_summary": proxy_summary,
        }

        # Environment information
        env_info = {
            "version": "SmolRouter v1.0.0",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.system(),
            "host": LISTEN_HOST,
            "port": str(LISTEN_PORT),
            "start_time": "Runtime Info",
            "uptime": "N/A",
        }

        return templates.TemplateResponse(
            request,
            "system.html",
            {
                "title": "System Configuration",
                "current_page": "system",
                "settings": settings,
                "providers": providers,
                "security": security,
                "routing": routing,
                "env_info": env_info,
            },
        )
    except Exception as e:
        logger.error(f"Error loading system dashboard: {e}")
        return HTMLResponse(content="<h1>Error loading system dashboard</h1>", status_code=500)


@app.get("/request/{request_id}", response_class=HTMLResponse)
async def request_detail(request_id: str, request: Request):
    """Detailed view of a specific request"""
    # Check WebUI access security
    get_webui_security().check_webui_access(request)

    try:
        # Convert to string in case it's a UUID object
        request_id_str = str(request_id)
        log_entry = await RequestLog.get_by_id(request_id_str)

        if log_entry is None:
            return HTMLResponse(
                content="<h1>Request Not Found</h1><p>The requested log entry does not exist.</p>", status_code=404
            )

        # Precompute elapsed_ms for inflight to avoid tz pitfalls in templates
        elapsed_ms = None
        if not getattr(log_entry, "completed_at", None):
            try:
                elapsed_ms = int((datetime.now() - log_entry.timestamp).total_seconds() * 1000)
            except Exception:
                elapsed_ms = None

        # Compute duplicates list (other requests with same body hash)
        duplicates = []
        try:
            body_hash = getattr(log_entry, "request_body_hash", None)
            if body_hash:
                from smolrouter.redis_backend import RedisRequestLog

                ids = await RedisRequestLog.get_recent_duplicate_request_ids(body_hash, limit=10)
                other_ids = [rid for rid in ids if rid != log_entry.id]
                for rid in other_ids[:10]:
                    dupe = await RequestLog.get_by_id(rid)
                    if dupe:
                        duplicates.append(
                            {
                                "id": dupe.id,
                                "timestamp": dupe.timestamp.isoformat(),
                                "status_code": dupe.status_code,
                                "source_ip": dupe.source_ip,
                            }
                        )
        except Exception:
            duplicates = []

        return templates.TemplateResponse(
            request,
            "request_detail.html",
            {
                "log": log_entry,
                "elapsed_ms": elapsed_ms,
                "request_body_str": getattr(log_entry, "request_body", b"").decode("utf-8")
                if getattr(log_entry, "request_body", None)
                else None,
                "response_body_str": getattr(log_entry, "response_body", b"").decode("utf-8")
                if getattr(log_entry, "response_body", None)
                else None,
                "duplicates": duplicates,
            },
        )
    except Exception as e:
        logger.error(f"Error rendering request detail: {e}")
        return HTMLResponse(content=f"<h1>Error</h1><p>Failed to load request details: {e}</p>", status_code=500)


@app.get(
    "/api/requests/{request_id}",
    responses={
        404: {"description": "Request not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_request_details(request_id: str, request: Request):
    """API endpoint for request details with JSON response"""
    # Check WebUI access security
    get_webui_security().check_webui_access(request)

    try:
        # Convert to string in case it's a UUID object
        request_id_str = str(request_id)
        log_entry = await RequestLog.get_by_id(request_id_str)

        # Prepare duplicate info
        duplicate_info = {
            "is_duplicate": getattr(log_entry, "is_duplicate", False),
            "duplicate_count": getattr(log_entry, "duplicate_count", 0),
            "request_body_hash": getattr(log_entry, "request_body_hash", None),
            "duplicates": [],
        }
        try:
            if duplicate_info["request_body_hash"]:
                from smolrouter.redis_backend import RedisRequestLog

                ids = await RedisRequestLog.get_recent_duplicate_request_ids(
                    duplicate_info["request_body_hash"], limit=10
                )
                other_ids = [rid for rid in ids if rid != log_entry.id]
                for rid in other_ids[:10]:
                    dupe = await RequestLog.get_by_id(rid)
                    if dupe:
                        duplicate_info["duplicates"].append(
                            {
                                "id": dupe.id,
                                "timestamp": dupe.timestamp.isoformat(),
                                "status_code": dupe.status_code,
                                "source_ip": dupe.source_ip,
                            }
                        )
        except Exception:
            pass

        return {
            "id": log_entry.id,
            "timestamp": log_entry.timestamp.isoformat(),
            "source_ip": log_entry.source_ip,
            "path": log_entry.path,
            "original_model": log_entry.original_model,
            "mapped_model": log_entry.mapped_model,
            "service_type": log_entry.service_type,
            "status_code": log_entry.status_code,
            "duration_ms": log_entry.duration_ms,
            "request_size": log_entry.request_size,
            "response_size": log_entry.response_size,
            "upstream_url": log_entry.upstream_url,
            "error_message": log_entry.error_message,
            "api_key_suffix": getattr(log_entry, "api_key_suffix", None),
            "proxy_used": getattr(log_entry, "proxy_used", None),
            "provider_id": getattr(log_entry, "provider_id", None),
            "api_key_index": getattr(log_entry, "api_key_index", None),
            "api_key_total": getattr(log_entry, "api_key_total", None),
            "request_body": log_entry.request_body.decode("utf-8") if log_entry.request_body else None,
            "response_body": log_entry.response_body.decode("utf-8") if log_entry.response_body else None,
            "duplicate": duplicate_info,
        }
    except RequestLog.DoesNotExist:
        raise HTTPException(status_code=404, detail="Request not found")
    except Exception as e:
        logger.error(f"Error fetching request details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/clients/{client_ip}", response_class=HTMLResponse)
async def client_dashboard(client_ip: str, request: Request):
    """Client-specific dashboard page"""
    # Check WebUI access security
    get_webui_security().check_webui_access(request)

    try:
        return templates.TemplateResponse(
            request,
            "client_dashboard.html",
            {"client_ip": client_ip, "current_page": "client"},
        )
    except Exception as e:
        logger.error(f"Error loading client dashboard for {client_ip}: {e}")
        return HTMLResponse(content=f"<h1>Error</h1><p>Failed to load client dashboard: {e}</p>", status_code=500)


@app.get("/testing", response_class=HTMLResponse)
async def testing_page(request: Request):
    """Testing page with chat UI and validation tools"""
    # Check WebUI access security
    get_webui_security().check_webui_access(request)

    try:
        return templates.TemplateResponse(
            request,
            "testing.html",
            {"current_page": "testing"},
        )
    except Exception as e:
        logger.error(f"Error loading testing page: {e}")
        return HTMLResponse(content=f"<h1>Error</h1><p>Failed to load testing page: {e}</p>", status_code=500)


@app.get(
    "/api/testing/models",
    responses={500: {"description": "Failed to fetch available models"}},
)
async def get_available_models_for_testing(request: Request):
    """API endpoint to get available models for testing UI"""
    # Check WebUI access security
    get_webui_security().check_webui_access(request)

    try:
        # Actually fetch models using container architecture
        global container
        if container is None:
            await init_new_architecture()

        if container is not None:
            source_ip = request.client.host if request.client else "unknown"
            auth_payload = None
            try:
                auth_payload = verify_request_auth(request)
            except Exception:
                pass
            client_context = container.create_client_context(
                ip=source_ip, auth_payload=auth_payload, headers=dict(request.headers)
            )
            mediator = await container.get_mediator()
            models = await mediator.get_available_models(client_context)
        else:
            models = []

        # Sort models alphabetically by provider group, then by model name
        sorted_models = sorted(models, key=lambda m: (m.provider_id, m.name))

        # Return simplified model list for dropdown
        model_list = [
            {
                "id": model.id,
                "name": model.name,
                "display_name": model.display_name,
                "provider": model.provider_id,
                "request_model": model.display_name,
            }
            for model in sorted_models
        ]

        return {"models": model_list}

    except Exception as e:
        logger.error(f"Error fetching models for testing: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch available models")


@app.get(
    "/api/clients/{client_ip}",
    responses={500: {"description": "Internal server error"}},
)
async def get_client_data(client_ip: str, request: Request, limit: int = 100):
    """API endpoint for client-specific data"""
    # Check WebUI access security
    get_webui_security().check_webui_access(request)

    try:
        # Get client-specific logs
        logs = list(
            RequestLog.select()
            .where(RequestLog.source_ip == client_ip)
            .order_by(RequestLog.timestamp.desc())
            .limit(limit)
        )

        # Calculate client-specific stats
        total_requests = RequestLog.select().where(RequestLog.source_ip == client_ip).count()

        successful_requests = (
            RequestLog.select()
            .where(
                (RequestLog.source_ip == client_ip) & (RequestLog.status_code >= 200) & (RequestLog.status_code < 400)
            )
            .count()
        )

        # Recent requests (last 24 hours)
        from datetime import datetime, timedelta

        since_24h = datetime.now() - timedelta(hours=24)
        recent_requests = (
            RequestLog.select().where((RequestLog.source_ip == client_ip) & (RequestLog.timestamp >= since_24h)).count()
        )

        # Inflight requests
        inflight_requests = (
            RequestLog.select().where((RequestLog.source_ip == client_ip) & (RequestLog.completed_at.is_null())).count()
        )

        # Get unique models used by this client
        models_used = list(
            RequestLog.select(RequestLog.original_model)
            .where((RequestLog.source_ip == client_ip) & (RequestLog.original_model.is_null(False)))
            .distinct()
            .limit(50)
        )

        # Convert logs to dict format
        logs_data = []
        for log in logs:
            # Calculate duration for pending requests
            duration_ms = log.duration_ms
            if duration_ms is None and log.status_code is None:
                # Pending request - calculate elapsed time
                elapsed_seconds = (datetime.now() - log.timestamp).total_seconds()
                duration_ms = int(elapsed_seconds * 1000)

            logs_data.append(
                {
                    "id": log.id,
                    "timestamp": log.timestamp.isoformat(),
                    "source_ip": log.source_ip,
                    "path": log.path,
                    "original_model": log.original_model,
                    "mapped_model": log.mapped_model,
                    "service_type": log.service_type,
                    "status_code": log.status_code,
                    "duration_ms": duration_ms,
                    "request_size": log.request_size,
                    "response_size": log.response_size,
                    "upstream_url": log.upstream_url,
                }
            )

        stats = {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "recent_requests": recent_requests,
            "inflight_requests": inflight_requests,
            "models_used": [model.original_model for model in models_used],
        }

        return {
            "client_ip": client_ip,
            "stats": stats,
            "logs": logs_data,
        }

    except Exception as e:
        logger.error(f"Error fetching client data for {client_ip}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# WebSocket connection manager for real-time updates
class ConnectionManager:
    def __init__(self):
        # Store connections with their client filter (if any)
        self.active_connections: list[tuple[WebSocket, Optional[str]]] = []

    async def connect(self, websocket: WebSocket, client_filter: Optional[str] = None):
        await websocket.accept()
        self.active_connections.append((websocket, client_filter))
        filter_info = f" (filtering for {client_filter})" if client_filter else ""
        logger.debug(f"WebSocket connected{filter_info}. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        # Find and remove the connection
        for i, (conn, _) in enumerate(self.active_connections):
            if conn == websocket:
                del self.active_connections[i]
                break
        logger.debug(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict, source_ip: Optional[str] = None):
        """Broadcast message to relevant connected clients"""
        if not self.active_connections:
            return

        disconnected = []
        for connection, client_filter in self.active_connections:
            # Skip if this connection is filtered for a different client
            if client_filter and source_ip and client_filter != source_ip:
                continue

            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            self.disconnect(websocket)


# Global connection manager
manager = ConnectionManager()


async def broadcast_request_event(event_type: str, log_entry=None):
    """Broadcast request events to WebSocket clients"""
    try:
        logger.debug(
            f"broadcast_request_event called: {event_type}, connected clients: {len(manager.active_connections)}"
        )
        if event_type == "new_request" and log_entry:
            event_data = {"type": "new_request", "data": _serialize_request_log(log_entry)}
            await manager.broadcast(event_data, source_ip=log_entry.source_ip)

        elif event_type == "request_completed" and log_entry:
            event_data = {"type": "request_completed", "data": _serialize_request_log(log_entry)}
            await manager.broadcast(event_data, source_ip=log_entry.source_ip)

        elif event_type == "dashboard_update":
            # Send updated dashboard stats
            stats = await get_log_stats()
            event_data = {"type": "dashboard_update", "stats": stats}
            await manager.broadcast(event_data)

    except Exception as e:
        logger.error(f"Failed to broadcast WebSocket event: {e}")


@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    await manager.connect(websocket)

    try:
        # Send initial data
        try:
            logs, parsed_query = await _get_dashboard_logs(limit=100, q=websocket.query_params.get("q"))
            stats = await get_log_stats()

            initial_data = {
                "type": "dashboard_update",
                "logs": [_serialize_request_log(log) for log in logs],
                "stats": stats,
                "filter": parsed_query.to_meta(matched_count=len(logs), limit=100),
            }
            await manager.send_personal_message(initial_data, websocket)

        except DashboardFilterError as e:
            await manager.send_personal_message(
                {
                    "type": "filter_error",
                    "error": "invalid_filter",
                    "message": str(e),
                    "invalid_terms": e.invalid_terms,
                },
                websocket,
            )
        except Exception as e:
            logger.error(f"Failed to send initial dashboard data: {e}")

        # Keep connection alive and handle pings
        while True:
            try:
                # Wait for ping or other messages from client
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "ping":
                    await manager.send_personal_message({"type": "pong"}, websocket)
                elif message.get("type") == "refresh":
                    # Client requested refresh - send current data
                    # (Same logic as initial data - could be refactored)
                    pass

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

    finally:
        manager.disconnect(websocket)


@app.websocket("/ws/clients/{client_ip}")
async def websocket_client_dashboard(websocket: WebSocket, client_ip: str):
    """WebSocket endpoint for client-specific real-time updates"""
    await manager.connect(websocket, client_filter=client_ip)

    try:
        # Send initial client-specific data
        try:
            # Get client-specific logs
            logs = list(
                RequestLog.select()
                .where(RequestLog.source_ip == client_ip)
                .order_by(RequestLog.timestamp.desc())
                .limit(100)
            )

            # Calculate client-specific stats
            total_requests = RequestLog.select().where(RequestLog.source_ip == client_ip).count()

            successful_requests = (
                RequestLog.select()
                .where(
                    (RequestLog.source_ip == client_ip)
                    & (RequestLog.status_code >= 200)
                    & (RequestLog.status_code < 400)
                )
                .count()
            )

            # Recent requests (last 24 hours)
            from datetime import datetime, timedelta

            since_24h = datetime.now() - timedelta(hours=24)
            recent_requests = (
                RequestLog.select()
                .where((RequestLog.source_ip == client_ip) & (RequestLog.timestamp >= since_24h))
                .count()
            )

            # Inflight requests
            inflight_requests = (
                RequestLog.select()
                .where((RequestLog.source_ip == client_ip) & (RequestLog.completed_at.is_null()))
                .count()
            )

            # Format logs with real-time duration calculation
            formatted_logs = []
            for log in logs:
                duration_ms = log.duration_ms
                if duration_ms is None and log.status_code is None:
                    # Pending request - calculate elapsed time
                    elapsed_seconds = (datetime.now() - log.timestamp).total_seconds()
                    duration_ms = int(elapsed_seconds * 1000)

                formatted_logs.append(
                    {
                        "id": log.id,
                        "timestamp": log.timestamp.isoformat(),
                        "source_ip": log.source_ip,
                        "path": log.path,
                        "original_model": log.original_model,
                        "mapped_model": log.mapped_model,
                        "service_type": log.service_type,
                        "status_code": log.status_code,
                        "duration_ms": duration_ms,
                        "request_size": log.request_size,
                        "response_size": log.response_size,
                        "upstream_url": log.upstream_url,
                    }
                )

            stats = {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "recent_requests": recent_requests,
                "inflight_requests": inflight_requests,
            }

            initial_data = {
                "type": "client_dashboard_update",
                "logs": formatted_logs,
                "stats": stats,
                "client_ip": client_ip,
            }
            await manager.send_personal_message(initial_data, websocket)

        except Exception as e:
            logger.error(f"Failed to send initial client dashboard data for {client_ip}: {e}")

        # Keep connection alive and handle pings
        while True:
            try:
                # Wait for ping or other messages from client
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "ping":
                    await manager.send_personal_message({"type": "pong"}, websocket)
                elif message.get("type") == "refresh":
                    # Client requested refresh - send current data
                    # (Same logic as initial data - could be refactored)
                    pass

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error for client {client_ip}: {e}")
                break

    finally:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=LISTEN_HOST, port=LISTEN_PORT)
