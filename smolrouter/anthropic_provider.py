"""
Anthropic Claude provider for SmolRouter

Provides OpenAI-compatible access to Anthropic Claude models with:
- API key passthrough from client requests (preferred)
- Fallback to configured provider keys
- Per-model request/token statistics
- OpenAI to Anthropic format translation
"""

import httpx
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from smolrouter.interfaces import IModelProvider, ProviderConfig, ModelInfo

logger = logging.getLogger(__name__)

@dataclass
class AnthropicConfig(ProviderConfig):
    """Configuration for Anthropic provider"""
    api_keys: List[str] = None  # Will be initialized in __post_init__ if None
    api_keys_file: Optional[str] = None
    max_requests_per_day: Optional[int] = None  # Not used for rotation, just monitoring
    timeout: float = 30.0

    def _post_init_anthropic(self):
        """Load API keys from file if specified"""
        # Initialize api_keys if not provided
        if self.api_keys is None:
            self.api_keys = []

        if self.api_keys_file:
            try:
                with open(self.api_keys_file, 'r') as f:
                    file_keys = [line.strip() for line in f if line.strip()]
                    self.api_keys.extend(file_keys)
                logger.info(f"Loaded {len(file_keys)} API keys from {self.api_keys_file}")
            except Exception as e:
                logger.error(f"Failed to load API keys from {self.api_keys_file}: {e}")

@dataclass
class ModelStats:
    """Statistics for a single model"""
    model: str
    requests_today: int = 0
    tokens_today: int = 0
    last_request: Optional[datetime] = None
    last_error: Optional[str] = None
    error_count: int = 0

class AnthropicProvider(IModelProvider):
    """Anthropic Claude provider with API key passthrough support"""

    def __init__(self, config: AnthropicConfig):
        self.config = config
        self.config._post_init_anthropic()

        # Simple per-model statistics (no per-key tracking needed)
        self.model_stats: Dict[str, ModelStats] = {}

        logger.info(f"Initialized Anthropic provider with {len(self.config.api_keys)} fallback keys")

    async def discover_models(self) -> List[ModelInfo]:
        """Discover available Anthropic models"""
        # Common Anthropic Claude models
        models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]

        return [
            ModelInfo(
                model_id=model,
                model_name=model,
                aliases=[model],
                provider_id=self.get_provider_id(),
                endpoint=self.get_endpoint()
            )
            for model in models
        ]

    async def health_check(self) -> bool:
        """Check if Anthropic API is reachable"""
        try:
            # Try a simple request to check connectivity
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("https://api.anthropic.com/v1/health")
                # Anthropic doesn't have a health endpoint, so we'll just check if the base URL is reachable
                return response.status_code in [200, 404]  # 404 is expected for /health
        except Exception as e:
            logger.warning(f"Anthropic health check failed: {e}")
            return False

    async def make_request(self, request_data: dict, client_headers: dict) -> dict:
        """Make request to Anthropic API with OpenAI compatibility"""

        model = request_data.get('model', '')

        # Get or create model stats
        if model not in self.model_stats:
            self.model_stats[model] = ModelStats(model=model)

        stats = self.model_stats[model]

        # Determine API key to use (client passthrough preferred)
        api_key = self._get_api_key(client_headers)
        if not api_key:
            raise Exception("No Anthropic API key available - pass via Authorization header or configure fallback keys")

        # Convert OpenAI format to Anthropic format
        anthropic_request = self._convert_openai_to_anthropic(request_data)

        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01'
        }

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    'https://api.anthropic.com/v1/messages',
                    headers=headers,
                    json=anthropic_request
                )

                if response.status_code != 200:
                    error_msg = f"Anthropic API error {response.status_code}: {response.text}"
                    stats.last_error = error_msg
                    stats.error_count += 1
                    raise Exception(error_msg)

                anthropic_response = response.json()

                # Update statistics
                stats.requests_today += 1
                stats.last_request = datetime.now()

                # Extract token usage
                if 'usage' in anthropic_response:
                    input_tokens = anthropic_response['usage'].get('input_tokens', 0)
                    output_tokens = anthropic_response['usage'].get('output_tokens', 0)
                    stats.tokens_today += input_tokens + output_tokens

                # Convert Anthropic response to OpenAI format
                return self._convert_anthropic_to_openai(anthropic_response, model)

        except Exception as e:
            stats.last_error = str(e)
            stats.error_count += 1
            raise

    def _get_api_key(self, client_headers: dict) -> Optional[str]:
        """Get API key from client headers or fallback to configured keys"""

        # Check for client-provided API key
        auth_header = client_headers.get('authorization', '')
        if auth_header.startswith('Bearer sk-ant-'):
            return auth_header.replace('Bearer ', '')

        # Fallback to first configured key if available
        if self.config.api_keys:
            return self.config.api_keys[0]

        return None

    def _convert_openai_to_anthropic(self, openai_request: dict) -> dict:
        """Convert OpenAI format to Anthropic format"""

        messages = openai_request.get('messages', [])

        # Extract system message if present
        system_message = None
        user_messages = []

        for msg in messages:
            if msg.get('role') == 'system':
                system_message = msg.get('content', '')
            else:
                user_messages.append(msg)

        anthropic_request = {
            'model': openai_request.get('model', 'claude-3-sonnet-20240229'),
            'messages': user_messages,
            'max_tokens': openai_request.get('max_tokens', 1024)
        }

        # Add system message if present
        if system_message:
            anthropic_request['system'] = system_message

        # Map optional parameters
        if 'temperature' in openai_request:
            anthropic_request['temperature'] = openai_request['temperature']

        if 'top_p' in openai_request:
            anthropic_request['top_p'] = openai_request['top_p']

        return anthropic_request

    def _convert_anthropic_to_openai(self, anthropic_response: dict, model: str) -> dict:
        """Convert Anthropic response to OpenAI format"""

        # Extract content from Anthropic response
        content = ""
        if 'content' in anthropic_response and anthropic_response['content']:
            # Anthropic returns content as a list of content blocks
            for block in anthropic_response['content']:
                if block.get('type') == 'text':
                    content += block.get('text', '')

        # Determine finish reason
        finish_reason = 'stop'
        if anthropic_response.get('stop_reason') == 'max_tokens':
            finish_reason = 'length'
        elif anthropic_response.get('stop_reason') == 'stop_sequence':
            finish_reason = 'stop'

        # Build OpenAI-compatible response
        openai_response = {
            'id': f"chatcmpl-{int(time.time())}.{int(time.time() * 1000) % 1000:06d}",
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': model,
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': content
                },
                'finish_reason': finish_reason
            }],
            'usage': {
                'prompt_tokens': anthropic_response.get('usage', {}).get('input_tokens', 0),
                'completion_tokens': anthropic_response.get('usage', {}).get('output_tokens', 0),
                'total_tokens': (
                    anthropic_response.get('usage', {}).get('input_tokens', 0) +
                    anthropic_response.get('usage', {}).get('output_tokens', 0)
                )
            }
        }

        return openai_response

    def get_provider_id(self) -> str:
        """Get provider identifier"""
        return self.config.name

    def get_provider_type(self) -> str:
        """Get provider type"""
        return "anthropic"

    def get_endpoint(self) -> str:
        """Get provider endpoint"""
        return "https://api.anthropic.com"

    def get_stats(self) -> dict:
        """Get provider statistics (simplified format)"""
        return {
            'provider_type': 'anthropic',
            'models': {
                model: {
                    'requests_today': stats.requests_today,
                    'tokens_today': stats.tokens_today,
                    'last_request': stats.last_request.isoformat() if stats.last_request else None,
                    'last_error': stats.last_error,
                    'error_count': stats.error_count
                }
                for model, stats in self.model_stats.items()
            },
            'total_configured_keys': len(self.config.api_keys)
        }

    def get_api_key_stats(self) -> dict:
        """Get API key statistics (compatible with dashboard format)"""
        # For Anthropic, we track by model rather than by key since keys are passed through
        # Format to match the expected dashboard structure
        stats = {
            "provider_stats": {
                "models": {
                    model: {
                        "requests_today": model_stats.requests_today,
                        "tokens_today": model_stats.tokens_today,
                        "last_request": model_stats.last_request.isoformat() if model_stats.last_request else None,
                        "last_error": model_stats.last_error,
                        "error_count": model_stats.error_count,
                        "status": "error" if model_stats.error_count > 5 else "available"
                    }
                    for model, model_stats in self.model_stats.items()
                },
                "total_configured_keys": len(self.config.api_keys),
                "passthrough_mode": True  # Indicates this provider uses passthrough
            }
        }
        return stats