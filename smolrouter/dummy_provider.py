"""
Dummy LLM provider for load testing and development

Provides realistic responses with configurable delays for testing SmolRouter
performance under various conditions without hitting real LLM APIs.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from smolrouter.interfaces import IModelProvider, ProviderConfig, ModelInfo

logger = logging.getLogger(__name__)


@dataclass
class DummyConfig(ProviderConfig):
    """Configuration for dummy provider"""

    response_delay_ms: int = 250  # Configurable delay in milliseconds (250ms default)
    failure_rate: float = 0.0  # 0.0 = never fail, 1.0 = always fail
    response_tokens: int = 150  # Default response token count
    max_requests_per_day: Optional[int] = None  # Not used, just for compatibility


@dataclass
class DummyStats:
    """Statistics for dummy provider"""

    model: str
    requests_today: int = 0
    tokens_today: int = 0
    last_request: Optional[datetime] = None
    avg_response_time: float = 0.0
    error_count: int = 0


class DummyProvider(IModelProvider):
    """Dummy LLM provider for testing and development"""

    def __init__(self, config: DummyConfig):
        self.config = config
        self.model_stats: Dict[str, DummyStats] = {}

        logger.info(
            "Initialized Dummy provider with %sms delay, %.1f%% failure rate",
            config.response_delay_ms,
            config.failure_rate * 100,
        )

    async def discover_models(self) -> List[ModelInfo]:
        """Discover available dummy models"""
        models = [
            "dummy-fast-3.5",
            "dummy-standard-4.0",
            "dummy-slow-4.0",
            "dummy-tiny-1.0",
            "dummy-large-8b",
            "dummy-xl-70b",
        ]

        return [
            ModelInfo(
                id=f"{model}@{self.get_provider_id()}",
                name=model,
                aliases=[model, model.replace("dummy-", "test-")],
                provider_id=self.get_provider_id(),
                provider_type=self.get_provider_type(),
                endpoint=self.get_endpoint(),
            )
            for model in models
        ]

    async def health_check(self) -> bool:
        """Check dummy provider health (always healthy unless configured otherwise)"""
        return True

    async def make_request(self, request_data: dict, client_headers: dict) -> dict:
        """Make dummy request with configurable delay and responses"""

        model = request_data.get("model", "dummy-standard-4.0")
        if client_headers:
            logger.debug("Dummy request received with %d client headers", len(client_headers))

        # Get or create model stats
        if model not in self.model_stats:
            self.model_stats[model] = DummyStats(model=model)

        stats = self.model_stats[model]

        # Simulate processing delay
        delay_seconds = self.config.response_delay_ms / 1000.0
        await asyncio.sleep(delay_seconds)

        # Simulate failures if configured
        import random

        if random.random() < self.config.failure_rate:
            stats.error_count += 1
            raise RuntimeError(f"Dummy provider simulated failure (rate: {self.config.failure_rate * 100:.1f}%)")

        # Extract prompt for realistic response
        messages = request_data.get("messages", [])
        if messages:
            last_message = messages[-1].get("content", "Hello")
        else:
            last_message = request_data.get("prompt", "Hello")

        # Generate dummy response content
        response_content = self._generate_dummy_response(last_message, model)

        # Calculate token counts
        prompt_tokens = len(str(messages).split()) if messages else len(last_message.split())
        completion_tokens = self.config.response_tokens
        total_tokens = prompt_tokens + completion_tokens

        # Update statistics
        stats.requests_today += 1
        stats.tokens_today += total_tokens
        stats.last_request = datetime.now()

        # Build OpenAI-compatible response
        response = {
            "id": f"dummy-{int(time.time())}-{random.randint(1000, 9999)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": response_content}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

        return response

    def _generate_dummy_response(self, prompt: str, model: str) -> str:
        """Generate simple dummy response content"""

        # Simple response: "You said: <prompt>"
        logger.debug("Generating dummy response for model %s", model)
        return f"You said: {prompt}"

    def get_provider_id(self) -> str:
        """Get provider identifier"""
        return self.config.name

    def get_provider_type(self) -> str:
        """Get provider type"""
        return "dummy"

    def get_endpoint(self) -> str:
        """Get provider endpoint"""
        return "dummy://localhost/test"

    def get_stats(self) -> dict:
        """Get provider statistics"""
        return {
            "provider_type": "dummy",
            "config": {
                "response_delay_ms": self.config.response_delay_ms,
                "failure_rate": self.config.failure_rate,
                "response_tokens": self.config.response_tokens,
            },
            "models": {
                model: {
                    "requests_today": stats.requests_today,
                    "tokens_today": stats.tokens_today,
                    "last_request": stats.last_request.isoformat() if stats.last_request else None,
                    "avg_response_time": stats.avg_response_time,
                    "error_count": stats.error_count,
                }
                for model, stats in self.model_stats.items()
            },
        }

    def get_api_key_stats(self) -> dict:
        """Get API key statistics (dummy provider doesn't use real keys)"""
        return {
            "provider_stats": {
                "dummy_mode": True,
                "models": {
                    model: {
                        "requests_today": model_stats.requests_today,
                        "tokens_today": model_stats.tokens_today,
                        "last_request": model_stats.last_request.isoformat() if model_stats.last_request else None,
                        "error_count": model_stats.error_count,
                        "status": "available",
                        "simulated_delay_ms": self.config.response_delay_ms,
                    }
                    for model, model_stats in self.model_stats.items()
                },
                "total_configured_keys": 1,  # Dummy value
                "dummy_provider": True,
            }
        }
