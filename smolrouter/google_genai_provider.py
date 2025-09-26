"""
Google Generative AI provider implementation.

Provides integration with Google's Generative AI API, supporting multiple API keys
with intelligent rotation based on requests-per-day (RPD) quotas.
"""

import logging
import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin
from dataclasses import dataclass, field
import pytz

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from google.api_core.exceptions import ResourceExhausted, PermissionDenied, InvalidArgument

from .interfaces import IModelProvider, ModelInfo, ProviderConfig, ClientContext
from .database import ApiKeyQuota
from .rate_limiter import google_genai_funnel

logger = logging.getLogger(__name__)


@dataclass
class ApiKeyModelStats:
    """Statistics for a single API key + model combination"""
    api_key: str
    model: str
    requests_today: int = 0
    tokens_today: int = 0
    last_request: Optional[datetime] = None
    last_error: Optional[str] = None
    error_count: int = 0
    quota_exhausted_at: Optional[datetime] = None
    invalid_key: bool = False  # Permanently mark invalid/expired keys

    @property
    def key_model_id(self) -> str:
        """Unique identifier for this key+model combination"""
        return f"{self.api_key[:8]}.../{self.model}"

    def is_day_reset_needed(self) -> bool:
        """Check if we need to reset daily counters (Pacific timezone reset)"""
        pacific_tz = pytz.timezone('US/Pacific')
        now_pacific = datetime.now(pacific_tz)
        now_date = now_pacific.date()

        # Check against last request time
        if self.last_request:
            last_request_pacific = self.last_request.replace(tzinfo=pacific_tz) if self.last_request.tzinfo is None else self.last_request.astimezone(pacific_tz)
            last_request_date = last_request_pacific.date()
            if now_date > last_request_date:
                return True

        # Also check against quota exhaustion time (important for 429 recovery)
        if self.quota_exhausted_at:
            exhausted_pacific = self.quota_exhausted_at.replace(tzinfo=pacific_tz) if self.quota_exhausted_at.tzinfo is None else self.quota_exhausted_at.astimezone(pacific_tz)
            exhausted_date = exhausted_pacific.date()
            if now_date > exhausted_date:
                return True

        return False

    def reset_daily_stats(self):
        """Reset daily counters"""
        self.requests_today = 0
        self.tokens_today = 0
        self.error_count = 0
        self.quota_exhausted_at = None  # Clear exhaustion marker
        logger.info(f"Reset daily stats for API key {self.api_key[:8]}...")


@dataclass
class GoogleGenAIConfig(ProviderConfig):
    """Extended configuration for Google GenAI provider"""
    api_keys: List[str] = field(default_factory=list)
    api_keys_file: Optional[str] = None  # Path to file containing API keys
    max_requests_per_day: int = 1500  # Free tier limit
    max_tokens_per_minute: int = 32000  # Free tier limit

    def __init__(self, **kwargs):
        # Extract Google-specific fields before calling parent
        self.api_keys = kwargs.pop('api_keys', [])
        self.api_keys_file = kwargs.pop('api_keys_file', None)
        self.max_requests_per_day = kwargs.pop('max_requests_per_day', 1500)
        self.max_tokens_per_minute = kwargs.pop('max_tokens_per_minute', 32000)

        # Set required fields for base class
        if 'url' not in kwargs:
            kwargs['url'] = 'https://generativelanguage.googleapis.com'

        # Call parent constructor
        super().__init__(**kwargs)
        self._post_init_google()

    def _post_init_google(self):
        if not self.api_keys and not self.api_keys_file:
            raise ValueError("Either api_keys or api_keys_file must be provided")

        # Load API keys from file if specified
        if self.api_keys_file:
            try:
                with open(self.api_keys_file, 'r') as f:
                    file_keys = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                self.api_keys.extend(file_keys)
                logger.info(f"Loaded {len(file_keys)} API keys from {self.api_keys_file}")
            except Exception as e:
                logger.error(f"Failed to load API keys from {self.api_keys_file}: {e}")
                raise

        if not self.api_keys:
            raise ValueError("No valid API keys found")


class GoogleGenAIProvider(IModelProvider):
    """Provider for Google Generative AI models with intelligent API key rotation"""

    # No mappings - let Google handle their own model aliasing
    MODEL_MAPPINGS = {}

    def __init__(self, config: GoogleGenAIConfig):
        if not isinstance(config, GoogleGenAIConfig):
            # Convert regular ProviderConfig to GoogleGenAIConfig
            if hasattr(config, 'api_keys'):
                config = GoogleGenAIConfig(**config.__dict__)
            else:
                raise ValueError("GoogleGenAIProvider requires GoogleGenAIConfig")

        self.config = config
        self.config.type = "google-genai"

        # Database persistence - no more in-memory stats storage

        # Cache for discovered models
        self._cached_models: Optional[List[ModelInfo]] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=30)

        logger.info(f"Initialized GoogleGenAI provider with {len(self.config.api_keys)} API keys")

    def get_provider_id(self) -> str:
        return self.config.name

    def get_provider_type(self) -> str:
        return "google-genai"

    def get_endpoint(self) -> str:
        return "https://generativelanguage.googleapis.com"

    def _get_pacific_date(self) -> str:
        """Get current date in Pacific timezone as YYYY-MM-DD string"""
        pacific_tz = pytz.timezone('US/Pacific')
        return datetime.now(pacific_tz).date().strftime('%Y-%m-%d')

    def _get_quota_record(self, api_key: str, model_name: str) -> ApiKeyQuota:
        """Get or create quota record for an API key + model combination"""
        pacific_date = self._get_pacific_date()
        quota, created = ApiKeyQuota.get_or_create_quota(
            api_key=api_key,
            provider_name=self.config.name,
            model_name=model_name,
            pacific_date=pacific_date
        )
        return quota

    def _select_best_api_key(self, model_name: str) -> str:
        """
        Select the API key with lowest usage for the given model.

        Returns the first key from the set of keys with the lowest request count FOR THIS MODEL.
        Order among equals doesn't matter - just consistent selection.
        """
        # Group keys by their request count for THIS MODEL, excluding exhausted/error keys
        available_keys = []
        exhausted_keys = []
        error_prone_keys = []

        for key in self.config.api_keys:
            quota = self._get_quota_record(key, model_name)

            # Skip if key is permanently invalid/expired
            if quota.invalid_key:
                logger.debug(f"API key {key[:8]}... marked as invalid, skipping")
                continue

            # Check if quota should be reset due to date change (defensive check)
            pacific_tz = pytz.timezone('US/Pacific')
            pacific_date = datetime.now(pacific_tz).strftime('%Y-%m-%d')

            # If the quota hasn't been reset today, consider it fresh
            actual_requests_today = quota.requests_today if quota.last_reset_date == pacific_date else 0

            # Check if key has hit daily limit for this model (either by count or 429 response)
            if actual_requests_today >= self.config.max_requests_per_day:
                exhausted_keys.append(key)
                logger.debug(f"API key {key[:8]}... exhausted for {model_name} ({actual_requests_today}/{self.config.max_requests_per_day}) reset_date={quota.last_reset_date} today={pacific_date}")
                continue

            # Skip keys with too many recent errors for this model
            if quota.error_count > 20:  # Increased threshold
                error_prone_keys.append(key)
                logger.debug(f"API key {key[:8]}... too many errors for {model_name} ({quota.error_count})")
                continue

            # Check for recent quota errors for this model (even if not at limit)
            # But skip this check if quota should have reset today
            if quota.last_error and self._is_quota_exhausted_error(quota.last_error) and quota.last_reset_date == pacific_date:
                # If the error was recent (within last hour) AND the quota hasn't reset, be cautious
                if quota.quota_exhausted_at:
                    # quota_exhausted_at could be datetime object or string from database
                    pacific_tz = pytz.timezone('US/Pacific')

                    if isinstance(quota.quota_exhausted_at, str):
                        # Parse datetime string from database
                        try:
                            quota_exhausted_dt = datetime.fromisoformat(quota.quota_exhausted_at.replace('Z', '+00:00'))
                            if quota_exhausted_dt.tzinfo is None:
                                quota_exhausted_pacific = pytz.utc.localize(quota_exhausted_dt).astimezone(pacific_tz)
                            else:
                                quota_exhausted_pacific = quota_exhausted_dt.astimezone(pacific_tz)
                        except ValueError:
                            # Skip malformed timestamp
                            continue
                    else:
                        # Handle datetime object
                        if quota.quota_exhausted_at.tzinfo is None:
                            quota_exhausted_pacific = pytz.utc.localize(quota.quota_exhausted_at).astimezone(pacific_tz)
                        else:
                            quota_exhausted_pacific = quota.quota_exhausted_at.astimezone(pacific_tz)

                    if (datetime.now(pacific_tz) - quota_exhausted_pacific).seconds < 3600:
                        logger.debug(f"API key {key[:8]}... recent quota error for {model_name}, skipping temporarily")
                        continue

            available_keys.append((key, actual_requests_today))

        if not available_keys:
            # All keys exhausted for this model - provide detailed status
            total_keys = len(self.config.api_keys)
            logger.error(f"🚫 ALL {total_keys} API KEYS EXHAUSTED FOR MODEL {model_name}:")
            logger.error(f"   - Quota exhausted: {len(exhausted_keys)} keys")
            logger.error(f"   - Error-prone: {len(error_prone_keys)} keys")

            # Calculate seconds until quota reset (midnight Pacific time)
            pacific_tz = pytz.timezone('US/Pacific')
            now_pacific = datetime.now(pacific_tz)
            tomorrow_pacific = (now_pacific + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            seconds_until_reset = int((tomorrow_pacific - now_pacific).total_seconds())

            logger.error(f"⏰ All API keys exhausted. Quota resets in {seconds_until_reset}s at midnight Pacific")

            # Raise a specific exception that the container can catch and convert to 429
            from google.api_core.exceptions import ResourceExhausted
            raise ResourceExhausted(
                f"All {total_keys} API keys exhausted for model {model_name}. "
                f"Quota resets in {seconds_until_reset} seconds at midnight Pacific time.",
                errors=[{
                    'reason': 'QUOTA_EXHAUSTED',
                    'retry_after_seconds': seconds_until_reset
                }]
            )

        # Sort by request count, then by key name for consistent ordering
        available_keys.sort(key=lambda x: (x[1], x[0]))

        # Select the first key from the lowest usage group
        best_key = available_keys[0][0]
        lowest_count = available_keys[0][1]

        # Count how many keys have the same lowest usage
        equal_usage_count = sum(1 for _, count in available_keys if count == lowest_count)

        logger.debug(f"Selected API key {best_key[:8]}... for {model_name} with {lowest_count} requests today "
                    f"({equal_usage_count} keys at this usage level)")

        return best_key

    def _update_api_key_stats(self, api_key: str, model_name: str, success: bool, tokens: int = 0, error: str = None):
        """Update statistics for an API key + model combination after a request"""
        quota = self._get_quota_record(api_key, model_name)

        # Database handles timezone automatically

        if success:
            quota.mark_request_success(tokens=tokens)
            logger.info(f"API key {api_key[:8]}... successful request for {model_name}: {quota.requests_today}/{self.config.max_requests_per_day} RPD, {tokens} tokens")
        else:
            # Check error type and handle appropriately
            if self._is_invalid_key_error(error):
                # Mark invalid across all models for this key
                key_hash = ApiKeyQuota.hash_api_key(api_key)
                ApiKeyQuota.update(invalid_key=True).where(
                    (ApiKeyQuota.api_key_hash == key_hash) &
                    (ApiKeyQuota.provider_name == self.config.name)
                ).execute()
                logger.error(f"🚫 API key {api_key[:8]}... INVALID/EXPIRED: {error}")

            # Check if this is a quota exhaustion error
            elif self._is_quota_exhausted_error(error):
                # Mark this key as exhausted for this model regardless of our internal counter
                quota.mark_request_failure(error=error, quota_exhausted=True)
                logger.error(f"🚫 API key {api_key[:8]}... QUOTA EXHAUSTED (429) for {model_name}: Hard marked as depleted")

                # Extract retry delay if available
                retry_delay = self._extract_retry_delay(error)
                if retry_delay:
                    logger.warning(f"🕒 Google suggests retry in {retry_delay}s for {api_key[:8]}... / {model_name}")
                else:
                    logger.warning(f"🕒 Key {api_key[:8]}... / {model_name} exhausted, will reset at midnight Pacific")
            else:
                # Regular error - still count the request
                quota.mark_request_failure(error=error)

            logger.warning(f"API key {api_key[:8]}... error #{quota.error_count} for {model_name}: {error}")

        # Log quota status if approaching limits for this model
        quota_percentage = (quota.requests_today / self.config.max_requests_per_day) * 100
        if quota_percentage >= 80:
            logger.warning(f"API key {api_key[:8]}... / {model_name} approaching daily limit: {quota_percentage:.1f}% used ({quota.requests_today}/{self.config.max_requests_per_day})")
        elif quota_percentage >= 100:
            logger.error(f"🚫 API key {api_key[:8]}... / {model_name} DAILY LIMIT REACHED: {quota.requests_today}/{self.config.max_requests_per_day}")

    def _is_invalid_key_error(self, error_msg: str) -> bool:
        """Check if error indicates invalid/expired API key"""
        if not error_msg:
            return False
        error_lower = error_msg.lower()
        invalid_key_indicators = [
            "permission denied",
            "api key not valid",
            "invalid api key",
            "api_key_invalid",
            "authentication failed",
            "unauthorized",
            "forbidden",
            "invalid_argument",  # Often used for bad keys in Google APIs
            "credentials are missing or invalid",
            "api key expired"
        ]
        return any(indicator in error_lower for indicator in invalid_key_indicators)

    def _is_quota_exhausted_error(self, error_msg: str) -> bool:
        """Check if error indicates quota exhaustion"""
        if not error_msg:
            return False

        error_lower = error_msg.lower()
        quota_indicators = [
            "429",
            "resource_exhausted",
            "quota exceeded",
            "current quota",
            "quota.*exceeded",
            "requests per day",
            "free_tier_requests"
        ]

        return any(indicator in error_lower for indicator in quota_indicators)

    def _extract_retry_delay(self, error_msg: str) -> Optional[float]:
        """Extract retry delay from Google error message"""
        if not error_msg:
            return None

        import re
        # Look for patterns like "retry in 20.915074628s" or "retryDelay': '20s'"
        patterns = [
            r'retry in (\d+(?:\.\d+)?)s',
            r'retryDelay.*?(\d+(?:\.\d+)?)s',
            r'Please retry.*?(\d+(?:\.\d+)?).*?s'
        ]

        for pattern in patterns:
            match = re.search(pattern, error_msg, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name to Google GenAI format"""
        return self.MODEL_MAPPINGS.get(model_name, model_name)

    async def health_check(self) -> bool:
        """Check if at least one API key is working"""
        for api_key in self.config.api_keys:
            try:
                genai.configure(api_key=api_key)
                # Try to list models as a health check
                models = genai.list_models()
                list(models)  # Force evaluation
                return True
            except Exception as e:
                logger.debug(f"Health check failed for API key {api_key[:8]}...: {e}")
                continue
        return False

    async def discover_models(self) -> List[ModelInfo]:
        """Discover available models from Google GenAI API or return static list"""
        # Check cache first
        if (self._cached_models and self._cache_time and
            datetime.now() - self._cache_time < self._cache_ttl):
            return self._cached_models

        # Try to discover models using available API keys (live discovery)
        for api_key in self.config.api_keys:
            try:
                genai.configure(api_key=api_key)
                models = []

                # List available models
                for model in genai.list_models():
                    if 'generateContent' in model.supported_generation_methods:
                        model_name = model.name.split('/')[-1]  # Extract model name from full path

                        # Create model info
                        model_info = ModelInfo(
                            id=f"{model_name}@{self.get_provider_id()}",
                            name=model_name,
                            provider_id=self.get_provider_id(),
                            provider_type=self.get_provider_type(),
                            endpoint=self.get_endpoint(),
                            aliases=[model_name],
                            metadata={
                                'full_name': model.name,
                                'display_name': model.display_name,
                                'description': getattr(model, 'description', ''),
                                'supported_methods': model.supported_generation_methods,
                                'input_token_limit': getattr(model, 'input_token_limit', None),
                                'output_token_limit': getattr(model, 'output_token_limit', None)
                            }
                        )
                        models.append(model_info)
                        logger.debug(f"Discovered Google GenAI model: {model_info.id}")

                # Cache the results
                self._cached_models = models
                self._cache_time = datetime.now()

                logger.info(f"Discovered {len(models)} models from Google GenAI provider {self.get_provider_id()}")
                return models

            except Exception as e:
                logger.error(f"Error discovering models with API key {api_key[:8]}...: {e}")
                continue

        # All API keys failed - fall back to static model list
        logger.warning("Live model discovery failed for all API keys, falling back to static model list")
        return self._get_static_google_genai_models()

    def _get_static_google_genai_models(self) -> List[ModelInfo]:
        """Return static list of Google GenAI models from JSON file"""
        import json
        import os
        from pathlib import Path

        try:
            # Get the absolute path to the models JSON file
            current_dir = Path(__file__).parent
            models_file = current_dir / "models" / "google-genai-models-2025-september.json"

            if not models_file.exists():
                logger.error(f"Static Google GenAI models file not found: {models_file}")
                return []

            with open(models_file, 'r') as f:
                data = json.load(f)

            models = []
            for model_data in data.get('models', []):
                # Only include models that support text generation
                supported_methods = model_data.get('supportedGenerationMethods', [])
                if 'generateContent' not in supported_methods:
                    continue  # Skip embedding and other non-text generation models

                # Extract model name from full path (models/model-name -> model-name)
                full_name = model_data.get('name', '')
                model_name = full_name.split('/')[-1] if '/' in full_name else full_name

                # Create model info
                model_info = ModelInfo(
                    id=f"{model_name}@{self.get_provider_id()}",
                    name=model_name,
                    provider_id=self.get_provider_id(),
                    provider_type=self.get_provider_type(),
                    endpoint=self.get_endpoint(),
                    aliases=[model_name],
                    metadata={
                        'full_name': full_name,
                        'display_name': model_data.get('displayName', model_name),
                        'description': model_data.get('description', ''),
                        'supported_methods': supported_methods,
                        'input_token_limit': model_data.get('inputTokenLimit'),
                        'output_token_limit': model_data.get('outputTokenLimit'),
                        'version': model_data.get('version'),
                        'temperature': model_data.get('temperature'),
                        'top_p': model_data.get('topP'),
                        'top_k': model_data.get('topK'),
                        'max_temperature': model_data.get('maxTemperature'),
                        'thinking': model_data.get('thinking', False)
                    }
                )
                models.append(model_info)
                logger.debug(f"Static Google GenAI model: {model_info.id}")

            logger.info(f"Loaded {len(models)} static models for Google GenAI provider {self.get_provider_id()}")
            return models

        except Exception as e:
            logger.error(f"Error loading static Google GenAI models: {e}")
            return []

    def _convert_openai_to_genai_request(self, openai_request: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Convert OpenAI request format to Google GenAI format"""
        model_name = self._normalize_model_name(openai_request.get('model', ''))

        # Extract messages
        messages = openai_request.get('messages', [])
        if not messages:
            raise ValueError("No messages provided in request")

        # Convert messages to Google GenAI format
        contents = []
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')

            # Map OpenAI roles to Google GenAI roles
            if role == 'system':
                # System messages can be added as user messages with special formatting
                contents.append({'role': 'user', 'parts': [{'text': f"System: {content}"}]})
            elif role == 'assistant':
                contents.append({'role': 'model', 'parts': [{'text': content}]})
            else:  # user
                contents.append({'role': 'user', 'parts': [{'text': content}]})

        # Build generation config
        generation_config = {}

        # Map common parameters
        if 'temperature' in openai_request:
            generation_config['temperature'] = openai_request['temperature']
        if 'max_tokens' in openai_request:
            generation_config['max_output_tokens'] = openai_request['max_tokens']
        if 'top_p' in openai_request:
            generation_config['top_p'] = openai_request['top_p']

        # Google GenAI doesn't support streaming in the same way, so we'll handle that separately
        genai_request = {
            'contents': contents,
            'generation_config': generation_config
        }

        return model_name, genai_request

    def _convert_genai_to_openai_response(self, genai_response: GenerateContentResponse,
                                        original_model: str) -> Dict[str, Any]:
        """Convert Google GenAI response to OpenAI format"""
        try:
            # Extract text content
            text_content = ""
            if genai_response.candidates:
                for part in genai_response.candidates[0].content.parts:
                    if hasattr(part, 'text'):
                        text_content += part.text

            # Extract usage information
            usage = {}
            if hasattr(genai_response, 'usage_metadata') and genai_response.usage_metadata:
                usage = {
                    'prompt_tokens': getattr(genai_response.usage_metadata, 'prompt_token_count', 0),
                    'completion_tokens': getattr(genai_response.usage_metadata, 'candidates_token_count', 0),
                    'total_tokens': getattr(genai_response.usage_metadata, 'total_token_count', 0)
                }

            # Build OpenAI-compatible response
            openai_response = {
                'id': f"chatcmpl-{datetime.now().timestamp()}",
                'object': 'chat.completion',
                'created': int(datetime.now().timestamp()),
                'model': original_model,
                'choices': [
                    {
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'content': text_content
                        },
                        'finish_reason': 'stop'
                    }
                ],
                'usage': usage
            }

            return openai_response

        except Exception as e:
            logger.error(f"Error converting GenAI response to OpenAI format: {e}")
            raise

    async def generate_completion(self, openai_request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate completion using Google GenAI API"""
        original_model = openai_request.get('model', '')

        try:
            # Convert request format
            model_name, genai_request = self._convert_openai_to_genai_request(openai_request)

            # Select best API key
            api_key = self._select_best_api_key(model_name)
            genai.configure(api_key=api_key)

            # Initialize the model
            model = genai.GenerativeModel(model_name)

            # Acquire rate limiting slot (blocks if needed)
            await google_genai_funnel.acquire_slot()

            try:
                # Generate content
                response = await asyncio.to_thread(
                    model.generate_content,
                    genai_request['contents'],
                    generation_config=genai_request['generation_config']
                )
            finally:
                # Always release slot, even on error
                google_genai_funnel.release_slot()

            # Convert response back to OpenAI format
            openai_response = self._convert_genai_to_openai_response(response, original_model)

            # Update API key statistics
            tokens_used = openai_response.get('usage', {}).get('total_tokens', 0)
            self._update_api_key_stats(api_key, model_name, success=True, tokens=tokens_used)

            return openai_response

        except ResourceExhausted as e:
            error_msg = f"Google GenAI quota exhausted: {e}"
            # Don't update stats if this is our own "all keys exhausted" exception
            if 'api_key' in locals() and 'model_name' in locals():
                self._update_api_key_stats(api_key, model_name, success=False, error=error_msg)

            # Check if this ResourceExhausted has retry timing info
            retry_after_seconds = None
            if hasattr(e, 'errors') and e.errors:
                for error_detail in e.errors:
                    if isinstance(error_detail, dict) and 'retry_after_seconds' in error_detail:
                        retry_after_seconds = error_detail['retry_after_seconds']
                        break

            # Create a custom exception that the container can handle with proper HTTP response
            quota_error = Exception(error_msg)
            quota_error.retry_after_seconds = retry_after_seconds
            raise quota_error
        except PermissionDenied as e:
            error_msg = f"Google GenAI permission denied: {e}"
            self._update_api_key_stats(api_key, model_name, success=False, error=error_msg)
            raise Exception(error_msg)
        except InvalidArgument as e:
            error_msg = f"Google GenAI invalid argument: {e}"
            self._update_api_key_stats(api_key, model_name, success=False, error=error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Google GenAI error: {e}"
            if 'api_key' in locals() and 'model_name' in locals():
                self._update_api_key_stats(api_key, model_name, success=False, error=error_msg)
            raise Exception(error_msg)

    def get_api_key_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get current API key usage statistics (grouped by API key, showing all models)"""
        stats = {}

        # Get all quota records for this provider
        quotas = ApiKeyQuota.select().where(
            ApiKeyQuota.provider_name == self.config.name
        )

        # Group stats by API key
        for quota in quotas:
            # Use hash for privacy but show some chars for identification
            key_display = f"{quota.api_key_hash[:8]}..."
            model = quota.model_name

            if key_display not in stats:
                stats[key_display] = {
                    'models': {},
                    'total_requests_today': 0,
                    'total_tokens_today': 0,
                    'total_errors': 0,
                    'daily_limit_per_model': self.config.max_requests_per_day
                }

            quota_percentage = (quota.requests_today / self.config.max_requests_per_day) * 100
            is_exhausted = quota.requests_today >= self.config.max_requests_per_day or quota.quota_exhausted_at is not None

            stats[key_display]['models'][model] = {
                'requests_today': quota.requests_today,
                'tokens_today': quota.tokens_today,
                'error_count': quota.error_count,
                'last_request': quota.updated_at.isoformat() if quota.updated_at and hasattr(quota.updated_at, 'isoformat') else str(quota.updated_at) if quota.updated_at else None,
                'last_error': quota.last_error,
                'quota_percentage': quota_percentage,
                'quota_exhausted': is_exhausted,
                'quota_exhausted_at': quota.quota_exhausted_at.isoformat() if quota.quota_exhausted_at and hasattr(quota.quota_exhausted_at, 'isoformat') else str(quota.quota_exhausted_at) if quota.quota_exhausted_at else None,
                'status': 'invalid' if quota.invalid_key else ('exhausted' if is_exhausted else 'available')
            }

            # Aggregate totals for this API key
            stats[key_display]['total_requests_today'] += quota.requests_today
            stats[key_display]['total_tokens_today'] += quota.tokens_today
            stats[key_display]['total_errors'] += quota.error_count

        # Add unused API keys (those not yet used with any model)
        # This ensures we show total capacity, not just used keys
        used_key_hashes = set()
        for quota in quotas:
            used_key_hashes.add(quota.api_key_hash)

        for api_key in self.config.api_keys:
            key_hash = ApiKeyQuota.hash_api_key(api_key)
            if key_hash not in used_key_hashes:
                key_display = f"{key_hash[:8]}..."
                stats[key_display] = {
                    'models': {},
                    'total_requests_today': 0,
                    'total_tokens_today': 0,
                    'total_errors': 0,
                    'daily_limit_per_model': self.config.max_requests_per_day
                }

        # Add rate limiter statistics
        stats['_rate_limiter'] = google_genai_funnel.stats

        return stats