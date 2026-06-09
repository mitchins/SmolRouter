"""Unit tests for the pure helper functions/methods in
smolrouter.google_genai_provider.

These cover the request/response conversion, error classification, quota-limit,
retry-delay parsing, and proxy URL helpers without making any network calls or
touching the Google SDK. The provider is constructed with a dummy api key;
none of these methods perform I/O.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from smolrouter import google_genai_provider as ggp
from smolrouter.google_genai_provider import (
    GoogleGenAIConfig,
    GoogleGenAIProvider,
    _format_optional_datetime,
    _quota_status,
    _to_pacific_datetime,
)


def _make_provider(**kwargs):
    config_kwargs = {"name": "test-google", "type": "google-genai", "enabled": True, "api_keys": ["test-key"]}
    config_kwargs.update(kwargs)
    return GoogleGenAIProvider(GoogleGenAIConfig(**config_kwargs))


@pytest.fixture
def provider():
    return _make_provider()


# ==========================================================================
# module-level datetime / quota helpers
# ==========================================================================


def test_to_pacific_datetime_naive_assume_local():
    naive = datetime(2026, 1, 1, 12, 0, 0)
    result = _to_pacific_datetime(naive, assume_utc=False)
    assert result.tzinfo is not None


def test_to_pacific_datetime_naive_assume_utc():
    naive = datetime(2026, 1, 1, 12, 0, 0)
    result = _to_pacific_datetime(naive, assume_utc=True)
    # Converted from UTC to Pacific -> wall-clock hour shifts back
    assert result.tzinfo is not None
    assert result.hour != 12


def test_to_pacific_datetime_aware_is_converted():
    aware = datetime(2026, 1, 1, 20, 0, 0, tzinfo=timezone.utc)
    result = _to_pacific_datetime(aware)
    assert result.utcoffset() != aware.utcoffset()


def test_format_optional_datetime():
    assert _format_optional_datetime(None) is None
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert _format_optional_datetime(dt) == dt.isoformat()
    assert _format_optional_datetime("raw") == "raw"


def test_quota_status():
    assert _quota_status(is_invalid=True, is_exhausted=False) == "invalid"
    assert _quota_status(is_invalid=False, is_exhausted=True) == "exhausted"
    assert _quota_status(is_invalid=False, is_exhausted=False) == "available"


# ==========================================================================
# OpenAI -> GenAI request conversion
# ==========================================================================


def test_convert_content_to_parts_string(provider):
    assert provider._convert_openai_content_to_parts("hello") == [{"text": "hello"}]


def test_convert_content_to_parts_text_list(provider):
    content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert provider._convert_openai_content_to_parts(content) == [{"text": "a"}, {"text": "b"}]


def test_convert_content_to_parts_image_data_uri(provider):
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]
    parts = provider._convert_openai_content_to_parts(content)
    assert parts == [{"inline_data": {"mime_type": "image/png", "data": "QUJD"}}]


def test_convert_content_to_parts_remote_image_skipped(provider):
    content = [{"type": "image_url", "image_url": {"url": "https://example.com/i.png"}}]
    assert provider._convert_openai_content_to_parts(content) == []


def test_convert_content_to_parts_non_list_non_str(provider):
    assert provider._convert_openai_content_to_parts(42) == []


def test_convert_message_roles(provider):
    user = provider._convert_openai_message_to_genai_content({"role": "user", "content": "hi"})
    assert user == {"role": "user", "parts": [{"text": "hi"}]}

    assistant = provider._convert_openai_message_to_genai_content({"role": "assistant", "content": "ok"})
    assert assistant["role"] == "model"

    system = provider._convert_openai_message_to_genai_content({"role": "system", "content": "be nice"})
    assert system["role"] == "user"
    assert system["parts"][0]["text"] == "System: be nice"


def test_convert_message_empty_string_still_produces_part(provider):
    # An empty string yields a (single, empty) text part, so the message is kept.
    msg = provider._convert_openai_message_to_genai_content({"role": "user", "content": ""})
    assert msg == {"role": "user", "parts": [{"text": ""}]}


def test_convert_message_no_parts_returns_none(provider):
    # Content that produces zero parts (only an unsupported remote image) -> dropped.
    content = [{"type": "image_url", "image_url": {"url": "https://example.com/i.png"}}]
    assert provider._convert_openai_message_to_genai_content({"role": "user", "content": content}) is None


def test_convert_openai_to_genai_request(provider):
    request = {
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.5,
        "max_tokens": 256,
        "top_p": 0.9,
    }
    model_name, genai_request = provider._convert_openai_to_genai_request(request)
    assert model_name == "gemini-2.5-flash"
    assert genai_request["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert genai_request["generation_config"] == {
        "temperature": 0.5,
        "max_output_tokens": 256,
        "top_p": 0.9,
    }


def test_convert_openai_to_genai_request_no_messages_raises(provider):
    with pytest.raises(ValueError, match="No messages"):
        provider._convert_openai_to_genai_request({"model": "x", "messages": []})


# ==========================================================================
# GenAI -> OpenAI response conversion
# ==========================================================================


def test_extract_genai_text_top_level(provider):
    resp = SimpleNamespace(text="direct text")
    assert provider._extract_genai_text(resp) == "direct text"


def test_extract_genai_text_from_candidates(provider):
    part = SimpleNamespace(text="part-text")
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]))
    resp = SimpleNamespace(text="", candidates=[candidate])
    assert provider._extract_genai_text(resp) == "part-text"


def test_extract_genai_text_no_candidates(provider):
    resp = SimpleNamespace(text="", candidates=None)
    assert provider._extract_genai_text(resp) == ""


def test_extract_genai_usage(provider):
    meta = SimpleNamespace(prompt_token_count=10, candidates_token_count=5, total_token_count=15)
    resp = SimpleNamespace(usage_metadata=meta)
    assert provider._extract_genai_usage(resp) == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def test_extract_genai_usage_missing(provider):
    assert provider._extract_genai_usage(SimpleNamespace(usage_metadata=None)) == {}


def test_convert_genai_to_openai_response(provider):
    meta = SimpleNamespace(prompt_token_count=3, candidates_token_count=2, total_token_count=5)
    genai_resp = SimpleNamespace(text="answer", usage_metadata=meta)
    out = provider._convert_genai_to_openai_response(genai_resp, "gemini-2.5-flash")
    assert out["object"] == "chat.completion"
    assert out["model"] == "gemini-2.5-flash"
    assert out["choices"][0]["message"]["content"] == "answer"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["total_tokens"] == 5


# ==========================================================================
# error classification
# ==========================================================================


def test_is_quota_exhausted_error(provider):
    assert provider._is_quota_exhausted_error("Error 429: RESOURCE_EXHAUSTED") is True
    assert provider._is_quota_exhausted_error("quota exceeded for the day") is True
    assert provider._is_quota_exhausted_error("some other error") is False
    assert provider._is_quota_exhausted_error(None) is False


def test_is_invalid_key_error(provider):
    assert provider._is_invalid_key_error(status_code=403) is True
    assert provider._is_invalid_key_error("API key not valid") is True
    assert provider._is_invalid_key_error("PERMISSION_DENIED") is True
    assert provider._is_invalid_key_error("transient network blip") is False
    assert provider._is_invalid_key_error(None) is False


def test_extract_retry_delay(provider):
    assert provider._extract_retry_delay("Please retry in 20.5s") == 20.5
    assert provider._extract_retry_delay("retryDelay': '30s'") == 30.0
    assert provider._extract_retry_delay("no delay mentioned") is None
    assert provider._extract_retry_delay(None) is None


def test_extract_status_code_from_exception(provider):
    assert provider._extract_status_code_from_exception(Exception("Got 403 permission")) == 403
    assert provider._extract_status_code_from_exception(Exception("429 quota")) == 429
    assert provider._extract_status_code_from_exception(Exception("401 unauthorized")) == 401
    assert provider._extract_status_code_from_exception(Exception("teapot")) is None


def test_extract_retry_after_seconds(provider):
    err = Exception("boom")
    err.errors = [{"retry_after_seconds": 12}]
    assert provider._extract_retry_after_seconds(err) == 12
    assert provider._extract_retry_after_seconds(Exception("plain")) is None


def test_is_proxy_connectivity_error(provider):
    assert provider._is_proxy_connectivity_error(httpx.ConnectError("refused")) is True
    assert provider._is_proxy_connectivity_error(Exception("Connection refused by host")) is True
    assert provider._is_proxy_connectivity_error(Exception("All connection attempts failed")) is True
    assert provider._is_proxy_connectivity_error(Exception("bad request 400")) is False


# ==========================================================================
# quota limits / model name normalization
# ==========================================================================


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gemma-3-27b", 14400),
        ("gemini-2.5-flash-lite", 1000),
        ("gemini-2.5-flash", 20),
        ("gemini-2.0-flash-exp", 5),
        ("gemini-2.0-flash", 20),
        ("gemini-1.5-pro", 50),
        ("gemini-1.5-flash", 1000),
        ("something-preview", 5),
    ],
)
def test_get_model_daily_limit(provider, model, expected):
    assert provider.get_model_daily_limit(model) == expected


def test_get_model_daily_limit_unknown_uses_config_default(provider):
    assert provider.get_model_daily_limit("totally-unknown") == provider.config.max_requests_per_day


def test_normalize_model_name_passthrough(provider):
    assert provider._normalize_model_name("gemini-2.5-flash") == "gemini-2.5-flash"


# ==========================================================================
# model info builders / google name extraction
# ==========================================================================


def test_extract_google_model_name():
    assert GoogleGenAIProvider._extract_google_model_name("models/gemini-2.5-flash") == "gemini-2.5-flash"
    assert GoogleGenAIProvider._extract_google_model_name("gemini-2.5-flash") == "gemini-2.5-flash"


def test_build_google_model_info(provider):
    info = provider._build_google_model_info("gemini-2.5-flash", {"thinking": True})
    assert info.name == "gemini-2.5-flash"
    assert info.provider_type == "google-genai"
    assert info.metadata == {"thinking": True}
    assert "gemini-2.5-flash" in info.aliases


def test_build_live_model_metadata(provider):
    model = SimpleNamespace(
        name="models/gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        description="fast",
        input_token_limit=1000,
        output_token_limit=2000,
    )
    meta = provider._build_live_google_model_metadata(model, ["generateContent"])
    assert meta["full_name"] == "models/gemini-2.5-flash"
    assert meta["display_name"] == "Gemini 2.5 Flash"
    assert meta["supported_methods"] == ["generateContent"]


def test_build_static_model_metadata(provider):
    data = {"name": "models/gemini-2.0-flash", "description": "d", "thinking": True, "inputTokenLimit": 5}
    meta = provider._build_static_google_model_metadata(data, ["generateContent"])
    assert meta["full_name"] == "models/gemini-2.0-flash"
    assert meta["display_name"] == "gemini-2.0-flash"  # falls back to extracted name
    assert meta["thinking"] is True
    assert meta["input_token_limit"] == 5


# ==========================================================================
# proxy url helpers
# ==========================================================================


class _FakeProxyConfig:
    def __init__(self, url):
        self._url = url

    def to_httpx_proxy(self):
        return self._url


def test_proxy_config_to_url(provider):
    assert provider._proxy_config_to_url(None) is None
    assert provider._proxy_config_to_url(_FakeProxyConfig("http://p:8080")) == "http://p:8080"


def test_mask_proxy_url(provider):
    assert provider._mask_proxy_url(None) is None
    masked = provider._mask_proxy_url("http://user:pass@proxy:8080")
    assert "user" not in masked and "pass" not in masked
    assert "proxy:8080" in masked
    assert provider._mask_proxy_url("not-a-url") == "not-a-url"
