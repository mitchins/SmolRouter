"""Unit tests for the pure helper functions/methods in
smolrouter.google_genai_provider.

These cover the request/response conversion, error classification, quota-limit,
retry-delay parsing, and proxy URL helpers without making any network calls or
touching the Google SDK. The provider is constructed with a dummy api key;
none of these methods perform I/O.
"""

import base64
import io
import wave
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from smolrouter import google_genai_provider as ggp
from smolrouter.google_genai_provider import (
    GoogleGenAICompletionContext,
    GoogleGenAIConfig,
    GoogleGenAIRequestError,
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
    # Naive value is reinterpreted as Pacific without shifting the wall clock.
    assert result.tzinfo.key == "America/Los_Angeles"
    assert result.tzname() == "PST"  # Jan 1 -> standard time
    assert result.hour == 12


def test_to_pacific_datetime_naive_assume_utc():
    naive = datetime(2026, 1, 1, 12, 0, 0)
    result = _to_pacific_datetime(naive, assume_utc=True)
    # 2026-01-01 12:00 UTC -> 04:00 PST (UTC-8)
    assert result.tzinfo.key == "America/Los_Angeles"
    assert result.hour == 4


def test_to_pacific_datetime_aware_is_converted():
    aware = datetime(2026, 1, 1, 20, 0, 0, tzinfo=timezone.utc)
    result = _to_pacific_datetime(aware)
    # 2026-01-01 20:00 UTC -> 12:00 PST
    assert result.tzinfo.key == "America/Los_Angeles"
    assert result.hour == 12
    assert result.utcoffset() != aware.utcoffset()


def test_to_pacific_datetime_dst_summer():
    # June is in Pacific Daylight Time (UTC-7): 2026-06-01 12:00 UTC -> 05:00 PDT.
    result = _to_pacific_datetime(datetime(2026, 6, 1, 12, 0, 0), assume_utc=True)
    assert result.tzname() == "PDT"
    assert result.hour == 5


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


def test_convert_content_to_parts_input_audio(provider):
    content = [{"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}}]
    assert provider._convert_openai_content_to_parts(content) == [
        {"inline_data": {"mime_type": "audio/wav", "data": "QUJD"}}
    ]


def test_audio_format_to_mime_type_defaults(provider):
    assert provider._audio_format_to_mime_type("mp3") == "audio/mpeg"
    assert provider._audio_format_to_mime_type("aiff") == "audio/aiff"
    assert provider._audio_format_to_mime_type("custom") == "audio/custom"
    assert provider._audio_format_to_mime_type("") == "audio/wav"


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


def test_convert_openai_to_genai_request_accepts_max_completion_tokens(provider):
    request = {
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 123,
    }

    _model_name, genai_request = provider._convert_openai_to_genai_request(request)

    assert genai_request["generation_config"]["max_output_tokens"] == 123


def test_convert_openai_to_genai_request_accepts_responses_string_input(provider):
    request = {
        "model": "gemini-2.5-flash",
        "input": "hello from responses",
        "max_output_tokens": 77,
    }

    model_name, genai_request = provider._convert_openai_to_genai_request(request, endpoint="/v1/responses")

    assert model_name == "gemini-2.5-flash"
    assert genai_request["contents"] == [{"role": "user", "parts": [{"text": "hello from responses"}]}]
    assert genai_request["generation_config"]["max_output_tokens"] == 77


def test_convert_openai_to_genai_request_accepts_embeddings_input(provider):
    request = {
        "model": "gemini-embedding-001",
        "input": ["hello", "world"],
        "dimensions": 768,
    }

    model_name, genai_request = provider._convert_openai_to_genai_request(request, endpoint="/v1/embeddings")

    assert model_name == "gemini-embedding-001"
    assert genai_request["contents"] == ["hello", "world"]
    assert genai_request["config"]["output_dimensionality"] == 768


def test_convert_openai_to_genai_request_accepts_responses_multimodal_input(provider):
    request = {
        "model": "gemini-2.5-flash",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "transcribe this"},
                    {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}},
                ],
            }
        ],
    }

    _model_name, genai_request = provider._convert_openai_to_genai_request(request, endpoint="/v1/responses")

    assert genai_request["contents"] == [
        {
            "role": "user",
            "parts": [
                {"text": "transcribe this"},
                {"inline_data": {"mime_type": "audio/wav", "data": "QUJD"}},
            ],
        }
    ]


def test_convert_openai_to_genai_request_prepends_responses_instructions(provider):
    request = {
        "model": "gemini-2.5-flash",
        "input": "hello from responses",
        "instructions": "reply tersely",
    }

    _model_name, genai_request = provider._convert_openai_to_genai_request(request, endpoint="/v1/responses")

    assert genai_request["contents"] == [
        {"role": "user", "parts": [{"text": "System: reply tersely"}]},
        {"role": "user", "parts": [{"text": "hello from responses"}]},
    ]


def test_convert_openai_to_genai_request_no_messages_raises(provider):
    with pytest.raises(ValueError, match="No messages"):
        provider._convert_openai_to_genai_request({"model": "x", "messages": []})


def test_is_tts_request_detects_modalities_audio(provider):
    request = {"model": "gemini-3.1-flash-tts-preview", "messages": [], "modalities": ["audio"]}
    assert provider._is_tts_request(request) is True


def test_is_tts_request_detects_response_format_audio(provider):
    request = {"model": "gemini-3.1-flash-tts-preview", "messages": [], "response_format": {"type": "audio"}}
    assert provider._is_tts_request(request) is True


def test_build_tts_generation_config_uses_default_voice(provider):
    config = provider._build_tts_generation_config({"audio": {"format": "wav"}})

    assert config.response_modalities == ["AUDIO"]
    assert config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Kore"


def test_build_tts_generation_config_uses_supplied_voice(provider):
    config = provider._build_tts_generation_config({"audio": {"voice": "Puck", "format": "wav"}})

    assert config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Puck"


def test_get_tts_audio_format_defaults_when_explicitly_null(provider):
    assert provider._get_tts_audio_format({"audio": {"format": None}}) == "pcm"


def test_build_tts_generation_config_supports_two_speakers(provider):
    config = provider._build_tts_generation_config(
        {
            "audio": {
                "format": "wav",
                "speakers": [
                    {"speaker": "Joe", "voice": "Kore"},
                    {"speaker": "Jane", "voice": "Puck"},
                ],
            }
        }
    )

    speaker_configs = config.speech_config.multi_speaker_voice_config.speaker_voice_configs
    assert len(speaker_configs) == 2
    assert speaker_configs[0].speaker == "Joe"
    assert speaker_configs[0].voice_config.prebuilt_voice_config.voice_name == "Kore"
    assert speaker_configs[1].speaker == "Jane"
    assert speaker_configs[1].voice_config.prebuilt_voice_config.voice_name == "Puck"


def test_validate_tts_request_rejects_three_speakers(provider):
    request = {
        "model": "gemini-3.1-flash-tts-preview",
        "messages": [{"role": "user", "content": "hello"}],
        "modalities": ["audio"],
        "audio": {
            "format": "wav",
            "speakers": [
                {"speaker": "Joe", "voice": "Kore"},
                {"speaker": "Jane", "voice": "Puck"},
                {"speaker": "Sam", "voice": "Leda"},
            ],
        },
    }

    with pytest.raises(ValueError, match="at most 2 speakers"):
        provider._validate_tts_request(request, endpoint="/v1/chat/completions")


def test_validate_tts_request_rejects_mixed_output_modalities(provider):
    request = {
        "model": "gemini-3.1-flash-tts-preview",
        "messages": [{"role": "user", "content": "hello"}],
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
    }

    with pytest.raises(ValueError, match="audio-only output"):
        provider._validate_tts_request(request, endpoint="/v1/chat/completions")


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}],
        [{"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}}],
    ],
)
def test_validate_tts_request_rejects_non_text_input(provider, content):
    request = {
        "model": "gemini-3.1-flash-tts-preview",
        "messages": [{"role": "user", "content": content}],
        "modalities": ["audio"],
        "audio": {"format": "wav"},
    }

    with pytest.raises(ValueError, match="only support text input"):
        provider._validate_tts_request(request, endpoint="/v1/chat/completions")


def test_extract_tts_text_from_message_defaults_null_role_to_user(provider):
    message = {"role": None, "content": "hello"}
    assert provider._extract_tts_text_from_message(message) == "hello"


def test_extract_tts_text_from_message_ignores_null_text(provider):
    message = {"role": "user", "content": [{"type": "text", "text": None}]}
    assert provider._extract_tts_text_from_message(message) == ""


# ==========================================================================
# image generation validation
# ==========================================================================


def test_image_generation_model_allowlist(provider):
    assert provider._is_image_generation_model("gemini-3.1-flash-image")
    assert not provider._is_image_generation_model("gemini-2.0-flash")
    assert provider._is_imagen_generation_model("imagen-4.0-generate-001")
    assert provider._is_imagen_generation_model("imagen-4.0-fast-generate-001")
    assert not provider._is_imagen_generation_model("gemini-2.0-flash")


def test_validate_image_request_requires_supported_model(provider):
    context = GoogleGenAICompletionContext(
        original_model="gemini-2.0-flash",
        model_name="gemini-2.0-flash",
        observation_id="obs-1",
        api_key="test-key",
        api_key_suffix="abcd1234",
    )

    with pytest.raises(
        GoogleGenAIRequestError,
        match="model 'gemini-2.0-flash' does not support image generation",
    ):
        provider._validate_image_generation_request(
            {"model": "gemini-2.0-flash", "prompt": "sunset"},
            context,
        )


@pytest.mark.parametrize(
    "image_request, expected_message",
    [
        ({"model": "gemini-2.5-flash-image"}, "require a non-empty prompt"),
        ({"model": "gemini-2.5-flash-image", "prompt": ""}, "require a non-empty prompt"),
        ({"model": "gemini-2.5-flash-image", "prompt": "ok", "response_format": "base64"}, "response_format"),
        (
            {"model": "gemini-2.5-flash-image", "prompt": "ok", "n": 0},
            "n must be a positive integer",
        ),
    ],
)
def test_validate_image_request_rejects_invalid_fields(provider, image_request, expected_message):
    context = GoogleGenAICompletionContext(
        original_model=image_request.get("model", ""),
        model_name=image_request.get("model", ""),
        observation_id="obs-1",
        api_key="test-key",
        api_key_suffix="abcd1234",
    )

    with pytest.raises(GoogleGenAIRequestError, match=expected_message):
        provider._validate_image_generation_request(image_request, context)


def test_validate_image_request_rejects_imagen_url_response_format(provider):
    context = GoogleGenAICompletionContext(
        original_model="imagen-4.0-generate-001",
        model_name="imagen-4.0-generate-001",
        observation_id="obs-1",
        api_key="test-key",
        api_key_suffix="abcd1234",
    )

    with pytest.raises(
        GoogleGenAIRequestError,
        match="response_format='url' is unsupported for Imagen models",
    ):
        provider._validate_image_generation_request(
            {
                "model": "imagen-4.0-generate-001",
                "prompt": "sunset",
                "response_format": "url",
            },
            context,
        )


def test_validate_imagen_request_rejects_non_exact_size_without_native_config(provider):
    context = GoogleGenAICompletionContext(
        original_model="imagen-4.0-generate-001",
        model_name="imagen-4.0-generate-001",
        observation_id="obs-1",
        api_key="test-key",
        api_key_suffix="abcd1234",
    )

    with pytest.raises(
        GoogleGenAIRequestError,
        match="only support top-level sizes that map to supported provider aspect ratios",
    ):
        provider._validate_image_generation_request(
            {
                "model": "imagen-4.0-generate-001",
                "prompt": "sunset",
                "size": "1920x1080",
            },
            context,
        )


@pytest.mark.parametrize(
    ("request_size", "expected_aspect_ratio"),
    [
        ("1024x1024", "1:1"),
        ("1024x1536", "3:4"),
        ("1536x1024", "4:3"),
    ],
)
def test_build_imagen_generation_request_maps_supported_size(provider, request_size, expected_aspect_ratio):
    context = GoogleGenAICompletionContext(
        original_model="imagen-4.0-generate-001",
        model_name="imagen-4.0-generate-001",
        observation_id="obs-1",
        api_key="test-key",
        api_key_suffix="abcd1234",
    )

    image_request = provider._build_imagen_generation_request(
        {
            "model": "imagen-4.0-generate-001",
            "prompt": "portrait of a robot",
            "size": request_size,
            "n": 2,
            "response_format": "b64_json",
            "extra_body": {
                "google": {"imagen": {"image_size": "2K", "aspect_ratio": expected_aspect_ratio}}
            },
        },
        context,
    )

    assert image_request["instances"] == [{"prompt": "portrait of a robot"}]
    assert image_request["parameters"]["sampleCount"] == 2
    assert image_request["parameters"]["imageSize"] == "2K"
    assert image_request["parameters"]["aspectRatio"] == expected_aspect_ratio


def test_build_imagen_generation_request_defaults_to_openai_single_image(provider):
    context = GoogleGenAICompletionContext(
        original_model="imagen-4.0-generate-001",
        model_name="imagen-4.0-generate-001",
        observation_id="obs-1",
        api_key="test-key",
        api_key_suffix="abcd1234",
    )

    image_request = provider._build_imagen_generation_request(
        {
            "model": "imagen-4.0-generate-001",
            "prompt": "sunset",
            "size": "1024x1024",
        },
        context,
    )

    assert image_request["parameters"]["sampleCount"] == 1


def test_normalize_imagen_parameters_merges_output_options(provider):
    normalized = provider._normalize_imagen_parameters(
        {
            "output_mime_type": "image/jpeg",
            "outputOptions": {"compressionQuality": 87},
        }
    )

    assert normalized["outputOptions"] == {
        "mimeType": "image/jpeg",
        "compressionQuality": 87,
    }


def test_validate_imagen_request_rejects_unsupported_native_parameter(provider):
    context = GoogleGenAICompletionContext(
        original_model="imagen-4.0-generate-001",
        model_name="imagen-4.0-generate-001",
        observation_id="obs-1",
        api_key="test-key",
        api_key_suffix="abcd1234",
    )

    with pytest.raises(GoogleGenAIRequestError, match="unsupported Imagen parameter"):
        provider._validate_image_generation_request(
            {
                "model": "imagen-4.0-generate-001",
                "prompt": "sunset",
                "extra_body": {"google": {"imagen": {"sample_count": 4}}},
            },
            context,
        )


def test_build_imagen_generation_request_normalizes_person_generation(provider):
    context = GoogleGenAICompletionContext(
        original_model="imagen-4.0-generate-001",
        model_name="imagen-4.0-generate-001",
        observation_id="obs-1",
        api_key="test-key",
        api_key_suffix="abcd1234",
    )

    image_request = provider._build_imagen_generation_request(
        {
            "model": "imagen-4.0-generate-001",
            "prompt": "sunset",
            "extra_body": {"google": {"imagen": {"person_generation": "ALLOW_ADULT"}}},
        },
        context,
    )

    assert image_request["parameters"]["personGeneration"] == "allow_adult"


def test_convert_imagen_response_to_openai_images(provider):
    converted = provider._convert_imagen_response_to_openai_images(
        {
            "predictions": [
                {"bytesBase64Encoded": "QUJD"},
                {"bytesBase64Encoded": "RUZH"},
            ]
        }
    )

    assert converted["data"] == [{"b64_json": "QUJD"}, {"b64_json": "RUZH"}]
    assert "created" in converted


def test_validate_image_request_accepts_supported_optional_fields(provider):
    context = GoogleGenAICompletionContext(
        original_model="gemini-2.5-flash-image",
        model_name="gemini-2.5-flash-image",
        observation_id="obs-1",
        api_key="test-key",
        api_key_suffix="abcd1234",
    )

    provider._validate_image_generation_request(
        {
            "model": "gemini-2.5-flash-image",
            "prompt": "sunset",
            "size": "1024x1024",
            "response_format": "b64_json",
            "user": "user-123",
            "extra_body": {"google": {"safety_settings": []}},
        },
        context,
    )


def test_build_image_generation_request_forwards_user(provider):
    context = GoogleGenAICompletionContext(
        original_model="gemini-2.5-flash-image",
        model_name="gemini-2.5-flash-image",
        observation_id="obs-1",
        api_key="test-key",
        api_key_suffix="abcd1234",
    )

    image_request = provider._build_image_generation_request(
        {
            "model": "gemini-2.5-flash-image",
            "prompt": "sunset",
            "user": "user-123",
        },
        context,
    )

    assert image_request["user"] == "user-123"


def test_validate_image_request_rejects_unknown_fields(provider):
    context = GoogleGenAICompletionContext(
        original_model="gemini-2.5-flash-image",
        model_name="gemini-2.5-flash-image",
        observation_id="obs-1",
        api_key="test-key",
        api_key_suffix="abcd1234",
    )

    with pytest.raises(GoogleGenAIRequestError, match="unsupported image request field"):
        provider._validate_image_generation_request(
            {"model": "gemini-2.5-flash-image", "prompt": "sunset", "quality": "high"},
            context,
        )


@pytest.mark.asyncio
async def test_generate_image_rejects_imagen_size_conflict_before_context_init(provider):
    provider._initialize_request_context = AsyncMock(side_effect=AssertionError("should not initialize context"))

    with pytest.raises(
        GoogleGenAIRequestError,
        match="top-level size and extra_body.google.imagen.aspect_ratio conflict",
    ):
        await provider.generate_image(
            {
                "model": "imagen-4.0-generate-001",
                "prompt": "sunset",
                "size": "1024x1024",
                "extra_body": {"google": {"imagen": {"aspect_ratio": "16:9"}}},
            }
        )

    assert provider._initialize_request_context.await_count == 0


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


def test_extract_genai_audio_bytes(provider):
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(inline_data=SimpleNamespace(mime_type="audio/pcm", data=b"\x01\x02\x03\x04")),
                        SimpleNamespace(text="ignored"),
                    ]
                )
            )
        ]
    )

    assert provider._extract_genai_audio_bytes(response) == b"\x01\x02\x03\x04"


def test_extract_genai_audio_bytes_uses_first_candidate_only(provider):
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(inline_data=SimpleNamespace(mime_type="audio/pcm", data=b"\x01\x02"))]
                )
            ),
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(inline_data=SimpleNamespace(mime_type="audio/pcm", data=b"\x03\x04"))]
                )
            ),
        ]
    )

    assert provider._extract_genai_audio_bytes(response) == b"\x01\x02"


def test_pcm_to_wav_bytes_wraps_pcm(provider):
    pcm = b"\x00\x00\x01\x00\x02\x00\x03\x00"
    wav_bytes = provider._pcm_to_wav_bytes(pcm)

    assert wav_bytes[:4] == b"RIFF"
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getframerate() == ggp.TTS_SAMPLE_RATE_HZ
        assert wav_file.getnchannels() == ggp.TTS_CHANNELS
        assert wav_file.getsampwidth() == ggp.TTS_SAMPLE_WIDTH_BYTES
        assert wav_file.readframes(4) == pcm


def test_convert_genai_to_openai_response(provider):
    meta = SimpleNamespace(prompt_token_count=3, candidates_token_count=2, total_token_count=5)
    genai_resp = SimpleNamespace(text="answer", usage_metadata=meta)
    out = provider._convert_genai_to_openai_response(genai_resp, "gemini-2.5-flash")
    assert out["id"].startswith("chatcmpl-")
    assert isinstance(out["created"], int)
    assert out["object"] == "chat.completion"
    assert out["model"] == "gemini-2.5-flash"
    assert out["choices"][0]["message"]["content"] == "answer"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["total_tokens"] == 5


def test_convert_genai_to_responses_response(provider):
    meta = SimpleNamespace(prompt_token_count=3, candidates_token_count=2, total_token_count=5)
    genai_resp = SimpleNamespace(text="answer", usage_metadata=meta)
    out = provider._convert_genai_to_responses_response(genai_resp, "gemini-2.5-flash")

    assert out["id"].startswith("resp-")
    assert out["object"] == "response"
    assert out["status"] == "completed"
    assert out["model"] == "gemini-2.5-flash"
    assert out["output_text"] == "answer"
    assert out["output"][0]["content"][0]["type"] == "output_text"
    assert out["usage"]["total_tokens"] == 5
    assert out["usage"]["input_tokens"] == 3
    assert out["usage"]["output_tokens"] == 2


def test_convert_genai_to_responses_response_uses_unique_ids(provider):
    meta = SimpleNamespace(prompt_token_count=3, candidates_token_count=2, total_token_count=5)
    genai_resp = SimpleNamespace(text="answer", usage_metadata=meta)

    out1 = provider._convert_genai_to_responses_response(genai_resp, "gemini-2.5-flash")
    out2 = provider._convert_genai_to_responses_response(genai_resp, "gemini-2.5-flash")

    assert out1["id"] != out2["id"]
    assert out1["output"][0]["id"] != out2["output"][0]["id"]


def test_convert_genai_to_openai_audio_chat_response(provider):
    meta = SimpleNamespace(prompt_token_count=3, candidates_token_count=2, total_token_count=5)
    genai_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(inline_data=SimpleNamespace(mime_type="audio/pcm", data=b"\x01\x02\x03\x04"))]
                )
            )
        ],
        usage_metadata=meta,
    )

    out = provider._convert_genai_to_openai_audio_chat_response(
        genai_resp, "gemini-3.1-flash-tts-preview", "wav"
    )

    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"]["content"] is None
    assert out["choices"][0]["message"]["audio"]["format"] == "wav"
    assert out["choices"][0]["message"]["audio"]["mime_type"] == "audio/wav"
    assert base64.b64decode(out["choices"][0]["message"]["audio"]["data"])[:4] == b"RIFF"
    assert out["usage"]["total_tokens"] == 5


def test_convert_genai_to_responses_audio_response(provider):
    meta = SimpleNamespace(prompt_token_count=3, candidates_token_count=2, total_token_count=5)
    genai_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(inline_data=SimpleNamespace(mime_type="audio/pcm", data=b"\x01\x02\x03\x04"))]
                )
            )
        ],
        usage_metadata=meta,
    )

    out = provider._convert_genai_to_responses_audio_response(
        genai_resp, "gemini-3.1-flash-tts-preview", "pcm"
    )

    assert out["object"] == "response"
    assert out["output"][0]["content"][0]["type"] == "output_audio"
    assert out["output"][0]["content"][0]["audio"]["format"] == "pcm"
    assert out["output"][0]["content"][0]["audio"]["mime_type"] == "audio/pcm"
    assert base64.b64decode(out["output"][0]["content"][0]["audio"]["data"]) == b"\x01\x02\x03\x04"
    assert out["usage"]["total_tokens"] == 5


# ==========================================================================
# error classification
# ==========================================================================


def test_is_quota_exhausted_error(provider):
    assert provider._is_quota_exhausted_error("Error 429: RESOURCE_EXHAUSTED") is True
    assert provider._is_quota_exhausted_error("quota exceeded for the day") is True
    assert provider._is_quota_exhausted_error("some other error") is False
    assert provider._is_quota_exhausted_error(None) is False


# Real per-minute (RPM) 429 body as returned by Google free tier (gemini-3.1-flash-lite).
# The quotaMetric is identical for RPM and RPD; only the quotaId distinguishes them.
RPM_429 = (
    "Google GenAI error: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
    "'You exceeded your current quota. \\n* Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 15, "
    "model: gemini-3.1-flash-lite\\nPlease retry in 21.424365033s.', 'status': 'RESOURCE_EXHAUSTED', "
    "'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': "
    "[{'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaValue': '15'}]}, "
    "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '21s'}]}}"
)

RPD_429 = (
    "Google GenAI error: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
    "'You exceeded your current quota.', 'status': 'RESOURCE_EXHAUSTED', 'details': "
    "[{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': "
    "[{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '1500'}]}]}}"
)


def test_is_per_minute_quota_error(provider):
    assert provider._is_per_minute_quota_error(RPM_429) is True
    assert provider._is_per_minute_quota_error(RPD_429) is False
    assert provider._is_per_minute_quota_error("429 quota exceeded retry in 12s") is False
    assert provider._is_per_minute_quota_error(None) is False


def test_is_per_day_quota_error(provider):
    assert provider._is_per_day_quota_error(RPD_429) is True
    assert provider._is_per_day_quota_error("requests per day exceeded") is True
    assert provider._is_per_day_quota_error(RPM_429) is False
    assert provider._is_per_day_quota_error(None) is False


def test_quota_cooldown_seconds_per_minute_uses_retry_delay(provider):
    # RPM -> honor Google's retryDelay (21s) plus the small buffer.
    cooldown = provider._quota_cooldown_seconds(RPM_429)
    assert cooldown == pytest.approx(21.424365033 + provider.RPM_COOLDOWN_BUFFER_SECONDS)


def test_quota_cooldown_seconds_per_minute_without_retry_delay(provider):
    rpm_no_delay = "429 RESOURCE_EXHAUSTED GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
    cooldown = provider._quota_cooldown_seconds(rpm_no_delay)
    assert cooldown == provider.DEFAULT_RPM_COOLDOWN_SECONDS + provider.RPM_COOLDOWN_BUFFER_SECONDS


def test_quota_cooldown_seconds_per_day_is_all_day(provider):
    # RPD and ambiguous quota errors return None -> caller benches until midnight Pacific.
    assert provider._quota_cooldown_seconds(RPD_429) is None
    assert provider._quota_cooldown_seconds("429 quota exceeded retry in 12s") is None


def test_is_invalid_key_error(provider):
    assert provider._is_invalid_key_error(status_code=403) is True
    assert provider._is_invalid_key_error("403 upgrade your account to a paid plan", status_code=403) is False
    assert provider._is_invalid_key_error("API key not valid") is True
    assert provider._is_invalid_key_error("PERMISSION_DENIED") is True
    assert provider._is_invalid_key_error("transient network blip") is False
    assert provider._is_invalid_key_error(None) is False


def test_extract_retry_delay(provider):
    assert provider._extract_retry_delay("Please retry in 20.5s") == 20.5  # pattern 1
    assert provider._extract_retry_delay("retryDelay': '30s'") == 30.0  # pattern 2
    assert provider._extract_retry_delay("Please retry after 7.5 seconds") == 7.5  # pattern 3
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
        ("imagen-4.0-fast-generate-001", 25),
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
    assert meta["description"] == "fast"
    assert meta["supported_methods"] == ["generateContent"]
    assert meta["input_token_limit"] == 1000
    assert meta["output_token_limit"] == 2000


def test_build_static_model_metadata(provider):
    data = {"name": "models/gemini-2.0-flash", "description": "d", "thinking": True, "inputTokenLimit": 5}
    meta = provider._build_static_google_model_metadata(data, ["generateContent"])
    assert meta["full_name"] == "models/gemini-2.0-flash"
    assert meta["display_name"] == "gemini-2.0-flash"  # falls back to extracted name
    assert meta["description"] == "d"
    assert meta["supported_methods"] == ["generateContent"]
    assert meta["thinking"] is True
    assert meta["input_token_limit"] == 5


def test_build_static_model_metadata_keeps_tts_context_limit_in_sync(provider):
    data = {
        "name": "models/gemini-3.1-flash-tts-preview",
        "description": "tts",
        "inputTokenLimit": 8192,
    }
    meta = provider._build_static_google_model_metadata(data, ["generateContent"])
    assert meta["supports_tts"] is True
    assert meta["max_context_tokens"] == 8192


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
