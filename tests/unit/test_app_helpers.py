"""Unit tests for the pure helper functions in smolrouter.app.

These functions carry the bulk of app.py's logic (URL validation, routing,
model rewriting, token estimation, Ollama<->OpenAI conversion, log
serialization, and proxy diagnostics) but were largely untested because the
endpoint tests exercise them only incidentally. Driving them directly with
plain fakes covers a large amount of app.py without a running server.
"""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from smolrouter import app


# ==========================================================================
# validate_url
# ==========================================================================


def test_validate_url_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        app.validate_url("", "DEFAULT_UPSTREAM")


def test_validate_url_passthrough_valid():
    assert app.validate_url("http://host:8000", "X") == "http://host:8000"


def test_validate_url_adds_missing_protocol():
    assert app.validate_url("host:8000", "X") == "http://host:8000"


def test_validate_url_fixes_duplicate_protocol():
    assert app.validate_url("http://http://host:8000", "X") == "http://host:8000"


def test_validate_url_rejects_non_http_scheme():
    with pytest.raises(ValueError, match="http or https"):
        app.validate_url("ftp://host", "X")


# ==========================================================================
# load_routes_config
# ==========================================================================


def _point_routes_config_at(monkeypatch, path):
    """Point both the env var and the module-level ROUTES_CONFIG string at path."""
    monkeypatch.setenv("ROUTES_CONFIG", str(path))
    monkeypatch.setattr(app, "ROUTES_CONFIG", str(path))


def test_load_routes_config_missing_file_returns_empty(monkeypatch, tmp_path):
    _point_routes_config_at(monkeypatch, tmp_path / "does_not_exist.yaml")
    assert app.load_routes_config() == {"routes": []}


def test_load_routes_config_valid_yaml(monkeypatch, tmp_path):
    cfg = tmp_path / "routes.yaml"
    cfg.write_text("routes:\n  - match:\n      model: gpt-4\n    route:\n      upstream: http://gpu\n")
    _point_routes_config_at(monkeypatch, cfg)
    result = app.load_routes_config()
    assert len(result["routes"]) == 1
    assert result["routes"][0]["route"]["upstream"] == "http://gpu"


def test_load_routes_config_valid_json(monkeypatch, tmp_path):
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({"routes": [{"match": {}, "route": {"upstream": "http://x"}}]}))
    _point_routes_config_at(monkeypatch, cfg)
    result = app.load_routes_config()
    assert len(result["routes"]) == 1


def test_load_routes_config_missing_routes_key(monkeypatch, tmp_path):
    cfg = tmp_path / "routes.yaml"
    cfg.write_text("servers:\n  alpha: http://alpha\n")
    _point_routes_config_at(monkeypatch, cfg)
    assert app.load_routes_config() == {"routes": []}


def test_load_routes_config_routes_not_a_list(monkeypatch, tmp_path):
    cfg = tmp_path / "routes.yaml"
    cfg.write_text("routes: not-a-list\n")
    _point_routes_config_at(monkeypatch, cfg)
    assert app.load_routes_config() == {"routes": []}


def test_load_routes_config_malformed_yaml_returns_empty(monkeypatch, tmp_path):
    cfg = tmp_path / "routes.yaml"
    cfg.write_text("routes: [unclosed\n")
    _point_routes_config_at(monkeypatch, cfg)
    assert app.load_routes_config() == {"routes": []}


# ==========================================================================
# model pattern matching / routing
# ==========================================================================


def test_matches_model_pattern_none_matches_all():
    assert app._matches_model_pattern(None, "anything") is True


def test_matches_model_pattern_exact():
    assert app._matches_model_pattern("gpt-4", "gpt-4") is True
    assert app._matches_model_pattern("gpt-4", "gpt-3") is False


def test_matches_model_pattern_regex():
    assert app._matches_model_pattern("/gpt-.*/", "gpt-4o") is True
    assert app._matches_model_pattern("/^claude/", "gpt-4o") is False


def test_matches_model_pattern_non_string_compares_equal():
    assert app._matches_model_pattern(42, "42") is False
    assert app._matches_model_pattern(42, 42) is True


def test_route_matches_request_host_mismatch():
    assert app._route_matches_request({"source_host": "a"}, "b", "m") is False


def test_route_matches_request_matches():
    assert app._route_matches_request({"model": "gpt-4"}, "h", "gpt-4") is True


def test_find_route_returns_default_when_no_routes(monkeypatch):
    monkeypatch.setattr(app, "ROUTES_CONFIG_DATA", {"routes": []})
    monkeypatch.setattr(app, "DEFAULT_UPSTREAM", "http://default:9000")
    upstream, override = app.find_route("1.2.3.4", "gpt-4")
    assert upstream == "http://default:9000"
    assert override is None


def test_find_route_matches_with_override(monkeypatch):
    monkeypatch.setattr(
        app,
        "ROUTES_CONFIG_DATA",
        {"routes": [{"match": {"model": "gpt-4"}, "route": {"upstream": "http://gpu", "model": "llama"}}]},
    )
    upstream, override = app.find_route("h", "gpt-4")
    assert upstream == "http://gpu"
    assert override == "llama"


def test_find_route_skips_route_without_upstream(monkeypatch):
    monkeypatch.setattr(app, "ROUTES_CONFIG_DATA", {"routes": [{"match": {"model": "gpt-4"}, "route": {}}]})
    monkeypatch.setattr(app, "DEFAULT_UPSTREAM", "http://default")
    upstream, override = app.find_route("h", "gpt-4")
    assert upstream == "http://default"


# ==========================================================================
# rewrite_model
# ==========================================================================


def test_rewrite_model_exact_match(monkeypatch):
    monkeypatch.setattr(app, "MODEL_MAP", {"gpt-4": "local-llama"})
    assert app.rewrite_model("gpt-4") == "local-llama"


def test_rewrite_model_regex_expand(monkeypatch):
    monkeypatch.setattr(app, "MODEL_MAP", {"/gpt-(.*)/": r"local-\1"})
    assert app.rewrite_model("gpt-4o") == "local-4o"


def test_rewrite_model_no_match_returns_original(monkeypatch):
    monkeypatch.setattr(app, "MODEL_MAP", {})
    assert app.rewrite_model("unknown") == "unknown"


# ==========================================================================
# should_strip_thinking_for_provider
# ==========================================================================


@pytest.mark.parametrize(
    "ptype,url,expected",
    [
        ("google-genai", "https://x", False),
        ("anthropic", "https://x", False),
        ("openai", "https://api.openai.com/v1", False),
        ("openai", "https://openai.azure.com", False),
        ("openai", "http://localhost:1234", True),
        ("ollama", "http://localhost:11434", True),
        ("mystery", "http://x", False),
    ],
)
def test_should_strip_thinking_for_provider(ptype, url, expected):
    assert app.should_strip_thinking_for_provider(ptype, url) is expected


# ==========================================================================
# token estimation
# ==========================================================================


def test_estimate_prompt_tokens_empty():
    assert app._estimate_prompt_tokens_from_request_body(None) == 0


def test_estimate_prompt_tokens_from_json():
    body = json.dumps({"messages": [{"role": "user", "content": "hello world"}]}).encode()
    assert app._estimate_prompt_tokens_from_request_body(body) > 0


def test_estimate_prompt_tokens_from_invalid_json_falls_back():
    assert app._estimate_prompt_tokens_from_request_body(b"plain text here") >= 0


def test_extract_completion_text_message_and_text():
    data = {
        "choices": [
            {"message": {"content": "abc"}},
            {"text": "def"},
            {"message": {"content": None}},
        ]
    }
    assert app._extract_completion_text(data) == "abcdef"


def test_estimate_completion_tokens_empty():
    assert app._estimate_completion_tokens_from_response_body(None) == 0


def test_estimate_completion_tokens_ollama_response_field():
    body = json.dumps({"response": "some text"}).encode()
    assert app._estimate_completion_tokens_from_response_body(body) > 0


def test_estimate_completion_tokens_openai_choices():
    body = json.dumps({"choices": [{"message": {"content": "hi there"}}]}).encode()
    assert app._estimate_completion_tokens_from_response_body(body) > 0


def test_estimate_completion_tokens_non_json_text():
    assert app._estimate_completion_tokens_from_response_body(b"raw text") > 0


def test_calculate_token_counts_uses_usage_when_present():
    data = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    prompt, completion, total = app._calculate_token_counts(data, None, None)
    assert (prompt, completion, total) == (10, 5, 15)


def test_calculate_token_counts_estimates_without_usage():
    req = json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode()
    resp = json.dumps({"choices": [{"message": {"content": "world"}}]}).encode()
    prompt, completion, total = app._calculate_token_counts({}, req, resp)
    assert total == prompt + completion
    assert prompt > 0 and completion > 0


# ==========================================================================
# misc small helpers
# ==========================================================================


def test_serialize_json_bytes():
    assert app._serialize_json_bytes({"a": 1}) == b'{"a": 1}'
    assert app._serialize_json_bytes(None) is None


def test_decode_log_body():
    assert app._decode_log_body(None) is None
    assert app._decode_log_body(b"hello") == "hello"
    assert app._decode_log_body(123) == "123"


def test_build_request_tracking_headers():
    assert app._build_request_tracking_headers(None) == {}
    assert app._build_request_tracking_headers(SimpleNamespace()) == {}
    entry = SimpleNamespace(request_id="abc-123")
    assert app._build_request_tracking_headers(entry) == {"x-smolrouter-uuid": "abc-123"}


# ==========================================================================
# response normalization (depends on module flags)
# ==========================================================================


def test_normalize_response_text_strips_thinking(monkeypatch):
    monkeypatch.setattr(app, "STRIP_THINKING", True)
    monkeypatch.setattr(app, "STRIP_JSON_MARKDOWN", False)
    out = app._normalize_response_text("<think>secret</think>answer")
    assert "secret" not in out
    assert "answer" in out


def test_normalize_openai_choice_message(monkeypatch):
    monkeypatch.setattr(app, "STRIP_THINKING", True)
    monkeypatch.setattr(app, "STRIP_JSON_MARKDOWN", False)
    choice = {"message": {"content": "<think>x</think>hi"}}
    app._normalize_openai_choice(choice)
    assert choice["message"]["content"].strip() == "hi"


def test_normalize_openai_choice_text(monkeypatch):
    monkeypatch.setattr(app, "STRIP_THINKING", True)
    monkeypatch.setattr(app, "STRIP_JSON_MARKDOWN", False)
    choice = {"text": "<think>x</think>done"}
    app._normalize_openai_choice(choice)
    assert choice["text"].strip() == "done"


def test_normalize_openai_response_content_noop_when_flags_off(monkeypatch):
    monkeypatch.setattr(app, "STRIP_THINKING", False)
    monkeypatch.setattr(app, "STRIP_JSON_MARKDOWN", False)
    data = {"choices": [{"message": {"content": "<think>x</think>hi"}}]}
    app._normalize_openai_response_content(data)
    # Unchanged because both flags disabled
    assert data["choices"][0]["message"]["content"] == "<think>x</think>hi"


# ==========================================================================
# Ollama <-> OpenAI conversion
# ==========================================================================


def test_build_ollama_openai_payload_chat(monkeypatch):
    monkeypatch.setattr(app, "ROUTES_CONFIG_DATA", {"routes": []})
    monkeypatch.setattr(app, "DEFAULT_UPSTREAM", "http://default")
    monkeypatch.setattr(app, "MODEL_MAP", {})
    monkeypatch.setattr(app, "DISABLE_THINKING", False)
    payload, upstream, original, final = app._build_ollama_openai_payload(
        "/api/chat", "1.2.3.4", {"model": "llama", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert payload["model"] == "llama"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert upstream == "http://default"
    assert original == "llama" and final == "llama"


def test_build_ollama_openai_payload_generate_wraps_prompt(monkeypatch):
    monkeypatch.setattr(app, "ROUTES_CONFIG_DATA", {"routes": []})
    monkeypatch.setattr(app, "DEFAULT_UPSTREAM", "http://default")
    monkeypatch.setattr(app, "MODEL_MAP", {})
    monkeypatch.setattr(app, "DISABLE_THINKING", True)
    payload, _, _, _ = app._build_ollama_openai_payload(
        "/api/generate", "ip", {"model": "llama", "prompt": "tell me a joke"}
    )
    assert payload["messages"][0] == {"role": "user", "content": "tell me a joke"}
    # DISABLE_THINKING appends a /no_think system message
    assert payload["messages"][-1] == {"role": "system", "content": "/no_think"}


def test_extract_openai_choice_content_variants():
    assert app._extract_openai_choice_content({"message": {"content": "m"}}) == "m"
    assert app._extract_openai_choice_content({"text": "t"}) == "t"
    assert app._extract_openai_choice_content({"delta": {"content": "d"}}, streaming=True) == "d"
    assert app._extract_openai_choice_content({}) == ""


def test_build_ollama_response(monkeypatch):
    monkeypatch.setattr(app, "STRIP_THINKING", False)
    monkeypatch.setattr(app, "STRIP_JSON_MARKDOWN", False)
    openai_data = {"created": 123, "choices": [{"message": {"content": "hello"}}]}
    result = app._build_ollama_response("llama", openai_data)
    assert result["model"] == "llama"
    assert result["response"] == "hello"
    assert result["done"] is True
    assert result["done_reason"] == "stop"


def test_ollama_done_chunk_is_terminal():
    chunk = app._ollama_done_chunk("llama")
    assert chunk.endswith(b"\n")
    parsed = json.loads(chunk)
    assert parsed["done"] is True
    assert parsed["model"] == "llama"


def test_convert_openai_stream_message_done():
    chunk, is_done = app._convert_openai_stream_message("llama", "[DONE]")
    assert is_done is True
    assert json.loads(chunk)["done"] is True


def test_convert_openai_stream_message_content(monkeypatch):
    monkeypatch.setattr(app, "STRIP_THINKING", False)
    monkeypatch.setattr(app, "STRIP_JSON_MARKDOWN", False)
    data = json.dumps({"created": 1, "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]})
    chunk, is_done = app._convert_openai_stream_message("llama", data)
    assert is_done is False
    parsed = json.loads(chunk)
    assert parsed["response"] == "hi"
    assert parsed["done_reason"] == "stop"


def test_convert_openai_stream_message_invalid_json():
    chunk, is_done = app._convert_openai_stream_message("llama", "{not json")
    assert chunk is None
    assert is_done is False


def test_split_next_sse_message_single_event():
    # Realistic path: one complete SSE event per buffer -> clean split, no remainder.
    msg, rest = app._split_next_sse_message("data: hello\n\n")
    assert msg == "data: hello"
    assert rest == ""


def test_split_next_sse_message_incomplete_buffer():
    # Without a terminator the buffer is returned untouched for the next chunk.
    assert app._split_next_sse_message("partial") == (None, "partial")


def test_split_next_sse_message_multi_event_keeps_remainder_intact():
    # Two events packed into one buffer: the first is returned and the second is
    # left fully intact for the next iteration (delimiter consumed exactly).
    msg, rest = app._split_next_sse_message("data: hello\n\ndata: world\n\n")
    assert msg == "data: hello"
    assert rest == "data: world\n\n"


def test_extract_sse_data_payload():
    assert app._extract_sse_data_payload("data: {}") == "{}"
    assert app._extract_sse_data_payload("event: ping") is None


def test_consume_ollama_sse_buffer_content_event(monkeypatch):
    monkeypatch.setattr(app, "STRIP_THINKING", False)
    monkeypatch.setattr(app, "STRIP_JSON_MARKDOWN", False)
    payload = json.dumps({"choices": [{"delta": {"content": "hi"}}]})
    chunks, remaining, is_done = app._consume_ollama_sse_buffer(f"data: {payload}\n\n", "llama")
    assert is_done is False
    assert len(chunks) == 1
    assert json.loads(chunks[0])["response"] == "hi"
    assert remaining == ""


def test_consume_ollama_sse_buffer_done_event():
    chunks, remaining, is_done = app._consume_ollama_sse_buffer("data: [DONE]\n\n", "llama")
    assert is_done is True
    assert len(chunks) == 1
    assert json.loads(chunks[0])["done"] is True


def test_consume_ollama_sse_buffer_skips_non_data_lines():
    chunks, remaining, is_done = app._consume_ollama_sse_buffer("event: ping\n\n", "llama")
    assert chunks == []
    assert is_done is False


def test_consume_ollama_sse_buffer_multiple_events_in_one_buffer(monkeypatch):
    monkeypatch.setattr(app, "STRIP_THINKING", False)
    monkeypatch.setattr(app, "STRIP_JSON_MARKDOWN", False)
    payload = json.dumps({"choices": [{"delta": {"content": "hi"}}]})
    buffer = f"data: {payload}\n\ndata: [DONE]\n\n"
    chunks, remaining, is_done = app._consume_ollama_sse_buffer(buffer, "llama")
    assert is_done is True
    assert len(chunks) == 2  # content chunk + terminal done chunk
    assert json.loads(chunks[0])["response"] == "hi"
    assert json.loads(chunks[1])["done"] is True


# ==========================================================================
# Log serialization helpers
# ==========================================================================


def _make_log_entry(**overrides):
    base = dict(
        id="log-1",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        source_ip="1.2.3.4",
        path="/v1/chat/completions",
        service_type="openai",
        original_model="gpt-4",
        mapped_model="llama",
        duration_ms=42,
        request_size=100,
        response_size=200,
        status_code=200,
        error_message=None,
        completed_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        upstream_url="http://up",
        method="POST",
        is_duplicate=False,
        duplicate_count=0,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        request_body=b'{"q": 1}',
        response_body=b'{"a": 2}',
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_timestamp_now_for_log_tz_aware():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert app._timestamp_now_for_log(ts).tzinfo is not None


def test_timestamp_now_for_log_naive():
    assert app._timestamp_now_for_log(None).tzinfo is None


def test_serialize_request_log_summary():
    out = app._serialize_request_log_summary(_make_log_entry())
    assert out["id"] == "log-1"
    assert out["status_code"] == 200
    assert out["timestamp"].startswith("2026-01-01")
    assert out["duration_ms"] == 42


def test_serialize_request_log_summary_pending_computes_duration():
    entry = _make_log_entry(
        status_code="pending",
        duration_ms=None,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=2),
    )
    out = app._serialize_request_log_summary(entry)
    assert out["status_code"] is None
    assert out["duration_ms"] >= 1000  # ~2s elapsed


def test_serialize_request_log_inflight_flag():
    entry = _make_log_entry(completed_at=None)
    out = app._serialize_request_log(entry)
    assert out["is_inflight"] is True
    assert out["upstream"] == "http://up"


def test_serialize_duplicate_request_log():
    out = app._serialize_duplicate_request_log(_make_log_entry())
    assert set(out) == {"id", "timestamp", "status_code", "source_ip"}


def test_serialize_request_detail_response():
    entry = _make_log_entry()
    dupe_info = {"is_duplicate": False, "duplicates": []}
    out = app._serialize_request_detail_response(entry, dupe_info)
    assert out["request_body"] == '{"q": 1}'
    assert out["response_body"] == '{"a": 2}'
    assert out["duplicate"] is dupe_info
    assert out["original_model"] == "gpt-4"


def test_serialize_performance_point_prefers_mapped_model():
    out = app._serialize_performance_point(_make_log_entry())
    assert out["model"] == "llama"  # mapped_model wins
    assert out["total_tokens"] == 15


def test_serialize_performance_point_falls_back_to_original():
    out = app._serialize_performance_point(_make_log_entry(mapped_model=None))
    assert out["model"] == "gpt-4"


def test_has_performance_metrics():
    assert app._has_performance_metrics(_make_log_entry()) is True
    assert app._has_performance_metrics(_make_log_entry(prompt_tokens=None)) is False
    assert app._has_performance_metrics(_make_log_entry(completed_at=None)) is False


def test_is_within_performance_window():
    recent = _make_log_entry(timestamp=datetime.now(timezone.utc) - timedelta(hours=1))
    old = _make_log_entry(timestamp=datetime.now(timezone.utc) - timedelta(hours=48))
    assert app._is_within_performance_window(recent, hours=24) is True
    assert app._is_within_performance_window(old, hours=24) is False
    assert app._is_within_performance_window(_make_log_entry(timestamp=None), hours=24) is False


def test_matches_performance_filters():
    entry = _make_log_entry()
    assert app._matches_performance_filters(entry, None, None) is True
    assert app._matches_performance_filters(entry, "llama", None) is True
    assert app._matches_performance_filters(entry, "other", None) is False
    assert app._matches_performance_filters(entry, None, "openai") is True
    assert app._matches_performance_filters(entry, None, "ollama") is False


# ==========================================================================
# Proxy diagnostics
# ==========================================================================


class _FakeProxyConfig:
    def __init__(self, url):
        self._url = url

    def to_httpx_proxy(self):
        return self._url


def test_proxy_config_to_url():
    assert app._proxy_config_to_url(None) is None
    assert app._proxy_config_to_url(_FakeProxyConfig("http://p:8080")) == "http://p:8080"
    assert app._proxy_config_to_url("http://literal") == "http://literal"


def test_mask_proxy_url_strips_credentials():
    masked = app._mask_proxy_url("http://user:pass@proxy:8080")
    assert "user" not in masked
    assert "pass" not in masked
    assert "proxy:8080" in masked


def test_mask_proxy_url_passthrough_and_none():
    assert app._mask_proxy_url(None) is None
    assert app._mask_proxy_url("not-a-url") == "not-a-url"


def test_build_proxy_entry_defaults():
    entry = app._build_proxy_entry("L", kind="pool", url="http://p")
    assert entry["label"] == "L"
    assert entry["kind"] == "pool"
    assert entry["status"] == "unknown"
    assert entry["failure_count"] == 0


def test_summarize_proxy_entries_counts():
    entries = [
        app._build_proxy_entry("a", kind="pool", url="u", status="healthy"),
        app._build_proxy_entry("b", kind="pool", url="u", status="unhealthy"),
        app._build_proxy_entry("c", kind="direct", url="direct", status="direct"),
    ]
    summary = app._summarize_proxy_entries(entries)
    assert summary["entry_count"] == 3
    assert summary["healthy_count"] == 1
    assert summary["unhealthy_count"] == 1
    assert summary["direct_entry_count"] == 1
    assert summary["proxy_count"] == 2  # non-direct


def test_build_generic_default_proxy_entry_present_and_absent():
    cfg = SimpleNamespace(proxy_config=_FakeProxyConfig("http://user:pw@d:8080"))
    entry = app._build_generic_default_proxy_entry(cfg)
    assert entry["kind"] == "default"
    assert "user" not in entry["url"]

    assert app._build_generic_default_proxy_entry(SimpleNamespace(proxy_config=None)) is None


def test_build_generic_model_override_entries():
    cfg = SimpleNamespace(
        per_model_proxy={
            "model-b": _FakeProxyConfig("http://b:1"),
            "model-a": _FakeProxyConfig("http://a:1"),
            "model-none": None,
        }
    )
    entries = app._build_generic_model_override_entries(cfg)
    # Sorted by model name, None proxy filtered out
    assert [e["model_name"] for e in entries] == ["model-a", "model-b"]


def test_build_generic_pool_entries_marks_direct_and_selected():
    provider = SimpleNamespace(_proxy_pool_index=1)
    cfg = SimpleNamespace(proxy_pool=[None, _FakeProxyConfig("http://p:1")])
    entries = app._build_generic_pool_entries(provider, cfg)
    assert entries[0]["status"] == "direct"
    assert entries[0]["selected_next"] is False
    assert entries[1]["kind"] == "pool"
    assert entries[1]["selected_next"] is True  # index 1 selected


def test_build_generic_provider_proxy_diagnostics_unconfigured():
    provider = SimpleNamespace(
        config=None,
        get_provider_id=lambda: "pid",
        get_provider_type=lambda: "openai",
    )
    diag = app._build_generic_provider_proxy_diagnostics(provider)
    assert diag["configured"] is False
    assert diag["provider_id"] == "pid"
    assert diag["summary"]["entry_count"] == 0


def test_build_generic_provider_proxy_diagnostics_configured():
    cfg = SimpleNamespace(
        proxy_config=_FakeProxyConfig("http://d:8080"),
        per_model_proxy={},
        proxy_pool=[_FakeProxyConfig("http://p:1")],
        proxy_pool_enabled=True,
    )
    provider = SimpleNamespace(
        config=cfg,
        _proxy_pool_index=0,
        get_provider_id=lambda: "pid",
        get_provider_type=lambda: "openai",
    )
    diag = app._build_generic_provider_proxy_diagnostics(provider)
    assert diag["configured"] is True
    assert diag["pool_enabled"] is True
    assert diag["next_pool_index"] == 1  # 0-based index + 1
    assert diag["default_proxy"] is not None


def test_build_proxy_configuration_report_filters_unconfigured():
    configured = SimpleNamespace(
        config=SimpleNamespace(
            proxy_config=_FakeProxyConfig("http://d:8080"),
            per_model_proxy={},
            proxy_pool=[],
            proxy_pool_enabled=False,
        ),
        _proxy_pool_index=None,
        get_provider_id=lambda: "configured",
        get_provider_type=lambda: "openai",
    )
    unconfigured = SimpleNamespace(
        config=None,
        get_provider_id=lambda: "bare",
        get_provider_type=lambda: "openai",
    )
    providers, summary = app._build_proxy_configuration_report([configured, unconfigured])
    ids = [p["provider_id"] for p in providers]
    assert "configured" in ids
    assert "bare" not in ids  # filtered because not configured


def test_build_proxy_configuration_report_uses_provider_diagnostics_hook():
    custom = SimpleNamespace(
        get_proxy_diagnostics=lambda: {"configured": True, "provider_id": "custom", "summary": {}},
        get_provider_id=lambda: "custom",
    )
    providers, _ = app._build_proxy_configuration_report([custom])
    assert providers[0]["provider_id"] == "custom"
