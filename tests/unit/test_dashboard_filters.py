from types import SimpleNamespace

import pytest

from smolrouter.dashboard_filters import filter_request_logs, matches_dashboard_filter, parse_dashboard_filter_query


def _make_log(**overrides):
    payload = {
        "source_ip": "192.168.1.100",
        "provider_id": "google-gen-ai",
        "service_type": "google-genai",
        "original_model": "gemma3-12b",
        "mapped_model": "gemma3-12b-it",
        "path": "/v1/chat/completions",
        "method": "POST",
        "status_code": 200,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


@pytest.mark.parametrize(
    ("query_text", "matches"),
    [
        ("host:192.168.1.100", True),
        ("host:10.0.0.1", False),
        ("provider:google-gen-ai", True),
        ("provider:google-genai", True),
        ("provider:anthropic", False),
        ("model:gemma3-12b", True),
        ("model:gemma3-12b-it", True),
        ("model:claude", False),
    ],
)
def test_matches_dashboard_filter_across_field_aliases(query_text, matches):
    log = _make_log()
    query = parse_dashboard_filter_query(query_text)

    assert matches_dashboard_filter(log, query) is matches


def test_filter_request_logs_ignores_blank_field_values():
    query = parse_dashboard_filter_query("provider:google")
    logs = [
        _make_log(provider_id=None, service_type=None),
        _make_log(provider_id="google-gen-ai", service_type=None),
    ]

    filtered = filter_request_logs(logs, query)

    assert filtered == [logs[1]]