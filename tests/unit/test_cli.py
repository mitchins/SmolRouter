"""Unit tests for smolrouter.cli argument parsing and main() wiring."""

import os
from unittest.mock import patch

import pytest

from smolrouter.cli import _build_parser, main


@pytest.fixture(autouse=True)
def clean_listen_env(monkeypatch):
    for key in ("ROUTES_CONFIG", "LISTEN_HOST", "LISTEN_PORT", "RELOAD"):
        monkeypatch.delenv(key, raising=False)
    yield


# --------------------------------------------------------------------------
# _build_parser defaults
# --------------------------------------------------------------------------


def test_parser_defaults():
    args = _build_parser().parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 1234
    assert args.reload is False
    assert args.routes_config is None


def test_parser_reads_env_defaults(monkeypatch):
    monkeypatch.setenv("LISTEN_HOST", "0.0.0.0")  # NOSONAR
    monkeypatch.setenv("LISTEN_PORT", "9999")
    args = _build_parser().parse_args([])
    assert args.host == "0.0.0.0"  # NOSONAR
    assert args.port == 9999


def test_parser_explicit_flags():
    args = _build_parser().parse_args(
        ["--config", "routes.yaml", "--host", "1.2.3.4", "--port", "8080", "--reload"]
    )
    assert args.routes_config == "routes.yaml"
    assert args.host == "1.2.3.4"
    assert args.port == 8080
    assert args.reload is True


# --------------------------------------------------------------------------
# main()
# --------------------------------------------------------------------------


def test_main_without_reload_imports_app_and_runs():
    fake_app = object()
    with (
        patch("uvicorn.run") as run,
        patch.dict("sys.modules"),
    ):
        # Patch the app object that main imports lazily
        with patch("smolrouter.app.app", fake_app):
            main(["--host", "127.0.0.1", "--port", "4321"])

    run.assert_called_once()
    # Called positionally with the imported app object
    assert run.call_args.args[0] is fake_app
    assert run.call_args.kwargs["host"] == "127.0.0.1"
    assert run.call_args.kwargs["port"] == 4321
    assert os.environ["LISTEN_HOST"] == "127.0.0.1"
    assert os.environ["LISTEN_PORT"] == "4321"


def test_main_with_reload_uses_import_string():
    with patch("uvicorn.run") as run:
        main(["--reload", "--port", "5555"])

    run.assert_called_once()
    assert run.call_args.args[0] == "smolrouter.app:app"
    assert run.call_args.kwargs["reload"] is True
    assert os.environ["RELOAD"] == "true"


def test_main_sets_routes_config_env():
    with patch("uvicorn.run"):
        with patch("smolrouter.app.app", object()):
            main(["--config", "/tmp/routes.yaml"])
    assert os.environ["ROUTES_CONFIG"] == "/tmp/routes.yaml"
