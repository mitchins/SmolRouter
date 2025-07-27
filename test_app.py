import pytest
import httpx
from httpx import AsyncClient
from app import app, rewrite_model, strip_think_chain_from_text, MODEL_MAP
import json
import respx


def load_mock_json(filename):
    with open(f"tests/mocks/{filename}", "r") as f:
        return json.load(f)


@pytest.fixture
def mock_openai_upstream():
    with respx.mock as respx_mock:
        respx_mock.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json=load_mock_json("openai_chat_completion_non_streaming.json")
            )
        )
        respx_mock.post("http://localhost:8000/v1/completions").mock(
            return_value=httpx.Response(
                200,
                json=load_mock_json("openai_completion_non_streaming.json")
            )
        )
        respx_mock.get("http://localhost:8000/v1/models").mock(
            return_value=httpx.Response(
                200,
                json=load_mock_json("openai_list_models.json")
            )
        )
        yield respx_mock


@pytest.fixture
def mock_ollama_upstream():
    with respx.mock as respx_mock:
        respx_mock.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(
                200,
                json=load_mock_json("ollama_list_models.json")
            )
        )
        yield respx_mock


@pytest.mark.asyncio
async def test_openai_chat_completions_non_streaming(mock_openai_upstream):
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }
        )
    assert response.status_code == 200
    data = response.json()
    assert "Hello, this is a test." in data["choices"][0]["message"]["content"]
    assert "<think>" not in data["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_openai_completions_non_streaming(mock_openai_upstream):
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/completions",
            json={
                "model": "text-davinci-003",
                "prompt": "Hello",
                "stream": False
            }
        )
    assert response.status_code == 200
    data = response.json()
    assert "This is a test." in data["choices"][0]["text"]
    assert "<think>" not in data["choices"][0]["text"]


@pytest.mark.asyncio
async def test_openai_list_models(mock_openai_upstream):
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "gpt-3.5-turbo" in [m["id"] for m in data["data"]]


@pytest.mark.asyncio
async def test_ollama_generate_non_streaming(mock_openai_upstream):
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/generate",
            json={
                "model": "llama2",
                "prompt": "Tell me a joke.",
                "stream": False
            }
        )
    assert response.status_code == 200
    data = response.json()
    assert "Hello, this is a test." in data["response"]
    assert "<think>" not in data["response"]
    assert data["done"] is True
    assert data["model"] == "llama2"


@pytest.mark.asyncio
async def test_ollama_chat_non_streaming(mock_openai_upstream):
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={
                "model": "mistral",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }
        )
    assert response.status_code == 200
    data = response.json()
    assert "Hello, this is a test." in data["response"]
    assert "<think>" not in data["response"]
    assert data["done"] is True
    assert data["model"] == "mistral"


@pytest.mark.asyncio
async def test_ollama_list_models(mock_ollama_upstream):
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/tags")
    assert response.status_code == 200
    data = response.json()
    assert "llama2" in [m["name"] for m in data["models"]]


def test_rewrite_model_exact_match():
    original_model_map = MODEL_MAP.copy()
    MODEL_MAP.update({"old-model": "new-model"})
    assert rewrite_model("old-model") == "new-model"
    MODEL_MAP.clear()
    MODEL_MAP.update(original_model_map)


def test_rewrite_model_regex_match():
    original_model_map = MODEL_MAP.copy()
    MODEL_MAP.update({"/old-(.*)/": "new-\\1"})
    assert rewrite_model("old-variant") == "new-variant"
    MODEL_MAP.clear()
    MODEL_MAP.update(original_model_map)


def test_rewrite_model_no_match():
    original_model_map = MODEL_MAP.copy()
    MODEL_MAP.update({"old-model": "new-model"})
    assert rewrite_model("unmapped-model") == "unmapped-model"
    MODEL_MAP.clear()
    MODEL_MAP.update(original_model_map)


def test_strip_think_chain_from_text():
    text_with_think = "Hello <think>internal thought</think> world."
    assert strip_think_chain_from_text(text_with_think) == "Hello world."

    text_with_multiple_think = "First <think>1</think> second <think>2</think>."
    assert strip_think_chain_from_text(text_with_multiple_think) == "First second."

    text_no_think = "Just a regular sentence."
    assert strip_think_chain_from_text(text_no_think) == "Just a regular sentence."

    text_only_think = "<think>only think</think>"
    assert strip_think_chain_from_text(text_only_think) == ""

    text_think_with_newlines = "Hello <think>\ninternal\nthought\n</think> world."
    assert strip_think_chain_from_text(text_think_with_newlines) == "Hello world."


def test_model_mapping_with_environment_variables():
    """Test that model mapping works with environment variable configuration"""
    # This tests the core model mapping logic without mocking the environment
    original_model_map = MODEL_MAP.copy()
    
    # Test exact mapping
    MODEL_MAP.update({"gpt-4": "claude-3-opus", "gpt-3.5-turbo": "claude-3-sonnet"})
    assert rewrite_model("gpt-4") == "claude-3-opus"
    assert rewrite_model("gpt-3.5-turbo") == "claude-3-sonnet"
    assert rewrite_model("unmapped") == "unmapped"
    
    # Test regex mapping
    MODEL_MAP.update({"/gpt-(.*)/": "claude-3-\\1"})
    assert rewrite_model("gpt-4o") == "claude-3-4o"
    
    MODEL_MAP.clear()
    MODEL_MAP.update(original_model_map)


def test_think_chain_stripping_edge_cases():
    """Test edge cases for think chain stripping"""
    # Test nested think tags (should not happen but test anyway)
    nested = "Text <think>outer <think>inner</think> more</think> end."
    result = strip_think_chain_from_text(nested)
    assert "<think>" not in result
    assert "Text" in result and "end." in result
    
    # Test empty content after stripping
    only_think = "<think>all content</think>"
    assert strip_think_chain_from_text(only_think) == ""
    
    # Test malformed tags
    malformed = "Text <think>unclosed tag"
    assert strip_think_chain_from_text(malformed) == "Text <think>unclosed tag"