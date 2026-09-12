from unittest.mock import AsyncMock, Mock

import pytest

from smolrouter.access_control import NoAccessControl
from smolrouter.interfaces import ModelInfo, ProviderConfig, SystemPromptConfig
from smolrouter.mediator import ModelMediator
from smolrouter.providers import ProviderFactory
from smolrouter.strategies import SimpleModelStrategy
from smolrouter.system_prompts import transform_system_prompt_messages


def test_system_prompt_modes_and_missing_system_message():
    original_messages = [
        {"role": "system", "content": "Original instructions."},
        {"role": "user", "content": "Describe this image."},
    ]

    append_result = transform_system_prompt_messages(
        original_messages, SystemPromptConfig(mode="append", content="Additional instructions.")
    )
    prepend_result = transform_system_prompt_messages(
        original_messages, SystemPromptConfig(mode="prepend", content="Framing instructions.")
    )
    replace_result = transform_system_prompt_messages(
        original_messages, SystemPromptConfig(mode="replace", content="Replacement instructions.")
    )
    missing_result = transform_system_prompt_messages(
        [{"role": "user", "content": "Hello"}], SystemPromptConfig(mode="append", content="Be concise.")
    )

    assert append_result.messages[0]["content"] == "Original instructions.\n\nAdditional instructions."
    assert prepend_result.messages[0]["content"] == "Framing instructions.\n\nOriginal instructions."
    assert replace_result.messages[0]["content"] == "Replacement instructions."
    assert missing_result.messages == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]
    assert append_result.applied is True
    assert original_messages[0]["content"] == "Original instructions."


def test_system_prompt_combines_system_messages_and_preserves_other_roles():
    result = transform_system_prompt_messages(
        [
            {"role": "developer", "content": "Developer context."},
            {"role": "system", "content": "First."},
            {"role": "user", "content": "Question."},
            {"role": "system", "content": "Second."},
        ],
        SystemPromptConfig(mode="append", content="Configured."),
    )

    assert result.messages == [
        {"role": "developer", "content": "Developer context."},
        {"role": "system", "content": "First.\n\nSecond.\n\nConfigured."},
        {"role": "user", "content": "Question."},
    ]


def test_system_prompt_preserves_router_marker_when_replacing():
    result = transform_system_prompt_messages(
        [
            {"role": "system", "content": "Application instructions."},
            {"role": "user", "content": "Question."},
            {"role": "system", "content": "/no_think"},
        ],
        SystemPromptConfig(mode="replace", content="Provider instructions."),
    )

    assert result.messages == [
        {"role": "system", "content": "Provider instructions."},
        {"role": "user", "content": "Question."},
        {"role": "system", "content": "/no_think"},
    ]


def test_system_prompt_inserts_before_user_when_router_marker_is_only_system_message():
    result = transform_system_prompt_messages(
        [{"role": "user", "content": "Question."}, {"role": "system", "content": "/no_think"}],
        SystemPromptConfig(mode="append", content="Provider instructions."),
    )

    assert result.messages == [
        {"role": "system", "content": "Provider instructions."},
        {"role": "user", "content": "Question."},
        {"role": "system", "content": "/no_think"},
    ]


@pytest.mark.parametrize(
    "messages, reason",
    [
        ([{"role": "system", "content": [{"type": "text", "text": "structured"}]}], "not a string"),
        ([{"role": "system", "content": "text", "name": "named"}], "semantic fields"),
    ],
)
def test_system_prompt_skips_unsupported_system_messages_without_mutation(messages, reason):
    original_messages = [dict(message) for message in messages]

    result = transform_system_prompt_messages(
        messages, SystemPromptConfig(mode="append", content="Configured.")
    )

    assert result.applied is False
    assert reason in result.skipped_reason
    assert result.messages == original_messages
    assert messages == original_messages


def test_provider_config_coerces_and_validates_system_prompt_policies():
    config = ProviderConfig(
        name="test-provider",
        type="openai",
        url="https://example.com",
        system_prompt={"mode": "APPEND", "content": "Default."},
        per_model_system_prompt={
            "canonical-model": {"mode": "replace", "content": "Specific."},
            "opted-out-model": None,
        },
    )

    assert config.system_prompt == SystemPromptConfig(mode="append", content="Default.")
    assert config.get_system_prompt_for_model("canonical-model") == SystemPromptConfig(
        mode="replace", content="Specific."
    )
    assert config.get_system_prompt_for_model("other-model") == config.system_prompt
    assert config.get_system_prompt_for_model("opted-out-model") is None

    with pytest.raises(ValueError, match="Unsupported system prompt mode"):
        ProviderConfig(
            name="invalid",
            type="openai",
            url="https://example.com",
            system_prompt={"mode": "rewrite", "content": "Nope."},
        )
    with pytest.raises(ValueError, match="non-empty string"):
        SystemPromptConfig(mode="append", content=" ")
    with pytest.raises(ValueError, match="model names"):
        ProviderConfig(
            name="invalid",
            type="openai",
            url="https://example.com",
            per_model_system_prompt={" ": {"mode": "append", "content": "Nope."}},
        )


def test_provider_factory_coerces_yaml_system_prompt_configuration():
    providers = ProviderFactory.create_providers_from_config(
        [
            {
                "name": "test-openai",
                "type": "openai",
                "url": "https://example.com",
                "api_key": None,
                "system_prompt": {"mode": "append", "content": "Default."},
                "per_model_system_prompt": {
                    "model-a": {"mode": "prepend", "content": "Specific."},
                },
            }
        ]
    )

    assert len(providers) == 1
    assert providers[0].config.get_system_prompt_for_model("model-a") == SystemPromptConfig(
        mode="prepend", content="Specific."
    )


@pytest.mark.asyncio
async def test_mediator_applies_canonical_model_policy_to_non_streaming_and_streaming_payloads():
    provider = Mock()
    provider.config = ProviderConfig(
        name="test-openai",
        type="openai",
        url="https://example.com",
        system_prompt={"mode": "append", "content": "Provider default."},
        per_model_system_prompt={"canonical-model": {"mode": "replace", "content": "Canonical policy."}},
    )
    provider.generate_completion = AsyncMock(return_value=({"id": "chatcmpl-test"}, 200))

    resolved_model = ModelInfo(
        id="canonical-model@test-openai",
        name="canonical-model",
        provider_id="test-openai",
        provider_type="openai",
        endpoint="https://example.com",
        aliases=["incoming-alias"],
    )
    mediator = ModelMediator(Mock(), SimpleModelStrategy({}), NoAccessControl())
    mediator.resolve_model_for_request = AsyncMock(return_value=resolved_model)
    mediator._get_provider_by_id = Mock(return_value=provider)
    payload = {
        "model": "incoming-alias",
        "stream": True,
        "messages": [{"role": "system", "content": "Application policy."}, {"role": "user", "content": "Hi"}],
    }

    response_data, status_code, upstream, _metadata = await mediator.route_request(
        "127.0.0.1", "incoming-alias", payload, "/v1/chat/completions", {}, 30.0
    )

    assert response_data["id"] == "chatcmpl-test"
    assert status_code == 200
    assert upstream == "openai:test-openai"
    forwarded_payload = provider.generate_completion.await_args.args[0]
    assert forwarded_payload["model"] == "canonical-model"
    assert forwarded_payload["stream"] is True
    assert forwarded_payload["messages"] == [{"role": "system", "content": "Canonical policy."}, {"role": "user", "content": "Hi"}]
    assert payload["model"] == "incoming-alias"
    assert payload["messages"][0]["content"] == "Application policy."


@pytest.mark.asyncio
async def test_mediator_skips_system_prompt_transform_for_responses_and_audio_requests():
    provider = Mock()
    provider.config = ProviderConfig(
        name="test-openai",
        type="openai",
        url="https://example.com",
        system_prompt={"mode": "append", "content": "Configured."},
    )
    provider.generate_completion = AsyncMock(return_value=({"id": "request"}, 200))
    resolved_model = ModelInfo(
        id="model@test-openai",
        name="model",
        provider_id="test-openai",
        provider_type="openai",
        endpoint="https://example.com",
    )

    for path, extra in [
        ("/v1/responses", {"instructions": "Original instructions."}),
        ("/v1/chat/completions", {"modalities": ["text", "audio"]}),
    ]:
        mediator = ModelMediator(Mock(), SimpleModelStrategy({}), NoAccessControl())
        mediator.resolve_model_for_request = AsyncMock(return_value=resolved_model)
        mediator._get_provider_by_id = Mock(return_value=provider)
        payload = {"model": "model", "messages": [{"role": "system", "content": "Original."}], **extra}

        await mediator.route_request("127.0.0.1", "model", payload, path, {}, 30.0)

        forwarded_payload = provider.generate_completion.await_args.args[0]
        assert forwarded_payload["messages"] == [{"role": "system", "content": "Original."}]
        provider.generate_completion.reset_mock()
