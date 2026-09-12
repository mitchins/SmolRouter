"""Pure helpers for provider-specific chat system prompt transformations."""

from dataclasses import dataclass
from typing import Any, List, Optional

from .interfaces import SystemPromptConfig


_ROUTER_SYSTEM_MARKERS = frozenset({"/no_think"})
_SYSTEM_MESSAGE_KEYS = frozenset({"role", "content"})


@dataclass(frozen=True)
class SystemPromptTransformResult:
    """Result of applying a system prompt policy to a messages list."""

    messages: List[Any]
    applied: bool = False
    skipped_reason: Optional[str] = None


def _copy_messages(messages: List[Any]) -> List[Any]:
    return [dict(message) if isinstance(message, dict) else message for message in messages]


def _classify_system_messages(messages: List[Any]):
    logical_indices = []
    logical_contents = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip().lower() != "system":
            continue

        content = message.get("content")
        if not isinstance(content, str):
            return [], [], "system message content is not a string"
        if set(message) - _SYSTEM_MESSAGE_KEYS:
            return [], [], "system message contains unsupported semantic fields"
        if content not in _ROUTER_SYSTEM_MARKERS:
            logical_indices.append(index)
            logical_contents.append(content)

    return logical_indices, logical_contents, None


def _compose_system_prompt(logical_content: str, policy: SystemPromptConfig) -> str:
    if not logical_content or policy.mode == "replace":
        return policy.content
    if policy.mode == "prepend":
        return f"{policy.content}\n\n{logical_content}"
    return f"{logical_content}\n\n{policy.content}"


def _rebuild_messages(messages: List[Any], transformed_message: dict, first_system_index: Optional[int]) -> List[Any]:
    result: List[Any] = []

    for index, message in enumerate(messages):
        if first_system_index == index:
            result.append(transformed_message)
        if isinstance(message, dict) and str(message.get("role", "")).strip().lower() == "system":
            if message.get("content") not in _ROUTER_SYSTEM_MARKERS:
                continue
        result.append(dict(message) if isinstance(message, dict) else message)

    if first_system_index is None:
        result.insert(0, transformed_message)
    return result


def transform_system_prompt_messages(
    messages: Any, policy: SystemPromptConfig
) -> SystemPromptTransformResult:
    """Apply a policy without mutating the input messages or their order.

    System messages with structured content or extra semantic fields are left
    untouched because flattening them could silently discard meaning. Router-
    owned marker messages such as ``/no_think`` are preserved separately from
    the logical application system prompt, including for ``replace``.
    """
    if not isinstance(messages, list):
        return SystemPromptTransformResult(
            messages=messages,
            skipped_reason="messages is not a list",
        )

    copied_messages = _copy_messages(messages)
    logical_indices, logical_contents, skipped_reason = _classify_system_messages(messages)
    if skipped_reason:
        return SystemPromptTransformResult(messages=copied_messages, skipped_reason=skipped_reason)

    logical_content = "\n\n".join(logical_contents)
    transformed_content = _compose_system_prompt(logical_content, policy)
    first_system_index = logical_indices[0] if logical_indices else None
    transformed_message = {"role": "system", "content": transformed_content}
    result = _rebuild_messages(messages, transformed_message, first_system_index)
    return SystemPromptTransformResult(messages=result, applied=True)
