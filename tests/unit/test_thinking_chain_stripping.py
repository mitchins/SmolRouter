"""
Unit tests for thinking chain stripping functionality.

Tests the provider-aware CoT removal logic with various tag formats and provider types.
"""

from smolrouter.app import strip_think_chain_from_text, should_strip_thinking_for_provider


class TestShouldStripThinkingForProvider:
    """Test the provider detection logic for thinking chain stripping."""

    def test_cloud_providers_never_strip(self):
        """Cloud providers should never have thinking chains stripped."""
        # Google GenAI
        assert not should_strip_thinking_for_provider("google-genai", "https://generativelanguage.googleapis.com")

        # Anthropic
        assert not should_strip_thinking_for_provider("anthropic", "https://api.anthropic.com")

    def test_openai_cloud_never_strip(self):
        """OpenAI cloud services should never have thinking chains stripped."""
        assert not should_strip_thinking_for_provider("openai", "https://api.openai.com")
        assert not should_strip_thinking_for_provider("openai", "https://oai.azure.com")
        assert not should_strip_thinking_for_provider("openai", "https://openai.azure.com")

    def test_openai_self_hosted_should_strip(self):
        """Self-hosted OpenAI-compatible servers should have thinking chains stripped."""
        assert should_strip_thinking_for_provider("openai", "http://localhost:8000")
        assert should_strip_thinking_for_provider("openai", "http://192.168.1.14:1234")
        assert should_strip_thinking_for_provider("openai", "https://my-local-server.com")

    def test_ollama_should_strip(self):
        """Ollama providers should have thinking chains stripped."""
        assert should_strip_thinking_for_provider("ollama", "http://localhost:11434")
        assert should_strip_thinking_for_provider("ollama", "http://ollama-server:11434")

    def test_unknown_provider_defaults_to_false(self):
        """Unknown provider types should default to not stripping."""
        assert not should_strip_thinking_for_provider("unknown", "http://localhost:8000")
        assert not should_strip_thinking_for_provider("", "http://localhost:8000")


class TestStripThinkChainFromText:
    """Test the thinking chain removal functionality."""

    def test_qwen_format_basic(self):
        """Test basic Qwen <think>...</think> format removal."""
        text = "Here's my answer. <think>Let me think about this...</think> The result is 42."
        expected = "Here's my answer.  The result is 42."
        assert strip_think_chain_from_text(text) == expected

    def test_qwen_format_multiline(self):
        """Test Qwen format with multiline thinking blocks."""
        text = """Question: What is 2+2?

<think>
Let me calculate this step by step:
2 + 2 = 4
This is basic arithmetic.
</think>

The answer is 4."""

        expected = """Question: What is 2+2?



The answer is 4."""

        assert strip_think_chain_from_text(text) == expected

    def test_smollm_bracket_format(self):
        """Test SmolLM [think]...[/think] format removal."""
        text = "I need to solve this. [think]Working through the logic here...[/think] The solution is X."
        expected = "I need to solve this.  The solution is X."
        assert strip_think_chain_from_text(text) == expected

    def test_xml_thinking_format(self):
        """Test XML <thinking>...</thinking> format removal."""
        text = "Problem analysis: <thinking>This requires careful consideration...</thinking> Final answer: Yes."
        expected = "Problem analysis:  Final answer: Yes."
        assert strip_think_chain_from_text(text) == expected

    def test_reasoning_format(self):
        """Test <reasoning>...</reasoning> format removal."""
        text = "Let me analyze: <reasoning>Step 1: ...\nStep 2: ...</reasoning> Conclusion: Done."
        expected = "Let me analyze:  Conclusion: Done."
        assert strip_think_chain_from_text(text) == expected

    def test_multiple_thinking_blocks(self):
        """Test removal of multiple thinking blocks."""
        text = "<think>First thought</think> Some text <think>Second thought</think> More text."
        expected = " Some text  More text."
        assert strip_think_chain_from_text(text) == expected

    def test_mixed_formats(self):
        """Test removal of mixed thinking tag formats."""
        text = "Start <think>Qwen thinking</think> middle [think]SmolLM thinking[/think] end."
        expected = "Start  middle  end."
        assert strip_think_chain_from_text(text) == expected

    def test_nested_tags_not_supported(self):
        """Test that nested tags are handled gracefully (removes outer only)."""
        text = "<think>Outer <think>inner</think> back to outer</think> After."
        # Should remove from first <think> to first </think>
        expected = " back to outer</think> After."
        assert strip_think_chain_from_text(text) == expected

    def test_unclosed_tags(self):
        """Test handling of unclosed thinking tags."""
        text = "Before <think>This tag is never closed and continues to the end"
        expected = "Before "
        assert strip_think_chain_from_text(text) == expected

    def test_preserve_formatting(self):
        """Test that all whitespace and formatting is preserved."""
        text = """# Title

Paragraph one with proper spacing.

<think>
Multi-line thinking
with indentation
    and various spacing
</think>

Paragraph two after thinking.

"Dialogue here," she said."""

        expected = """# Title

Paragraph one with proper spacing.



Paragraph two after thinking.

"Dialogue here," she said."""

        assert strip_think_chain_from_text(text) == expected

    def test_no_thinking_tags(self):
        """Test that text without thinking tags is unchanged."""
        text = "This is regular text with no thinking tags at all."
        assert strip_think_chain_from_text(text) == text

    def test_empty_thinking_tags(self):
        """Test handling of empty thinking tags."""
        text = "Before <think></think> after."
        expected = "Before  after."
        assert strip_think_chain_from_text(text) == expected

    def test_provider_aware_stripping_cloud(self):
        """Test that cloud providers don't get their content stripped."""
        text = "Cloud response <think>This should not be stripped</think> from Gemini."

        # Should NOT strip for cloud providers
        result = strip_think_chain_from_text(text, "google-genai", "https://generativelanguage.googleapis.com")
        assert result == text

        result = strip_think_chain_from_text(text, "anthropic", "https://api.anthropic.com")
        assert result == text

    def test_provider_aware_stripping_self_hosted(self):
        """Test that self-hosted providers do get their content stripped."""
        text = "Local response <think>This should be stripped</think> from local model."
        expected = "Local response  from local model."

        # Should strip for self-hosted providers
        result = strip_think_chain_from_text(text, "openai", "http://localhost:8000")
        assert result == expected

        result = strip_think_chain_from_text(text, "ollama", "http://localhost:11434")
        assert result == expected

    def test_backward_compatibility_without_provider_info(self):
        """Test that function works without provider info (backward compatibility)."""
        text = "Response <think>thinking content</think> text."
        expected = "Response  text."

        # Should strip when no provider info provided (original behavior)
        assert strip_think_chain_from_text(text) == expected
        assert strip_think_chain_from_text(text, None, None) == expected

    def test_case_sensitivity(self):
        """Test that tag matching is case sensitive."""
        text = "Text <THINK>uppercase thinking</THINK> more text."
        # Should NOT strip uppercase tags (case sensitive)
        assert strip_think_chain_from_text(text) == text

    def test_partial_tag_names(self):
        """Test that partial tag names are not matched."""
        text = "Text <thinking_hard>not a real tag</thinking_hard> more text."
        # Should NOT strip tags that aren't exact matches
        assert strip_think_chain_from_text(text) == text

    def test_performance_large_text(self):
        """Test performance with large text blocks."""
        # Create a large text with thinking blocks
        large_text = (
            "Start. "
            + "Regular text. " * 1000
            + "<think>"
            + "Thinking content. " * 100
            + "</think>"
            + "End text. " * 1000
        )

        result = strip_think_chain_from_text(large_text)

        # Should still work correctly
        assert "<think>" not in result
        assert "</think>" not in result
        assert "Start." in result
        assert "End text." in result
