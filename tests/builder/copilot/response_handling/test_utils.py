"""Unit tests for response handling utility functions."""

import pytest

from rasa.builder.copilot.response_handling.utils import (
    remove_prefix,
    remove_prefix_and_suffix,
    remove_suffix,
)


@pytest.mark.parametrize(
    "content,expected_content,expected_prefix",
    [
        # No prefix
        ("Normal content", "Normal content", None),
        ("", "", None),
        # Triple backtick prefix
        ("```code content", "code content", "```"),
        (
            "```content with newlines\nand more",
            "content with newlines\nand more",
            "```",
        ),
        # Triple backtick markdown prefix
        # Note: "```markdown" is checked before "```" due to correct iteration order
        (
            "```markdown\ncontent",
            "\ncontent",
            "```markdown",
        ),
        ("```markdowncontent", "content", "```markdown"),
        # Triple quote prefix
        ('"""quoted content', "quoted content", '"""'),
        ('"""content with quotes"', 'content with quotes"', '"""'),
        # Prefix at start only
        # Note: "```" at the start will be matched and removed
        ("```prefix but not at start", "prefix but not at start", "```"),
        ("Some text ```at end", "Some text ```at end", None),
        # Multiple prefix matches (should match first one found)
        # Note: "```markdown" is checked before "```" due to correct iteration order
        ("```markdown```", "```", "```markdown"),
    ],
)
def test_remove_prefix(
    content: str,
    expected_content: str,
    expected_prefix: str | None,
):
    """Test remove_prefix with various content scenarios.

    Args:
        content: Input content to test
        expected_content: Expected content after prefix removal
        expected_prefix: Expected prefix that was found and removed
    """
    result_content, result_prefix = remove_prefix(content)

    assert result_content == expected_content
    assert result_prefix == expected_prefix


def test_remove_prefix_markdown_order():
    """Test that "```markdown" prefix is matched before "```" prefix.

    This test verifies that the more specific prefix "```markdown" is checked
    before the general prefix "```" in the dictionary iteration order.
    Without the correct order, "```markdown\nHello```" would only remove "```",
    leaving "markdown\nHello```" instead of properly removing the full prefix.
    """
    content = "```markdown\nHello```"
    result_content, result_prefix = remove_prefix(content)

    # Should match the more specific "```markdown" prefix, not just "```"
    assert result_prefix == "```markdown"
    assert result_content == "\nHello```"

    # Verify that regular "```" prefix still works for non-markdown content
    content2 = "```code content"
    result_content2, result_prefix2 = remove_prefix(content2)
    assert result_prefix2 == "```"
    assert result_content2 == "code content"


@pytest.mark.parametrize(
    "content,expected_content,expected_suffix",
    [
        # No suffix
        ("Normal content", "Normal content", None),
        ("", "", None),
        # Triple backtick suffix
        ("code content```", "code content", "```"),
        (
            "content with newlines\nand more```",
            "content with newlines\nand more",
            "```",
        ),
        # Triple quote suffix
        ('quoted content"""', "quoted content", '"""'),
        ('"quoted content"""', '"quoted content', '"""'),
        # Suffix at end only
        ("```suffix but not at end", "```suffix but not at end", None),
        ("Some text at start```", "Some text at start", "```"),
        # Multiple suffix matches (should match first one found)
        ("```markdown```", "```markdown", "```"),
    ],
)
def test_remove_suffix(
    content: str,
    expected_content: str,
    expected_suffix: str | None,
):
    """Test remove_suffix with various content scenarios.

    Args:
        content: Input content to test
        expected_content: Expected content after suffix removal
        expected_suffix: Expected suffix that was found and removed
    """
    result_content, result_suffix = remove_suffix(content)

    assert result_content == expected_content
    assert result_suffix == expected_suffix


@pytest.mark.parametrize(
    "content,expected_content,expected_prefix,expected_suffix",
    [
        # No prefix or suffix
        ("Normal content", "Normal content", None, None),
        ("", "", None, None),
        # Prefix only
        ("```content", "content", "```", None),
        ('"""content', "content", '"""', None),
        # Suffix only
        ("content```", "content", None, "```"),
        ('content"""', "content", None, '"""'),
        # Both prefix and suffix
        ("```content```", "content", "```", "```"),
        ('"""content"""', "content", '"""', '"""'),
        # Note: "```markdown" is checked before "```" for prefix due to correct
        # iteration order
        ("```markdown\ncontent\n```", "\ncontent\n", "```markdown", "```"),
        # Prefix and suffix with matching pairs
        ("```code```", "code", "```", "```"),
        ('"""text"""', "text", '"""', '"""'),
        # Prefix and suffix with different types
        ('```content"""', "content", "```", '"""'),
        ('"""content```', "content", '"""', "```"),
        # Multiple occurrences (should only remove first prefix)
        # Prefix "```" matches first, but suffix "```" doesn't match at the end
        # because content ends with "suffix", not "```"
        ("```prefix```middle```suffix", "prefix```middle```suffix", "```", None),
    ],
)
def test_remove_prefix_and_suffix(
    content: str,
    expected_content: str,
    expected_prefix: str | None,
    expected_suffix: str | None,
):
    """Test remove_prefix_and_suffix with various content scenarios."""
    result_content, result_prefix, result_suffix = remove_prefix_and_suffix(content)

    assert result_content == expected_content
    assert result_prefix == expected_prefix
    assert result_suffix == expected_suffix
