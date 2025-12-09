"""Tests for general Langfuse telemetry utilities."""

from unittest.mock import MagicMock, patch

import pytest

from rasa.builder.telemetry.commit_langfuse_telemetry import (
    CommitMessageGenerationLangfuseTelemetry,
)


class TestCommitMessageGenerationLangfuseTelemetry:
    """Test cases for CommitMessageGenerationLangfuseTelemetry class."""

    @pytest.fixture
    def mock_langfuse_client(self):
        """Create a mock Langfuse client."""
        mock_client = MagicMock()
        mock_client.update_current_span = MagicMock()
        return mock_client

    @patch("langfuse.get_client")
    def test_update_commit_message_generation_input(
        self, mock_get_client, mock_langfuse_client
    ):
        """Test updating commit message generation input."""
        mock_get_client.return_value = mock_langfuse_client
        diff_output = "M domain/general/hello.yml"
        detailed_diff = """diff --git a/domain/general/hello.yml
 b/domain/general/hello.yml
index 4496aeb..6c36ccf 100644
--- a/domain/general/hello.yml
+++ b/domain/general/hello.yml
@@ -3,3 +3,3 @@ version: "3.1"
responses:
utter_hello:
- - text: "Hello there! Nice to meet you!"
+ - text: "Hello ! Nice to meet you!\""""
        prompt = """Generate a concise, descriptive Git commit message
for the following changes to a Rasa chatbot project.
The commit message should:
- Be in imperative mood (e.g., 'Add', 'Update', 'Fix', 'Remove')
- Be specific about what changed
- Be under 36 characters
- Focus on the most significant changes

File changes:
M domain/general/hello.yml

Detailed diff:
diff --git a/domain/general/hello.yml b/domain/general/hello.yml
index 4496aeb..6c36ccf 100644
--- a/domain/general/hello.yml
+++ b/domain/general/hello.yml
@@ -3,3 +3,3 @@ version: "3.1"
responses:
utter_hello:
- - text: "Hello there! Nice to meet you!"
+ - text: "Hello ! Nice to meet you!"

Generate only the commit message, nothing else:"""

        CommitMessageGenerationLangfuseTelemetry.update_commit_message_generation_input(
            diff_output=diff_output,
            detailed_diff=detailed_diff,
            prompt=prompt,
        )

        mock_langfuse_client.update_current_span.assert_called_once_with(
            input={
                "diff_output": diff_output,
                "detailed_diff": detailed_diff,
                "prompt": prompt,
            }
        )

    @patch("langfuse.get_client")
    def test_update_commit_message_generation_output(
        self, mock_get_client, mock_langfuse_client
    ):
        """Test updating commit message generation output."""
        mock_get_client.return_value = mock_langfuse_client
        response_content = "Update domain configuration\n"
        commit_message = "Update domain configuration"

        CommitMessageGenerationLangfuseTelemetry.update_commit_message_generation_output(
            response_content=response_content,
            commit_message=commit_message,
        )

        mock_langfuse_client.update_current_span.assert_called_once_with(
            output={
                "response_content": response_content,
                "commit_message": commit_message,
            }
        )
