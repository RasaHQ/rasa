import langfuse


class CommitMessageGenerationLangfuseTelemetry:
    """Telemetry utilities for commit generation traces."""

    @staticmethod
    def update_commit_message_generation_input(
        diff_output: str,
        detailed_diff: str,
        prompt: str,
    ) -> None:
        """Update the current Langfuse span with commit message generation input.

        Args:
            diff_output: The git diff output showing file changes.
            detailed_diff: The detailed git diff with line-by-line changes.
            prompt: The full prompt sent to the LLM.
        """
        langfuse_client = langfuse.get_client()
        langfuse_client.update_current_span(
            input={
                "diff_output": diff_output,
                "detailed_diff": detailed_diff,
                "prompt": prompt,
            }
        )

    @staticmethod
    def update_commit_message_generation_output(
        raw_response: str,
        commit_message: str,
    ) -> None:
        """Update the current Langfuse span with commit message generation output.

        Args:
            raw_response: The raw response from the LLM.
            commit_message: The cleaned and validated commit message.
        """
        langfuse_client = langfuse.get_client()
        langfuse_client.update_current_span(
            output={
                "raw_response": raw_response,
                "commit_message": commit_message,
            }
        )
