from rasa.builder.telemetry.langfuse_compat import with_langfuse


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
        with with_langfuse() as lf:
            if not lf:
                return
            langfuse_client = lf.get_client()
            langfuse_client.update_current_span(
                input={
                    "diff_output": diff_output,
                    "detailed_diff": detailed_diff,
                    "prompt": prompt,
                }
            )

    @staticmethod
    def update_commit_message_generation_output(
        response_content: str,
        commit_message: str,
    ) -> None:
        """Update the current Langfuse span with commit message generation output.

        Args:
            response_content: The response content from the LLM.
            commit_message: The cleaned and validated commit message.
        """
        with with_langfuse() as lf:
            if not lf:
                return
            langfuse_client = lf.get_client()
            langfuse_client.update_current_span(
                output={
                    "response_content": response_content,
                    "commit_message": commit_message,
                }
            )
