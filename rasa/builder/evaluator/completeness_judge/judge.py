"""Completeness Judge for evaluating response coverage of user requests.

This module provides functionality to evaluate whether a Copilot response
fully addressed all parts of a user's request by analyzing the extracted claims.
"""

import importlib
from typing import Any, Dict, List

import structlog
from jinja2 import Template

from rasa.builder.copilot.constants import ROLE_COPILOT, ROLE_COPILOT_INTERNAL
from rasa.builder.copilot.models import (
    ChatMessage,
    CopilotChatMessage,
    InternalCopilotRequestChatMessage,
    UserChatMessage,
)
from rasa.builder.evaluator.completeness_judge.models import (
    CompletenessJudgeInput,
    CompletenessJudgeResult,
)
from rasa.builder.evaluator.shared.base_judge import BaseLLMJudge
from rasa.builder.evaluator.shared.constants import (
    COMPLETENESS_JUDGE_PROMPTS_FILE,
    COMPLETENESS_JUDGE_PROMPTS_PACKAGE_NAME,
    COMPLETENESS_JUDGE_RESPONSE_SCHEMA_PATH,
)
from rasa.shared.constants import PACKAGE_NAME, ROLE_USER
from rasa.shared.utils.io import read_json_file

structlogger = structlog.get_logger()


class CompletenessJudge(BaseLLMJudge[CompletenessJudgeInput, CompletenessJudgeResult]):
    """Evaluates completeness of Copilot responses against user requests.

    This class uses an LLM to break down the user request into atomic parts
    and determine which claims address each part, enabling calculation of
    completeness scores.
    """

    def _get_prompt_template(self) -> Template:
        """Get the Jinja2 template for completeness evaluation prompts.

        Returns:
            Jinja2 Template instance for completeness judge
        """
        return Template(
            importlib.resources.read_text(
                f"{PACKAGE_NAME}.{COMPLETENESS_JUDGE_PROMPTS_PACKAGE_NAME}",
                COMPLETENESS_JUDGE_PROMPTS_FILE,
            )
        )

    def _get_response_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for completeness judge responses.

        Returns:
            Dictionary containing the JSON schema
        """
        return read_json_file(COMPLETENESS_JUDGE_RESPONSE_SCHEMA_PATH)

    def _render_prompt(self, input: CompletenessJudgeInput) -> str:
        """Render the prompt for the completeness judge.

        Args:
            input: The input containing user message, chat history, and claims

        Returns:
            The rendered prompt
        """
        # Format claims with their IDs for the prompt
        claims_json = input.claims.model_dump_json(indent=2)

        # Format chat history for the prompt
        chat_history_text = self._format_chat_history(input.chat_history)

        # Render the prompt
        prompt = self._prompt_template.render(
            user_message=input.user_message,
            chat_history=chat_history_text,
            claims=claims_json,
        )
        return prompt

    def _format_chat_history(self, chat_history: List[ChatMessage]) -> str:
        """Format chat history into a readable string for the prompt.

        Includes User, Copilot, and internal Copilot messages.

        Args:
            chat_history: List of ChatMessage objects

        Returns:
            Formatted string representation of chat history
        """
        role_to_prefix = {
            ROLE_USER: "User",
            ROLE_COPILOT: "Copilot",
            ROLE_COPILOT_INTERNAL: "Copilot (Internal)",
        }

        # Filter to include User, Copilot, and internal Copilot messages
        filtered_messages = [
            message
            for message in chat_history
            if message.role in [ROLE_USER, ROLE_COPILOT, ROLE_COPILOT_INTERNAL]
        ]

        if not filtered_messages:
            return "No previous conversation history."

        formatted_messages: List[str] = []
        for chat_message in filtered_messages:
            if isinstance(chat_message, UserChatMessage):
                role = role_to_prefix[ROLE_USER]
                content = chat_message.get_flattened_text_content()
            elif isinstance(chat_message, CopilotChatMessage):
                role = role_to_prefix[ROLE_COPILOT]
                content = chat_message.get_flattened_text_content()
            elif isinstance(chat_message, InternalCopilotRequestChatMessage):
                role = role_to_prefix[ROLE_COPILOT_INTERNAL]
                text_content = chat_message.get_flattened_text_content()
                log_content = chat_message.get_flattened_log_content()
                if text_content and log_content:
                    content = f"{text_content}\nLogs: {log_content}"
                elif text_content:
                    content = text_content
                elif log_content:
                    content = f"Logs: {log_content}"
                else:
                    continue
            else:
                continue

            formatted_messages.append(f"{role}: {content}")

        return "\n".join(formatted_messages)

    def _parse_result(self, response_data: Dict[str, Any]) -> CompletenessJudgeResult:
        """Parse the LLM response into a CompletenessJudgeResult.

        Args:
            response_data: The parsed JSON response from the LLM in the form
                of the CompletenessJudgeResult schema

        Returns:
            CompletenessJudgeResult
        """
        result = CompletenessJudgeResult.model_validate(response_data)

        structlogger.info(
            "evaluator.completeness_judge.parse_result.success",
            total_parts=len(result.verdicts),
            covered_parts=len(result.covered_parts),
            uncovered_parts=len(result.uncovered_parts),
            confidence=result.confidence,
        )

        return result
